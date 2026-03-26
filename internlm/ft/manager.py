# Copyright (c) InternLM. All rights reserved.
# Fault Tolerance Manager - wraps torchft Manager for InternEvo's training loop.
#
# Provides per-step heartbeat-based health detection, quorum management,
# and coordination across replica groups.

import logging
import os
import socket
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Global FT manager instance
_ft_manager: Optional["FTManager"] = None


def is_ft_enabled() -> bool:
    """Check if fault tolerance is enabled."""
    return _ft_manager is not None


def get_ft_manager() -> Optional["FTManager"]:
    """Get the global FT manager instance."""
    return _ft_manager


class FTManager:
    """
    Fault Tolerance Manager for InternEvo.

    Wraps torchft's Manager to provide:
    - Per-step heartbeat-based health detection via Lighthouse
    - Quorum management across replica groups
    - Automatic recovery coordination when workers fail
    - State dict save/load for live recovery

    Usage in training loop:
        ft_mgr = FTManager(config)
        ft_mgr.initialize(rank, world_size)

        for step in training_loop:
            if not ft_mgr.step_begin():
                # This step should be skipped (recovering)
                continue
            ... forward / backward ...
            if not ft_mgr.step_end():
                # Commit failed, step should be retried
                continue
    """

    def __init__(self, ft_config: Dict[str, Any]) -> None:
        """
        Args:
            ft_config: Fault tolerance configuration dict with keys:
                - enabled: bool, whether FT is enabled
                - lighthouse_addr: str, address of the Lighthouse server
                - min_replicas: int, minimum number of healthy replicas per step
                - replica_id: str, identifier for this replica group
                - heartbeat_interval_ms: int, heartbeat interval in milliseconds
                - timeout_sec: int, timeout for operations in seconds
                - quorum_timeout_sec: int, timeout for quorum in seconds
                - use_async_quorum: bool, whether to use async quorum
                - init_sync: bool, whether to sync weights on step 0
        """
        self._config = ft_config
        self._enabled = ft_config.get("enabled", False)
        self._manager = None  # torchft Manager instance
        self._ft_pg = None  # torchft ProcessGroup
        self._initialized = False
        self._step_count = 0
        self._is_healthy = True
        self._load_state_dict_fn = None
        self._state_dict_fn = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    @property
    def manager(self):
        """Access the underlying torchft Manager."""
        return self._manager

    @property
    def step_count(self) -> int:
        return self._step_count

    def set_state_dict_fns(
        self,
        load_state_dict: Callable[[Dict[str, Any]], None],
        state_dict: Callable[[], Dict[str, Any]],
    ) -> None:
        """Register state dict functions for live recovery."""
        self._load_state_dict_fn = load_state_dict
        self._state_dict_fn = state_dict
        if self._manager is not None:
            self._manager.register_state_dict_fn(
                "internevo", load_state_dict, state_dict
            )

    def initialize(
        self,
        rank: int,
        world_size: int,
        store_addr: Optional[str] = None,
        store_port: Optional[int] = None,
    ) -> None:
        """
        Initialize the fault tolerance manager.

        Must be called after torch.distributed is initialized.

        Args:
            rank: Local rank within replica group
            world_size: World size of replica group
            store_addr: TCPStore address (defaults to MASTER_ADDR)
            store_port: TCPStore port (defaults to MASTER_PORT)
        """
        if not self._enabled:
            logger.info("Fault tolerance is disabled, skipping initialization.")
            return

        try:
            from torchft.manager import Manager, WorldSizeMode
            from torchft.process_group import ProcessGroupGloo
        except ImportError:
            logger.error(
                "torchft is not installed. Install with: pip install torchft-nightly"
            )
            self._enabled = False
            return

        lighthouse_addr = self._config.get("lighthouse_addr", "")
        if not lighthouse_addr:
            lighthouse_addr = os.environ.get("TORCHFT_LIGHTHOUSE", "")
        if not lighthouse_addr:
            logger.error("No lighthouse address provided. Disabling fault tolerance.")
            self._enabled = False
            return

        min_replicas = self._config.get("min_replicas", 1)
        replica_id = self._config.get("replica_id", "")
        heartbeat_interval_ms = self._config.get("heartbeat_interval_ms", 100)
        timeout_sec = self._config.get("timeout_sec", 60)
        quorum_timeout_sec = self._config.get("quorum_timeout_sec", 60)
        connect_timeout_sec = self._config.get("connect_timeout_sec", 60)
        use_async_quorum = self._config.get("use_async_quorum", True)
        init_sync = self._config.get("init_sync", True)

        # Create fault-tolerant ProcessGroup (Gloo-based for resilience)
        self._ft_pg = ProcessGroupGloo(timeout=timedelta(seconds=timeout_sec))

        # Create checkpoint transport
        checkpoint_transport = self._create_checkpoint_transport()

        # Create the Manager
        self._manager = Manager(
            pg=self._ft_pg,
            load_state_dict=self._load_state_dict_fn,
            state_dict=self._state_dict_fn,
            min_replica_size=min_replicas,
            use_async_quorum=use_async_quorum,
            timeout=timedelta(seconds=timeout_sec),
            quorum_timeout=timedelta(seconds=quorum_timeout_sec),
            connect_timeout=timedelta(seconds=connect_timeout_sec),
            rank=rank,
            world_size=world_size,
            store_addr=store_addr,
            store_port=store_port,
            lighthouse_addr=lighthouse_addr,
            replica_id=replica_id,
            hostname=socket.gethostname(),
            heartbeat_interval=timedelta(milliseconds=heartbeat_interval_ms),
            checkpoint_transport=checkpoint_transport,
            init_sync=init_sync,
        )

        self._initialized = True

        # Set as global instance
        global _ft_manager
        _ft_manager = self

        logger.info(
            f"FTManager initialized: rank={rank}, world_size={world_size}, "
            f"lighthouse={lighthouse_addr}, replica_id={replica_id}, "
            f"min_replicas={min_replicas}"
        )

    def _create_checkpoint_transport(self):
        """Create the checkpoint transport based on config."""
        transport_type = self._config.get("checkpoint_transport", "pg")
        timeout_sec = self._config.get("timeout_sec", 60)

        if transport_type == "pg":
            try:
                from torchft.checkpointing.pg_transport import PGTransport

                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return PGTransport(
                    pg=self._ft_pg,
                    timeout=timedelta(seconds=timeout_sec),
                    device=device,
                )
            except ImportError:
                logger.warning(
                    "PGTransport not available, falling back to HTTPTransport"
                )

        # Default: HTTP transport
        from torchft.checkpointing import HTTPTransport

        return HTTPTransport(
            timeout=timedelta(seconds=timeout_sec),
            num_chunks=0,
        )

    def step_begin(self) -> bool:
        """
        Called at the beginning of each training step.

        Performs quorum check and determines if this worker should participate.

        Returns:
            True if this worker should proceed with the step,
            False if it should skip (e.g., during recovery).
        """
        if not self._enabled or not self._initialized:
            return True

        try:
            # Start quorum - this checks lighthouse for healthy workers
            # and triggers recovery if needed
            self._manager.start_quorum()
            self._is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Quorum check failed at step {self._step_count}: {e}")
            self._is_healthy = False
            return False

    def step_end(self) -> bool:
        """
        Called at the end of each training step.

        Commits the step result and checks if the step was successful.

        Returns:
            True if the step was committed successfully,
            False if it should be retried.
        """
        if not self._enabled or not self._initialized:
            self._step_count += 1
            return True

        try:
            should_commit = self._manager.should_commit()
            if should_commit:
                self._step_count += 1
                return True
            else:
                logger.warning(
                    f"Step {self._step_count} not committed, will retry."
                )
                return False
        except Exception as e:
            logger.warning(f"Commit check failed at step {self._step_count}: {e}")
            return False

    def report_error(self, e: Exception) -> None:
        """Report an error to the manager for fault tracking."""
        if self._enabled and self._initialized:
            self._manager.report_error(e)
            self._is_healthy = False

    def num_participants(self) -> int:
        """Get the number of participating replicas in current quorum."""
        if not self._enabled or not self._initialized:
            return 1
        return self._manager.num_participants()

    def is_participating(self) -> bool:
        """Check if this worker is participating in current step."""
        if not self._enabled or not self._initialized:
            return True
        return self._manager.is_participating()

    def errored(self) -> bool:
        """Check if an error has been reported."""
        if not self._enabled or not self._initialized:
            return False
        return self._manager.errored()

    def wrap_future(self, future):
        """Wrap a future with fault-tolerant error handling."""
        if not self._enabled or not self._initialized:
            return future
        return self._manager.wrap_future(future)

    def allreduce(self, tensor: torch.Tensor) -> Any:
        """
        Fault-tolerant allreduce across replica groups.

        This is used for cross-replica gradient synchronization.
        If an error occurs, it's tracked and the step will be retried.
        """
        if not self._enabled or not self._initialized:
            return None
        return self._manager.allreduce(tensor)

    def get_ft_process_group(self):
        """Get the fault-tolerant ProcessGroup."""
        return self._ft_pg

    def shutdown(self) -> None:
        """Shutdown the FT manager."""
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._initialized = False

        global _ft_manager
        if _ft_manager is self:
            _ft_manager = None

        logger.info("FTManager shutdown complete.")

    def state_dict(self) -> Dict[str, Any]:
        """Get manager state dict for checkpointing."""
        state = {"step_count": self._step_count}
        if self._manager is not None:
            state["manager_state"] = self._manager.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load manager state from checkpoint."""
        self._step_count = state.get("step_count", 0)
        if self._manager is not None and "manager_state" in state:
            self._manager.load_state_dict(state["manager_state"])
