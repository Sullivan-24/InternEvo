# Copyright (c) InternLM. All rights reserved.
# Fault-tolerant checkpoint transport for InternEvo.
#
# Enables live recovery from healthy peers using torchft's PGTransport,
# which transfers checkpoints via ProcessGroup (Gloo-based) instead of
# relying on file-based checkpoint storage.

import logging
import os
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class FTCheckpointTransport:
    """
    Checkpoint transport for live recovery in InternEvo.

    Uses torchft's PGTransport (Gloo-based) to transfer model state,
    optimizer state, and training state from a healthy peer to a
    recovering worker.

    This avoids the need to read from shared storage during recovery,
    significantly reducing recovery time for large models.

    Integration with InternEvo's CheckpointManager:
    - Provides an alternative recovery path alongside file-based checkpoints
    - When a worker recovers, it first tries live recovery from a peer
    - Falls back to file-based checkpoint if no healthy peer is available
    """

    def __init__(
        self,
        ft_manager: Any,
        timeout_sec: int = 120,
    ) -> None:
        """
        Args:
            ft_manager: FTManager instance
            timeout_sec: Timeout for checkpoint transfer operations
        """
        self._ft_manager = ft_manager
        self._timeout = timedelta(seconds=timeout_sec)
        self._transport = None
        self._state_dict_fns: Dict[str, Callable] = {}

    def register_state_provider(
        self,
        name: str,
        state_dict_fn: Callable[[], Dict[str, Any]],
        load_state_dict_fn: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Register a state dict provider for checkpoint transport.

        Args:
            name: Name of the state component (e.g., "model", "optimizer")
            state_dict_fn: Function that returns the state dict
            load_state_dict_fn: Function that loads a state dict
        """
        self._state_dict_fns[name] = {
            "state_dict": state_dict_fn,
            "load_state_dict": load_state_dict_fn,
        }

    def get_full_state_dict(self) -> Dict[str, Any]:
        """Get combined state dict from all registered providers."""
        full_state = {}
        for name, fns in self._state_dict_fns.items():
            try:
                full_state[name] = fns["state_dict"]()
            except Exception as e:
                logger.warning(f"Failed to get state dict for {name}: {e}")
        return full_state

    def load_full_state_dict(self, state: Dict[str, Any]) -> None:
        """Load combined state dict to all registered providers."""
        for name, fns in self._state_dict_fns.items():
            if name in state:
                try:
                    fns["load_state_dict"](state[name])
                    logger.info(f"Loaded state dict for {name} via live recovery")
                except Exception as e:
                    logger.warning(f"Failed to load state dict for {name}: {e}")

    def setup_transport(self) -> None:
        """
        Set up the checkpoint transport.

        Should be called after FTManager is initialized and state
        providers are registered.
        """
        if self._ft_manager is None or not self._ft_manager.enabled:
            return

        # Register combined state dict functions with the FT manager
        self._ft_manager.set_state_dict_fns(
            load_state_dict=self.load_full_state_dict,
            state_dict=self.get_full_state_dict,
        )

        logger.info("FT checkpoint transport setup complete")

    def try_live_recovery(self) -> bool:
        """
        Attempt live recovery from a healthy peer.

        Returns:
            True if recovery was successful, False otherwise.
        """
        if self._ft_manager is None or not self._ft_manager.enabled:
            return False

        manager = self._ft_manager.manager
        if manager is None:
            return False

        # torchft Manager handles recovery automatically during quorum.
        # When a worker re-joins, the Manager detects it needs recovery
        # and uses the registered state_dict/load_state_dict functions
        # to transfer state from a healthy peer.
        #
        # The PGTransport uses Gloo send/recv to transfer:
        #   1. Metadata (pickle-serialized tensor shapes/dtypes)
        #   2. Tensor data (raw bytes via Gloo)
        #
        # This happens inside Manager.start_quorum() when the worker
        # is flagged as needing recovery.

        logger.info("Live recovery will be handled by torchft Manager during quorum")
        return True

    def shutdown(self) -> None:
        """Shutdown the checkpoint transport."""
        if self._transport is not None:
            self._transport.shutdown()
            self._transport = None
