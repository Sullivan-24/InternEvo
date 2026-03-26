# Copyright (c) InternLM. All rights reserved.
# Fault-tolerant ProcessGroup integration for InternEvo.
#
# Wraps torchft's ProcessGroup implementations to provide:
# - Reconfigurable process groups that can be rebuilt after failures
# - Graceful error reporting instead of hard crashes
# - Gloo-based fallback for resilience

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class FTProcessGroupManager:
    """
    Manages fault-tolerant process groups for InternEvo.

    This wraps torchft's ProcessGroupGloo to provide reconfigurable
    process groups that can handle worker failures gracefully.

    Integration with InternEvo's ParallelMode system:
    - The DATA parallel group is replaced with a fault-tolerant version
    - Other groups (PIPELINE, TENSOR) remain as-is since they represent
      intra-replica communication
    - The FT group handles cross-replica coordination
    """

    def __init__(self, timeout_sec: int = 60) -> None:
        self._timeout = timedelta(seconds=timeout_sec)
        self._ft_groups: Dict[str, Any] = {}
        self._original_groups: Dict[str, Any] = {}
        self._initialized = False

    def create_ft_data_group(
        self,
        ranks: List[int],
        backend: str = "gloo",
    ) -> Any:
        """
        Create a fault-tolerant process group for data parallelism.

        This group can be reconfigured after worker failures without
        requiring a full restart.

        Args:
            ranks: List of global ranks in this group
            backend: Communication backend (gloo recommended for FT)

        Returns:
            Fault-tolerant process group
        """
        try:
            from torchft.process_group import ProcessGroupGloo

            ft_pg = ProcessGroupGloo(timeout=self._timeout)
            self._ft_groups["data"] = ft_pg
            logger.info(
                f"Created FT data process group with Gloo backend, "
                f"ranks={ranks}, timeout={self._timeout}"
            )
            return ft_pg
        except ImportError:
            logger.warning(
                "torchft ProcessGroupGloo not available. "
                "Using standard dist.new_group as fallback."
            )
            pg = dist.new_group(ranks)
            self._ft_groups["data"] = pg
            return pg

    def get_ft_group(self, name: str) -> Optional[Any]:
        """Get a fault-tolerant process group by name."""
        return self._ft_groups.get(name)

    def reconfigure_groups(self) -> None:
        """
        Reconfigure all fault-tolerant process groups after a failure.

        This is called by the FTManager when quorum changes.
        torchft's ProcessGroupGloo handles the reconfiguration internally
        through the Manager's quorum mechanism.
        """
        for name, pg in self._ft_groups.items():
            if hasattr(pg, "configure"):
                logger.info(f"Reconfiguring FT process group: {name}")
                # torchft's ProcessGroup.configure() is called automatically
                # by the Manager during quorum
            else:
                logger.warning(
                    f"Process group {name} does not support reconfiguration"
                )

    def shutdown(self) -> None:
        """Shutdown all fault-tolerant process groups."""
        for name, pg in self._ft_groups.items():
            if hasattr(pg, "shutdown"):
                pg.shutdown()
        self._ft_groups.clear()


def create_ft_process_group(
    ranks: List[int],
    backend: str = "gloo",
    timeout_sec: int = 60,
) -> Any:
    """
    Convenience function to create a fault-tolerant process group.

    Args:
        ranks: List of global ranks
        backend: Communication backend
        timeout_sec: Operation timeout in seconds

    Returns:
        A fault-tolerant process group
    """
    try:
        from torchft.process_group import ProcessGroupGloo

        return ProcessGroupGloo(timeout=timedelta(seconds=timeout_sec))
    except ImportError:
        logger.warning("torchft not available, using standard process group")
        return dist.new_group(ranks)
