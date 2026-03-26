# Copyright (c) InternLM. All rights reserved.
# Fault-tolerant gradient handler for InternEvo.
#
# Wraps the allreduce gradient operations with fault tolerance,
# using torchft's Manager.allreduce for cross-replica gradient sync.

import logging
from collections import defaultdict

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from internlm.core.context import global_context as gpc
from internlm.core.gradient_handler import BaseGradientHandler
from internlm.utils.common import get_current_device

logger = logging.getLogger(__name__)


class FTGradientHandler(BaseGradientHandler):
    """
    Fault-tolerant gradient handler for data parallel allreduce.

    Uses torchft's Manager.allreduce for cross-replica gradient
    synchronization, which gracefully handles worker failures
    instead of crashing the entire training job.

    If a worker fails during allreduce:
    - The error is tracked by the FT Manager
    - The step will not be committed
    - The failed worker will be reconfigured on the next quorum
    """

    def handle_gradient(self):
        """
        Perform fault-tolerant gradient allreduce across data parallel groups.
        """
        from internlm.ft.manager import get_ft_manager

        ft_mgr = get_ft_manager()
        if ft_mgr is None or not ft_mgr.enabled:
            # Fall back to standard allreduce
            self._standard_handle_gradient()
            return

        if ft_mgr.errored():
            # Skip gradient sync if already errored
            logger.warning("FT: Skipping gradient sync due to prior error")
            return

        try:
            # Bucketize gradients by type for efficient communication
            buckets = defaultdict(list)
            for param in self._model.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    buckets[tp].append(param)

            for tp, bucket in buckets.items():
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads).to(get_current_device())

                # Use FT manager's allreduce (fault-tolerant)
                work = ft_mgr.allreduce(coalesced)
                if work is not None:
                    work.wait()

                for buf, synced in zip(
                    grads, _unflatten_dense_tensors(coalesced, grads)
                ):
                    buf.copy_(synced)

        except Exception as e:
            logger.error(f"FT gradient allreduce failed: {e}")
            ft_mgr.report_error(e)

    def _standard_handle_gradient(self):
        """Standard gradient allreduce (non-FT fallback)."""
        if gpc.get_world_size(gpc.ParallelMode.DATA) > 1:
            buckets = defaultdict(list)
            for param in self._model.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    buckets[tp].append(param)

            for tp, bucket in buckets.items():
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads).to(get_current_device())
                dist.all_reduce(coalesced, op=dist.ReduceOp.AVG)
                for buf, synced in zip(
                    grads, _unflatten_dense_tensors(coalesced, grads)
                ):
                    buf.copy_(synced)
