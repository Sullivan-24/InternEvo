# Copyright (c) InternLM. All rights reserved.
# FT-aware training loop integration for InternEvo TrainerBuilder.
#
# This file provides the `ft_fit` method which replaces TrainerBuilder.fit()
# to add per-step fault tolerance via torchft's Manager.

import gc
import logging
import time

import torch

from internlm.core.context import global_context as gpc
from internlm.utils.gputest import empty_cache_and_diag

logger = logging.getLogger(__name__)


def ft_fit(trainer_builder):
    """
    Fault-tolerant training loop replacement for TrainerBuilder.fit().

    Wraps each training step with FT manager's step_begin/step_end
    for per-step health detection and recovery coordination.

    Args:
        trainer_builder: TrainerBuilder instance (self)
    """
    from internlm.ft.manager import get_ft_manager

    ft_mgr = get_ft_manager()

    if ft_mgr is None or not ft_mgr.enabled:
        # Fall back to original fit
        logger.info("FT not enabled, using original training loop")
        trainer_builder._original_fit()
        return

    logger.info("Starting fault-tolerant training loop")

    # Register state dict functions for live recovery
    _register_state_dicts(trainer_builder, ft_mgr)

    do_next = True
    if gpc.config.get("FAILURE", False):
        if gpc.get_global_rank() in gpc.config.get("FAILURE_GLOBAL_RANKS", []):
            do_next = False

    if do_next:
        trainer_builder.train()
        train_iter = iter(trainer_builder.train_dl)

        from internlm.train.pipeline import initialize_llm_profile

        with initialize_llm_profile(
            profiling=trainer_builder.profiling, start_time=trainer_builder.current_time
        ) as prof:
            gc.disable()
            for batch_count in range(
                trainer_builder.train_state.batch_count, gpc.config.data.total_steps
            ):
                # === FT Step Begin: Quorum check and health detection ===
                if not ft_mgr.step_begin():
                    logger.warning(
                        f"FT: Skipping step {batch_count} (recovering/unhealthy)"
                    )
                    continue

                try:
                    if trainer_builder._process_batch(batch_count, train_iter, prof):
                        break
                except Exception as e:
                    # Report error to FT manager for tracking
                    logger.error(f"FT: Error in step {batch_count}: {e}")
                    ft_mgr.report_error(e)
                    continue

                # === FT Step End: Commit check ===
                if not ft_mgr.step_end():
                    logger.warning(
                        f"FT: Step {batch_count} not committed, will retry"
                    )
                    # Step was not committed - the next iteration will retry
                    continue

    trainer_builder.ckpt_manager.wait_async_upload_finish()
    logger.info("Fault-tolerant training loop completed")


def _register_state_dicts(trainer_builder, ft_mgr):
    """Register model/optimizer/train_state for live recovery."""

    def state_dict():
        """Collect full training state for checkpoint transport."""
        state = {
            "model": _get_model_state(trainer_builder),
            "train_state": trainer_builder.train_state.state_dict(),
        }
        # Include optimizer state if available
        if hasattr(trainer_builder, "optimizer") and trainer_builder.optimizer is not None:
            try:
                state["optimizer"] = trainer_builder.optimizer.state_dict()
            except Exception as e:
                logger.warning(f"Failed to get optimizer state dict: {e}")
        return state

    def load_state_dict(state):
        """Load full training state from live recovery."""
        if "model" in state:
            _load_model_state(trainer_builder, state["model"])
        if "train_state" in state:
            trainer_builder.train_state.load_state_dict(state["train_state"])
        if "optimizer" in state:
            try:
                trainer_builder.optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state dict: {e}")

    ft_mgr.set_state_dict_fns(load_state_dict, state_dict)


def _get_model_state(trainer_builder):
    """Get model state dict."""
    model = trainer_builder.engine.model
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _load_model_state(trainer_builder, state):
    """Load model state dict."""
    model = trainer_builder.engine.model
    if hasattr(model, "module"):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
