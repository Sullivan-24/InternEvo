import gc
import logging
import signal
import time
import os
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

try:
    import redis
except ImportError:
    redis = None

from internlm.checkpoint.checkpoint_manager import CheckpointManager
from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode
from internlm.core.parallel.comm import initialize_offload_manager
from internlm.core.trainer import Trainer
from internlm.data.streaming.utils import streaming_simple_resume
from internlm.data.train_state import get_train_state
from internlm.eval.evaluation import evaluate_on_val_dls
from internlm.initialize.initialize_trainer import initialize_trainer
from internlm.model.losses.ce_loss import InternLoss
from internlm.model.metrics import AccPerplex
from internlm.monitor.monitor import send_alert_message
from internlm.train.pipeline import (
    get_scheduler_hooks,
    initialize_llm_profile,
    initialize_optimizer,
    initialize_parallel_communicator,
    inject_model,
    load_new_batch,
    record_current_batch_training_metrics,
    set_param_unique_tracking_name,
)
from internlm.utils.common import (
    BatchSkipper,
    check_cuda_env,
    enable_pytorch_expandable_segments,
    get_current_device,
    get_megatron_flops,
    launch_time,
)
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import get_parallel_log_file_name
from internlm.utils.simple_memory_profiler import SimpleMemoryProfiler
from internlm.utils.utils import DataType
from internlm.utils.writer import Writer

# global llm logger
logger = logging.getLogger(__file__)

# Greyhound fail-slow detection integration
_CHECK_PAUSE = False

def _failslow_pause_handler(signum, frame):
    """Handler for SIGUSR1 signal from Greyhound L2 control plane."""
    global _CHECK_PAUSE
    _CHECK_PAUSE = True
    if logger:
        logger.info(f"[Greyhound] Rank {gpc.get_global_rank()}: Received fail-slow signal, pausing at next iteration")

signal.signal(signal.SIGUSR1, _failslow_pause_handler)

class FailSlowHook:
    """Redis-based fail-slow injection hook for InternEvo.

    Mirrors Greyhound/Megatron's ClientWrapper: every iteration reads
    delay_time_{rank} from Redis and sleeps that many seconds, simulating
    a computation straggler.  Gracefully no-ops when Redis is unavailable.
    """

    def __init__(self, check_interval: int = 1):
        self.delay_time: float = 0.0
        self.rank = None
        self.client = None
        self.check_interval = check_interval
        self.count = 0
        self._init_redis()

    def _init_redis(self):
        if redis is None:
            return
        try:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            self.client = redis.StrictRedis(host=master_addr, port=6379, db=0)
            self.client.ping()
        except Exception:
            self.client = None

    def check(self):
        """Poll Redis for delay update, then sleep if delay is active.

        Call once per training iteration (before forward-backward).
        """
        self.count += 1
        if self.client is not None and self.count % self.check_interval == 0:
            if self.rank is None:
                self.rank = gpc.get_global_rank()
            try:
                val = self.client.get(f"delay_time_{self.rank}")
                if val is not None:
                    new_delay = float(val.decode())
                    if new_delay != self.delay_time:
                        logger.info(
                            f"[FailSlowHook] Rank {self.rank}: delay_time "                            f"{self.delay_time:.4f}s -> {new_delay:.4f}s"
                        )
                        self.delay_time = new_delay
            except Exception:
                pass

        if self.delay_time > 0:
            time.sleep(self.delay_time)




class TrainerBuilder(Trainer):
    """
    A class for building and managing InternEvo training workflow.

    `TrainerBuilder` extends the base `Trainer` class to include additional functionality
    for initializing and managing various components involved in the training process.
    This includes setting up logging, checkpoints, loss functions, optimizers, metrics,
    train states, and profiling tools. The class supports distributed training and allows
    for seamless management of training, evaluation, and checkpointing.

    Args:
        model (Union[torch.nn.Module, List[torch.nn.Module]]): The model to be trained.
        train_dl (DataLoader): DataLoader for training data.
        val_dls (Optional[Dict[str, DataLoader]], optional): DataLoaders for validation data.
        **kwargs: Additional keyword arguments including:
            - config (str): Path to the configuration file.
            - profiling (bool): Whether to enable profiling.
            - dataset_types (list): List of dataset types to be used for training.

    Methods:
        __init__: Initializes the `TrainerBuilder` with the model, data loaders, and other components.
        fit: Runs the training loop, processing batches and handling evaluation and checkpointing.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        train_dl: DataLoader,
        val_dls: Optional[Dict[str, DataLoader]] = None,
        **kwargs,
    ):
        """
        Initialize TrainerBuilder with necessary components for training.

        Args:
            model (Union[torch.nn.Module, List[torch.nn.Module]]): The model to be trained.
            train_dl (DataLoader): DataLoader for training data.
            val_dls (Optional[Dict[str, DataLoader]], optional): DataLoaders for validation data.
            **kwargs: Additional keyword arguments including:
                - config (str): Path to the configuration file.
                - profiling (bool): Whether to enable profiling.
                - dataset_types (list): List of dataset types to be used for training.

        """
        # set very_beginning_time
        self.very_beginning_time = time.time()
        # broadcast current_time and setup logging
        self.current_time = self._setup_time_and_logging()
        # load config_lines
        config_lines = self._read_config(kwargs["config"])

        # set tracking name for parameters
        set_param_unique_tracking_name(model)

        # inject model for amp and parallel training
        model = inject_model(model)

        # check cuda env
        check_cuda_env()

        # set torch expandable_segments
        enable_pytorch_expandable_segments()

        # initialize loss function
        criterion = self._initialize_criterion()

        # initialize isp communicator
        isp_communicator = initialize_parallel_communicator(model)

        # initialize cpu offload manager for selective checkpoint
        initialize_offload_manager(gpc.config.get("selective_checkpoint_offload", False))

        # initialize train state
        train_state = get_train_state(train_dl)

        # initialize optimizer
        optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

        # initialize checkpoint manager and try resume training
        self.ckpt_manager = self._initialize_checkpoint_manager(model, optimizer, lr_scheduler, train_dl, config_lines)
        self.ckpt_manager.try_resume_training(train_state, self.current_time)

        # initialize customed llm writer
        self.writer = self._initialize_writer(train_state, config_lines)

        # initialize metric for calculating accuracy and perplexity
        self.metric = self._initialize_metric(kwargs["dataset_types"])

        # initialize simple memory profiler
        self.memory_profiler = self._initialize_memory_profiler(model, optimizer, kwargs["profiling"])

        # initialize batch skipper
        self.batch_skipper = self._initialize_batch_skipper(train_state)

        # initialize trainer
        engine, scheduler = initialize_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            beta2_scheduler=beta2_scheduler,
            scheduler_hooks=get_scheduler_hooks(self.metric, optimizer, isp_communicator),
        )

        # set attributes
        self._set_attributes(
            kwargs["profiling"], train_dl, val_dls, train_state, optimizer, beta2_scheduler, isp_communicator
        )

        super().__init__(engine, scheduler)

        # Greyhound fail-slow injection hook (reads delay_time_{rank} from Redis)
        self.failslow_hook = FailSlowHook()

    def _setup_time_and_logging(self) -> str:
        current_time = launch_time()
        objs = [current_time]
        dist.broadcast_object_list(objs, src=0)
        current_time = objs[0].replace(":", ".")
        global logger
        logger = get_logger(
            __name__, launch_time=current_time, job_name=gpc.config.JOB_NAME, file_name=get_parallel_log_file_name()
        )
        return current_time

    def _read_config(self, config_path: str) -> list:
        with open(config_path, "r") as f:
            return f.readlines()

    def _initialize_criterion(self) -> InternLoss:
        return InternLoss(
            parallel_output=gpc.config.model.parallel_output,
            label_smoothing=gpc.config.loss.label_smoothing,
            op_type=gpc.config.loss.op_type,
        )

    def _initialize_checkpoint_manager(
        self, model, optimizer, lr_scheduler, train_dl, config_lines
    ) -> CheckpointManager:
        return CheckpointManager(
            ckpt_config=gpc.config.ckpt,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dl=train_dl,
            model_config=gpc.config.model,
            model_config_file="".join(config_lines),
            feishu_address=gpc.config.monitor.alert.feishu_alert_address,
        )

    def _initialize_writer(self, train_state, config_lines) -> Writer:
        return Writer(
            job_name=gpc.config.JOB_NAME,
            launch_time=self.current_time,
            file_name=get_parallel_log_file_name(),
            tensorboard_folder=gpc.config.tensorboard_folder,
            resume_tb_folder=train_state.resume_tb_folder,
            step_count=train_state.step_count,
            config=config_lines,
            logger=logger,
            enable_tb=gpc.config.enable_tb,
            queue_max_length=gpc.config.tensorboard.queue_max_length,
            total_steps=gpc.config.data.total_steps,
        )

    def _initialize_metric(self, dataset_types) -> AccPerplex:
        # initialize metric for calculating accuracy and perplexity
        # if isp mode, head output is parallel in sequence dim, metric dp group should be SP*DP
        # _dp_pg = (
        #     gpc.get_group(ParallelMode.ISP_DATA)
        #     if is_using_isp() and gpc.config.model.parallel_output
        #     else gpc.get_group(ParallelMode.DATA)
        # )
        # _tp_pg = dist.new_group([gpc.get_global_rank()]) if is_using_isp() else gpc.get_group(ParallelMode.TENSOR)
        _dp_pg = gpc.get_sub_group(ParallelMode.DATA)
        _tp_pg = gpc.get_sub_group(ParallelMode.TENSOR)
        return AccPerplex(
            device=get_current_device(),
            tp_pg=_tp_pg,
            dp_pg=_dp_pg,
            dataset_types=dataset_types,
        )

    def _initialize_memory_profiler(self, model, optimizer, profiling) -> Optional[SimpleMemoryProfiler]:
        if profiling:
            return SimpleMemoryProfiler(
                model,
                optimizer.optim,
                log_folder=f"RUN/{gpc.config.JOB_NAME}/{self.current_time}/memory_trace/rank{gpc.get_global_rank()}_"
                + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
                + f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
                + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
            )
        else:
            return None

    def _initialize_batch_skipper(self, train_state) -> BatchSkipper:
        skip_batches = gpc.config.data.skip_batches
        if gpc.config.data.type == DataType.streaming.name and gpc.config.ckpt.auto_resume:
            skip_batches = streaming_simple_resume(train_state)
        return BatchSkipper(skip_batches)

    def _set_attributes(self, profiling, train_dl, val_dls, train_state, optimizer, beta2_scheduler, isp_communicator):
        self.profiling = profiling
        self.train_dl = train_dl
        self.val_dls = val_dls
        self.train_state = train_state
        self.optimizer = optimizer
        self.beta2_scheduler = beta2_scheduler
        self.isp_communicator = isp_communicator

    def fit(self):
        """
        Run InternEvo training loop.
        """
        do_next = True
        if gpc.config.get("FAILURE",False):
            if gpc.get_global_rank() in gpc.config.get("FAILURE_GLOBAL_RANKS",[]):
                do_next = False
        if do_next:
            self.train()
            train_iter = iter(self.train_dl)

            with initialize_llm_profile(profiling=self.profiling, start_time=self.current_time) as prof:
                gc.disable()
                for batch_count in range(self.train_state.batch_count, gpc.config.data.total_steps):
                    if self._process_batch(batch_count, train_iter, prof):
                        break

        self.ckpt_manager.wait_async_upload_finish()

    def _process_batch(self, batch_count: int, train_iter, prof) -> bool:
        global _CHECK_PAUSE

        # Greyhound fail-slow detection: pause if signaled
        if _CHECK_PAUSE:
            _CHECK_PAUSE = False
            pause_duration = 0
            max_pause_duration = 300  # max 5 minutes

            if gpc.is_rank_for_log():
                logger.warning(f"[Greyhound] Batch {batch_count}: Detected fail-slow, pausing training for recovery...")

            # Wait for control plane to signal recovery via Redis
            if redis:
                try:
                    client = redis.StrictRedis(host="localhost", port=6379, db=0)
                    rank = gpc.get_global_rank()
                    pause_key = f"failslow_pause_rank_{rank}"

                    while pause_duration < max_pause_duration:
                        pause_flag = client.get(pause_key)
                        if pause_flag is None or pause_flag == b"false":
                            break
                        time.sleep(1)
                        pause_duration += 1

                    if pause_duration >= max_pause_duration:
                        logger.error(f"[Greyhound] Rank {rank}: Pause timeout, resuming anyway")
                    else:
                        logger.info(f"[Greyhound] Rank {rank}: Recovery complete after {pause_duration}s, resuming")
                except Exception as e:
                    logger.warning(f"[Greyhound] Redis pause check failed: {e}, resuming anyway")
            else:
                # Fallback: simple sleep if redis not available
                time.sleep(2)
                logger.info(f"[Greyhound] Batch {batch_count}: Resumed (redis unavailable, used simple sleep)")

        # Greyhound fail-slow injection: apply per-iteration compute delay if set
        self.failslow_hook.check()

        empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
        start_time = time.time()
        start_time_ = time.perf_counter()
        timer("one-batch").start()

        batch, train_iter = self._load_and_prepare_batch(batch_count, train_iter)
        if self.batch_skipper(batch_count):
            #TODO,chimear just use only one pipe result
            if gpc.is_rank_for_log():#mth rewrite just for hetpipe
                logger.info(f"Skip batch count:`{batch_count}`...")
            timer("one-batch").stop()
            return False

        timer("fwd-bwd").start()
        loss, moe_loss = self._forward_backward(batch)
        timer("fwd-bwd").stop()
        #TODO, how chimera do this
        timer("update-parameters").start()
        weight_update_start = time.time()
        success_update, grad_norm_groups = self._update_parameters()
        weight_update_time = time.time() - weight_update_start
        timer("update-parameters").stop()
        timer("one-batch").stop()

        # Append weight_update_time to profiling JSON if enabled
        if getattr(gpc.config, 'profile_fwd_bwd', False) and os.environ.get("CUDA_LAUNCH_BLOCKING") == "1" and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            import json
            output_dir = os.path.join("InternEvo/results/fwd_bwd_time", gpc.config.model_type, f"{gpc.config.PP_MODE}_l{gpc.config.NUM_LAYER}_hid{gpc.config.HIDDEN_SIZE}_seq{gpc.config.SEQ_LEN}_voc{gpc.config.VOCAB_SIZE}_mb{gpc.config.MICRO_NUM}", gpc.config.timestamp)
            output_file = os.path.join(output_dir, f"PP_rank_{gpc.get_local_rank(ParallelMode.PIPELINE)}.json")
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as _f:
                        data = json.load(_f)
                    # Append weight_update_time to latest iteration
                    if "iterations" in data and len(data["iterations"]) > 0:
                        data["iterations"][-1]["weight_update_time"] = weight_update_time
                    if "weight_update_times" not in data:
                        data["weight_update_times"] = []
                    data["weight_update_times"].append(weight_update_time)
                    data["avg_weight_update"] = sum(data["weight_update_times"]) / len(data["weight_update_times"])
                    with open(output_file, 'w') as _f:
                        json.dump(data, _f, indent=4)
                except Exception:
                    pass

        self._record_metrics(batch_count, batch, start_time, loss, moe_loss, success_update, grad_norm_groups)
        if self._should_evaluate():
            self._evaluate()

        if self.ckpt_manager.try_save_checkpoint(self.train_state):
            return True

        self._update_profilers(batch_count, prof)
        return False

    def _load_and_prepare_batch(self, batch_count: int, train_iter):
        batch, train_iter = load_new_batch(train_dl=self.train_dl, train_iter=train_iter, train_state=self.train_state)
        self.train_state.batch_count = batch_count
        self.train_state.num_consumed_samples_in_epoch += len(batch[1])
        if batch[0].get("type_ids", None) is not None:
            self.metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))
        return batch, train_iter

    def _forward_backward(self, batch):
        self.zero_grad()
        if hasattr(gpc.config.model, "num_experts"):
            _, _, loss, moe_loss, moe_z_loss = self.execute_schedule(
                batch, forward_only=False, return_loss=True, return_output_label=False,
            )
        else:
            _, _, loss = self.execute_schedule(batch, forward_only=False, return_loss=True, return_output_label=False,)
            moe_loss = None
        return loss, moe_loss

    def _update_parameters(self):
        # print(f"TENGHUI>>>>>>>>>RANK:{gpc.get_global_rank()}, device:{get_current_device()}, 2")
        trainer_result = self.step()
        # print(f"TENGHUI>>>>>>>>>RANK:{gpc.get_global_rank()}, device:{get_current_device()}, 2E")
        assert trainer_result is not None
        success_update, grad_norm_groups = trainer_result
        if success_update:
            self.train_state.step_count += 1
        else:
            self.train_state.inf_nan_skip_batches += 1
            if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():
                logger.warning(f"Warning: skip parameter update at step {self.train_state.batch_count}.")
                send_alert_message(
                    address=gpc.config.monitor.alert.feishu_alert_address,
                    message=f"Warning: skip parameter update at step {self.train_state.batch_count}.",
                )
        return success_update, grad_norm_groups

    def _record_metrics(self, batch_count: int, batch, start_time, loss, moe_loss, success_update, grad_norm_groups):
        get_tflops_func = partial(
            get_megatron_flops,
            checkpoint=gpc.config.model.checkpoint,
            seq_len=gpc.config.data["seq_len"],
            hidden_size=gpc.config.model.hidden_size,
            num_layers=gpc.config.model.num_layers,
            vocab_size=gpc.config.model.vocab_size,
            global_batch_size=gpc.config.data.micro_bsz
            * gpc.config.data.micro_num
            * gpc.get_world_size(ParallelMode.DATA),
            global_world_size=gpc.get_world_size(ParallelMode.GLOBAL),
            mlp_ratio=gpc.config.model["mlp_ratio"],
        )
        record_current_batch_training_metrics(
            get_tflops_func=get_tflops_func,
            logger=logger,
            writer=self.writer,
            success_update=success_update,
            batch_count=batch_count,
            batch=batch,
            train_state=self.train_state,
            optimizer=self.optimizer,
            beta2_scheduler=self.beta2_scheduler,
            engine=self.engine,
            start_time=start_time,
            very_begining_time=self.very_beginning_time,
            loss=loss,
            moe_loss=moe_loss,
            grad_norm=grad_norm_groups,
            metric=self.metric,
        )

    def _should_evaluate(self) -> bool:
        return (
            gpc.config.data.valid_every > 0
            and self.train_state.step_count > 0
            and self.train_state.step_count % gpc.config.data.valid_every == 0
        )

    def _evaluate(self):
        evaluate_on_val_dls(
            self,
            val_dls=self.val_dls,
            writer=self.writer,
            logger=logger,
            step_count=self.train_state.step_count,
        )

    def _update_profilers(self, batch_count: int, prof):
        if self.memory_profiler is not None:
            self.memory_profiler.step()
        if batch_count % 1 == 0:
            prof.step()
