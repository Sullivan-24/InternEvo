#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from preprocess import busy_wait_kernel 
from internlm.core.engine import Engine
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.scheduler import comm
from internlm.utils.common import (
    SchedulerHook,
    check_data_is_packed,
    get_current_device,
    move_to_device,
)
from internlm.utils.logger import get_logger
from internlm.utils.timeout import llm_timeout

from .base_scheduler import BaseScheduler

logger = get_logger(__file__)


def get_tensor_shape():
    if hasattr(gpc.config, "TENSOR_SHAPE"):
        return gpc.config.TENSOR_SHAPE

    if not gpc.is_initialized(ParallelMode.PIPELINE):
        return None

    if (
        hasattr(gpc.config.data, "seq_len")
        and hasattr(gpc.config.data, "micro_bsz")
        and hasattr(gpc.config.model, "hidden_size")
    ):
        if gpc.config.data.use_packed_dataset and gpc.is_evaluating is False:
            if gpc.config.parallel.sequence_parallel:
                sequence_world_size = gpc.get_world_size(ParallelMode.TENSOR)
                tensor_shape = (
                    1,
                    gpc.config.data["seq_len"] * gpc.config.data["micro_bsz"] // sequence_world_size,
                    gpc.config.model["hidden_size"],
                )
            else:
                tensor_shape = (
                    1,
                    gpc.config.data["seq_len"] * gpc.config.data["micro_bsz"],
                    gpc.config.model["hidden_size"],
                )
        else:
            if gpc.config.parallel.sequence_parallel:
                sequence_world_size = gpc.get_world_size(ParallelMode.TENSOR)
                tensor_shape = (
                    gpc.config.data["micro_bsz"],
                    gpc.config.data["seq_len"] // sequence_world_size,
                    gpc.config.model["hidden_size"],
                )
            else:
                tensor_shape = (
                    gpc.config.data["micro_bsz"],
                    gpc.config.data["seq_len"],
                    gpc.config.model["hidden_size"],
                )
        return torch.Size(tensor_shape)
    else:
        return None


def pack_return_tensors(return_tensors):
    output, label = tuple(zip(*return_tensors))
    if isinstance(output[0], torch.Tensor):
        output = torch.cat(output, dim=0)
    elif isinstance(output[0], (list, tuple)):
        output = tuple(torch.cat(tensors, dim=0) for tensors in zip(*output))
    else:
        raise TypeError("Output of model must be tensor or list/tuple of tensors")
    if isinstance(label[0], torch.Tensor):
        label = torch.cat(label, dim=0)
    elif isinstance(label[0], dict):
        merged_label = {k: [] for k in label[0].keys()}
        for d in label:
            for k, v in d.items():
                merged_label[k].append(v)
        label = {k: torch.cat(v, dim=0) for k, v in merged_label.items()}
    return output, label


@contextmanager
def switch_virtual_pipeline_parallel_rank(rank):
    prev_rank = gpc.virtual_pipeline_parallel_rank
    try:
        gpc.set_virtual_pipeline_parallel_rank(rank)
        yield
    finally:
        gpc.set_virtual_pipeline_parallel_rank(prev_rank)


@contextmanager
def switch_optimizer_grad_sync_skip_mode(optimizer, skip: bool = True):
    prev_mode = optimizer.skip_grad_reduce
    try:
        optimizer.skip_grad_reduce = skip
        yield
    finally:
        optimizer.skip_grad_reduce = prev_mode


class HydraPipelineScheduler(BaseScheduler):
    """
    A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        dtype (torch.dtype): Type of data. torch.float by default.
        data_process_func (Callable, optional):
            The post processing function which receives a micro batch of data, and it will be executed
            in `load_micro_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (Optional[List[SchedulerHook]], optional): List of scheduler hooks.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        assert num_microbatches > 0, f"expected num_microbatches to be larger then 1, but got {num_microbatches}"

        assert not isinstance(
            tensor_shape, int
        ), "tensor_shape type should be one of Union[torch.Size, List[int], Tuple[int]]."

        super().__init__(data_process_func=data_process_func)

        self.num_microbatches = num_microbatches
        self.dtype = dtype
        self._hooks = scheduler_hooks

        self._tensor_shape = (
            tensor_shape if tensor_shape is None or isinstance(tensor_shape, torch.Size) else torch.Size(tensor_shape)
        )

        self.scatter_gather_tensors = scatter_gather_tensors and gpc.is_using_parallel_mode(ParallelMode.TENSOR)

        if gpc.config.parallel.sequence_parallel:
            self.scatter_gather_tensors = False

        # cache for the batch data
        self.batch_data = None
        self.last_label = None

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape

    def pre_processing(self, engine):
        self.dtype = gpc.config.model.get("dtype", torch.half)

    @staticmethod
    def _call_engine(engine, data):  # pylint: disable=W0237
        if data is None:
            return None

        if isinstance(data, torch.Tensor):
            return engine(data)
        elif isinstance(data, (list, tuple)):
            return engine(*data)
        elif isinstance(data, dict):
            stage_output = data.pop("stage_output", None)
            if stage_output is None:
                return engine(**data)
            elif isinstance(stage_output, torch.Tensor):
                return engine(stage_output, **data)
            elif isinstance(stage_output, (tuple, list)):
                return engine(*stage_output, **data)
            else:
                raise TypeError(
                    f"Expected stage_output to be of type torch.Tensor, list, or tuple, "
                    f"but got {type(stage_output)}"
                )
        else:
            raise TypeError(f"Expected data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")

    def load_batch(self, engine, data_iter):
        # Pipeline schedule just puts data in memory,
        batch_data, actual_batch_size = engine.load_batch(data_iter, to_gpu=True)

        # Even if 'use_flash_attn' is False, the data seen when the 'load_batch' is called is still packed,
        # because internlm's current train dataset is packed, even using dummy data.
        # The unpack operation is performed in load_micro_batch().
        if check_data_is_packed(batch_data):
            micro_num = actual_batch_size
        else:
            micro_num = actual_batch_size // gpc.config.data["micro_bsz"]

        self.microbatch_offset = 0
        self.batch_size = actual_batch_size
        self.batch_data, self.batch_label = batch_data
        self.bsz_stride = self.batch_size // micro_num
        # 'num_microbatches' is no longer an initialization parameter,
        # but is determined on the fly by the Scheduler.
        self.num_microbatches = micro_num  # Rampup or variable bsz size.

    def load_micro_batch(self):
        micro_batch_data, micro_batch_label = self._load_micro_batch(
            data=self.batch_data, label=self.batch_label, offset=self.microbatch_offset, bsz_stride=self.bsz_stride
        )

        if self.data_process_func:
            micro_batch_data, micro_batch_label = self.data_process_func(micro_batch_data, micro_batch_label)

        micro_batch_data["label"] = micro_batch_label
        self.microbatch_offset += self.bsz_stride

        return move_to_device(micro_batch_data)

    def _get_data_label_for_current_step(self, stage_output, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            assert not self._config.parallel["pipeline"].get("mode", "1F1B") == "ZBV"
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                # for the first stage, we use the data from the
                # dataloader output by default
                data, label = micro_batch_data
            else:
                # for non-first stage, we use the output passed
                # by the previous as the model input
                data = stage_output
                _, label = micro_batch_data
        # normally this way
        elif isinstance(micro_batch_data, dict):
            label = micro_batch_data.pop("label", None)
            data = {"stage_output": stage_output, **micro_batch_data}

        return data, label  # pylint: disable=E0606

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _get_current_microbatch_id(self, step_id: int) -> int:
        """
        Get the current microbatch ID based on the step ID.
        In 1f1b scheduler, the microbatch ID is the same as the step ID,
        but it is important to note that the step ID is calculated separately
        for forward and backward passes.
        """
        return step_id

    def _forward_step(
        self,
        engine,
        input_obj,
        return_tensors,
        return_output_label=True,
        accum_loss=None,
        accum_moe_loss=None,
        chunk_id=0,
    ):
        """
        Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
            accum_moe_loss (optional): Where accumulated moe loss stores.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        if chunk_id == 0:
            micro_batch_data = self.load_micro_batch()
            data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)
            self.last_label = label
            self._call_hooks("before_forward", data)
            if hasattr(gpc.config.model, "num_experts"):
                output_obj, moe_losses = self._call_engine(engine.model[chunk_id], data)
            else:
                output_obj = self._call_engine(engine.model[chunk_id], data)
            self._call_hooks("after_forward", output_obj)

        if chunk_id == 1:
        # if gpc.is_last_rank(ParallelMode.PIPELINE):
            self._call_hooks("post_helper_func", input_obj, self.last_label)
            if return_output_label:
                return_tensors.append((input_obj, self.last_label))
            # if accum_loss is not None:
            self._call_hooks("before_criterion", input_obj, self.last_label)
            loss = self._call_engine_criterion(engine, input_obj, self.last_label)
            self._call_hooks("after_criterion", loss)

            loss_reduced = loss / self.num_microbatches
            accum_loss.add_(loss_reduced.detach())
            output_obj = loss_reduced

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            moe_loss = sum(moe_losses) * gpc.config.loss.moe_loss_coeff

            # the moe_loss is computed among the "tensor" group if sequence parallel is enabled,
            # so we need to do allreduce
            if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
                dist.all_reduce(moe_loss, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
                moe_loss.div_(gpc.get_world_size(ParallelMode.TENSOR))
            moe_loss /= self.num_microbatches
            accum_moe_loss.add_(moe_loss.detach())
        else:
            moe_loss = None

        if gpc.config["HETER"] and gpc.get_local_rank(ParallelMode.PIPELINE) >= gpc.config["PP_SIZE"] // 2:
            import time
            busy_wait_kernel(gpc.config["SLEEP_TIME"])
        return output_obj, moe_loss

    def _backward_step(self, engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss=None):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            step_id (int): The ID of the current step.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()

        # Backward pass.

        # Only the last microbatch does syncing grad.
        skip_grad_sync = self._get_current_microbatch_id(step_id) != self.num_microbatches - 1

        self._call_hooks("before_backward", output_obj, output_obj_grad)
        with switch_optimizer_grad_sync_skip_mode(engine.optimizer, skip_grad_sync):
            if moe_loss is None or moe_loss.item() == 0.0:
                if output_obj_grad is None:
                    engine.backward(output_obj)
                else:
                    engine.backward_by_grad(output_obj, output_obj_grad)
            else:
                if output_obj_grad is None:
                    engine.backward(output_obj + moe_loss)
                else:
                    # scale the latent loss
                    moe_loss = moe_loss * engine.optimizer.loss_scale
                    # we perform chain rule here by projecting the grad to the direction of
                    # [output_obj_grad, 1], Because moe_loss have no relation with subsequent
                    # layer, we set it to None (will be ragarded as 1).
                    engine.backward_by_grad([output_obj, moe_loss], [output_obj_grad, None])

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)
        self._call_hooks("after_backward", input_obj_grad)
        if gpc.config["HETER"] and gpc.get_local_rank(ParallelMode.PIPELINE) >= gpc.config["PP_SIZE"] // 2:
            busy_wait_kernel(gpc.config["SLEEP_TIME"])
        return input_obj_grad

    def _forward_only_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function performs forward only computation process. The scheduling of microbatches is similar to the
        warmup phase, where each microbatch first receives the forward input from the previous stage, then performs
        the forward computation, and finally passes the forward computation output to the next stage. There are two
        special cases to note:
        1. The first stage of the pipeline does not need to receive forward input; its input comes from the dataloader.
        2. The last stage of the pipeline does not need to send forward output; its output is returned to the user code
           for processing.

        Args:
            engine (colossalai.engine.Engine): internlm engine for training and inference.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                output, label, and accumulated loss.
        """
        # Input, output tensors only need to be saved when doing backward passes
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            accum_moe_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_moe_loss = None

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        need_forward_meta = self.tensor_shape is None

        # Run all forward passes.
        for _ in range(self.num_microbatches):
            # Receive input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj, _ = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.
                # Send the forward computation output to the next stage
                assert output_obj.dtype == self.dtype
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

            if accum_loss is not None:
                accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        # print(f"engine.model type:{type(engine.model)}\n, pp_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)} model:{engine.model}")
        pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
        # accum_loss = (
        #     torch.zeros(1, device=get_current_device())
        #     if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
        #     else None
        # )
        accum_loss = (
            torch.zeros(1, device=get_current_device())
        )
        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            accum_moe_loss = torch.zeros(1, device=get_current_device())
        else:
            accum_moe_loss = None

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        nmb = self.num_microbatches
        mid_groups = []
        for i in range(0, nmb, pp_size):
            mid_group = []
            for j in range(pp_size):
                mb_id = i + j
                if mb_id < nmb:
                    mid_group.append(mb_id)
            mid_groups.append(mid_group)

        for mid_group in mid_groups:
            # run a group of F
            for mid in mid_group:
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    if forward_recv_shapes is None:
                        forward_recv_shapes = comm.recv_obj_meta()
                    input_obj = comm.recv_forward(
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                else:
                    input_obj = None

                # print(f"PP RANK:{gpc.get_local_rank(ParallelMode.PIPELINE)}, mid={mid}, F received")
                output_obj, moe_loss = self._forward_step(
                    engine=engine,
                    input_obj=input_obj,
                    return_tensors=return_tensors,
                    return_output_label=return_output_label,
                    accum_loss=accum_loss,
                    accum_moe_loss=accum_moe_loss,
                    chunk_id=0,
                )
                # if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]
                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False
                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    assert output_obj.dtype == self.dtype
                    comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)
                    # print(f"PP RANK:{gpc.get_local_rank(ParallelMode.PIPELINE)}, mid={mid}, F sended")
                
                input_objs.append(input_obj)
                output_objs.append(output_obj)
                moe_losses.append(moe_loss)

            # send/recv fwd 可以用scatter替换
            if pp_rank == pp_size - 1:
                for pid in range(pp_size - 1):
                    comm.send_forward(output_objs[pid], next_rank=pid, scatter_gather_tensors=self.scatter_gather_tensors)
                    print(f"pp_rank:{pp_size-1} send fwd to pp_rank {pid}.")
                input_obj = input_objs[-1]
            else:
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    prev_rank = pp_size - 1,
                )
                print(f"pp_rank:{pp_rank} receive fwd from pp_rank {pp_size-1}.")

            output_obj, moe_loss = self._forward_step(
                engine=engine,
                input_obj=input_obj,
                return_tensors=return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
                chunk_id=1,
            )
            input_obj_grad = self._backward_step(
                engine, 0, input_obj, output_obj, None, moe_loss
            )

            # send/recv bwd
            head_b_objs = []
            if pp_rank == pp_size - 1:
                for pid in range(pp_size - 1):
                    head_b_obj = comm.recv_backward(backward_recv_shapes, next_rank=pid, dtype=self.dtype, scatter_gather_tensors=self.scatter_gather_tensors)
                    head_b_objs.append(head_b_obj)
                    print(f"pp_rank:{pp_size-1} recv bwd from pp_rank {pid}.")
            else:
                comm.send_backward(
                    input_obj_grad,
                    prev_rank = pp_size - 1,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
                print(f"pp_rank:{pp_rank} send bwd to pp_rank {pp_size-1}.")

            # run a group of B
            for mid in mid_group:
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)
                moe_loss = moe_losses.pop(0)

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    output_obj_grad = comm.recv_backward(
                        backward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    # print(f"PP RANK:{gpc.get_local_rank(ParallelMode.PIPELINE)}, mid={mid}, B received")
                else:
                    output_obj_grad = head_b_objs[mid % len(head_b_objs)]

                input_obj_grad = self._backward_step(
                    engine, mid, input_obj, output_obj, output_obj_grad, moe_loss
                )

                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)
                    # print(f"PP RANK:{gpc.get_local_rank(ParallelMode.PIPELINE)}, mid={mid}, B sended")

        
        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

            if accum_loss is not None:
                accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    @llm_timeout(func_name="nointerleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        Returns a tuple with losses if the last stage, an empty tuple otherwise.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss), loss and label could be None.
                The loss would be returned only in the last stage. And the moe_loss is accumulated from all stages.
        """

        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        # Load data first
        self.load_batch(engine, data_iter)
        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            self._accum_loss = torch.zeros(1, device=get_current_device())

        if forward_only:
            output, label, accum_loss, accum_moe_loss = self._forward_only_step(
                engine, return_loss, return_output_label
            )
        else:
            output, label, accum_loss, accum_moe_loss = self._forward_backward_step(
                engine, return_loss, return_output_label
            )

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss