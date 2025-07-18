#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import queue
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.scheduler import comm
from internlm.utils.common import SchedulerHook, get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp
from internlm.utils.utils import ModuleType, Step
from .pipeline_scheduler_1f1b import (
    InterleavedPipelineScheduler,do_compute,
    PipelineScheduler,
    pack_return_tensors,
)

from .pipeline_scheduler_zb import WeightGradStore,ZeroBubblePipelineVShapeScheduler

import queue
import time
import json
import os

logger = get_logger(__file__)


def write_json(jsonpath, content):
    if gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        with open(jsonpath, 'a',encoding='utf-8') as f:
            json.dump(content, f)
            f.write('\n')

def _get_localrankid_by_placement(stage_id: int, stage_placement:list) -> int:
    for device_id in range(len(stage_placement)):
        for stage_in_device in stage_placement[device_id]:
            if stage_in_device == stage_id:
                return device_id

def _get_localrankid_mutistream(stream: int, stage_id: int, stage_placement:list) -> int:
    for device_id in range(len(stage_placement)):
        if stage_id == stage_placement[device_id][stream]:
            return device_id

def _get_chunkid_by_stages(stage_id: int, stages:list) -> int:
    for index, value in enumerate(stages):
        if stage_id == value:
            return index
    raise ValueError(f"stage_id {stage_id} not found in stages list")
def _get_chunkid_by_stage_placement(stage_id: int, stage_placement:list) -> int:
    for i in range(len(stage_placement)):
        stages = stage_placement[i]
        for j in range(len(stages)):
            if stage_id == stages[j]:
                return j
    raise ValueError(f"stage_id {stage_id} not found in stage_placement list")

class UnifiedSingleChunkPipelineScheduler(PipelineScheduler):
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
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        comm_graph: List[List[tuple]] = None,
    ):
        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        self.split_backward = gpc.config.split_backward
        if self.split_backward:
            gpc.config.scheduler_type = ModuleType.ZBH1.value
            WeightGradStore.set_pp_mode("ZBH1")
            WeightGradStore.set_optim(optimizer)
        else:
            gpc.config.scheduler_type = ModuleType.ONEFONEB.value
        self.last_stage = len(unified_scheduler)-1
        self.first_stage = 0
        file_path = f"./jsonResult/async/{gpc.config.scheduler_type}_pp{gpc.pipeline_parallel_size}_mb{self.num_microbatches}"
        os.makedirs(file_path, exist_ok=True)
        gpc._config['jsonpath'] = file_path + "/iter_0_opeartion_list.json"
        self.unified_scheduler = unified_scheduler
        self.comm_graph = comm_graph

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes
        async_communicator_recv_forward_queue = queue.Queue()
        async_communicator_recv_backward_queue = queue.Queue()
        recv_forward_start_buffer_queue = queue.Queue()
        recv_backward_start_buffer_queue = queue.Queue()
        input_objs = queue.Queue()
        output_objs = queue.Queue()
        moe_losses = queue.Queue()
        return_tensors = queue.Queue()
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )
        accum_moe_loss = torch.zeros(1, device=get_current_device())

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        #rank_info
        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        stage_id = local_rank
        steps = self.unified_scheduler[local_rank]
        comm_list = self.comm_graph[local_rank]
        num_steps = len(steps)
        jsonpath = gpc._config['jsonpath']

        for s in range(num_steps):
            step_type, microbatch_id, stage_id, _, _, _ = steps[s]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']

            for before_ops in before_recv_list:
                op=before_ops[0]
                if op == Step.FORWARD.value:
                    recv_f_buffer = comm.AsynCommunicator(
                            recv_prev_shape=forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                    recv_f_buffer.start()
                    recv_forward_start_buffer_queue.put(
                        recv_f_buffer
                    )
   
                elif op == Step.BACKWARD.value:
                    recv_b_buffer = comm.AsynCommunicator(
                            recv_next_shape=backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                    recv_b_buffer.start()
                    recv_backward_start_buffer_queue.put(recv_b_buffer)
            if step_type == Step.FORWARD.value:# Forward pass
                # Receive the input from the previous stage 
                if stage_id>self.first_stage:
                    if async_communicator_recv_forward_queue.qsize()>0:
                        input_obj = async_communicator_recv_forward_queue.get()
                    else:
                        input_obj, _ = recv_forward_start_buffer_queue.get().wait_and_receive()
                else:
                    input_obj = None

                # Perform forward computation
                #start_time = time.perf_counter()
                output_obj, moe_loss = self._forward_step(
                    engine,
                    input_obj,
                    return_tensors,
                    return_output_label=return_output_label,
                    accum_loss=accum_loss,
                    accum_moe_loss=accum_moe_loss,
                )
                #end_time = time.perf_counter()
                #json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                #write_json(jsonpath, json_content)
                if recv_forward_start_buffer_queue.qsize()>0:
                    recv_f_tensor, _ = recv_forward_start_buffer_queue.get().wait_and_receive()
                    async_communicator_recv_forward_queue.put(recv_f_tensor)
                if recv_backward_start_buffer_queue.qsize()>0:
                    _, recv_b_tensor = recv_backward_start_buffer_queue.get().wait_and_receive()
                    async_communicator_recv_backward_queue.put(recv_b_tensor)

                if stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        backward_recv_shapes = output_obj.shape
                    else:
                        backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]    
                    if need_forward_meta:
                        comm.send_obj_meta(output_obj)
                        need_forward_meta = False  # send only once.
                # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
                # forward computation

                if stage_id < self.last_stage :
                    assert output_obj.dtype == self.dtype

                send_forward_once = True
                if local_rank%2 == 0 and send_forward_once and stage_id < self.last_stage :
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    ).start()
                    send_forward_once = False

                for after_ops  in  after_recv_list:
                    op = after_ops[0]
                    if op == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                            recv_prev_shape=forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        recv_f_buffer.start()
                        recv_forward_start_buffer_queue.put(recv_f_buffer)
                    elif op == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=backward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        recv_b_buffer.start()
                        recv_backward_start_buffer_queue.put(recv_b_buffer)
                if send_forward_once and stage_id < self.last_stage :
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    ).start()

                input_objs.put(input_obj)
                output_objs.put(output_obj)
                moe_losses.put(moe_loss)

            elif step_type == Step.BACKWARD.value:# Backward pass
                input_obj = input_objs.get()
                output_obj = output_objs.get()
                moe_loss = moe_losses.get()
                
                if stage_id<self.last_stage:
                    if async_communicator_recv_backward_queue.qsize()>0:
                        output_obj_grad = async_communicator_recv_backward_queue.get()
                    else:
                        _, output_obj_grad = recv_backward_start_buffer_queue.get().wait_and_receive()
                else:
                    output_obj_grad = None
            
                #start_time = time.perf_counter()
                input_obj_grad = self._backward_step(
                    engine, microbatch_id, input_obj, output_obj, output_obj_grad, moe_loss
                )
                #end_time = time.perf_counter()
                #json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                #write_json(jsonpath, json_content)
                if recv_forward_start_buffer_queue.qsize()>0:
                    recv_f_tensor, _ = recv_forward_start_buffer_queue.get().wait_and_receive()
                    async_communicator_recv_forward_queue.put(recv_f_tensor)
                if recv_backward_start_buffer_queue.qsize()>0:
                    _, recv_b_tensor = recv_backward_start_buffer_queue.get().wait_and_receive()
                    async_communicator_recv_backward_queue.put(recv_b_tensor)
                send_backward_once = True
                if local_rank%2 == 0 and send_backward_once and stage_id > self.first_stage:
                    comm.AsynCommunicator(
                            object_send_prev=input_obj_grad,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        ).start()   
                    send_backward_once = False

                for after_ops  in  after_recv_list:
                    op = after_ops[0]
                    if op == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                            recv_prev_shape=forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        recv_f_buffer.start()
                        recv_forward_start_buffer_queue.put(recv_f_buffer)
                    elif op == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=backward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        recv_b_buffer.start()
                        recv_backward_start_buffer_queue.put(recv_b_buffer)
                if send_backward_once and stage_id > self.first_stage:
                    comm.AsynCommunicator(
                            object_send_prev=input_obj_grad,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        ).start()
                WeightGradStore.flush()

            elif step_type == Step.WEIGHT.value: # Weight update
                #start_time = time.perf_counter()
                WeightGradStore.pop()
                #end_time = time.perf_counter()
                #json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                #write_json(jsonpath, json_content)
                for after_ops  in  after_recv_list:
                    op = after_ops[0]
                    if op == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                                recv_prev_shape=forward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        recv_f_buffer.start()
                        recv_forward_start_buffer_queue.put(recv_f_buffer)

                    elif op == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                            recv_next_shape=backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        recv_b_buffer.start()
                        recv_backward_start_buffer_queue.put(recv_b_buffer)

        output, label = pack_return_tensors(return_tensors) if return_tensors.qsize() > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

class UnifiedMultipleChunksPipelineScheduler(ZeroBubblePipelineVShapeScheduler):
    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        stage_placement: List[List[int]] = None,
        comm_graph: List[List[tuple]] = None,
        scheduler_type: str = None,
        split_backward: bool = False,
        layerwise: bool = False,
        first_stage: int = None,
        last_stage: int = None,
    ):
        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        
        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        assert scheduler_type is not None
        self.unified_scheduler = unified_scheduler
        self.stage_placement = stage_placement
        self.comm_graph = comm_graph
        self.scheduler_type = scheduler_type
        self.split_backward = split_backward
        self.layerwise = layerwise
        self.local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        self.num_layers = gpc.config.model.get("num_layers", torch.half)
        self.first_stage = first_stage
        self.last_stage = last_stage
        file_path = f"./jsonResult/async/{scheduler_type}_pp{gpc.pipeline_parallel_size}_chunk{num_chunks}_mb{num_microbatches}"
        os.makedirs(file_path, exist_ok=True)
        gpc._config['jsonpath'] = file_path+f"/iter0_rank{self.local_rank}_opeartion_list.json"
        if split_backward:
            self._backward_step_num = [0]*num_chunks
            self._num_microbatches = num_microbatches
        WeightGradStore.set_weight_grad_queue(num_chunks=num_chunks, num_microbatches=num_microbatches)

    def _clear_state(self) -> None:
        super()._clear_state()
        self._special_chunk0_forward = True
        self._chunk1_need_recv_prev_chunk1_grad = True
        local_placement = gpc.config.stage_placement[gpc.get_local_rank(ParallelMode.PIPELINE)]
        self._backward_step_num = [0]*len(local_placement)#self.num_chunks

    def recv_all(self,recv_forward_queue_list,recv_backward_queue_list,recvlist):
        for ops in recvlist:
            recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = ops
            recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
            if recv_op_type == Step.FORWARD.value:
                recv_f_buffer = comm.AsynCommunicator(
                            recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                            prev_rank=recv_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            # stage_id = recv_stage_id,
                            # microbatch_id = recv_microbatch_id,
                            # step_type = recv_op_type,
                            # local_rank = self.local_rank,
                            # chunk_id = recv_chunk_id,                      
                        )
                recv_f_buffer.start()

                if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                    recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                else:
                    recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)

            elif recv_op_type == Step.BACKWARD.value:
                recv_b_buffer = comm.AsynCommunicator(
                        recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                        next_rank=recv_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = recv_stage_id,
                        # microbatch_id = recv_microbatch_id,
                        # step_type = recv_op_type,
                        # local_rank = self.local_rank,
                        # chunk_id = recv_chunk_id,
                    )
                recv_b_buffer.start()

                if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                    recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                else:
                    recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)

        return recv_forward_queue_list, recv_backward_queue_list

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes

        # Used for tensor meta information communication
        local_rank = self.local_rank
        global_rank = gpc.get_global_rank()
        stage_placement=self.stage_placement
        steps = self.unified_scheduler[local_rank]
        stages_in_this_device = stage_placement[local_rank]
        chunks = self._num_chunks
        comm_list = self.comm_graph[local_rank]
        chunk_to_prev_stage_id = [-100 for _ in range(chunks)]
        chunk_to_next_stage_id = [self.last_stage+1 for _ in range(chunks)]
        chunk_to_prev_local_rank = [-100 for _ in range(chunks)]
        chunk_to_next_local_rank = [gpc.pipeline_parallel_size for _ in range(chunks)]
        chunk_to_prev_global_rank = [-100 for _ in range(chunks)]
        chunk_to_next_global_rank = [gpc.world_size for _ in range(chunks)]
        recv_backward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_queue_list = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_forward_queue = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_backward_queue = [queue.Queue() for _ in range(chunks)]
        input_obj_grad_map = [
        [[] for _ in range(self.num_microbatches)] 
        for _ in range(chunks)
        ]
        jsonpath = gpc._config['jsonpath']
        for i in range(chunks):
            chunk_id = i
            stage_id=stages_in_this_device[chunk_id]
            prev_stage = stage_id - 1
            next_stage = stage_id + 1
            chunk_to_prev_stage_id[chunk_id] = prev_stage
            chunk_to_next_stage_id[chunk_id] = next_stage
            if prev_stage >= 0:
                chunk_to_prev_local_rank[chunk_id] = _get_localrankid_by_placement(prev_stage,stage_placement)
                chunk_to_prev_global_rank[chunk_id] = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE, chunk_to_prev_local_rank[chunk_id])
            if next_stage <= self.last_stage:
                chunk_to_next_local_rank[chunk_id] = _get_localrankid_by_placement(next_stage,stage_placement)
                chunk_to_next_global_rank[chunk_id] = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,chunk_to_next_local_rank[chunk_id])

        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, chunk_id, startTime, end_time = steps[s]
            prev_stage = chunk_to_prev_stage_id[chunk_id]
            next_stage = chunk_to_next_stage_id[chunk_id]
            prev_global_rank = chunk_to_prev_global_rank[chunk_id]
            next_global_rank = chunk_to_next_global_rank[chunk_id]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']

            recv_forward_queue_list, recv_backward_queue_list = \
                self.recv_all(recv_forward_queue_list,recv_backward_queue_list,before_recv_list)
            if step_type == Step.FORWARD.value:# Forward pass
                input_obj = None
                if stage_id>self.first_stage:
                    if async_communicator_recv_forward_queue[chunk_id].qsize()>0:
                        input_obj = async_communicator_recv_forward_queue[chunk_id].get()
                    else:
                        input_obj,_ = recv_forward_queue_list[chunk_id].get().wait_and_receive()
                    self._input_objs[chunk_id].append(input_obj)

                # Perform forward computation
                # start_time  = time.time()
                # start_time_ = time.perf_counter()
                output_obj = self._forward_step(engine, chunk_id, input_obj)  
                # end_time = time.perf_counter()
                # json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute"}
                #                 # "start_time":start_time, "timespan":(end_time - start_time_), "output_obj_shape": output_obj.shape if output_obj is not None else 0, \
                #                 # "self.output_obj_shape": self.output_obj_shape if self.output_obj_shape is not None else 0}
                # write_json(jsonpath, json_content)
                if stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

                    assert self._output_obj_shapes[chunk_id] == self._input_obj_shapes[chunk_id]

                    if self._send_tensor_shape_flags[chunk_id]:
                        comm.send_obj_meta(output_obj,next_global_rank)
                        self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.
                if self.scheduler_type == ModuleType.INTERLEAVED.value and stage_id in self.stage_placement[-1] and stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id+1] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id+1] = [out_tensor.shape for out_tensor in output_obj]

                if stage_id > self.first_stage and self._input_obj_shapes[chunk_id] is None:
                    self._input_obj_shapes[chunk_id] = comm.recv_obj_meta(prev_rank=prev_global_rank)

                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)

                send_forward_once = True
                if global_rank == next_global_rank:
                    async_communicator_recv_forward_queue[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    #recv_forward_queue_list[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    send_forward_once = False
                elif local_rank%2 == 0 and stage_id<self.last_stage:
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                    ).start()
                    send_forward_once = False

                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                                    recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=recv_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                    # stage_id = recv_stage_id,
                                    # microbatch_id = recv_microbatch_id,
                                    # step_type = recv_op_type,
                                    # local_rank = self.local_rank,
                                    # chunk_id = recv_chunk_id,
                                )
                        recv_f_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)                        
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                                next_rank=recv_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                # stage_id = recv_stage_id,
                                # microbatch_id = recv_microbatch_id,
                                # step_type = recv_op_type,
                                # local_rank = self.local_rank,
                                # chunk_id = recv_chunk_id,
                            )
                        recv_b_buffer.start()

                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)                   

                if stage_id<self.last_stage and send_forward_once:
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                    ).start()

            elif step_type == Step.BACKWARD.value:# Backwaqsize

                if stage_id<self.last_stage:
                    if async_communicator_recv_backward_queue[chunk_id].qsize()>0:
                        output_obj_grad = async_communicator_recv_backward_queue[chunk_id].get()
                    else:
                        _, output_obj_grad = recv_backward_queue_list[chunk_id].get().wait_and_receive()
                    self._output_obj_grads[chunk_id].append(output_obj_grad)

                # start_time = time.time()
                # start_time_ = time.perf_counter()
                if self.split_backward:
                    origin_skip = engine.optimizer.skip_grad_reduce
                    input_obj_grad = self._schedule_backward(engine, chunk_id, microbatch_id)
                    input_obj_grad_map[chunk_id][microbatch_id] = input_obj_grad
                else:
                    input_obj_grad = InterleavedPipelineScheduler._backward_step(self, engine, chunk_id, microbatch_id)
                # end_time = time.perf_counter()
                # json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "input_obj_grad_shape": input_obj_grad.shape if input_obj_grad is not None else 0}
                #                 #"start_time":start_time, "timespan":(end_time - start_time_), "input_obj_grad_shape": input_obj_grad.shape if input_obj_grad is not None else 0}
                # write_json(jsonpath, json_content)
                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)

                send_backward_once = True
                if global_rank == prev_global_rank:
                    async_communicator_recv_backward_queue[chunk_id-1].put(input_obj_grad)
                    #recv_backward_queue_list[chunk_id-1].put(input_obj_grad)
                    send_backward_once = False
                elif local_rank%2 == 0 and stage_id > self.first_stage :
                    comm.AsynCommunicator(
                            object_send_prev=input_obj_grad,
                            prev_rank=prev_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            # stage_id = stage_id,
                            # microbatch_id = microbatch_id,
                            # step_type = step_type,
                            # local_rank = self.local_rank,
                            # chunk_id = chunk_id
                        ).start()
                    send_backward_once = False
                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                                    recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=recv_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                    # stage_id = recv_stage_id,
                                    # microbatch_id = recv_microbatch_id,
                                    # step_type = recv_op_type,
                                    # local_rank = self.local_rank,
                                    # chunk_id = recv_chunk_id,
                                )
                        recv_f_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                                next_rank=recv_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                # stage_id = recv_stage_id,
                                # microbatch_id = recv_microbatch_id,
                                # step_type = recv_op_type,
                                # local_rank = self.local_rank,
                                # chunk_id = recv_chunk_id,
                            )
                        recv_b_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)

                if stage_id > self.first_stage and send_backward_once:
                    comm.AsynCommunicator(
                        object_send_prev=input_obj_grad,
                        prev_rank=prev_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                    ).start()

            elif step_type == Step.WEIGHT.value: #Weight update
                # start_time = time.time()
                # start_time_ = time.perf_counter()
                WeightGradStore.pop(chunk_id=chunk_id,microbatch_id=microbatch_id)
                self._call_hooks("after_backward",input_obj_grad_map[chunk_id][microbatch_id])
                input_obj_grad_map[chunk_id][microbatch_id]=0
                engine.optimizer.skip_grad_reduce = origin_skip
                # end_time = time.perf_counter()
                # json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "input_obj_grad.shape":input_obj_grad_map[chunk_id][microbatch_id].shape if input_obj_grad_map[chunk_id][microbatch_id] is not None else 0}#"operation":"compute", "start_time":start_time, "timespan":(end_time - start_time_)}
                # write_json(jsonpath, json_content)
                recv_forward_queue_list, recv_backward_queue_list = \
                    self.recv_all(recv_forward_queue_list,recv_backward_queue_list,after_recv_list)

class UnifiedHetPipelineScheduler(ZeroBubblePipelineVShapeScheduler):
    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        stage_placement: List[List[int]] = None,
        comm_graph: List[List[tuple]] = None,
        scheduler_type: ModuleType = None,
        split_backward: bool = False,
        layerwise: bool = False,
        first_stage: int = None,
        last_stage: int = None
    ):
        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )

        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        assert scheduler_type is not None
        self.unified_scheduler = unified_scheduler
        self.stage_placement = stage_placement
        self.comm_graph = comm_graph
        self.scheduler_type = scheduler_type
        self.split_backward = split_backward
        self.layerwise = layerwise
        self.local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        self.num_layers = gpc.config.model.get("num_layers", torch.half)
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.input_obj_shape = self._input_obj_shapes[0]
        self.output_obj_shape = None
        file_path = f"InternEvo/jsonResult/het/{scheduler_type}_pp{gpc.pipeline_parallel_size}_layers{self.num_layers}_mb{num_microbatches}"
        os.makedirs(file_path, exist_ok=True)

        gpc._config['jsonpath'] = file_path+f"/iter0_rank{self.local_rank}_opeartion_list.json"
        #TODO, num_chunks is different in different rank
        if split_backward:
            self._backward_step_num = [0]*num_chunks
            self._num_microbatches = num_microbatches
        WeightGradStore.set_weight_grad_queue(num_chunks=num_chunks, num_microbatches=num_microbatches)

    def _clear_state(self) -> None:
        super()._clear_state()
        self._special_chunk0_forward = True
        self._chunk1_need_recv_prev_chunk1_grad = True
        local_placement = gpc.config.stage_placement[gpc.get_local_rank(ParallelMode.PIPELINE)]
        self._backward_step_num = [0]*len(local_placement)#self.num_chunks

    def _forward_step(self, engine, chunk_id, input_obj=None, stage_id = None):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if stage_id == self.first_stage and len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]):
            self._input_objs[chunk_id].append(None)

        if input_obj is None:
            input_obj = self._input_objs[chunk_id][-1]

        if stage_id > self.first_stage:
            assert input_obj is not None, f"{gpc.get_global_rank()} input is None"
        micro_batch_data = self.load_micro_batch(chunk_id)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        if hasattr(gpc.config.model, "num_experts"):
            output_obj, moe_losses = self._call_engine(engine.model[chunk_id], data)
        else:
            output_obj = self._call_engine(engine.model[chunk_id], data)
        # Convert output_obj to fp32 when last model chunk of last stage
        if stage_id == self.last_stage and isinstance(engine.model[chunk_id], NaiveAMPModel):
            output_obj = engine.model[chunk_id].convert_to_fp32(output_obj)
        self._call_hooks("after_forward", output_obj)
        if gpc.config.heter:
            if gpc.get_local_rank(parallel_mode=ParallelMode.PIPELINE) >= gpc.get_world_size(ParallelMode.PIPELINE)/2:
                for _ in range(gpc.config.sleep_forward_time_perlayer):
                    do_compute()

        if stage_id == self.last_stage:
            self._call_hooks("post_helper_func", output_obj, label)

            if self._return_tensors is not None:
                self._return_tensors.append((output_obj, label))
            if self._accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                self._accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            moe_loss = sum(moe_losses) * gpc.config.loss.moe_loss_coeff

            # the moe_loss is computed among the "tensor" group if sequence parallel is enabled,
            # so we need to do allreduce
            if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
                dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.TENSOR))
            moe_loss /= self.num_microbatches

            if self._accum_moe_loss is not None:
                self._accum_moe_loss.add_(moe_loss.detach())
        else:
            moe_loss = None

        self._output_objs[chunk_id].append(output_obj)
        self._moe_losses[chunk_id].append(moe_loss)

        assert output_obj is not None, f"{gpc.get_global_rank()} chunk{chunk_id} output is None"

        return output_obj
    
    #this is for split backward
    def _schedule_backward(self, engine, chunk_id, stage_id, microbatch_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        self._backward_step_num[chunk_id] += 1
        if self._backward_step_num[chunk_id] == self._num_microbatches:
            skip_grad_sync = False
        else:
            skip_grad_sync = True

        #if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
        if stage_id == self.last_stage and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
        moe_loss = self._moe_losses[chunk_id].pop(0)

        # if not gpc.is_pipeline_last_stage():
        if stage_id < self.last_stage:
            assert output_obj_grad is not None
        #if not gpc.is_pipeline_first_stage():
        if stage_id > self.first_stage:
            assert input_obj is not None

        input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad, skip_grad_sync, moe_loss)

        WeightGradStore.flush(chunk_id=chunk_id,microbatch_id=microbatch_id)

        return input_obj_grad

    def recv_all(self,recv_forward_queue_list,recv_backward_queue_list,recvlist):
        for ops in recvlist:
            recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = ops
            recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
            if recv_op_type == Step.FORWARD.value:
                recv_f_buffer = comm.AsynCommunicator_unified(
                            recv_prev_shape=self.input_obj_shape,
                            prev_rank=recv_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            # stage_id = recv_stage_id,
                            # microbatch_id = recv_microbatch_id,
                            # step_type = recv_op_type,
                            # chunk_id = recv_chunk_id, 
                            local_rank = self.local_rank,
                            match_rank = recv_device_id, 
                            step_id = self.step_id,               
                        )
                recv_f_buffer.start()
                store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id+1,self.stage_placement[self.local_rank])
                recv_forward_queue_list[store_recv_chunk_id].put(recv_f_buffer)
            elif recv_op_type == Step.BACKWARD.value:
                recv_b_buffer = comm.AsynCommunicator_unified(
                        recv_next_shape=self.output_obj_shape,
                        next_rank=recv_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = recv_stage_id,
                        # microbatch_id = recv_microbatch_id,
                        # step_type = recv_op_type,
                        # local_rank = self.local_rank,
                        # chunk_id = recv_chunk_id,
                        local_rank = self.local_rank,
                        match_rank = recv_device_id, 
                        step_id = self.step_id,   
                    )
                recv_b_buffer.start()
                store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id-1,self.stage_placement[self.local_rank])
                recv_backward_queue_list[store_recv_chunk_id].put(recv_b_buffer)
        return recv_forward_queue_list, recv_backward_queue_list

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes

        # Used for tensor meta information communication
        local_rank = self.local_rank
        global_rank = gpc.get_global_rank()
        steps = self.unified_scheduler[local_rank]
        stages_in_this_device = self.stage_placement[local_rank]
        chunks = self._num_chunks
        comm_list = self.comm_graph[local_rank]
        chunk_to_prev_stage_id = [-100 for _ in range(chunks)]
        chunk_to_next_stage_id = [self.last_stage+1 for _ in range(chunks)]
        chunk_to_prev_local_rank = [-100 for _ in range(chunks)]
        chunk_to_next_local_rank = [gpc.pipeline_parallel_size for _ in range(chunks)]
        chunk_to_prev_global_rank = [-100 for _ in range(chunks)]
        chunk_to_next_global_rank = [gpc.world_size for _ in range(chunks)]
        recv_backward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_queue_list = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_forward_queue = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_backward_queue = [queue.Queue() for _ in range(chunks)]
        input_obj_grad_map = [
        [[] for _ in range(self.num_microbatches)] 
        for _ in range(chunks)
        ]
        jsonpath = gpc._config['jsonpath']

        for i in range(chunks):
            chunk_id = i
            stage_id=stages_in_this_device[chunk_id]
            prev_stage = stage_id - 1
            next_stage = stage_id + 1
            chunk_to_prev_stage_id[chunk_id] = prev_stage
            chunk_to_next_stage_id[chunk_id] = next_stage
            if prev_stage >= 0:
                chunk_to_prev_local_rank[chunk_id] = _get_localrankid_by_placement(prev_stage,self.stage_placement)
                chunk_to_prev_global_rank[chunk_id] = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE, chunk_to_prev_local_rank[chunk_id])
            if next_stage <= self.last_stage:
                chunk_to_next_local_rank[chunk_id] = _get_localrankid_by_placement(next_stage,self.stage_placement)
                chunk_to_next_global_rank[chunk_id] = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,chunk_to_next_local_rank[chunk_id])

        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, chunk_id, startTime, end_time = steps[s]
            prev_stage = chunk_to_prev_stage_id[chunk_id]
            next_stage = chunk_to_next_stage_id[chunk_id]
            prev_local_rank = chunk_to_next_local_rank[chunk_id]
            next_local_rank = chunk_to_next_local_rank[chunk_id]
            prev_global_rank = chunk_to_prev_global_rank[chunk_id]
            next_global_rank = chunk_to_next_global_rank[chunk_id]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']
            self.step_id = s
            recv_forward_queue_list, recv_backward_queue_list = \
                self.recv_all(recv_forward_queue_list,recv_backward_queue_list,before_recv_list)
            if step_type == Step.FORWARD.value:# Forward pass
                input_obj = None
                if stage_id>self.first_stage:
                    if async_communicator_recv_forward_queue[chunk_id].qsize()>0:
                        input_obj = async_communicator_recv_forward_queue[chunk_id].get()
                    else:
                        input_obj,_ = recv_forward_queue_list[chunk_id].get().wait_and_receive()
                    self._input_objs[chunk_id].append(input_obj)

                # Perform forward computation
                start_time  = time.time()
                start_time_ = time.perf_counter()
                output_obj = self._forward_step(engine, chunk_id, input_obj, stage_id)
                end_time = time.perf_counter()
                json_content = {"step_type":step_type, "local_rank":local_rank, "step_id":self.step_id, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "operation":"compute", \
                                "start_time":start_time, "timespan":(end_time - start_time_)}
                write_json(jsonpath, json_content)

                #TODO:add a flag to determine whether to do num_chunks 
                if stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self.output_obj_shape = output_obj.shape
                        # self._output_obj_shapes[chunk_id] = output_obj.shape
                    else:
                        # self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]
                        self.output_obj_shape = [out_tensor.shape for out_tensor in output_obj]

                    # assert self._output_obj_shapes[chunk_id] == self._input_obj_shapes[chunk_id]

                    # if self._send_tensor_shape_flags[chunk_id]:
                    #     comm.send_obj_meta(output_obj,next_global_rank)
                    #     self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.

                #FIX, this maybe wrong, it should be the shape of the next chunk
                # if stage_id < self.last_stage and _get_chunkid_by_stage_placement(stage_id+1,self.stage_placement)-1 == chunk_id:
                #     if isinstance(output_obj, torch.Tensor):
                #         self._output_obj_shapes[chunk_id+1] = output_obj.shape
                #     else:
                #         self._output_obj_shapes[chunk_id+1] = [out_tensor.shape for out_tensor in output_obj]

                # if stage_id > self.first_stage and self._input_obj_shapes[chunk_id] is None:
                #     self._input_obj_shapes[chunk_id] = comm.recv_obj_meta(prev_rank=prev_global_rank)

                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)

                send_forward_once = True
                if global_rank == next_global_rank:
                    #async_communicator_recv_forward_queue[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    store_send_chunk_id = _get_chunkid_by_stages(stage_id+1,self.stage_placement[self.local_rank])
                    async_communicator_recv_forward_queue[store_send_chunk_id].put(output_obj.clone().detach().requires_grad_())
                    #recv_forward_queue_list[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    send_forward_once = False
                elif local_rank%2 == 0 and stage_id<self.last_stage:
                    comm.AsynCommunicator_unified(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                        local_rank = self.local_rank,
                        match_rank = next_local_rank, 
                        step_id = self.step_id,
                    ).start()
                    send_forward_once = False

                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator_unified(
                                    recv_prev_shape=self.input_obj_shape,
                                    prev_rank=recv_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                    # stage_id = recv_stage_id,
                                    # microbatch_id = recv_microbatch_id,
                                    # step_type = recv_op_type,
                                    # local_rank = self.local_rank,
                                    # chunk_id = recv_chunk_id,
                                    local_rank = self.local_rank,
                                    match_rank = recv_device_id, 
                                    step_id = self.step_id,
                                )
                        recv_f_buffer.start()
                        store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id+1,self.stage_placement[self.local_rank])
                        recv_forward_queue_list[store_recv_chunk_id].put(recv_f_buffer)                   
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator_unified(
                                recv_next_shape=self.output_obj_shape,
                                next_rank=recv_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                # stage_id = recv_stage_id,
                                # microbatch_id = recv_microbatch_id,
                                # step_type = recv_op_type,
                                # chunk_id = recv_chunk_id,
                                local_rank = self.local_rank,
                                match_rank = recv_device_id, 
                                step_id = self.step_id,
                            )
                        recv_b_buffer.start()
                        store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id-1,self.stage_placement[self.local_rank])
                        recv_backward_queue_list[store_recv_chunk_id].put(recv_b_buffer)                 

                if stage_id<self.last_stage and send_forward_once:
                    comm.AsynCommunicator_unified(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                        local_rank = self.local_rank,
                        match_rank = next_local_rank, 
                        step_id = self.step_id,               
                    ).start()

            elif step_type == Step.BACKWARD.value:# Backward pass

                if stage_id<self.last_stage:
                    if async_communicator_recv_backward_queue[chunk_id].qsize()>0:
                        output_obj_grad = async_communicator_recv_backward_queue[chunk_id].get()
                    else:
                        _, output_obj_grad = recv_backward_queue_list[chunk_id].get().wait_and_receive()
                    self._output_obj_grads[chunk_id].append(output_obj_grad)


                start_time = time.time()
                start_time_ = time.perf_counter()
                if self.split_backward:
                    origin_skip = engine.optimizer.skip_grad_reduce
                    input_obj_grad = self._schedule_backward(engine, chunk_id, stage_id, microbatch_id)
                    input_obj_grad_map[chunk_id][microbatch_id]=input_obj_grad
                else:
                    input_obj_grad = InterleavedPipelineScheduler._backward_step_(self, engine, chunk_id, microbatch_id, stage_id)
                end_time = time.perf_counter()
                json_content = {"step_type":step_type, "local_rank":local_rank, "step_id":self.step_id, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "operation":"compute", \
                                "start_time":start_time, "timespan":(end_time - start_time_)}
                write_json(jsonpath, json_content)

                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)

                send_backward_once = True
                if global_rank == prev_global_rank:
                    #async_communicator_recv_backward_queue[chunk_id-1].put(input_obj_grad)
                    store_send_chunk_id = _get_chunkid_by_stages(stage_id-1,self.stage_placement[self.local_rank])
                    async_communicator_recv_backward_queue[store_send_chunk_id].put(input_obj_grad)
                    #recv_backward_queue_list[chunk_id-1].put(input_obj_grad)
                    send_backward_once = False
                elif local_rank%2 == 0 and stage_id > self.first_stage :
                    comm.AsynCommunicator_unified(
                            object_send_prev=input_obj_grad,
                            prev_rank=prev_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            # stage_id = stage_id,
                            # microbatch_id = microbatch_id,
                            # step_type = step_type,
                            # local_rank = self.local_rank,
                            # chunk_id = chunk_id
                            local_rank = self.local_rank,
                            match_rank = prev_local_rank, 
                            step_id = self.step_id,
                        ).start()
                    send_backward_once = False
                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    recv_global_rank = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id)
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator_unified(
                                    recv_prev_shape=self.input_obj_shape,
                                    prev_rank=recv_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                    # stage_id = recv_stage_id,
                                    # microbatch_id = recv_microbatch_id,
                                    # step_type = recv_op_type,
                                    # local_rank = self.local_rank,
                                    # chunk_id = recv_chunk_id,
                                    local_rank = self.local_rank,
                                    match_rank = recv_device_id, 
                                    step_id = self.step_id,
                                )
                        recv_f_buffer.start()
                        store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id+1,self.stage_placement[self.local_rank])
                        recv_forward_queue_list[store_recv_chunk_id].put(recv_f_buffer)    
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator_unified(
                                recv_next_shape=self.output_obj_shape,
                                next_rank=recv_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                # stage_id = recv_stage_id,
                                # microbatch_id = recv_microbatch_id,
                                # step_type = recv_op_type,
                                # local_rank = self.local_rank,
                                # chunk_id = recv_chunk_id,
                                local_rank = self.local_rank,
                                match_rank = recv_device_id, 
                                step_id = self.step_id,
                            )
                        recv_b_buffer.start()
                        store_recv_chunk_id = _get_chunkid_by_stages(recv_stage_id-1,self.stage_placement[self.local_rank])
                        recv_backward_queue_list[store_recv_chunk_id].put(recv_b_buffer)
                if stage_id > self.first_stage and send_backward_once:
                    comm.AsynCommunicator_unified(
                        object_send_prev=input_obj_grad,
                        prev_rank=prev_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        # stage_id = stage_id,
                        # microbatch_id = microbatch_id,
                        # step_type = step_type,
                        # local_rank = self.local_rank,
                        # chunk_id = chunk_id
                        local_rank = self.local_rank,
                        match_rank = prev_local_rank, 
                        step_id = self.step_id,
                    ).start()
            elif step_type == Step.WEIGHT.value: #Weight update
                start_time = time.time()
                start_time_ = time.perf_counter()
                WeightGradStore.pop(chunk_id=chunk_id,microbatch_id=microbatch_id)
                self._call_hooks("after_backward",input_obj_grad_map[chunk_id][microbatch_id])
                if gpc.config.heter:
                    if gpc.get_local_rank(parallel_mode=ParallelMode.PIPELINE) >= gpc.get_world_size(ParallelMode.PIPELINE)/2:
                        for _ in range(gpc.config.sleep_backward_time_perlayer):
                            do_compute()
                input_obj_grad_map[chunk_id][microbatch_id] = 0
                engine.optimizer.skip_grad_reduce = origin_skip
                end_time = time.perf_counter()
                json_content = {"step_type":step_type, "local_rank":local_rank, "step_id":self.step_id, "chunk_id":chunk_id, "stage_id": stage_id, "microbatch_id":microbatch_id, "operation":"compute", "start_time":start_time, "timespan":(end_time - start_time_)}
                write_json(jsonpath, json_content)
                recv_forward_queue_list, recv_backward_queue_list = \
                    self.recv_all(recv_forward_queue_list,recv_backward_queue_list,after_recv_list)

    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

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

        gpc.set_virtual_pipeline_parallel_rank(0)

        self.load_batch(engine, data_iter)

        if return_loss and gpc.devices_have_lastStage:
            self._accum_loss = torch.zeros(1, device=get_current_device())
        
        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            self._accum_moe_loss = torch.zeros(1, device=get_current_device())

        if return_output_label:
            self._return_tensors = []

        if forward_only:
            self._forward_only_step(engine)
        else:
            self._forward_backward_step(engine)

        if return_output_label and len(self._return_tensors) > 0:
            output, label = pack_return_tensors(self._return_tensors)
        else:
            output, label = (None, None)

        accum_loss = self._accum_loss
        accum_moe_loss = self._accum_moe_loss

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(self._accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))
            accum_moe_loss = self._accum_moe_loss

            if accum_loss is not None:
                accum_loss += self._accum_moe_loss

        self._clear_state()

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss

class UnifiedMultipleStreamsPipelineScheduler(ZeroBubblePipelineVShapeScheduler):
    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        stage_placement: List[List[int]] = None,
        comm_graph: List[List[tuple]] = None,
    ):
        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        self.unified_scheduler = unified_scheduler
        self.stage_placement = stage_placement
        self.comm_graph = comm_graph
        self.last_stage = max([stage_id for _, stage_id in stage_placement])
        #self.last_stage = max(max(row) for row in stage_placement)
        self.first_stage = min(min(row) for row in stage_placement)
        self.scheduler_type = gpc.config.scheduler_type
        self.split_backward = gpc.config.split_backward
        assert self.scheduler_type == ModuleType.CHIMERA.value
        file_path = f"./jsonResult/async/{self.scheduler_type}_pp{gpc.pipeline_parallel_size}_chunk{num_chunks}_mb{num_microbatches}"
        os.makedirs(file_path, exist_ok=True)
        gpc._config['jsonpath'] = file_path + f"/iter_0_opeartion_list.json"
    
    def recv_all(self,recv_forward_queue_list,recv_backward_queue_list,recvlist):
        for ops in recvlist:
            recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = ops
            if recv_op_type == Step.FORWARD.value:
                recv_f_buffer = comm.AsynCommunicator(
                            recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                            prev_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                recv_f_buffer.start()
                if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                    recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                else:
                    recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)

            elif recv_op_type == Step.BACKWARD.value:
                recv_b_buffer = comm.AsynCommunicator(
                        recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                        next_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                recv_b_buffer.start()
                if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                    recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                else:
                    recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)

        return recv_forward_queue_list, recv_backward_queue_list

    def _schedule_backward(self, engine, chunk_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        self._backward_step_num[chunk_id] += 1
        if self._backward_step_num[chunk_id] == self._num_microbatches:
            skip_grad_sync = False
        else:
            skip_grad_sync = True

        if gpc.is_pipeline_last_stage_mutistream() and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
        moe_loss = self._moe_losses[chunk_id].pop(0)

        if not gpc.is_pipeline_last_stage_mutistream():
            assert output_obj_grad is not None
        if not gpc.is_pipeline_first_stage_mutistream():
            assert input_obj is not None

        input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad, skip_grad_sync, moe_loss)

        WeightGradStore.flush()

        return input_obj_grad
 
    def _forward_step(self, engine, chunk_id, input_obj=None):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_first_stage_mutistream() and len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]):
            self._input_objs[chunk_id].append(None)

        if input_obj is None:
            input_obj = self._input_objs[chunk_id][-1]

        if not gpc.is_pipeline_first_stage_mutistream():
            assert input_obj is not None, f"{gpc.get_global_rank()} input is None"
        micro_batch_data = self.load_micro_batch(chunk_id)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        if hasattr(gpc.config.model, "num_experts"):
            output_obj, moe_losses = self._call_engine(engine.model[chunk_id], data)
        else:
            output_obj = self._call_engine(engine.model[chunk_id], data)
        # Convert output_obj to fp32 when last model chunk of last stage
        if gpc.is_pipeline_last_stage_mutistream(ignore_virtual=False) and isinstance(engine.model[chunk_id], NaiveAMPModel):
            output_obj = engine.model[chunk_id].convert_to_fp32(output_obj)
        self._call_hooks("after_forward", output_obj)

        if gpc.is_pipeline_last_stage_mutistream():
            self._call_hooks("post_helper_func", output_obj, label)

            if self._return_tensors is not None:
                self._return_tensors.append((output_obj, label))
            if self._accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                self._accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            moe_loss = sum(moe_losses) * gpc.config.loss.moe_loss_coeff

            # the moe_loss is computed among the "tensor" group if sequence parallel is enabled,
            # so we need to do allreduce
            if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
                dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.TENSOR))
            moe_loss /= self.num_microbatches

            if self._accum_moe_loss is not None:
                self._accum_moe_loss.add_(moe_loss.detach())
        else:
            moe_loss = None

        self._output_objs[chunk_id].append(output_obj)
        self._moe_losses[chunk_id].append(moe_loss)

        assert output_obj is not None, f"{gpc.get_global_rank()} chunk{chunk_id} output is None"

        return output_obj

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes

        # Used for tensor meta information communication
        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        global_rank = gpc.get_global_rank()
        stage_placement=self.stage_placement
        steps = self.unified_scheduler[local_rank]
        stages_in_this_device = stage_placement[local_rank]
        comm_list = self.comm_graph[local_rank]
        chunks = len(stages_in_this_device)
        chunk_to_prev_stage_id = [-1 for _ in range(chunks)]
        chunk_to_next_stage_id = [gpc.world_size for _ in range(chunks)]
        chunk_to_prev_global_rank = [-1 for _ in range(chunks)]
        chunk_to_next_global_rank = [gpc.world_size for _ in range(chunks)]
        recv_backward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_queue_list = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_forward_queue = [queue.Queue() for _ in range(chunks)]
        async_communicator_recv_backward_queue = [queue.Queue() for _ in range(chunks)]
        input_obj_grad_queue_list = [queue.Queue() for _ in range(chunks)]
        jsonpath = gpc._config['jsonpath']

        for i in range(chunks):
            chunk_id = i
            stage_id=stages_in_this_device[chunk_id]
            prev_stage = stage_id - 1
            next_stage = stage_id + 1
            chunk_to_prev_stage_id[chunk_id] = prev_stage
            chunk_to_next_stage_id[chunk_id] = next_stage
            if prev_stage >= 0:
                chunk_to_prev_global_rank[chunk_id] =  gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,_get_localrankid_mutistream(chunk_id,prev_stage,stage_placement))
            if next_stage <= self.last_stage:
                chunk_to_next_global_rank[chunk_id] = gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,_get_localrankid_mutistream(chunk_id,next_stage,stage_placement))
        
        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, chunk_id, startTime, end_time = steps[s]
            prev_stage = chunk_to_prev_stage_id[chunk_id]
            next_stage = chunk_to_next_stage_id[chunk_id]
            prev_global_rank = chunk_to_prev_global_rank[chunk_id]
            next_global_rank = chunk_to_next_global_rank[chunk_id]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']

            recv_forward_queue_list, recv_backward_queue_list = \
                self.recv_all(recv_forward_queue_list,recv_backward_queue_list,before_recv_list)
            if step_type == Step.FORWARD.value:# Forward pass
                input_obj = None
                if stage_id>self.first_stage:
                    if async_communicator_recv_forward_queue[chunk_id].qsize()>0:
                        input_obj = async_communicator_recv_forward_queue[chunk_id].get()
                    else:
                        input_obj,_ = recv_forward_queue_list[chunk_id].get().wait_and_receive()
                    self._input_objs[chunk_id].append(input_obj)

                # Perform forward computation
                start_time = time.perf_counter()
                output_obj = self._forward_step(engine, chunk_id, input_obj)  
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)

                if stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

                    assert self._output_obj_shapes[chunk_id] == self._input_obj_shapes[chunk_id]

                    if self._send_tensor_shape_flags[chunk_id]:
                        comm.send_obj_meta(output_obj,next_global_rank)
                        self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.
                if self.scheduler_type == ModuleType.INTERLEAVED.value and stage_id in self.stage_placement[-1] and stage_id < self.last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id+1] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id+1] = [out_tensor.shape for out_tensor in output_obj]

                if stage_id > self.first_stage and self._input_obj_shapes[chunk_id] is None:
                    self._input_obj_shapes[chunk_id] = comm.recv_obj_meta(prev_rank=prev_global_rank)

                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)
                send_forward_once = True
                if global_rank == next_global_rank:
                    async_communicator_recv_forward_queue[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    send_forward_once = False
                elif local_rank%2 == 0 and stage_id<self.last_stage:
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors
                    ).start()
                    send_forward_once = False

                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                                    recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                        recv_f_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)                        
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                                next_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        recv_b_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)                   

                if stage_id<self.last_stage and send_forward_once:
                    comm.AsynCommunicator(
                        object_send_next=output_obj,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors
                    ).start()

            elif step_type == Step.BACKWARD.value:# Backwaqsize

                if stage_id<self.last_stage:
                    if async_communicator_recv_backward_queue[chunk_id].qsize()>0:
                        output_obj_grad = async_communicator_recv_backward_queue[chunk_id].get()
                    else:
                        _, output_obj_grad = recv_backward_queue_list[chunk_id].get().wait_and_receive()
                    self._output_obj_grads[chunk_id].append(output_obj_grad)

                start_time = time.perf_counter()
                if self.split_backward:
                    origin_skip = engine.optimizer.skip_grad_reduce
                    input_obj_grad = self._schedule_backward(engine, chunk_id)
                else:
                    input_obj_grad = InterleavedPipelineScheduler._backward_step(self, engine, chunk_id, microbatch_id)

                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)
                
                input_obj_grad_queue_list[chunk_id].put(input_obj_grad)
                for chunk in range (chunks):
                    if recv_forward_queue_list[chunk].qsize()>0:
                        recv_f_tensor, _ = recv_forward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_forward_queue[chunk].put(recv_f_tensor)
                    if recv_backward_queue_list[chunk].qsize()>0:
                        _, recv_b_tensor = recv_backward_queue_list[chunk].get().wait_and_receive()
                        async_communicator_recv_backward_queue[chunk].put(recv_b_tensor)

                send_backward_once = True
                if global_rank == prev_global_rank:
                    async_communicator_recv_backward_queue[chunk_id-1].put(input_obj_grad)
                    send_backward_once = False
                elif local_rank%2 == 0 and stage_id > self.first_stage :
                    comm.AsynCommunicator(
                            object_send_prev=input_obj_grad,
                            prev_rank=prev_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors
                        ).start()
                    send_backward_once = False
                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = after_ops      
                    if recv_op_type == Step.FORWARD.value:
                        recv_f_buffer = comm.AsynCommunicator(
                                    recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                        recv_f_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(recv_f_buffer)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(recv_f_buffer)
                    elif recv_op_type == Step.BACKWARD.value:
                        recv_b_buffer = comm.AsynCommunicator(
                                recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                                next_rank=gpc.get_global_rank_by_local_rank(ParallelMode.PIPELINE,recv_device_id),
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        recv_b_buffer.start()
                        if self.scheduler_type == ModuleType.INTERLEAVED.value and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(recv_b_buffer)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(recv_b_buffer)

                if stage_id > self.first_stage and send_backward_once:
                    comm.AsynCommunicator(
                        object_send_prev=input_obj_grad,
                        prev_rank=prev_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors
                    ).start()

            elif step_type == Step.WEIGHT.value: # Weight update
                start_time = time.perf_counter()
                WeightGradStore.pop()
                self._call_hooks("after_backward",input_obj_grad_queue_list[chunk_id].get())
                engine.optimizer.skip_grad_reduce = origin_skip
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)
                recv_forward_queue_list, recv_backward_queue_list = \
                    self.recv_all(recv_forward_queue_list,recv_backward_queue_list,after_recv_list)

    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

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

        gpc.set_virtual_pipeline_parallel_rank(0)

        self.load_batch(engine, data_iter)

        if return_loss and gpc.is_pipeline_last_stage_mutistream(ignore_virtual=True):
            self._accum_loss = torch.zeros(1, device=get_current_device())

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            self._accum_moe_loss = torch.zeros(1, device=get_current_device())

        if return_output_label:
            self._return_tensors = []

        if forward_only:
            self._forward_only_step(engine)
        else:
            self._forward_backward_step(engine)

        if return_output_label and len(self._return_tensors) > 0:
            output, label = pack_return_tensors(self._return_tensors)
        else:
            output, label = (None, None)

        accum_loss = self._accum_loss
        accum_moe_loss = self._accum_moe_loss

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(self._accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))
            accum_moe_loss = self._accum_moe_loss

            if accum_loss is not None:
                accum_loss += self._accum_moe_loss

        self._clear_state()

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss
