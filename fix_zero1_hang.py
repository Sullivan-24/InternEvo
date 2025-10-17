#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import torch
import torch.distributed as dist
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

def fixed_warmup_process_group():
    """Fixed version of warmup_process_group that handles hangs"""
    
    if not dist.is_initialized():
        logger.warning("Distributed environment not initialized, skipping warmup")
        return
    
    logger.info("Starting fixed warmup process group...")
    
    # 创建测试buffer
    buffer = torch.ones([64], device=get_current_device())
    
    # 定义要测试的并行模式
    parallel_modes = [
        (ParallelMode.DATA, "DATA"),
        (ParallelMode.TENSOR, "TENSOR"),
        (ParallelMode.PIPELINE, "PIPELINE"),
        (ParallelMode.ZERO1, "ZERO1"),
        (ParallelMode.MODEL, "MODEL"),
        (ParallelMode.ZERO3_DP, "ZERO3_DP"),
        (ParallelMode.EXPERT_DATA, "EXPERT_DATA"),
        (ParallelMode.EXPERT, "EXPERT"),
    ]
    
    for mode, mode_name in parallel_modes:
        if gpc.is_initialized(mode):
            logger.info(f"Testing {mode_name} parallel mode...")
            
            try:
                # 获取进程组信息
                group = gpc.get_group(mode)
                local_rank = gpc.get_local_rank(mode)
                world_size = gpc.get_world_size(mode)
                ranks_in_group = gpc.get_ranks_in_group(mode)
                
                logger.info(f"{mode_name} - Local rank: {local_rank}, World size: {world_size}")
                logger.info(f"{mode_name} - Ranks in group: {ranks_in_group}")
                
                # 首先测试barrier
                logger.info(f"Testing {mode_name} barrier...")
                dist.barrier(group=group)
                logger.info(f"{mode_name} barrier successful")
                
                # 然后测试all_reduce
                test_buffer = buffer.clone()
                logger.info(f"Testing {mode_name} all_reduce...")
                
                success = safe_all_reduce(test_buffer, group, mode_name)
                if success:
                    logger.info(f"{mode_name} all_reduce successful")
                else:
                    logger.error(f"{mode_name} all_reduce failed")
                    
            except Exception as e:
                logger.error(f"{mode_name} test failed: {e}")
                continue
    
    # 最后做全局barrier
    try:
        logger.info("Final global barrier...")
        dist.barrier()
        logger.info("Warmup process group completed successfully")
    except Exception as e:
        logger.error(f"Final barrier failed: {e}")
    
    # 清理
    del buffer
    torch.cuda.empty_cache()

def safe_all_reduce(buffer, group, mode_name, timeout=30):
    """Safe all_reduce with timeout and error handling"""
    
    import signal
    import threading
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler():
        raise TimeoutException(f"{mode_name} all_reduce timed out after {timeout}s")
    
    # 使用线程实现超时
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    
    try:
        start_time = time.time()
        dist.all_reduce(buffer, group=group)
        end_time = time.time()
        
        timer.cancel()  # 取消超时器
        
        logger.info(f"{mode_name} all_reduce completed in {end_time - start_time:.2f}s")
        return True
        
    except TimeoutException as e:
        timer.cancel()
        logger.error(str(e))
        return False
        
    except Exception as e:
        timer.cancel()
        logger.error(f"{mode_name} all_reduce failed: {e}")
        return False

def diagnose_zero1_hang():
    """Diagnose ZERO1 specific hang issues"""
    
    logger.info("=== ZERO1 Hang Diagnosis ===")
    
    if not gpc.is_initialized(ParallelMode.ZERO1):
        logger.error("ZERO1 parallel mode not initialized!")
        return
    
    # 获取ZERO1配置信息
    try:
        zero1_rank = gpc.get_local_rank(ParallelMode.ZERO1)
        zero1_world_size = gpc.get_world_size(ParallelMode.ZERO1)
        zero1_ranks = gpc.get_ranks_in_group(ParallelMode.ZERO1)
        zero1_group = gpc.get_group(ParallelMode.ZERO1)
        
        global_rank = dist.get_rank()
        global_world_size = dist.get_world_size()
        
        logger.info(f"Global rank: {global_rank}/{global_world_size}")
        logger.info(f"ZERO1 rank: {zero1_rank}/{zero1_world_size}")
        logger.info(f"ZERO1 ranks in group: {zero1_ranks}")
        
        # 检查配置参数
        tensor_parallel_size = gpc.get_world_size(ParallelMode.TENSOR) if gpc.is_initialized(ParallelMode.TENSOR) else 1
        pipeline_parallel_size = gpc.get_world_size(ParallelMode.PIPELINE) if gpc.is_initialized(ParallelMode.PIPELINE) else 1
        data_parallel_size = gpc.get_world_size(ParallelMode.DATA) if gpc.is_initialized(ParallelMode.DATA) else 1
        
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"Data parallel size: {data_parallel_size}")
        logger.info(f"ZERO1 parallel size: {zero1_world_size}")
        
        # 检查配置一致性
        expected_world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if global_world_size != expected_world_size:
            logger.warning(f"World size mismatch: {global_world_size} != {expected_world_size}")
        
        # 测试ZERO1组内通信
        logger.info("Testing ZERO1 group communication...")
        
        # 1. 测试barrier
        logger.info("Testing ZERO1 barrier...")
        dist.barrier(group=zero1_group)
        logger.info("ZERO1 barrier successful")
        
        # 2. 测试all_gather（比all_reduce更安全）
        logger.info("Testing ZERO1 all_gather...")
        tensor = torch.tensor([global_rank], device=get_current_device())
        gathered = [torch.zeros_like(tensor) for _ in range(zero1_world_size)]
        dist.all_gather(gathered, tensor, group=zero1_group)
        
        gathered_ranks = [t.item() for t in gathered]
        logger.info(f"Gathered ranks: {gathered_ranks}")
        logger.info(f"Expected ranks: {zero1_ranks}")
        
        if gathered_ranks == zero1_ranks:
            logger.info("ZERO1 all_gather test PASSED")
        else:
            logger.error("ZERO1 all_gather test FAILED")
        
        # 3. 最后测试all_reduce
        logger.info("Testing ZERO1 all_reduce...")
        buffer = torch.ones([64], device=get_current_device())
        success = safe_all_reduce(buffer, zero1_group, "ZERO1", timeout=10)
        
        if success:
            logger.info("ZERO1 all_reduce test PASSED")
        else:
            logger.error("ZERO1 all_reduce test FAILED")
            
    except Exception as e:
        logger.error(f"ZERO1 diagnosis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def check_common_hang_causes():
    """Check common causes of all_reduce hangs"""
    
    logger.info("=== Checking Common Hang Causes ===")
    
    # 1. 检查NCCL环境变量
    nccl_vars = [
        "NCCL_DEBUG", "NCCL_ASYNC_ERROR_HANDLING", "NCCL_TIMEOUT",
        "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE"
    ]
    
    logger.info("NCCL Environment Variables:")
    for var in nccl_vars:
        value = os.environ.get(var, "Not set")
        logger.info(f"  {var}: {value}")
    
    # 2. 检查设备状态
    if torch.cuda.is_available():
        device = get_current_device()
        logger.info(f"Current device: {device}")
        logger.info(f"Device count: {torch.cuda.device_count()}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # 3. 检查进程组状态
    if dist.is_initialized():
        logger.info(f"Backend: {dist.get_backend()}")
        logger.info(f"Global rank: {dist.get_rank()}")
        logger.info(f"World size: {dist.get_world_size()}")
    
    # 4. 检查网络连接
    import socket
    hostname = socket.gethostname()
    logger.info(f"Hostname: {hostname}")

if __name__ == "__main__":
    import os
    
    # 设置调试环境
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    logger.info("Starting ZERO1 hang fix diagnosis...")
    
    check_common_hang_causes()
    diagnose_zero1_hang()
    
    logger.info("Running fixed warmup...")
    fixed_warmup_process_group()