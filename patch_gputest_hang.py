#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Patch for internlm/utils/gputest.py to fix ZERO1 all_reduce hang issue
"""

import time
import torch
import torch.distributed as dist
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

def safe_all_reduce_with_timeout(buffer, group, mode_name, timeout=30):
    """
    Safe all_reduce operation with timeout and error handling
    
    Args:
        buffer: tensor to reduce
        group: process group
        mode_name: name of parallel mode for logging
        timeout: timeout in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    import signal
    import threading
    
    class AllReduceTimeoutError(Exception):
        pass
    
    def timeout_handler():
        raise AllReduceTimeoutError(f"{mode_name} all_reduce timed out after {timeout}s")
    
    # 使用线程计时器实现超时
    timer = threading.Timer(timeout, timeout_handler)
    
    try:
        logger.info(f"Starting {mode_name} all_reduce with timeout {timeout}s...")
        
        # 首先测试group barrier确保所有进程都到达
        logger.debug(f"Testing {mode_name} barrier before all_reduce...")
        timer.start()
        dist.barrier(group=group)
        timer.cancel()
        
        # 执行all_reduce
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        
        start_time = time.time()
        dist.all_reduce(buffer, group=group)
        end_time = time.time()
        
        timer.cancel()
        
        logger.info(f"{mode_name} all_reduce completed successfully in {end_time - start_time:.2f}s")
        return True
        
    except AllReduceTimeoutError as e:
        if timer.is_alive():
            timer.cancel()
        logger.error(f"{mode_name} all_reduce timed out: {e}")
        return False
        
    except Exception as e:
        if timer.is_alive():
            timer.cancel()
        logger.error(f"{mode_name} all_reduce failed: {e}")
        return False

def patched_warmup_process_group():
    """
    Patched version of warmup_process_group that handles hangs properly
    """
    # Prevent OOM from nccl communication.
    if not dist.is_initialized():
        logger.warning("Distributed environment not initialized, skipping warmup")
        return
    
    logger.info("Starting patched warmup process group...")
    
    buffer = torch.ones([64], device=get_current_device())
    
    # 定义并行模式和超时时间
    parallel_modes = [
        (ParallelMode.DATA, "DATA", 10),
        (ParallelMode.TENSOR, "TENSOR", 10),
        (ParallelMode.PIPELINE, "PIPELINE", 15),  # Pipeline可能需要更长时间
        (ParallelMode.ZERO1, "ZERO1", 20),        # ZERO1最容易卡住，给更长时间
        (ParallelMode.MODEL, "MODEL", 10),
        (ParallelMode.ZERO3_DP, "ZERO3_DP", 15),
        (ParallelMode.EXPERT_DATA, "EXPERT_DATA", 10),
        (ParallelMode.EXPERT, "EXPERT", 10),
    ]
    
    successful_modes = []
    failed_modes = []
    
    for mode, mode_name, timeout in parallel_modes:
        if gpc.is_initialized(mode):
            logger.info(f"Testing {mode_name} parallel mode...")
            
            try:
                # 获取进程组信息
                group = gpc.get_group(mode)
                local_rank = gpc.get_local_rank(mode)
                world_size = gpc.get_world_size(mode)
                ranks_in_group = gpc.get_ranks_in_group(mode)
                
                logger.info(f"{mode_name} - Local rank: {local_rank}/{world_size}, "
                           f"Ranks: {ranks_in_group}")
                
                # 创建测试buffer
                test_buffer = buffer.clone()
                
                # 执行安全的all_reduce
                success = safe_all_reduce_with_timeout(test_buffer, group, mode_name, timeout)
                
                if success:
                    successful_modes.append(mode_name)
                    logger.info(f"{mode_name} warmup successful")
                else:
                    failed_modes.append(mode_name)
                    logger.error(f"{mode_name} warmup failed")
                    
            except Exception as e:
                failed_modes.append(mode_name)
                logger.error(f"{mode_name} warmup exception: {e}")
                continue
        else:
            logger.debug(f"{mode_name} parallel mode not initialized, skipping")
    
    # 报告结果
    logger.info(f"Warmup completed - Successful: {successful_modes}, Failed: {failed_modes}")
    
    # 最后的全局barrier（如果有失败的模式，跳过以避免死锁）
    if not failed_modes:
        try:
            logger.info("Final global barrier...")
            dist.barrier()
            logger.info("All warmup operations completed successfully")
        except Exception as e:
            logger.error(f"Final barrier failed: {e}")
    else:
        logger.warning("Skipping final barrier due to failed modes")
    
    # 清理
    del buffer
    torch.cuda.empty_cache()

def diagnose_before_all_reduce(mode, mode_name):
    """
    Diagnostic function to run before problematic all_reduce operations
    """
    logger.info(f"=== Diagnosing {mode_name} before all_reduce ===")
    
    try:
        group = gpc.get_group(mode)
        local_rank = gpc.get_local_rank(mode)
        world_size = gpc.get_world_size(mode)
        ranks_in_group = gpc.get_ranks_in_group(mode)
        global_rank = dist.get_rank()
        
        logger.info(f"Global rank: {global_rank}")
        logger.info(f"{mode_name} local rank: {local_rank}")
        logger.info(f"{mode_name} world size: {world_size}")
        logger.info(f"{mode_name} ranks in group: {ranks_in_group}")
        logger.info(f"{mode_name} group object: {group}")
        
        # 测试简单的all_gather来验证连通性
        test_tensor = torch.tensor([global_rank], device=get_current_device())
        gathered = [torch.zeros_like(test_tensor) for _ in range(world_size)]
        
        logger.info(f"Testing {mode_name} all_gather connectivity...")
        dist.all_gather(gathered, test_tensor, group=group)
        
        gathered_ranks = [t.item() for t in gathered]
        logger.info(f"Gathered ranks: {gathered_ranks}")
        
        if set(gathered_ranks) == set(ranks_in_group):
            logger.info(f"{mode_name} connectivity test PASSED")
            return True
        else:
            logger.error(f"{mode_name} connectivity test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"{mode_name} diagnosis failed: {e}")
        return False

# 使用示例：替换原始的warmup_process_group调用
if __name__ == "__main__":
    # 在实际使用中，你可以这样调用：
    # from patch_gputest_hang import patched_warmup_process_group
    # patched_warmup_process_group()
    
    logger.info("ZERO1 hang patch loaded successfully")
    logger.info("Use patched_warmup_process_group() to replace the original function")