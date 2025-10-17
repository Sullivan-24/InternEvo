#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import torch
import torch.distributed as dist
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

def debug_zero1_hang():
    """Debug ZERO1 all_reduce hang issue"""
    
    logger.info("=== ZERO1 Hang Debug Analysis ===")
    
    # 1. 检查分布式环境
    if not dist.is_initialized():
        logger.error("Distributed environment not initialized!")
        return False
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"Rank: {rank}, World Size: {world_size}")
    
    # 2. 检查ZERO1进程组状态
    zero1_initialized = gpc.is_initialized(ParallelMode.ZERO1)
    logger.info(f"ZERO1 initialized: {zero1_initialized}")
    
    if not zero1_initialized:
        logger.error("ZERO1 parallel mode not initialized!")
        return False
    
    # 3. 获取ZERO1进程组信息
    try:
        zero1_group = gpc.get_group(ParallelMode.ZERO1)
        zero1_rank = gpc.get_local_rank(ParallelMode.ZERO1)
        zero1_world_size = gpc.get_world_size(ParallelMode.ZERO1)
        zero1_ranks = gpc.get_ranks_in_group(ParallelMode.ZERO1)
        
        logger.info(f"ZERO1 Group: {zero1_group}")
        logger.info(f"ZERO1 Local Rank: {zero1_rank}")
        logger.info(f"ZERO1 World Size: {zero1_world_size}")
        logger.info(f"ZERO1 Ranks in Group: {zero1_ranks}")
        
    except Exception as e:
        logger.error(f"Failed to get ZERO1 group info: {e}")
        return False
    
    # 4. 检查所有进程是否都到达了这个点
    logger.info("Checking if all processes reach this point...")
    try:
        # 使用全局barrier确保所有进程都到达这里
        dist.barrier()
        logger.info("All processes reached barrier")
    except Exception as e:
        logger.error(f"Global barrier failed: {e}")
        return False
    
    # 5. 测试ZERO1组内的barrier
    logger.info("Testing ZERO1 group barrier...")
    try:
        dist.barrier(group=zero1_group)
        logger.info("ZERO1 group barrier successful")
    except Exception as e:
        logger.error(f"ZERO1 group barrier failed: {e}")
        return False
    
    # 6. 测试小规模all_reduce
    logger.info("Testing small all_reduce operation...")
    try:
        buffer = torch.ones([1], device=get_current_device()) * rank
        logger.info(f"Before all_reduce: buffer = {buffer.item()}")
        
        # 设置超时
        start_time = time.time()
        dist.all_reduce(buffer, group=zero1_group)
        end_time = time.time()
        
        logger.info(f"After all_reduce: buffer = {buffer.item()}, time = {end_time - start_time:.2f}s")
        
        # 验证结果
        expected_sum = sum(zero1_ranks)
        if abs(buffer.item() - expected_sum) < 1e-6:
            logger.info("Small all_reduce test PASSED")
            return True
        else:
            logger.error(f"Small all_reduce test FAILED: expected {expected_sum}, got {buffer.item()}")
            return False
            
    except Exception as e:
        logger.error(f"Small all_reduce test failed: {e}")
        return False

def debug_with_timeout():
    """Debug with timeout mechanism"""
    
    import signal
    
    def timeout_handler(signum, frame):
        logger.error("ZERO1 all_reduce operation timed out!")
        raise TimeoutError("Operation timed out")
    
    # 设置30秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        result = debug_zero1_hang()
        signal.alarm(0)  # 取消超时
        return result
    except TimeoutError:
        logger.error("Debug operation timed out - likely a hang issue")
        return False
    except Exception as e:
        signal.alarm(0)
        logger.error(f"Debug operation failed: {e}")
        return False

def safe_all_reduce_with_retry(buffer, group, max_retries=3, timeout=30):
    """Safe all_reduce with retry mechanism"""
    
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("all_reduce timed out")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting all_reduce (attempt {attempt + 1}/{max_retries})")
            
            # 设置超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            # 执行all_reduce
            dist.all_reduce(buffer, group=group)
            
            # 取消超时
            signal.alarm(0)
            
            logger.info("all_reduce completed successfully")
            return True
            
        except TimeoutError:
            logger.warning(f"all_reduce attempt {attempt + 1} timed out")
            signal.alarm(0)
            
            if attempt < max_retries - 1:
                logger.info("Retrying after barrier synchronization...")
                try:
                    dist.barrier(group=group)
                    time.sleep(1)
                except:
                    logger.warning("Barrier synchronization failed")
                    
        except Exception as e:
            signal.alarm(0)
            logger.error(f"all_reduce attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
    
    logger.error("All all_reduce attempts failed")
    return False

def check_process_synchronization():
    """Check if all processes in ZERO1 group are synchronized"""
    
    if not gpc.is_initialized(ParallelMode.ZERO1):
        logger.error("ZERO1 not initialized")
        return False
    
    rank = dist.get_rank()
    zero1_group = gpc.get_group(ParallelMode.ZERO1)
    zero1_ranks = gpc.get_ranks_in_group(ParallelMode.ZERO1)
    
    logger.info(f"Rank {rank} checking synchronization with ZERO1 group: {zero1_ranks}")
    
    # 创建一个包含当前时间戳的tensor
    timestamp = torch.tensor([time.time()], device=get_current_device())
    
    try:
        # 尝试all_gather来检查所有进程是否都活跃
        gathered_timestamps = [torch.zeros_like(timestamp) for _ in zero1_ranks]
        dist.all_gather(gathered_timestamps, timestamp, group=zero1_group)
        
        logger.info("Process synchronization check passed")
        for i, ts in enumerate(gathered_timestamps):
            logger.info(f"Rank {zero1_ranks[i]} timestamp: {ts.item()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Process synchronization check failed: {e}")
        return False

if __name__ == "__main__":
    # 运行调试
    logger.info("Starting ZERO1 hang debug...")
    
    # 检查进程同步
    check_process_synchronization()
    
    # 运行超时调试
    debug_with_timeout()