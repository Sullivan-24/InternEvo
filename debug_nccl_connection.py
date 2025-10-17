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

def debug_nccl_connection():
    """Debug NCCL connection issues with comprehensive diagnostics"""
    
    # Set NCCL debug environment variables
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Increase timeout
    os.environ["NCCL_TIMEOUT"] = "300"
    
    # Network interface configuration
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    
    # Disable problematic features for debugging
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    logger.info("NCCL Debug environment variables set")
    
    if not dist.is_initialized():
        logger.error("Distributed environment not initialized")
        return False
    
    try:
        # Test basic connectivity first
        logger.info("Testing basic NCCL connectivity...")
        buffer = torch.ones([64], device=get_current_device())
        
        # Test different parallel modes with retry mechanism
        parallel_modes = [
            (ParallelMode.DATA, "DATA"),
            (ParallelMode.TENSOR, "TENSOR"), 
            (ParallelMode.PIPELINE, "PIPELINE"),
            (ParallelMode.ZERO1, "ZERO1"),
            (ParallelMode.MODEL, "MODEL"),
        ]
        
        for mode, mode_name in parallel_modes:
            if gpc.is_initialized(mode):
                success = test_parallel_mode_with_retry(buffer, mode, mode_name)
                if not success:
                    logger.error(f"Failed to establish NCCL connection for {mode_name}")
                    return False
                    
        logger.info("All NCCL connections tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"NCCL connection test failed: {e}")
        return False

def test_parallel_mode_with_retry(buffer, mode, mode_name, max_retries=3, retry_delay=5):
    """Test parallel mode with retry mechanism"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Testing {mode_name} parallel mode (attempt {attempt + 1}/{max_retries})")
            
            group = gpc.get_group(mode)
            dist.all_reduce(buffer.clone(), group=group)
            
            logger.info(f"{mode_name} parallel mode test successful")
            return True
            
        except Exception as e:
            logger.warning(f"{mode_name} test attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"{mode_name} test failed after {max_retries} attempts")
                return False
    
    return False

def check_network_connectivity():
    """Check basic network connectivity between nodes"""
    
    if not dist.is_initialized():
        return False
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"Rank {rank}/{world_size} checking network connectivity...")
    
    # Get all ranks in the same node
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Simple connectivity test
    try:
        tensor = torch.tensor([rank], device=get_current_device())
        dist.all_reduce(tensor)
        expected_sum = sum(range(world_size))
        
        if tensor.item() == expected_sum:
            logger.info("Basic network connectivity test passed")
            return True
        else:
            logger.error(f"Network connectivity test failed. Expected {expected_sum}, got {tensor.item()}")
            return False
            
    except Exception as e:
        logger.error(f"Network connectivity test failed: {e}")
        return False

def get_nccl_diagnostics():
    """Get NCCL diagnostic information"""
    
    diagnostics = {
        "rank": dist.get_rank() if dist.is_initialized() else "Not initialized",
        "world_size": dist.get_world_size() if dist.is_initialized() else "Not initialized", 
        "backend": dist.get_backend() if dist.is_initialized() else "Not initialized",
        "device": get_current_device(),
        "nccl_version": torch.cuda.nccl.version() if torch.cuda.is_available() else "CUDA not available",
    }
    
    # Environment variables
    nccl_env_vars = [
        "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_TIMEOUT",
        "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"
    ]
    
    for var in nccl_env_vars:
        diagnostics[var] = os.environ.get(var, "Not set")
    
    logger.info("NCCL Diagnostics:")
    for key, value in diagnostics.items():
        logger.info(f"  {key}: {value}")
    
    return diagnostics

if __name__ == "__main__":
    # Run comprehensive NCCL diagnostics
    get_nccl_diagnostics()
    check_network_connectivity()
    debug_nccl_connection()