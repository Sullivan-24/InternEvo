#!/bin/bash

###############################################################################
# InternEvo 训练启动脚本 - 包含 NCCL 优化配置
# 用于解决 NCCL AllReduce 超时问题
###############################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 基础配置
# ============================================================================

# 训练配置文件路径
CONFIG_FILE="${1:-configs/7B_sft.py}"

# 节点配置
NNODES=${NNODES:-1}              # 节点数量
NODE_RANK=${NODE_RANK:-0}       # 当前节点 rank
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # 每个节点的 GPU 数量

# Master 地址配置
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ============================================================================
# NCCL 基础配置
# ============================================================================

# 调试级别: WARN (生产), INFO (调试), TRACE (详细调试)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL}  # 只记录初始化和集合通信

# 超时配置 (秒)
# 默认 1800 秒 (30 分钟),如果频繁超时可以增加
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

# 异步错误处理
export NCCL_ASYNC_ERROR_HANDLING=1

# ============================================================================
# 网络配置 - 根据实际硬件选择
# ============================================================================

# 检测是否有 InfiniBand
if command -v ibstat &> /dev/null && ibstat 2>/dev/null | grep -q "State: Active"; then
    echo "检测到 InfiniBand,配置 NCCL 使用 IB..."
    
    # 启用 InfiniBand
    export NCCL_IB_DISABLE=0
    
    # IB 设备配置 (根据实际硬件调整)
    # 使用 ibstat 查看可用设备
    # export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
    
    # GID Index (通常是 3)
    export NCCL_IB_GID_INDEX=3
    
    # IB 超时 (毫秒)
    export NCCL_IB_TIMEOUT=23
    
    # IB 重试次数
    export NCCL_IB_RETRY_CNT=7
    
else
    echo "未检测到 InfiniBand,使用以太网..."
    
    # 禁用 InfiniBand
    export NCCL_IB_DISABLE=1
    
    # 指定网络接口 (排除 lo 和 docker)
    # 根据实际情况调整,使用 ip addr 查看可用接口
    export NCCL_SOCKET_IFNAME=^lo,docker
    # 或者明确指定: export NCCL_SOCKET_IFNAME=eth0
    
    # Socket 网络优化
    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_SOCKET_NTHREADS=4
fi

# ============================================================================
# NCCL 性能优化
# ============================================================================

# 通道数配置
export NCCL_MAX_NCHANNELS=16     # 最大通道数
export NCCL_MIN_NCHANNELS=4      # 最小通道数

# P2P 通信 (GPU 直接通信)
export NCCL_P2P_DISABLE=0        # 启用 P2P
export NCCL_P2P_LEVEL=NVL        # NVLink 优先级: NVL > PIX > SYS

# NVLink 配置 (如果硬件支持)
export NCCL_NVLS_ENABLE=0        # 通常设为 0,除非硬件明确支持

# 跨节点通信优化
export NCCL_CROSS_NIC=1          # 允许跨 NIC 通信
export NCCL_NET_GDR_LEVEL=5      # GPU Direct RDMA 级别

# 缓冲区大小 (字节)
export NCCL_BUFFSIZE=8388608     # 8MB,可根据网络带宽调整

# 协议选择
# export NCCL_PROTO=Simple       # Simple (默认) 或 LL/LL128

# ============================================================================
# NCCL 算法选择 (高级配置)
# ============================================================================

# AllReduce 算法: Ring, Tree, CollNet
# export NCCL_ALGO=Ring

# 树形算法配置
# export NCCL_NTHREADS=512       # NCCL 线程数
# export NCCL_LL_THRESHOLD=0     # Low-Latency 算法阈值

# ============================================================================
# CUDA 和 GPU 配置
# ============================================================================

# CUDA 设备顺序
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CUDA 启动阻塞 (有助于调试)
# export CUDA_LAUNCH_BLOCKING=1  # 仅在调试时启用,影响性能

# GPU 内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# InternEvo 特定配置
# ============================================================================

# 禁用 InternEvo 额外的超时检查,使用 NCCL 内置超时
export INTERNLM_ENABLE_TIMEOUT=0

# ============================================================================
# 错误处理和监控
# ============================================================================

# 启用核心转储 (如果需要调试崩溃)
# ulimit -c unlimited

# 设置日志输出
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "训练配置信息"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "节点数量: $NNODES"
echo "每节点 GPU: $NPROC_PER_NODE"
echo "总 GPU 数: $((NNODES * NPROC_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "日志目录: $LOG_DIR"
echo "NCCL 超时: $NCCL_TIMEOUT 秒"
echo "========================================"
echo ""

# 显示 NCCL 配置
echo "NCCL 环境变量:"
env | grep NCCL | sort
echo ""

# ============================================================================
# 启动训练
# ============================================================================

# 如果需要在后台运行,添加 & 和 nohup
# nohup torchrun ... > "$LOG_DIR/train.log" 2>&1 &

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_DIR/train_rank${NODE_RANK}.log"

# 检查退出状态
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "训练成功完成!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "训练失败,退出码: $EXIT_CODE"
    echo "检查日志: $LOG_DIR/train_rank${NODE_RANK}.log"
    echo "========================================"
    
    # 如果是 NCCL 超时错误,提供诊断建议
    if grep -q "NCCL.*timeout\|ProcessGroupNCCL" "$LOG_DIR/train_rank${NODE_RANK}.log"; then
        echo ""
        echo "检测到 NCCL 超时错误!"
        echo "建议运行诊断脚本: bash diagnose_nccl_timeout.sh"
        echo "查看故障排查文档: NCCL_TIMEOUT_TROUBLESHOOTING.md"
    fi
fi

exit $EXIT_CODE

###############################################################################
# 使用说明:
#
# 单节点训练:
#   bash train_with_nccl_optimizations.sh configs/7B_sft.py
#
# 多节点训练:
#   # 节点 0 (master):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=node0_ip bash train_with_nccl_optimizations.sh
#
#   # 节点 1:
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=node0_ip bash train_with_nccl_optimizations.sh
#
# 调试模式:
#   NCCL_DEBUG=INFO bash train_with_nccl_optimizations.sh
#
# 增加超时时间:
#   NCCL_TIMEOUT=3600 bash train_with_nccl_optimizations.sh  # 60 分钟
#
###############################################################################
