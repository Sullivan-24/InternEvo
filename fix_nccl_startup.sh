#!/bin/bash

# NCCL Connection Abort Error Fix Script
# 用于解决 socketStartConnect: Connect to 10.102.199.30<54079> failed : Software caused connection abort

echo "=== NCCL Connection Debug Setup ==="

# 1. 设置NCCL调试环境变量
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_ASYNC_ERROR_HANDLING=1

# 2. 增加超时时间
export NCCL_TIMEOUT=300
export INTERNLM_ENABLE_TIMEOUT=1

# 3. 网络接口配置
# 排除虚拟网卡，只使用物理网卡
export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0

# 4. 禁用可能有问题的功能进行排查
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=1  # 禁用P2P通信

# 5. 设置NCCL网络参数
export NCCL_MIN_NRINGS=1
export NCCL_MAX_NRINGS=1
export NCCL_SOCKET_NTHREADS=1

# 6. 检查网络连通性
echo "=== Network Connectivity Check ==="
if command -v ping &> /dev/null; then
    echo "Testing connectivity to problematic IP..."
    ping -c 3 10.102.199.30 || echo "WARNING: Cannot ping 10.102.199.30"
fi

if command -v telnet &> /dev/null; then
    echo "Testing port connectivity..."
    timeout 5 telnet 10.102.199.30 54079 || echo "WARNING: Cannot connect to 10.102.199.30:54079"
fi

# 7. 显示网络接口信息
echo "=== Network Interface Information ==="
ip addr show | grep -E "^[0-9]+:|inet "

# 8. 检查防火墙状态
echo "=== Firewall Status ==="
if command -v ufw &> /dev/null; then
    sudo ufw status
elif command -v iptables &> /dev/null; then
    sudo iptables -L | head -20
fi

# 9. 设置重试机制
export NCCL_RETRY_COUNT=3
export NCCL_RETRY_DELAY=5

echo "=== Environment Variables Set ==="
env | grep NCCL | sort

echo "=== Starting Training with NCCL Debug ==="

# 10. 启动训练（替换为您的实际启动命令）
# 示例：
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK \
#          --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#          train.py --config configs/your_config.py

# 如果仍然失败，尝试以下步骤：
echo "=== If training still fails, try these steps ==="
echo "1. Check if other processes are using the port: lsof -i :54079"
echo "2. Try different network interface: export NCCL_SOCKET_IFNAME=eth0"
echo "3. Enable IB if available: unset NCCL_IB_DISABLE"
echo "4. Check node time synchronization: ntpdate -q pool.ntp.org"
echo "5. Restart network service: sudo systemctl restart networking"