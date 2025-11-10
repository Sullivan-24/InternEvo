#!/bin/bash

# NCCL 超时问题诊断脚本
# 用法: bash diagnose_nccl_timeout.sh

echo "========================================"
echo "NCCL 超时问题诊断工具"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 NCCL 版本
echo "1. 检查 NCCL 版本..."
python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 版本: {torch.version.cuda}'); print(f'NCCL 版本: {torch.cuda.nccl.version()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 无法获取 NCCL 版本信息${NC}"
else
    echo -e "${GREEN}✓ NCCL 版本信息获取成功${NC}"
fi
echo ""

# 2. 检查 GPU 状态
echo "2. 检查 GPU 状态..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
    echo ""
    
    echo "GPU 错误统计:"
    nvidia-smi --query-gpu=index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠ 此 GPU 可能不支持 ECC${NC}"
    fi
    echo ""
    
    echo "GPU 拓扑:"
    nvidia-smi topo -m
    echo -e "${GREEN}✓ GPU 状态检查完成${NC}"
else
    echo -e "${RED}✗ nvidia-smi 命令不可用${NC}"
fi
echo ""

# 3. 检查网络配置
echo "3. 检查网络配置..."
echo "网络接口:"
ip addr show | grep -E "^[0-9]+:|inet " | grep -v "127.0.0.1"
echo ""

if command -v ibstat &> /dev/null; then
    echo "InfiniBand 状态:"
    ibstat | grep -E "State|Rate"
    echo -e "${GREEN}✓ InfiniBand 可用${NC}"
else
    echo -e "${YELLOW}⚠ 未检测到 InfiniBand (可能使用以太网)${NC}"
fi
echo ""

# 4. 检查 NCCL 环境变量
echo "4. 检查 NCCL 环境变量..."
nccl_vars=$(env | grep NCCL | sort)
if [ -z "$nccl_vars" ]; then
    echo -e "${YELLOW}⚠ 未设置 NCCL 环境变量 (将使用默认值)${NC}"
else
    echo "当前 NCCL 环境变量:"
    echo "$nccl_vars"
    echo -e "${GREEN}✓ 已设置 NCCL 环境变量${NC}"
fi
echo ""

# 5. 检查系统资源
echo "5. 检查系统资源..."
echo "内存使用:"
free -h
echo ""

echo "CPU 负载:"
uptime
echo ""

echo "磁盘空间:"
df -h | grep -E "Filesystem|/dev/"
echo -e "${GREEN}✓ 系统资源检查完成${NC}"
echo ""

# 6. 检查 PyTorch 分布式配置
echo "6. 检查 PyTorch 分布式配置..."
echo "当前分布式相关环境变量:"
env | grep -E "MASTER_ADDR|MASTER_PORT|WORLD_SIZE|RANK|LOCAL_RANK|NODE_RANK" | sort
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ 未设置分布式环境变量${NC}"
fi
echo ""

# 7. 建议的 NCCL 优化配置
echo "========================================"
echo "建议的 NCCL 优化配置"
echo "========================================"
cat << 'EOF'

# 基础配置
export NCCL_DEBUG=WARN                    # 生产环境用 WARN,调试用 INFO
export NCCL_ASYNC_ERROR_HANDLING=1       # 启用异步错误处理
export NCCL_TIMEOUT=1800                  # 30 分钟超时

# 如果使用 InfiniBand:
export NCCL_IB_DISABLE=0                 # 启用 IB
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1    # 根据实际设备调整
export NCCL_IB_GID_INDEX=3               # 根据网络配置调整

# 如果使用以太网:
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0         # 指定网络接口

# 性能优化
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_DISABLE=0                # 启用 P2P 通信

# 调试模式(出现问题时使用)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

EOF
echo ""

# 8. 运行简单的 NCCL 测试
echo "========================================"
echo "是否运行 NCCL 通信测试? (需要多 GPU)"
echo "========================================"
echo "按 Enter 跳过,输入 'y' 运行测试:"
read -t 10 run_test || run_test=""

if [ "$run_test" = "y" ] || [ "$run_test" = "Y" ]; then
    echo "运行 NCCL 测试..."
    
    # 创建测试脚本
    cat > /tmp/nccl_test.py << 'PYEOF'
import torch
import torch.distributed as dist
import sys
import time

def test_nccl():
    try:
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"[Rank {rank}/{world_size}] 初始化成功")
        
        # 创建测试张量
        tensor = torch.ones(1).cuda() * rank
        print(f"[Rank {rank}] 测试张量值: {tensor.item()}")
        
        # 测试 all_reduce
        start = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elapsed = time.time() - start
        
        expected = sum(range(world_size))
        print(f"[Rank {rank}] AllReduce 结果: {tensor.item()}, 耗时: {elapsed*1000:.2f}ms, 期望值: {expected}")
        
        if abs(tensor.item() - expected) < 1e-5:
            print(f"[Rank {rank}] ✓ 测试通过")
            return True
        else:
            print(f"[Rank {rank}] ✗ 测试失败")
            return False
            
    except Exception as e:
        print(f"[Rank {rank if 'rank' in locals() else '?'}] ✗ 错误: {e}")
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    success = test_nccl()
    sys.exit(0 if success else 1)
PYEOF
    
    # 检测 GPU 数量
    num_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [ $num_gpus -gt 1 ]; then
        echo "检测到 $num_gpus 个 GPU,运行测试..."
        timeout 30 torchrun --nproc_per_node=$num_gpus /tmp/nccl_test.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ NCCL 通信测试通过${NC}"
        else
            echo -e "${RED}✗ NCCL 通信测试失败${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 只有 $num_gpus 个 GPU,跳过多 GPU 测试${NC}"
    fi
    
    # 清理
    rm -f /tmp/nccl_test.py
else
    echo "跳过 NCCL 测试"
fi
echo ""

# 9. 总结和建议
echo "========================================"
echo "诊断总结"
echo "========================================"
echo ""
echo "如果遇到 NCCL 超时问题,建议:"
echo "1. 检查所有节点的网络连通性"
echo "2. 确认所有 GPU 工作正常,无硬件故障"
echo "3. 设置合适的 NCCL 环境变量(参考上面的建议配置)"
echo "4. 启用 NCCL_DEBUG=INFO 获取详细日志"
echo "5. 检查是否有负载不均衡的情况"
echo "6. 查看详细的故障排查文档: NCCL_TIMEOUT_TROUBLESHOOTING.md"
echo ""
echo "========================================"
