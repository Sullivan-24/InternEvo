# NCCL AllReduce 超时问题排查与解决方案

## 问题描述

```
[rank8]: Watchdog caught collective operation timeout: 
WorkNCCL(SeqNum=8, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) 
ran for 1800057 milliseconds before timing out.
```

发生在训练 step=420,Rank 8 的 ALLREDUCE 操作超时 30 分钟。

---

## 可能的原因

### 1. **网络通信问题** (最常见)
- 节点间网络连接不稳定
- 网络拥塞或带宽不足
- InfiniBand/RDMA 配置问题

### 2. **负载不均衡**
- 某个 rank 的计算负载显著高于其他 rank
- GPU 性能不一致(硬件故障或节流)
- 数据加载不均匀导致某些 rank 延迟

### 3. **硬件故障**
- GPU 故障或性能降级
- GPU 内存不足导致频繁的 swap
- NVLink/PCIe 通信问题

### 4. **同步问题**
- 代码中存在不匹配的集合通信
- 某些 rank 陷入死锁
- 不同 rank 执行路径不一致

### 5. **NCCL 配置问题**
- NCCL 环境变量设置不当
- NCCL 版本兼容性问题

---

## 诊断步骤

### 1. 检查日志
```bash
# 查看所有 rank 的日志,特别是 rank 8 及其周围的 rank
grep -E "rank[7-9]" training.log

# 查看是否有其他错误信息
grep -i "error\|warning\|timeout" training.log
```

### 2. 检查网络连接
```bash
# 在所有节点上运行网络测试
# 使用 ib_write_bw 测试 InfiniBand 带宽(如果使用 IB)
ib_write_bw

# 检查网络延迟
ping <other_node_ip>
```

### 3. 检查 GPU 状态
```bash
# 在每个节点上检查 GPU 健康状态
nvidia-smi
nvidia-smi -q | grep -E "Temperature|Power|ECC"

# 检查 GPU 拓扑
nvidia-smi topo -m
```

### 4. 检查 NCCL 配置
```bash
# 查看当前 NCCL 环境变量
env | grep NCCL
```

---

## 解决方案

### 方案 1: 增加 NCCL 超时时间 (临时缓解)

修改训练启动脚本,设置更长的超时时间:

```bash
export NCCL_TIMEOUT=3600  # 60 分钟
```

或在训练配置中修改 `internlm/utils/timeout.py` 中的默认值。

**注意**: 这只是临时方案,不解决根本问题。

### 方案 2: 优化 NCCL 配置 (推荐)

```bash
# 启用 NCCL 调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 启用异步错误处理
export NCCL_ASYNC_ERROR_HANDLING=1

# 优化网络传输
export NCCL_IB_DISABLE=0  # 启用 InfiniBand(如果有)
export NCCL_IB_HCA=mlx5   # 指定 IB 设备

# 如果使用以太网
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 设置更激进的重试策略
export NCCL_MAX_NCHANNELS=4
export NCCL_MIN_NCHANNELS=1
```

### 方案 3: 检查和修复网络问题

1. **验证网络连通性**:
```bash
# 在各节点间进行 NCCL 测试
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 128M -f 2 -g <num_gpus>
```

2. **优化网络拓扑**:
- 确保使用高速网络(InfiniBand 或 100GbE+)
- 检查交换机配置
- 确认没有网络防火墙阻止 NCCL 通信

### 方案 4: 检查训练负载均衡

1. **添加监控**:
```python
# 在训练代码中添加性能监控
import time
import torch.distributed as dist

start_time = time.time()
# ... 训练步骤 ...
elapsed = time.time() - start_time

if dist.get_rank() == 8:
    print(f"Rank 8 step time: {elapsed:.2f}s")
```

2. **检查数据加载**:
- 确保所有 rank 的数据加载速度一致
- 使用足够的 dataloader workers
- 检查数据预处理是否存在瓶颈

### 方案 5: 启用 NCCL 容错机制

修改 `internlm/utils/timeout.py`:

```python
# 始终启用异步错误处理
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# 或者在启动脚本中设置
export INTERNLM_ENABLE_TIMEOUT=1
export NCCL_TIMEOUT=300  # 5 分钟,更早发现问题
```

### 方案 6: 代码层面的优化

1. **添加超时检测和重试机制**:
```python
import torch.distributed as dist
from contextlib import contextmanager

@contextmanager
def nccl_timeout_guard(timeout_sec=300):
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError(f"NCCL operation timeout after {timeout_sec}s")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# 使用示例
with nccl_timeout_guard(300):
    dist.all_reduce(tensor, group=group)
```

2. **添加进度日志**:
在 `internlm/train/pipeline.py` 的关键位置添加日志:
```python
if gpc.get_global_rank() == 8:
    logger.info(f"Rank 8: Before allreduce at step {step}")
# ... allreduce 操作 ...
if gpc.get_global_rank() == 8:
    logger.info(f"Rank 8: After allreduce at step {step}")
```

---

## 快速诊断命令

```bash
# 1. 查看 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 2. 运行简单的 NCCL 测试
torchrun --nproc_per_node=8 -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor)
print(f'Rank {dist.get_rank()} OK')
"

# 3. 检查 GPU 错误
nvidia-smi --query-gpu=index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv
```

---

## 推荐的启动配置

```bash
#!/bin/bash

# NCCL 优化配置
export NCCL_DEBUG=WARN  # 生产环境使用 WARN,调试时用 INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800  # 30 分钟

# 网络优化(根据实际硬件调整)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker

# 性能优化
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0  # 根据硬件支持情况

# InternEvo 特定配置
export INTERNLM_ENABLE_TIMEOUT=0  # 禁用额外的超时检查,使用 NCCL 自带的

# 启动训练
torchrun --nproc_per_node=8 --nnodes=<num_nodes> train.py --config <config_file>
```

---

## 紧急恢复步骤

如果训练频繁超时无法继续:

1. **从检查点恢复**:
```bash
# 使用上一个成功的检查点
python train.py --config <config> --resume <checkpoint_path>
```

2. **减少并行度**:
- 临时降低数据并行度
- 减少 pipeline 并行或 tensor 并行

3. **隔离故障节点**:
- 识别并移除故障的 GPU 或节点
- 使用 `CUDA_VISIBLE_DEVICES` 排除故障设备

---

## 相关文件

- **超时配置**: `internlm/utils/timeout.py`
- **分布式初始化**: `internlm/core/context/parallel_context.py`
- **进程组初始化**: `internlm/core/context/process_group_initializer.py`
- **通信工具**: `internlm/core/parallel/comm/utils.py`

---

## 参考资料

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [PyTorch Distributed 故障排查](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 环境变量完整列表](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
