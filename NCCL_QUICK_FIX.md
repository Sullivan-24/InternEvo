# NCCL 超时问题快速修复指南

## 🚨 紧急情况 - 立即尝试

如果训练刚刚因为 NCCL 超时失败,立即尝试以下步骤:

### 1. 快速重启训练(增加超时时间)

```bash
export NCCL_TIMEOUT=3600  # 增加到 60 分钟
export NCCL_DEBUG=INFO    # 启用详细日志
python train.py --config <your_config> --resume <last_checkpoint>
```

### 2. 使用优化的启动脚本

```bash
# 使用提供的优化脚本
bash train_with_nccl_optimizations.sh configs/<your_config>.py
```

### 3. 运行诊断

```bash
# 运行诊断脚本找出问题
bash diagnose_nccl_timeout.sh
```

---

## 📋 常见原因及对应解决方案

### 原因 1: 网络不稳定 (最常见 ~60%)

**症状**:
- 随机出现超时
- 不同 step 超时
- 多个 rank 报错

**快速修复**:
```bash
# 增加超时和重试
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# 如果使用以太网,指定网络接口
export NCCL_SOCKET_IFNAME=eth0  # 替换为实际接口
```

### 原因 2: GPU 硬件故障 (~20%)

**症状**:
- 总是同一个 rank 超时
- GPU 温度过高
- ECC 错误

**快速修复**:
```bash
# 检查 GPU 状态
nvidia-smi
nvidia-smi --query-gpu=temperature.gpu,ecc.errors.uncorrected.aggregate.total --format=csv

# 排除故障 GPU (假设 GPU 3 故障)
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
```

### 原因 3: 负载不均衡 (~15%)

**症状**:
- 训练速度不一致
- 某些 rank 明显慢于其他 rank

**快速修复**:
```bash
# 优化数据加载
# 在配置文件中增加 dataloader workers
data = dict(
    num_workers=8,  # 增加 worker 数量
    prefetch_factor=4,
)

# 启用异步数据加载
```

### 原因 4: NCCL 配置不当 (~5%)

**症状**:
- 初始化就失败
- 所有 step 都很慢

**快速修复**:
```bash
# 重置为保守配置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # 先禁用 IB 测试
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MAX_NCHANNELS=4
```

---

## 🔧 分步骤调试流程

### Step 1: 收集信息 (5 分钟)

```bash
# 运行诊断脚本
bash diagnose_nccl_timeout.sh > diagnosis.log

# 检查训练日志
grep -E "rank.*timeout|ProcessGroupNCCL|NCCL error" training.log

# 记录问题模式
# - 哪个 rank 超时?
# - 哪个 step 超时?
# - 是否可复现?
```

### Step 2: 基础排查 (10 分钟)

```bash
# 1. 检查网络连通性
ping <other_node_ip>

# 2. 检查 GPU 健康
nvidia-smi
nvidia-smi -q | grep -E "Temperature|Power|ECC"

# 3. 测试 NCCL 通信
# 创建测试脚本
cat > test_nccl.py << 'EOF'
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor)
print(f"Rank {dist.get_rank()} OK")
EOF

# 运行测试
torchrun --nproc_per_node=8 test_nccl.py
```

### Step 3: 应用修复 (5 分钟)

根据 Step 1-2 的发现,选择对应的修复方案:

**网络问题** → 增加超时 + 优化网络配置
```bash
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=23
```

**GPU 问题** → 排除故障 GPU
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7  # 排除 GPU 3
```

**负载问题** → 优化数据加载
```python
# 修改配置文件
data.num_workers = 8
```

### Step 4: 验证修复 (测试运行)

```bash
# 启用详细日志
export NCCL_DEBUG=INFO

# 短时间测试
python train.py --config <config> --steps 100

# 如果成功,恢复正常训练
export NCCL_DEBUG=WARN
python train.py --config <config> --resume <checkpoint>
```

---

## 📊 预防性措施

### 训练前检查清单

- [ ] 运行 `diagnose_nccl_timeout.sh`
- [ ] 测试 NCCL 通信
- [ ] 检查所有 GPU 健康状态
- [ ] 验证网络连通性
- [ ] 设置合适的 NCCL 环境变量
- [ ] 准备监控脚本

### 推荐的 NCCL 配置

```bash
# 生产环境推荐配置
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# 根据网络类型选择
# InfiniBand:
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# 以太网:
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# 性能优化
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_DISABLE=0
```

### 监控建议

在训练脚本中添加监控:
```python
import time
import torch.distributed as dist

# 记录每个 step 的时间
step_start = time.time()
# ... 训练代码 ...
step_time = time.time() - step_start

if step_time > 300:  # 超过 5 分钟警告
    logger.warning(f"Rank {dist.get_rank()} step {step} took {step_time:.2f}s")
```

---

## 🆘 仍然无法解决?

### 收集以下信息寻求帮助:

1. **环境信息**:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
nvidia-smi
```

2. **NCCL 配置**:
```bash
env | grep NCCL | sort
```

3. **错误日志**:
```bash
# 完整的错误堆栈
grep -A 20 "ProcessGroupNCCL" training.log
```

4. **网络拓扑**:
```bash
nvidia-smi topo -m
ibstat  # 如果使用 InfiniBand
```

5. **诊断报告**:
```bash
bash diagnose_nccl_timeout.sh > full_diagnosis.txt
```

### 联系渠道

- GitHub Issues: 附上以上信息
- 飞书群: 提供诊断报告
- 查看文档: `NCCL_TIMEOUT_TROUBLESHOOTING.md`

---

## 📚 相关文件

- **详细故障排查**: `NCCL_TIMEOUT_TROUBLESHOOTING.md`
- **诊断脚本**: `diagnose_nccl_timeout.sh`
- **优化启动脚本**: `train_with_nccl_optimizations.sh`

---

## ⚡ 一键修复命令

```bash
# 如果你赶时间,直接运行这个:
export NCCL_TIMEOUT=3600 && \
export NCCL_DEBUG=INFO && \
export NCCL_ASYNC_ERROR_HANDLING=1 && \
bash train_with_nccl_optimizations.sh configs/<your_config>.py
```

这会:
✅ 增加超时到 60 分钟
✅ 启用详细日志
✅ 应用所有优化配置
✅ 自动检测网络类型
✅ 从上次检查点恢复

---

**最后更新**: 2025-11-10
**适用于**: InternEvo / InternLM 训练框架
