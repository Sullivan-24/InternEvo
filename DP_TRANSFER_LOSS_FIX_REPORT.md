# DP_Transfer Loss对齐问题分析与修复报告

## 问题背景

在实现不同DP group之间的pipeline调度时，当某个rank出现fail-stop，其workload被转移到其他DP group的相同pipeline stage执行。虽然程序可以正常运行，但loss曲线无法与优化前对齐。

## 根本原因分析

经过详细的代码分析，发现了**三个核心问题**导致loss无法对齐：

---

## 🔴 问题1：Loss归一化不正确（最关键）

### 问题位置
- 文件：`internlm/core/scheduler/pipeline_scheduler_1f1b.py`
- 行号：335

### 问题代码
```python
loss_reduced = loss / self.num_microbatches
```

### 问题分析

当DP_Transfer=True时：
- 总的microbatch数量：`MICRO_NUM = num_microbatches_per_dp * DP_SIZE`
- 在forward/backward中临时设置：`self.num_microbatches = num_microbatches_per_dp`
- Loss除以`num_microbatches_per_dp`进行归一化

**关键问题**：
1. 对于接收额外workload的DP组，实际处理的microbatch数 > `num_microbatches_per_dp`
2. 但loss仍然除以`num_microbatches_per_dp`
3. **导致loss被人为放大**

**举例说明**：
```
假设：
- num_microbatches_per_dp = 4
- DP_SIZE = 2
- 总microbatch数 = 8

正常情况：
- DP group 0处理4个microbatch，loss除以4
- DP group 1处理4个microbatch，loss除以4

Workload转移后：
- DP group 0处理6个microbatch（接收了2个额外的）
- 但loss仍除以4，导致loss = actual_loss * 6/4 = actual_loss * 1.5
- Loss被人为放大了50%！
```

### 修复方案

1. **统计实际处理的microbatch数量**：
   - 在`UnifiedSingleChunkPipelineScheduler`初始化时统计
   - 存储在`gpc.config.actual_num_forward_microbatches`

2. **使用实际数量进行loss归一化**：
   ```python
   actual_num_microbatches = self.num_microbatches
   if gpc.config.DP_Transfer and hasattr(gpc.config, 'actual_num_forward_microbatches'):
       actual_num_microbatches = gpc.config.actual_num_forward_microbatches
   
   loss_reduced = loss / actual_num_microbatches
   ```

---

## 🔴 问题2：梯度同步时机错误

### 问题位置
- 文件：`internlm/core/scheduler/pipeline_scheduler_1f1b.py`
- 行号：406

### 问题代码
```python
if gpc.config.DP_Transfer:
    self.num_microbatches = gpc.config.num_microbatches_per_dp
skip_grad_sync = self._get_current_microbatch_id(step_id%self.num_microbatches) != self.num_microbatches - 1
```

### 问题分析

梯度同步的正确逻辑：
- 在处理完所有assigned microbatches之前，应该累积梯度（skip_grad_sync=True）
- 只在最后一个microbatch的backward时同步梯度（skip_grad_sync=False）

**当前实现的问题**：
1. 使用`step_id % self.num_microbatches`来判断
2. 当实际处理的microbatch数 ≠ `num_microbatches_per_dp`时：
   - 可能在中途错误地触发梯度同步
   - 或在最后没有正确触发同步

**举例说明**：
```
假设一个rank处理6个microbatch，但num_microbatches_per_dp=4：

step_id:  0  1  2  3  4  5
step_id%4: 0  1  2  3  0  1
是否同步: 否 否 否 是 否 否  <-- 在step 3错误地同步了！
实际应该: 否 否 否 否 否 是  <-- 应该在step 5同步
```

### 修复方案

1. **记录backward workload的索引**：
   ```python
   backward_workload_indices = [i for i, w in enumerate(self.workloads) 
                                if w["workload_type"] == WorkloadType.BACKWARD.value]
   last_backward_index = backward_workload_indices[-1]
   ```

2. **创建判断函数**：
   ```python
   def is_last_microbatch_func(workload_step_id):
       return workload_step_id == last_backward_index
   ```

3. **在backward时使用正确的判断**：
   ```python
   if gpc.config.DP_Transfer and hasattr(gpc.config, 'is_last_microbatch_func'):
       skip_grad_sync = not gpc.config.is_last_microbatch_func(step_id)
   ```

4. **传递正确的workload索引**：
   - 在`UnifiedSingleChunkPipelineScheduler`中调用`_backward_step`时
   - 传递workload索引`s`而不是`microbatch_id`

---

## 🟡 问题3：数据并行梯度平均问题

### 问题分析

在标准的数据并行训练中：
- 所有DP rank处理相同数量的样本
- 梯度在DATA parallel组内简单平均：`grad = sum(all_grads) / dp_size`

当workload转移后：
- 不同DP rank处理的样本数不同
- 如果仍然简单平均，会导致梯度scale错误

**举例说明**：
```
假设2个DP rank，batch_size=4（每个样本梯度为1）：

正常情况：
- Rank 0: 4个样本, grad=4
- Rank 1: 4个样本, grad=4
- 平均后: (4+4)/2 = 4 ✓

Workload转移后：
- Rank 0: 6个样本, grad=6
- Rank 1: 2个样本, grad=2
- 简单平均: (6+2)/2 = 4 ✓（恰好正确）
- 但如果比例不同，结果会错误
```

### 建议方案

虽然在某些情况下简单平均可能恰好正确，但为了更robust，建议：

1. **在每个rank上记录实际处理的token数量**
2. **使用AllReduce汇总总token数**
3. **按token数量加权平均梯度**：
   ```python
   # 伪代码
   total_tokens = all_reduce_sum(local_tokens)
   grad = grad * (local_tokens / total_tokens) * dp_size
   ```

---

## 修复文件清单

已修改的文件：

1. **internlm/core/scheduler/pipeline_scheduler_1f1b.py**
   - 移除了临时修改`self.num_microbatches`的代码
   - 使用`actual_num_forward_microbatches`进行loss归一化
   - 使用`is_last_microbatch_func`判断梯度同步时机

2. **internlm/core/scheduler/pipeline_scheduler_unified.py**
   - 在初始化时统计实际的forward workload数量
   - 创建`is_last_microbatch_func`函数
   - 修改backward调用，传递正确的workload索引

---

## 测试建议

修复后，建议进行以下测试：

### 1. Loss一致性测试
```bash
# 测试1：无workload转移的基准run
DP_Transfer=False python train.py --config config.py

# 测试2：有workload转移的run
DP_Transfer=True python train.py --config config.py

# 比较两者的loss曲线是否对齐
```

### 2. 检查关键指标
- **Loss值**：应该与baseline对齐（允许小的数值误差）
- **Gradient norm**：应该与baseline接近
- **训练稳定性**：loss曲线应该平滑，无异常spike
- **收敛速度**：应该与baseline相当

### 3. 调试日志
可以添加以下日志来验证修复：
```python
if gpc.is_rank_for_log():
    logger.info(f"Rank {gpc.get_global_rank()}: "
                f"actual_microbatches={gpc.config.actual_num_forward_microbatches}, "
                f"original_microbatches={gpc.config.num_microbatches_per_dp}")
```

---

## 预期效果

修复后应该达到：
1. ✅ Loss曲线与无workload转移时对齐
2. ✅ 梯度在正确的时机同步
3. ✅ 每个rank按实际处理的数据量正确归一化loss
4. ✅ 训练稳定，无异常loss spike

---

## 潜在风险和注意事项

### 1. 性能影响
- 额外的workload统计开销极小（只在初始化时执行一次）
- 无额外通信开销

### 2. 兼容性
- 修改对非DP_Transfer模式完全兼容（通过条件判断保护）
- 不影响现有的其他调度器

### 3. Edge Cases
需要注意的特殊情况：
- **All workload转移到一个rank**：loss应该等于总loss
- **Unbalanced transfer**：不同rank处理差异很大的workload数
- **Multiple failures**：多个rank同时fail-stop

---

## 总结

这三个问题的修复是递进关系：

1. **Loss归一化**（问题1）：确保每个rank的loss scale正确
2. **梯度同步时机**（问题2）：确保梯度在正确的时候累积和同步
3. **梯度平均权重**（问题3）：确保不同rank的梯度按数据量正确平均

三者共同作用才能保证在workload转移场景下loss曲线的正确对齐。

---

## 作者说明

本报告基于对以下代码的深入分析：
- `internlm/core/scheduler/pipeline_scheduler_1f1b.py`
- `internlm/core/scheduler/pipeline_scheduler_unified.py`
- `internlm/data/tokenized/batch_sampler.py`
- `internlm/train/pipeline.py`

修复日期：2025-11-11
