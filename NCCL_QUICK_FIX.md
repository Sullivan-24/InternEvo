# NCCL 库找不到问题 - 快速修复指南

## 问题描述

错误信息：
```
Error: /usr/lib/x86_64-linux-gnu/libnccl.so.2: cannot open shared object file: No such file or directory
```

即使在 `~/.bashrc` 中设置了 `LD_LIBRARY_PATH`，程序仍然找不到 NCCL 库。

## 根本原因

**`~/.bashrc` 只在交互式 shell 中加载**，在以下情况下不会自动加载：
- 通过脚本启动程序
- 使用 SLURM/PBS 作业调度系统
- SSH 执行远程命令
- cron 任务
- systemd 服务

## 快速修复方案

### 🔧 方案 1：使用诊断脚本（推荐第一步）

```bash
# 运行诊断脚本，找到 NCCL 库的实际位置
./diagnose_nccl.sh
```

这将显示：
- NCCL 库文件在哪里
- 当前环境配置
- 具体的修复建议

### 🚀 方案 2：使用自动修复的启动脚本（最简单）

```bash
# 直接使用提供的启动脚本
./run_train_with_nccl.sh --config configs/7B_sft.py

# 或者带其他参数
./run_train_with_nccl.sh --config configs/7B_sft.py --launcher slurm
```

这个脚本会自动：
1. 激活 conda 环境
2. 查找 NCCL 库
3. 设置正确的 LD_LIBRARY_PATH
4. 启动训练

### 🔨 方案 3：使用环境配置文件

```bash
# 在运行训练前 source 这个文件
source setup_nccl_env.sh

# 然后正常运行训练
python train.py --config configs/7B_sft.py
```

### ✏️ 方案 4：手动修复（如果知道 NCCL 路径）

运行诊断脚本后，如果找到了 NCCL 库，例如在：
```
~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib/libnccl.so.2
```

则在启动训练前执行：
```bash
# 激活 conda 环境
conda activate internevo

# 设置库路径（使用您实际找到的路径）
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# 验证
python -c "import torch; print('NCCL version:', torch.cuda.nccl.version())"

# 运行训练
python train.py --config configs/7B_sft.py
```

## 使用 SLURM 时的修复

在您的 SLURM 作业脚本中：

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate internevo

# 方法 A: source 环境配置（推荐）
source /path/to/workspace/setup_nccl_env.sh

# 或者 方法 B: 使用启动脚本
# ./run_train_with_nccl.sh --config configs/7B_sft.py

# 或者 方法 C: 手动设置（使用实际路径）
# export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# 运行训练
python train.py --config configs/7B_sft.py
```

## 永久修复（可选）

如果您想让设置永久生效，修改 `~/.bash_profile`（而不是 `~/.bashrc`）：

```bash
# 编辑 ~/.bash_profile
nano ~/.bash_profile

# 添加以下内容（使用您实际的路径）
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/internevo/lib:$LD_LIBRARY_PATH

# 保存后，重新登录或执行
source ~/.bash_profile
```

## 常见 NCCL 库位置

按优先级排序：

1. **PyTorch 自带**（推荐使用）
   ```
   ~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib/
   ```

2. **Conda 环境**
   ```
   ~/miniconda3/envs/internevo/lib/
   ```

3. **Nvidia 包**
   ```
   ~/miniconda3/envs/internevo/lib/python3.10/site-packages/nvidia/nccl/lib/
   ```

4. **系统安装**
   ```
   /usr/lib/x86_64-linux-gnu/
   ```

## 验证修复

运行以下命令确认 NCCL 可用：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('NCCL version:', torch.cuda.nccl.version())"
```

预期输出类似：
```
PyTorch: 2.1.0+cu121
CUDA available: True
NCCL version: (2, 18, 3)
```

## 故障排除

如果仍然有问题：

1. **确认库文件存在**
   ```bash
   ls -la ~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib/libnccl.so*
   ```

2. **检查文件权限**
   ```bash
   # 应该有读取权限
   ls -la <NCCL库路径>
   ```

3. **检查环境变量是否真的设置了**
   ```bash
   echo $LD_LIBRARY_PATH
   ```

4. **使用绝对路径**
   ```bash
   # 不要使用 ~ 符号，使用完整路径
   export LD_LIBRARY_PATH=/home/username/miniconda3/envs/internevo/lib:$LD_LIBRARY_PATH
   ```

## 为什么 bashrc 不起作用？

`.bashrc` vs `.bash_profile` vs 启动脚本：

| 启动方式 | ~/.bashrc | ~/.bash_profile | 启动脚本 |
|---------|-----------|-----------------|---------|
| 交互式登录 | ❌ | ✅ | ✅ |
| SSH 命令 | ❌ | ❌ | ✅ |
| SLURM 作业 | ❌ | ❌ | ✅ |
| Cron 任务 | ❌ | ❌ | ✅ |
| 手动终端 | ✅ | ✅ | ✅ |

**结论**：对于自动化任务和作业调度，使用启动脚本或在脚本中显式设置环境变量最可靠。

## 联系支持

如果以上方法都不能解决问题，请提供以下信息：

```bash
# 运行并保存输出
./diagnose_nccl.sh > nccl_diagnosis.txt 2>&1
```

将 `nccl_diagnosis.txt` 发送给支持团队。
