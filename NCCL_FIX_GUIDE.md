# NCCL 库找不到问题的解决方案

## 问题诊断

您遇到的错误：`/usr/lib/x86_64-linux-gnu/libnccl.so.2: cannot open shared object file: No such file or directory`

## 常见原因

1. **bashrc 未被加载**：在非交互式 shell（如通过脚本、cron、systemd 启动）中，`~/.bashrc` 不会自动加载
2. **路径不存在**：设置的路径可能实际不存在
3. **库文件名不匹配**：库文件可能是 `libnccl.so.2.x.x` 而不是 `libnccl.so.2`

## 解决方案

### 步骤 1：验证 NCCL 库是否存在

```bash
# 检查您设置的路径中是否真的有 NCCL 库
ls -la /mnt/shared-storage-user/ailab-sys/matenghui/miniconda3/envs/internevo/lib/python3.10/site-packages/nvidia/nccl/lib/

# 或者在整个 conda 环境中搜索
find ~/miniconda3/envs/internevo -name "libnccl.so*" 2>/dev/null
```

### 步骤 2：找到正确的 NCCL 库路径

NCCL 库通常在以下位置之一：

```bash
# PyTorch 自带的 NCCL
~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib/

# Conda 安装的 NCCL
~/miniconda3/envs/internevo/lib/

# 系统安装的 NCCL
/usr/lib/x86_64-linux-gnu/
```

查找命令：
```bash
# 在 conda 环境中查找
find ~/miniconda3/envs/internevo -name "libnccl.so*"

# 检查 PyTorch 目录
ls -la ~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib/ | grep nccl
```

### 步骤 3：修复方法

#### 方法 1：使用启动脚本（推荐）

创建一个启动脚本 `run_train.sh`：

```bash
#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate internevo

# 设置 LD_LIBRARY_PATH（请替换为实际路径）
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib:$LD_LIBRARY_PATH

# 验证 NCCL 库可以找到
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldconfig -p | grep nccl || echo "Warning: NCCL not found in system cache"

# 运行训练脚本
python train.py "$@"
```

使用方法：
```bash
chmod +x run_train.sh
./run_train.sh --config configs/7B_sft.py
```

#### 方法 2：修改 ~/.bashrc 并确保加载

编辑 `~/.bashrc`：

```bash
# 在文件末尾添加
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib:$LD_LIBRARY_PATH
```

然后在启动训练前**显式加载**：
```bash
source ~/.bashrc
python train.py --config configs/7B_sft.py
```

#### 方法 3：使用 ~/.bash_profile（用于非交互式会话）

编辑或创建 `~/.bash_profile`：

```bash
# 加载 .bashrc
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# 或直接在这里设置
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/miniconda3/envs/internevo/lib:$LD_LIBRARY_PATH
```

#### 方法 4：创建符号链接（如果有 root 权限）

```bash
# 找到实际的 NCCL 库位置
NCCL_PATH=$(find ~/miniconda3/envs/internevo -name "libnccl.so.2*" | head -1)
echo "Found NCCL at: $NCCL_PATH"

# 创建符号链接（需要 sudo）
sudo ln -s $NCCL_PATH /usr/lib/x86_64-linux-gnu/libnccl.so.2
```

#### 方法 5：在 Python 脚本中设置（临时方案）

在 `train.py` 的开头添加：

```python
import os
import sys

# 设置 LD_LIBRARY_PATH
nccl_path = os.path.expanduser("~/miniconda3/envs/internevo/lib/python3.10/site-packages/torch/lib")
os.environ['LD_LIBRARY_PATH'] = f"{nccl_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# 注意：这可能不会对已经加载的库生效
```

### 步骤 4：验证修复

运行以下命令验证：

```bash
# 激活环境
conda activate internevo

# 检查环境变量
echo $LD_LIBRARY_PATH

# 测试 NCCL 是否可用
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('NCCL available:', torch.cuda.nccl.version())"
```

## 调试命令

```bash
# 查看当前加载的库
ldd $(which python) | grep nccl

# 查看 PyTorch 使用的 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 检查系统 NCCL 库
ldconfig -p | grep nccl

# 查找所有 NCCL 库
find /usr -name "libnccl.so*" 2>/dev/null
find ~ -name "libnccl.so*" 2>/dev/null
```

## 注意事项

1. **路径必须存在**：确保设置的路径中确实有 `libnccl.so.2` 文件
2. **使用绝对路径**：避免使用 `~` 符号，改用完整路径如 `/home/username/...`
3. **多路径设置**：可以设置多个路径，用 `:` 分隔
4. **环境激活顺序**：先激活 conda 环境，再设置 LD_LIBRARY_PATH
5. **slurm/pbs 任务**：在作业脚本中显式设置环境变量

## 推荐的最佳实践

创建一个环境配置文件 `env_setup.sh`：

```bash
#!/bin/bash

# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate internevo

# 设置库路径
export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

echo "Environment setup complete"
echo "PyTorch location: $(python -c 'import torch; print(torch.__file__)')"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
```

使用：
```bash
source env_setup.sh
python train.py --config configs/7B_sft.py
```
