#!/bin/bash
# NCCL 库路径修复的训练启动脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "InternEvo 训练启动脚本（带 NCCL 修复）"
echo "========================================="

# 1. 激活 conda 环境
echo ""
echo "[1/5] 激活 conda 环境..."
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate internevo
    echo "✓ Conda 环境已激活: internevo"
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate internevo
    echo "✓ Conda 环境已激活: internevo"
else
    echo "警告: 未找到 conda 初始化脚本，尝试直接使用当前环境"
fi

# 2. 查找 NCCL 库
echo ""
echo "[2/5] 查找 NCCL 库..."

NCCL_PATHS=()

# 检查 PyTorch 自带的 NCCL
PYTORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "$PYTORCH_LIB" ] && [ -d "$PYTORCH_LIB" ]; then
    if ls "$PYTORCH_LIB"/libnccl.so* 1> /dev/null 2>&1; then
        NCCL_PATHS+=("$PYTORCH_LIB")
        echo "✓ 找到 PyTorch NCCL: $PYTORCH_LIB"
    fi
fi

# 检查 conda 环境的 lib 目录
CONDA_PREFIX_LIB="$CONDA_PREFIX/lib"
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX_LIB" ]; then
    if ls "$CONDA_PREFIX_LIB"/libnccl.so* 1> /dev/null 2>&1; then
        NCCL_PATHS+=("$CONDA_PREFIX_LIB")
        echo "✓ 找到 Conda NCCL: $CONDA_PREFIX_LIB"
    fi
fi

# 检查用户指定的路径（如果在 bashrc 中设置了）
USER_NCCL_PATH="/mnt/shared-storage-user/ailab-sys/matenghui/miniconda3/envs/internevo/lib/python3.10/site-packages/nvidia/nccl/lib"
if [ -d "$USER_NCCL_PATH" ]; then
    if ls "$USER_NCCL_PATH"/libnccl.so* 1> /dev/null 2>&1; then
        NCCL_PATHS+=("$USER_NCCL_PATH")
        echo "✓ 找到用户 NCCL: $USER_NCCL_PATH"
    fi
fi

# 3. 设置 LD_LIBRARY_PATH
echo ""
echo "[3/5] 设置 LD_LIBRARY_PATH..."

for path in "${NCCL_PATHS[@]}"; do
    export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
done

if [ ${#NCCL_PATHS[@]} -eq 0 ]; then
    echo "⚠ 警告: 未找到 NCCL 库，程序可能会失败"
    echo "   尝试继续运行..."
else
    echo "✓ LD_LIBRARY_PATH 已设置"
fi

# 4. 验证环境
echo ""
echo "[4/5] 验证环境..."
echo "Python: $(which python)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '未知')"

# 尝试验证 NCCL
python -c "import torch; print('NCCL 版本:', torch.cuda.nccl.version())" 2>/dev/null && echo "✓ NCCL 可用" || echo "⚠ NCCL 验证失败"

echo ""
echo "当前 LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5
echo ""

# 5. 启动训练
echo "[5/5] 启动训练..."
echo "========================================="
echo ""

# 运行 Python 脚本，传递所有参数
python train.py "$@"
