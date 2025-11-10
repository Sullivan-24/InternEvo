#!/bin/bash
# NCCL 环境配置脚本 - 使用方法: source setup_nccl_env.sh

echo "配置 NCCL 环境..."

# 方法 1: 使用 Python 自动检测 PyTorch 的库路径
if command -v python &> /dev/null; then
    TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
    if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
        echo "✓ 已添加 PyTorch 库路径: $TORCH_LIB"
    fi
fi

# 方法 2: 使用 conda 环境的 lib 目录
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "✓ 已添加 Conda 库路径: $CONDA_PREFIX/lib"
fi

# 方法 3: 添加用户自定义路径（如果存在）
USER_PATHS=(
    "/mnt/shared-storage-user/ailab-sys/matenghui/miniconda3/envs/internevo/lib"
    "$HOME/miniconda3/envs/internevo/lib"
)

for path in "${USER_PATHS[@]}"; do
    if [ -d "$path" ]; then
        export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
        echo "✓ 已添加自定义路径: $path"
    fi
done

# 验证 NCCL
echo ""
echo "验证 NCCL 库..."
if ldconfig -p 2>/dev/null | grep -q nccl; then
    echo "✓ 系统已注册 NCCL 库"
fi

# 显示最终的 LD_LIBRARY_PATH
echo ""
echo "当前 LD_LIBRARY_PATH 前 5 个路径:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5
echo ""
echo "环境配置完成！现在可以运行训练脚本了。"
