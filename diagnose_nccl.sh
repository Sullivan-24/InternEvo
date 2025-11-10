#!/bin/bash
# NCCL 问题诊断脚本

echo "========================================="
echo "NCCL 问题诊断工具"
echo "========================================="

# 1. 检查 Python 环境
echo ""
echo "[1] Python 环境信息:"
echo "---"
which python
python --version
echo ""

# 2. 检查 PyTorch 和 CUDA
echo "[2] PyTorch 和 CUDA 信息:"
echo "---"
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'PyTorch 路径: {torch.__file__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
try:
    print(f'NCCL 版本: {torch.cuda.nccl.version()}')
except Exception as e:
    print(f'NCCL 检查失败: {e}')
" 2>&1
echo ""

# 3. 搜索 NCCL 库文件
echo "[3] 搜索 NCCL 库文件:"
echo "---"

# 在 conda 环境中搜索
if [ -n "$CONDA_PREFIX" ]; then
    echo "在 Conda 环境中搜索: $CONDA_PREFIX"
    find "$CONDA_PREFIX" -name "libnccl.so*" 2>/dev/null | while read file; do
        echo "  找到: $file"
        ls -lh "$file"
    done
fi

# 在用户目录搜索
echo ""
echo "在用户目录搜索: $HOME"
find "$HOME" -name "libnccl.so*" 2>/dev/null | head -10 | while read file; do
    echo "  找到: $file"
    ls -lh "$file"
done

# 在系统目录搜索
echo ""
echo "在系统目录搜索: /usr"
find /usr -name "libnccl.so*" 2>/dev/null | while read file; do
    echo "  找到: $file"
    ls -lh "$file"
done
echo ""

# 4. 检查当前环境变量
echo "[4] 当前环境变量:"
echo "---"
echo "LD_LIBRARY_PATH:"
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | nl
else
    echo "  (未设置)"
fi
echo ""

echo "CONDA_PREFIX: ${CONDA_PREFIX:-未设置}"
echo "PATH (前 5 个):"
echo "$PATH" | tr ':' '\n' | head -5 | nl
echo ""

# 5. 检查系统库缓存
echo "[5] 系统库缓存中的 NCCL:"
echo "---"
ldconfig -p 2>/dev/null | grep nccl || echo "  未在系统库缓存中找到 NCCL"
echo ""

# 6. 测试动态链接
echo "[6] 测试 Python 的动态链接库:"
echo "---"
ldd $(which python) | grep nccl || echo "  Python 未直接链接 NCCL（这是正常的）"
echo ""

# 7. 提供建议
echo "========================================="
echo "诊断完成！"
echo ""
echo "如果上述搜索找到了 NCCL 库，请将包含该库的目录添加到 LD_LIBRARY_PATH"
echo ""
echo "推荐的修复命令："
echo "1. 如果找到了库文件，例如在 /path/to/lib/libnccl.so.2"
echo "   export LD_LIBRARY_PATH=/path/to/lib:\$LD_LIBRARY_PATH"
echo ""
echo "2. 或者使用提供的启动脚本："
echo "   ./run_train_with_nccl.sh --config configs/7B_sft.py"
echo ""
echo "3. 或者 source 环境配置文件："
echo "   source setup_nccl_env.sh"
echo "   python train.py --config configs/7B_sft.py"
echo "========================================="
