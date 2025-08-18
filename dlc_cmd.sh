#!/bin/bash

# 设置环境变量
source /conda_env/llm-torch2.1-flash2.2.1.sh
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
export CUDA_LAUNCH_BLOCKING=1
# 定义参数
MODEL_TYPE="llama2"
MODEL_SIZE="7B"
train_file="InternEvo/train.py"
config_file="InternEvo/configs/${MODEL_SIZE}_${MODEL_TYPE}_test.py"
profiling="--profiling"  # 如果需要，可以取消此行的注释
# profiling=""
# 运行训练
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nproc_per_node=4 --nnodes=$WORLD_SIZE --node_rank=$RANK $train_file \
--config $config_file --launcher torch $profiling