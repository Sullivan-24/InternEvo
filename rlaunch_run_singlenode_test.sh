#!/bin/bash

set -e 
set -u

# export NCCL_DEBUG=INFO
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_DEVICE_MAX_CONNECTIONS=0
# export CUDA_LAUNCH_BLOCKING=1
# export OMP_NUM_THREADS=1

NNODES=1
NPROC_PER_NODE=8
MODEL_SIZE=7B_llama2
CONFIG_FILE=configs/7B_llama2_test.py
LOG_DIR=./attn_record
DATA_NAME=github
DATA_SIZE=all
BUCKET_SIZE=B512
SEQ_LEN=4*1024
MICRO_NUM=8
DP_TP_PP=dp2_tp2_pp2
BUCKET_MODE="round_robin"
PROFILE="profile_True"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  
LOG_FILE=${LOG_DIR}/${DATA_NAME}/${MODEL_SIZE}/${SEQ_LEN}/M${BUCKET_MODE}_${BUCKET_SIZE}_mb${MICRO_NUM}/${DP_TP_PP}_${PROFILE}_${TIMESTAMP}.log
# LOG_FILE=test_8k_32_tp2pp4_sych.log

mkdir -p "$(dirname "${LOG_FILE}")"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    train.py \
    --config ${CONFIG_FILE} \
    --launcher "torch" \
    2>&1 | tee ${LOG_FILE}