#!/bin/bash
# launch_with_detector.sh  (示例)
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1

# path to your built probe .so and control plane wheel
export LD_PRELOAD=detector/build/libncclprobe.so
export CONTROL_PLANE_WHL_PATH=detector/dist/control_plane-1.0-py3-none-any.whl

LOGDIR=trainlog/log_$(date +%Y%m%d_%H%M%S)
mkdir -p ${LOGDIR}
export NCCLPROBE_LOG_PATH=${LOGDIR}
export GLOBAL_CONTROLLER_LOG_PATH=${LOGDIR}
export LOCAL_CONTROLLER_LOG_PATH=${LOGDIR}

# start redis (only once on master)
redis-server --save "" --appendonly no --bind 127.0.0.1 &
sleep 1

# launch training with torchrun (nproc_per_node 按你机器修改)
torchrun --nproc_per_node=8 --nnodes=1 train.py --config configs/7B_llama2.py --launcher torch