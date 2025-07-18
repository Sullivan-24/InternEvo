#!/bin/bash

ENV_PATH="/conda_env/llm-torch2.1-flash2.2.1.sh"
IMAGE="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/paieflops/jiaopenglong:jpl-py310-torch2-1-flash2-2-1-cu118-accl"

DLC_CONFIG="/cpfs01/user/matenghui/InternEvo/demo_dlc.config"

# gpu numbers
GPU_NUMS=8

# job name
JOB_NAME="demo"
current_time=$(date  "+%Y-%m-%d-%H:%M:%S")

# your cmd
# 可以构建任务shell 脚本，然后
# DLC_CMD="source ${ENV_PATH} && \
# torchrun --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
# --nproc_per_node=8 --nnodes=\$WORLD_SIZE --node_rank=\$RANK InternEvo/train.py \
# --config InternEvo/configs/42B_llama2.py --launcher torch \
# 2>&1 | tee InternEvo/debug_log/42B_llama2/${current_time}_.log"

# 优先级， 1-4，4最高
PRIORITY=4

PARTITION="llm_s"
WORKSPACE_ID="wsbuzbigeh1hjmst"

# PARTITION="llm_ddd"
# WORKSPACE_ID="ws1ujefpjyfgqjwp"
DLC_PATH="/cpfs01/user/matenghui/dlc"
#log_path="InternEvo/test_het/14B_llama2/unified_block"
log_path="InternEvo/test/7B_gemma"
# 判断路径是否为目录，若不存在则递归创建
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "目录已创建：$log_path"
else
    echo "目录已存在：$log_path"
fi

function do_dsw() {
    echo "do_dsw (only support job whose worldsize % 8 == 0 or worldsize < 8)"
    worker_cpu_total=40
    worker_mem_total=500

    if [[ $GPU_NUMS -lt 8 ]]; then
        num_nodes=1
        num_tasks_per_node=${GPU_NUMS}
        let node_mems=${worker_mem_total}*GPU_NUMS/8
        let cpu_nums=${worker_cpu_total}*GPU_NUMS/8
        shared_memory="10Gi"
        worker_gpu=${GPU_NUMS}
    else
        let num_nodes=GPU_NUMS/8
        num_tasks_per_node=8
        node_mems=${worker_mem_total}
        cpu_nums=${worker_cpu_total}
        shared_memory="200Gi"
        worker_gpu=8
    fi

    ${DLC_PATH} create job --config ${DLC_CONFIG} \
--kind PyTorchJob \
--name ${JOB_NAME} \
--priority $PRIORITY \
--worker_count $num_nodes \
--worker_cpu $cpu_nums \
--worker_gpu $worker_gpu \
--worker_memory "${node_mems}Gi" \
--worker_image ${IMAGE} \
--workspace_id ${WORKSPACE_ID} \
--worker_shared_memory ${shared_memory} \
--command "bash InternEvo/dlc_cmd.sh 2>&1 | tee ${log_path}/${current_time}.log"
# --command "${DLC_CMD}"
}

do_dsw
