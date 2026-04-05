#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.model.builder import create_model
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args
from internlm.core.context import ParallelMode
import os
import time
from internlm.accelerator import get_accelerator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import abc
import sys
import time
import torch
from datetime import datetime
import dataclasses
import redis
from math import sqrt, floor
import subprocess
import argparse
import re
import socket

def run_and_log_internevo(internevo_cmd_args, log_file_handle, log_file_dir):
    # Start the subprocess
    print(internevo_cmd_args)
    my_env = os.environ
    my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    my_env['OMP_NUM_THREADS'] = '1'
    my_env['LD_PRELOAD'] = '/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/build/libncclprobe.so'
    my_env['CONTROL_PLANE_WHL_PATH'] = '/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/dist/control_plane-1.0-py3-none-any.whl'
    my_env['NCCLPROBE_LOG_PATH'] = log_file_dir
    my_env['GLOBAL_CONTROLLER_LOG_PATH'] = log_file_dir
    my_env['LOCAL_CONTROLLER_LOG_PATH'] = log_file_dir
    # process = subprocess.Popen(internevo_cmd_args, text=True)
    process = main(internevo_cmd_args)
    iteration_time_pattern = re.compile(r'iteration \(ms\): ([\d.]+)')
    current_iteration_pattern = re.compile(r'iteration\s+(\d+)/')
    log_file_handle.write("Iteration, IterationTime(ms)\n")
    output = None
    # Read the output line by line
    while True:
        try:
            if output is not None:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and ('[2024' in output):
                    # Find the iteration time and current iteration
                    iteration_time_match = iteration_time_pattern.search(output)
                    current_iteration_match = current_iteration_pattern.search(output)
                    if iteration_time_match and current_iteration_match:
                        iteration_time = iteration_time_match.group(1)
                        current_iteration = current_iteration_match.group(1)
                        # Log the parsed information to the file
                        log_entry = f"{current_iteration}, {iteration_time}\n"
                        print(log_entry)
                        log_file_handle.write(log_entry)
                        log_file_handle.flush()  # Ensure it writes immediately
        except KeyboardInterrupt:
            process.terminate()
            log_file_handle.write("Training terminated\n")
            log_file_handle.flush()
            break


def initialize_detector():
    os.chdir('./')
    args = parse_args()
    log_file_dir = "trainlog"
    master = "localhost"  # os.getenv("MASTER_ADDR")
    nnodes = 1  # int(os.getenv("WORLD_SIZE"))
    rank = 0  # int(os.getenv("RANK"))

    # start redis
    redis_cmd = ["redis-server", "--save", "\"\"", "--appendonly", "no", "--bind", f"{master}"]
    if rank == 0:
        redis_proc = subprocess.Popen(redis_cmd)
        redis_logstr = "Rank 0 starts redis: [" + " ".join(redis_cmd) + "]\n"
        time.sleep(2)
        time_str = str(datetime.now()).replace(" ", '_').replace('-', '_').replace(':', '_').replace('.', '_')
        log_file_dir = log_file_dir + '/log_' + time_str
        client = redis.StrictRedis(host=master, port=6379, db=0)
        client.set("trainlog_dir", log_file_dir)
    else:
        redis_proc = None
        redis_logstr = ""
        client = redis.StrictRedis(host=master, port=6379, db=0)
        while True:
            try:
                logdir_ret = client.get("trainlog_dir")
                if len(logdir_ret) > 5:
                    break
            except:
                pass
        print("Logdir worker!", logdir_ret)
        log_file_dir = logdir_ret.decode()
    
    log_file_dir += f"_rank{rank}"
    if not os.path.exists(log_file_dir):
        os.mkdir(log_file_dir)
    log_file_path = log_file_dir + f"/internevo_output_{rank}.log"

    num_gpus = torch.cuda.device_count()
    gpu_properties = torch.cuda.get_device_properties("cuda:0")
    gpu_memory = gpu_properties.total_memory / (1024**3)  # GB
    total_gmem = gpu_memory * num_gpus
    # Find a proper hidden size to fulfill the GPU memory
    hsize = 1024 #int(1024 * (floor(sqrt(total_gmem / 18) * 2) / 2))
    hostname = socket.gethostname()
    ipaddr = socket.gethostbyname(hostname)

    info_str = f"***** log={log_file_path}, master={master}, nnodes={nnodes}, rank={rank},\
                num_gpus={num_gpus}, gpu_type={gpu_properties.name} gpu_memory={gpu_memory},\
                total_gmem={total_gmem}, hidden_size={hsize}, [My IP={ipaddr} Master IP={master}]\n"
    print(info_str)

    with open(log_file_path, 'w') as log_file:
        log_file.write(info_str)
        log_file.write(redis_logstr)
        log_file.flush()
        run_and_log_internevo(args, log_file, log_file_dir)

    if redis_proc:
        redis_proc.terminate()

internlm_accelerator=get_accelerator()
@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    model = create_model(model_type=gpc.config.model_type)

    # print(model)

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    merged_args = {**vars(args), "dataset_types": dataset_types}
    trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)
    
    trainer.fit()

if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    initialize_detector()