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
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--logdir', type=str, default='/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/trainlog')
#     parser.add_argument('--iter', type=int, default=10000)
#     # parser.add_argument('--nnodes', type=int, default=1)
#     # parser.add_argument('--rank', type=int, default=0)
#     # parser.add_argument('--master', type=str, default='localhost')
#     return parser.parse_args()

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

    tp = {1:1, 2:1, 4:1, 8:1}
    pp = {1:1, 2:1, 4:1, 8:1}

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
    # initialize model
    # assert gpc.config.HETER is not None
    # if args.profiling:
    #     if (gpc.config.HETER) or (gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0):
    #         internlm_accelerator.memory._record_memory_history()
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
    

    # if args.profiling:
    #     if (gpc.config.HETER) or (gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0):
    #         end_time = time.strftime("%Y-%m-%d-%H:%M",time.localtime(time.time()))
    #         snapshot_dir_path = f"./Memory traces/{end_time}/pp_rank{gpc.get_local_rank(ParallelMode.PIPELINE)}"
    #         snapshot_file_name = (
    #             f"snapshot{gpc.get_global_rank()}_recomp{gpc._config['model']['checkpoint']}_mb{gpc.micro_num}_"
    #             + f"tp{gpc.expert_tensor_parallel_size}_pp{gpc.pipeline_parallel_size}_{gpc._config['parallel']['pipeline']['mode']}_chunks{gpc._config['model']['num_chunks']}_"
    #             + f"seq{gpc._config['data']['seq_len']}_hidden{gpc._config['model']['hidden_size']}.pickle"
    #         )
    #         os.makedirs(snapshot_dir_path, exist_ok=True)
    #         internlm_accelerator.memory._dump_snapshot(os.path.join(snapshot_dir_path, snapshot_file_name))

if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    initialize_detector()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import os
# import subprocess
# import atexit
# import signal
# import time
# from datetime import datetime

# from internlm.core.context import global_context as gpc
# from internlm.core.trainer_builder import TrainerBuilder
# from internlm.data import (
#     build_train_loader_with_data_type,
#     build_valid_loader_with_data_type,
# )
# from internlm.initialize import initialize_distributed_env
# from internlm.model.builder import create_model
# from internlm.monitor import internevo_monitor
# from internlm.utils.common import parse_args

# # -------------------------------------------------------------------------
# # Detector integration (from InternEvo/run_training_dp.py)
# #
# # This block starts an external detector process (InternEvo's detector)
# # and ensures environment variables expected by the probe are set so that
# # when InternEvo is started under torchrun the detector / ncclprobe can
# # capture logs. The exact detector start command/path should be adjusted
# # to your deployment. Comments explain where to change paths.
# # -------------------------------------------------------------------------

# def start_detector_if_requested(log_dir: str):
#     """
#     Start the InternEvo detector (or ncclprobe controller) as a subprocess.

#     NOTE:
#     - Adjust `detector_start_cmd` to match the detector startup script in your setup.
#     - If detector is provided as a wheel, or has a different entrypoint, change accordingly.
#     - The environment variables set here mirror those used in InternEvo's run scripts so
#       the LD_PRELOAD probe and control-plane paths will be available to the training processes.
#     """
#     # Example environment variables used by InternEvo to inject nccl probe and control plane
#     my_env = os.environ.copy()
#     my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
#     my_env['OMP_NUM_THREADS'] = '1'
#     # Path to libncclprobe.so injected via LD_PRELOAD. Set to your actual build path.
#     my_env['LD_PRELOAD'] = '/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/build/libncclprobe.so'
#     # Path to control plane wheel (if used by detector). Adjust as needed.
#     my_env['CONTROL_PLANE_WHL_PATH'] = '/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/dist/control_plane-1.0-py3-none-any.whl'
#     # Where ncclprobe / controller will write logs (detector reads these). Use the supplied log_dir.
#     my_env['NCCLPROBE_LOG_PATH'] = log_dir
#     my_env['GLOBAL_CONTROLLER_LOG_PATH'] = log_dir
#     my_env['LOCAL_CONTROLLER_LOG_PATH'] = log_dir

#     # Make sure the log dir exists
#     os.makedirs(log_dir, exist_ok=True)

#     # Example detector start command.
#     # Replace this with the actual detector start command in your InternEvo deployment,
#     # for example it might be something like:
#     #    ["python", "-u", "detector/global_controller.py", "--log-dir", log_dir]
#     # or it might be an installed entrypoint:
#     #    ["python", "-m", "control_plane", "--log-dir", log_dir]
#     #
#     # The snippet below is intentionally generic — please update `detector_start_cmd`.
#     detector_start_cmd = [
#         "python",
#         "-u",
#         # Path to a detector start script. CHANGE THIS to your real detector entrypoint.
#         "/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/detector/start_detector.py",
#         "--log-dir",
#         log_dir,
#     ]

#     try:
#         proc = subprocess.Popen(detector_start_cmd, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     except FileNotFoundError:
#         # If the detector start script is not found, log a helpful message and don't crash.
#         print(f"[Detector] Detector start script not found at {detector_start_cmd[2]}. Skipping detector start.")
#         return None

#     # Register a cleanup hook to terminate detector on process exit
#     def _terminate_detector():
#         try:
#             if proc.poll() is None:
#                 proc.terminate()
#                 # give it a moment, then kill if needed
#                 time.sleep(1.0)
#                 if proc.poll() is None:
#                     proc.kill()
#         except Exception:
#             pass

#     atexit.register(_terminate_detector)

#     # Also ensure signals terminate detector
#     def _signal_handler(signum, frame):
#         _terminate_detector()
#         raise SystemExit()

#     signal.signal(signal.SIGINT, _signal_handler)
#     signal.signal(signal.SIGTERM, _signal_handler)

#     # Optionally, read a few lines from detector stdout to surface startup status
#     # (non-blocking read would be preferable for production; kept simple here)
#     try:
#         # give detector a short moment to emit startup logs
#         time.sleep(0.5)
#         if proc.stdout:
#             # drain small amount of available output
#             for _ in range(5):
#                 line = proc.stdout.readline()
#                 if not line:
#                     break
#                 print(f"[Detector stdout] {line.strip()}")
#     except Exception:
#         pass

#     print(f"[Detector] Started detector with PID {proc.pid} (log_dir={log_dir})")
#     return proc

# # -------------------------------------------------------------------------
# # End detector integration block
# # -------------------------------------------------------------------------


# @internevo_monitor(feishu_alert=True, clean_run=True)
# def main(args):
#     # initialize model
#     model = create_model(model_type=gpc.config.model_type)

#     # initialize train dataloader
#     train_dl, dataset_types = build_train_loader_with_data_type()

#     # initialize validation dataloader
#     val_dls = build_valid_loader_with_data_type()

#     # build trainer
#     merged_args = {**vars(args), "dataset_types": dataset_types}
#     trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)

#     # training
#     trainer.fit()


# if __name__ == "__main__":
#     args = parse_args()

#     # Initialize distributed environment
#     initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
#     assert hasattr(gpc, "config") and gpc.config is not None

#     # --- Detector startup: integrate InternEvo detector into InternEvo startup ---
#     # Choose a directory for detector logs. You can customize this path.
#     # InternEvo typically uses a timestamped directory; we follow a simple default here.
#     detector_log_dir = os.path.join(os.getcwd(), "trainlog")
#     detector_proc = start_detector_if_requested(detector_log_dir)
#     # -------------------------------------------------------------------------

#     # Run the main function with parsed arguments
#     main(args)

#     # When main returns, if detector was started, terminate it gracefully
#     if detector_proc:
#         try:
#             if detector_proc.poll() is None:
#                 detector_proc.terminate()
#                 time.sleep(0.5)
#                 if detector_proc.poll() is None:
#                     detector_proc.kill()
#         except Exception:
#             pass