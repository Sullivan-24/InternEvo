from datetime import datetime
import json
def set_pp_mode(JOB_NAME, pp_size, layer_num, chunk_num, seq_len):
    adap_partition = False
    if SCHEDULE == 0:
        pp_mode = "unified"
    elif SCHEDULE == 1:
        pp_mode = "1f1b"
    elif SCHEDULE == 2:
        pp_mode = "1f1b"
        chunk_num = layer_num // pp_size
    elif SCHEDULE == 3:
        pp_mode = "zbh1"
        split_backward = True
    elif SCHEDULE == 4:
        pp_mode = "1f1b"
        adap_partition = True
    else:
        raise ValueError("Wrong Schedule.")
    flag = "I-" if pp_mode == "1f1b" and chunk_num > 1 else ""
    print(f"{JOB_NAME}, {seq_len}, {flag}{pp_mode}, {chunk_num}, Ada Model Partition: {adap_partition}")
    return pp_mode, chunk_num, adap_partition

MODEL_NAME = "13B_llama2"
# MODEL_NAME = "14B_qwen2"
file_path = f'/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/PipelineSimulator/schedule_results/{MODEL_NAME}/runtime.json'
with open(file_path, 'r', encoding='utf-8') as f:
    runtime_info = json.load(f)

# Extract variable assignments from runtime_info
num_microbatches = runtime_info.get("num_microbatches")
pp_size = runtime_info.get("PP_SIZE")
stage_placement = runtime_info.get("stage_placement")
scheduler_type = runtime_info.get("scheduler_type")
split_backward = runtime_info.get("split_backward")
first_stage = runtime_info.get("first_stage")
last_stage = runtime_info.get("last_stage")
pp_ranks_containing_last_stage = runtime_info.get("pp_ranks_containing_last_stage")
recomp_stages = runtime_info.get("recomp_stages")
dp_size = runtime_info.get("DP_SIZE")
DP_Transfer = runtime_info.get("DP_Transfer")
layer_partition = runtime_info.get("layer_partition")
unified_scheduler = runtime_info.get("unified_scheduler")
comm_graph = runtime_info.get("comm_graph")
MODEL_NAME = runtime_info.get("MODEL_NAME")
SEQ_LEN = runtime_info.get("SEQ_LEN")
NUM_LAYER = runtime_info.get("NUM_LAYER")
tp_size = runtime_info.get("TP_SIZE")
FAILURE = runtime_info.get("FAILURE")
FALCON = runtime_info.get("FALCON")
HETER = runtime_info.get("HETER")
HETER_RATIOS = runtime_info.get("HETER_RATIOS")
slow_ratio_map = runtime_info.get("slow_ratio_map")
Failure_ranks_map = runtime_info.get("Failure_ranks_map")
transfer_info = runtime_info.get("transfer_info")
Available_ranks_map = runtime_info.get("Available_ranks_map")
FAILURE_GLOBAL_RANKS = runtime_info.get("FAILURE_GLOBAL_RANKS")
Failure_ranks_info = runtime_info.get("Failure_ranks_info")
HETER_GLOBAL_RANKS = runtime_info.get("HETER_GLOBAL_RANKS")
Heter_ranks_map = runtime_info.get("Heter_ranks_map")
Heter_ranks_info = runtime_info.get("Heter_ranks_info")
slow_ratio_dict = runtime_info.get("slow_ratio_dict")
slow_ratio_dict = {int(k): v for k, v in slow_ratio_dict.items()}
per_stage_layer_num = runtime_info.get("per_stage_layer_num")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
DO_ALERT = False
HID_FAC = 1
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble
MICRO_NUM = num_microbatches
evaluation = True
profile_all_rank = False
profile_fwd_bwd = False
layerwise = False

SCHEDULE = 3
HETER = False
FAILURE = False
if SCHEDULE == 0:
    DP_Transfer=True
    MICRO_NUM = num_microbatches*dp_size
else:
    DP_Transfer = False 

SLEEP_TIME_Profiles = [[x / per_stage_layer_num for x in sublist] for sublist in [[45,70],[45,70,20]]] #per stage sleep time profile, we need compute per layer sleep time
SLEEP_TIME = SLEEP_TIME_Profiles[0]
if SCHEDULE == 3 or (SCHEDULE == 0 and split_backward):
    SLEEP_TIME=SLEEP_TIME_Profiles[1]