import os
from preprocess import generate_
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

DO_ALERT = False

profile_all_rank = True
profile_fwd_bwd = True
DP_Transfer = False
layerwise = False

SEQ_LEN = 2048
SCHEDULE = 0
if SCHEDULE not in (0, 1, 2, 3, 4):
    print("Note: Env PP_MODE not set, set PP_MODE to default 1 (1f1b).")
    SCHEDULE = 1

DP_SIZE = 2
PP_SIZE = 4
TP_SIZE = 1

# MICRO_BSZ = int(8/DP_SIZE) # maintain the same global bsz, global_batch_size=gpc.config.data.micro_bsz* gpc.config.data.micro_num* gpc.get_world_size(ParallelMode.DATA)
MICRO_NUM = PP_SIZE*4

HETER= True
HETER_DEVICE = [[False for _ in range(PP_SIZE)] for _ in range(DP_SIZE)]
HETER_DEVICE[0][2] = True
# HETER_DEVICE[1][1] = True
# HETER_DEVICE[2][3] = True
slow_ratio = 2
SLEEP_TIME = [32,64,0]
SLEEP_TIME = [int(sleep_time*slow_ratio) for sleep_time in SLEEP_TIME]

FALCON = False
# num_microbatches_all_dp = [14,18]

HID_FAC = 1
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble

FAILURE= True
FAILURE_DP_ID = [0,1]
FAILURE_PP_ID = [1,2]
FAILURE_GLOBAL_RANKS = []
if FAILURE:
    for failure_index in range(len(FAILURE_DP_ID)):
        failure_dp = FAILURE_DP_ID[failure_index]
        failure_pp = FAILURE_PP_ID[failure_index]
        FAILURE_GLOBAL_RANKS.append(failure_pp*DP_SIZE+failure_dp) #!!!!!!! easy wrong

if FALCON or FAILURE:
    SCHEDULE = 0

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
    elif SCHEDULE == 4:
        pp_mode = "1f1b"
        adap_partition = True
    else:
        raise ValueError("Wrong Schedule.")

    flag = "I-" if pp_mode == "1f1b" and chunk_num > 1 else ""
    print(f"{JOB_NAME}, {seq_len}, {flag}{pp_mode}, {chunk_num}, Ada Model Partition: {adap_partition}")
    return pp_mode, chunk_num, adap_partition

if SCHEDULE == 0:
    num_microbatches, pp_size, stage_placement, scheduler_type, \
    split_backward, unified_scheduler, comm_graph, first_stage, \
    last_stage, Devices_containing_last_stage, recomp_stages, dp_size, \
    DP_Transfer,num_microbatches_per_dp = generate_()
    assert dp_size == DP_SIZE
    assert pp_size == PP_SIZE
    MICRO_NUM = num_microbatches

if FALCON:
    DP_Transfer = True
    MICRO_NUM = num_microbatches_per_dp*DP_SIZE