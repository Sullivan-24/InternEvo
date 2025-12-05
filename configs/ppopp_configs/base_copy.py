import os
from preprocess import generate_
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

DO_ALERT = False

profile_all_rank = True
profile_fwd_bwd = True
DP_Transfer = False
layerwise = False
split_backward = False

SEQ_LEN = 4096
SCHEDULE = 3
if SCHEDULE not in (0, 1, 2, 3, 4):
    print("Note: Env PP_MODE not set, set PP_MODE to default 1 (1f1b).")
    SCHEDULE = 1

DP_SIZE = 4
PP_SIZE = 4
TP_SIZE = 1

# MICRO_BSZ = int(8/DP_SIZE) # maintain the same global bsz, global_batch_size=gpc.config.data.micro_bsz* gpc.config.data.micro_num* gpc.get_world_size(ParallelMode.DATA)
MICRO_NUM = PP_SIZE*4

HETER= False
HETER_DEVICE = [[False for _ in range(PP_SIZE)] for _ in range(DP_SIZE)]
HETER_DEVICE[0][1] = True
HETER_DEVICE[1][2] = True


FALCON = False

HID_FAC = 1
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble

FAILURE= False
FAILURE_TP_ID = [0]
FAILURE_DP_ID = [1]
FAILURE_PP_ID = [2]
FAILURE_GLOBAL_RANKS = []
if FAILURE:
    for failure_index in range(len(FAILURE_DP_ID)):
        failure_tp = FAILURE_TP_ID[failure_index]
        failure_dp = FAILURE_DP_ID[failure_index]
        failure_pp = FAILURE_PP_ID[failure_index]
        # FAILURE_GLOBAL_RANKS.append(failure_pp*DP_SIZE+failure_dp) #!!!!!!! easy wrong
        FAILURE_GLOBAL_RANKS.append(failure_pp*(DP_SIZE*TP_SIZE)+failure_dp*TP_SIZE+failure_tp) #!!!!!!! easy wrong

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
        split_backward = True
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

slow_ratio = 1#0.825#*0.625
SLEEP_TIME_Profiles =[[40,60],[40,60,0]]#[[50,100],[50,80,20]]#[32,64,0]
SLEEP_TIME = SLEEP_TIME_Profiles[0]
if SCHEDULE == 3 or (SCHEDULE == 0 and split_backward):
    SLEEP_TIME=SLEEP_TIME_Profiles[1]
SLEEP_TIME = [int(sleep_time*slow_ratio) for sleep_time in SLEEP_TIME]