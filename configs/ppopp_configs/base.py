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

MICRO_NUM = PP_SIZE * 2
HETER= True
HETER_DEVICE = [[False for _ in range(PP_SIZE)] for _ in range(DP_SIZE)]
HETER_DEVICE[0][2] = True
# HETER_DEVICE[1][2] = True
#2k: 1f1b [30,50,0] , zbh1 [30,42,8], upp onechunk
SLEEP_TIME = [30,48,12]#[30,60,0]#[40,80,0]#f\b\w ms

# FAILURE = True
# FAILURE_DEVICE = [[False for _ in range(PP_SIZE)] for _ in range(DP_SIZE)]
# FAILURE_DEVICE[1][1] = 
DP_Transfer = False

FALCON = False
num_microbatches_all_dp = [8,8]

HID_FAC = 1
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble

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