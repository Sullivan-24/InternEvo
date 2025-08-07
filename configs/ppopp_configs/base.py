import os
from preprocess import generate_
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

DO_ALERT = False
HETER = True
SLEEP_TIME = 0
layerwise = False
profile_fwd_bwd = False
SEQ_LEN = 2048*1
SCHEDULE = 4
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