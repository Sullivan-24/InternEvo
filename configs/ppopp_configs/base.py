import os
from preprocess import generate_
DO_ALERT = False
HETER = True
SLEEP_TIME = 0
layerwise = False
profile_fwd_bwd = True
SEQ_LEN = 2048
SCHEDULE = 0
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble
ALPA = False


def set_pp_mode(JOB_NAME, pp_size, layer_num, chunk_num, seq_len, adap_partition):
    if SCHEDULE == 0:
        pp_mode = "unified"
    elif SCHEDULE == 1:
        pp_mode = "1f1b"
    elif SCHEDULE == 2:
        pp_mode = "zbh1"
    elif SCHEDULE == 3:
        pp_mode = "1f1b"
        chunk_num = layer_num // pp_size
        assert not adap_partition, f"I-1F1B does not support adaptive model partition."
    else:
        raise ValueError("Wrong Schedule.")

    flag = "I-" if pp_mode == "1f1b" and chunk_num > 1 else ""
    print(f"{JOB_NAME}, {seq_len}, {flag}{pp_mode}, {chunk_num}, Ada Model Partition: {adap_partition}")
    return pp_mode, chunk_num