from preprocess import generate_
from datetime import datetime
import math

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
DO_ALERT = False
HID_FAC = 1
OVERLAP_SYNC_GRAD = False # should be disabled when enable zerobubble

evaluation = False
profile_all_rank = False
profile_fwd_bwd = False
DP_Transfer = False
layerwise = False
split_backward = False

SEQ_LEN = 4096
DP_SIZE = 1
PP_SIZE = 16
TP_SIZE = 4
SCHEDULE = 3
if SCHEDULE not in (0, 1, 2, 3, 4):
    print("Note: Env PP_MODE not set, set PP_MODE to default 1 (1f1b).")
    SCHEDULE = 1
# MICRO_BSZ = int(8/DP_SIZE) # maintain the same global bsz, global_batch_size=gpc.config.data.micro_bsz* gpc.config.data.micro_num* gpc.get_world_size(ParallelMode.DATA)
MICRO_NUM = 8

FALCON = False
HETER= False
HETER_GLOBAL_RANKS = []
Heter_ranks_map = [[[] for _ in range(PP_SIZE) ] for _ in range(DP_SIZE)]
Heter_ranks_info = []
slow_ratio_dict={}
slow_ratio_map = [[0 for _ in range(PP_SIZE) ] for _ in range(DP_SIZE)]
# Heter_ranks_map[0][0] = [0]
# Heter_ranks_map[1][1] = [0]
# slow_ratio_map[0][0] = 2
# slow_ratio_map[1][1] = 2
# # Heter_ranks_map[1][3]   = [0]
# # Heter_ranks_map[1][9] = [0]
# Heter_ranks_map[0][2] = [0]
# # Heter_ranks_map[1][13]= [0]
# Heter_ranks_map[0][8] = [0]
# Heter_ranks_map[1][14] = [0]

# # # slow_ratio_map[1][3] = 1
# # # slow_ratio_map[1][9] = 0.5
# slow_ratio_map[0][2] = 2
# # # slow_ratio_map[1][13]=2
# slow_ratio_map[0][8]=1
# slow_ratio_map[1][14]=0.5
FAILURE = False
FAILURE_GLOBAL_RANKS = []
Failure_ranks_info = []
Failure_ranks_map = [[[] for _ in range(PP_SIZE) ] for _ in range(DP_SIZE)]
Available_ranks_map = [[[] for _ in range(PP_SIZE) ] for _ in range(DP_SIZE)]

if FAILURE:
    # Failure_ranks_map[0][2] = [1,2]
    # Failure_ranks_map[1][2] = [0,1]
    # Failure_ranks_map[2][3] = [0,1]
    # Failure_ranks_map[3][0] = [0,1]
    # Failure_ranks_map[1][3] = [0,1]
    # Failure_ranks_map[2][1] = [0,1]
    # Failure_ranks_map[3][1] = [0,1]

    # Failure_ranks_map[0][6] = [0,1]
    # # Failure_ranks_map[0][6] = [0,1]
    # Failure_ranks_map[1][5] = [0,1]
    # Failure_ranks_map[0][14] = [0,1]

    # Failure_ranks_map[0][1] = [0,1]
    # # Failure_ranks_map[0][4] = [0,1]
    # Failure_ranks_map[0][5] = [0,1]
    # # Failure_ranks_map[0][3] = [0,1]
    # # Failure_ranks_map[0][7] = [0,1]
    # Failure_ranks_map[1][2] = [0,1]
    # # Failure_ranks_map[1][0] = [0,1]
    # Failure_ranks_map[1][6] = [0,1]

    for dp_index, pp_ranks in enumerate(Failure_ranks_map):
        for pp_index, failure_tp_ranks in enumerate(pp_ranks):
            TPGroup = [i for i in range(TP_SIZE)]
            if len(failure_tp_ranks) == 0:
                Available_ranks_map[dp_index][pp_index] = TPGroup
                continue
            for failure_tp_rank in failure_tp_ranks:
                TPGroup.remove(failure_tp_rank)
                Failure_ranks_info.append(f"dp:{dp_index}, pp:{pp_index},tp:{failure_tp_rank}")
            if len(TPGroup)>0:
                while(math.log2(len(TPGroup))%1 != 0):
                    Failure_ranks_map[dp_index][pp_index].append(TPGroup.pop(-1))
            for failure_local_tp_rank in Failure_ranks_map[dp_index][pp_index]:
                FAILURE_GLOBAL_RANKS.append(pp_index*(DP_SIZE*TP_SIZE)+dp_index*TP_SIZE+failure_local_tp_rank)
            Available_ranks_map[dp_index][pp_index] = TPGroup
            # for tp_local_rank in TPGroup:
            #     Available_ranks_map[dp_index][pp_index].append(pp_index*(DP_SIZE*TP_SIZE)+dp_index*TP_SIZE+tp_local_rank)

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
#13B[[50,85],[50,80,5]] 40/4
#70B[[50,115],[50,90,25]] 80/16 layers
#30B[[60,150],[60,120,30]] 64/8
#7B [[120,180],[120,160,20]]] 32/2
layer_partition = []
with open("/mnt/petrelfs/xuhaoran/tenghui/InternEvo/executor_config/partition.txt", "r") as f:
    content = f.read().strip()
    layer_partition = eval(content)
    layer_partition = [8]*16
    layer_partition[0]=7
    layer_partition[-1] = 7 

per_stage_layer_num = 16 #!!!
SLEEP_TIME_Profiles = [[x / per_stage_layer_num for x in sublist] for sublist in [[120,180],[120,160,20]]] #per stage sleep time profile, we need compute per layer sleep time
SLEEP_TIME = SLEEP_TIME_Profiles[0]
if SCHEDULE == 3 or (SCHEDULE == 0 and split_backward):
    SLEEP_TIME=SLEEP_TIME_Profiles[1]

# SLEEP_TIME = [int(sleep_time*slow_ratio) for sleep_time in SLEEP_TIME]
if HETER:
    for dp_index,pps in enumerate(Heter_ranks_map):
        for pp_index,heter_local_tp_ranks in enumerate(pps):
            for heter_tp in heter_local_tp_ranks:
                Heter_global_rank = pp_index*(DP_SIZE*TP_SIZE)+dp_index*TP_SIZE+heter_tp
                slow_ratio_dict[Heter_global_rank] = slow_ratio_map[dp_index][pp_index]
                HETER_GLOBAL_RANKS.append(Heter_global_rank)
                Heter_ranks_info.append(f"dp{dp_index}, pp:{pp_index}, tp:{heter_tp}, slow_ratio:{slow_ratio_map[dp_index][pp_index]}")