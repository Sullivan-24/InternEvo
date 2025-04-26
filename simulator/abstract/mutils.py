from ..config import *


if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
    assert SCHEDULE_METHOD == Schedule.Layerwise, "SCHEDULE_METHOD should be Layerwise"

if SCHEDULE_METHOD in (Schedule.STANDARD_INTERLEAVED, Schedule.INTERLEAVED):
    assert CHUNK_NUM > 1, "Interleaved: CHUNK_NUM should be larger than 1"
elif SCHEDULE_METHOD in (Schedule.STANDARD_ZBH1, Schedule.ZBH1):
    assert CHUNK_NUM == 1, "ZBH1: CHUNK_NUM should be 1"
    assert SPLIT_BACKPROP, "Zerobubble needs bwd split"
elif SCHEDULE_METHOD in (Schedule.ONE_F_ONE_B, Schedule.STANDARD_1F1B):
    assert CHUNK_NUM == 1, "1F1B: CHUNK_NUM should be 1"
elif SCHEDULE_METHOD in (Schedule.Chimera, ):
    assert CHUNK_NUM == 1, "Chimera: CHUNK_NUM should be 1"
elif SCHEDULE_METHOD in (Schedule.STANDARD_ZBV, Schedule.ZBV, ):
    assert CHUNK_NUM == 2, "ZBV: CHUNK_NUM should be 2"

assert CHUNK_NUM <= (LAYER_NUM // DEVICE_NUM), f"Chunk num should be in [1,{LAYER_NUM//DEVICE_NUM}], but got {CHUNK_NUM}. "


STAGE_SEARCH_METHOD = StageSearchOrder.Random

GLOBAL_TIME = 0

def UPDATE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME+=1

def DECREASE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME-=1

def GET_TIME():
    global GLOBAL_TIME
    return GLOBAL_TIME

def RESET_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME = 0


def is_head_layer(sid, total_layer_num=LAYER_NUM, layerwise=LAYERWISE):
    if layerwise and total_layer_num + 1 == sid:
        return True
    return False

def is_last_stage(sid, total_stage_num=STAGE_NUM):
    if total_stage_num - 1 == sid:
        return True
    return False



