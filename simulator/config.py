from dataclasses import dataclass
from simulator.abstract.variables import *
from simulator.model_config import *
# --------------------- Solver config ---------------------
BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.CHIMERA
RUN_MODE = RunMode.SIM_SOLVE

SOLVING_TIME_LIMIT = 60 * 30
SCHEDULE_METHOD = Schedule.Layerwise
SCHEDULE_METHOD = Schedule.UnifiedPP
# SCHEDULE_METHOD = Schedule.STANDARD_INTERLEAVED
# SCHEDULE_METHOD = Schedule.STANDARD_1F1B
# SCHEDULE_METHOD = Schedule.STANDARD_ZBH1
# SCHEDULE_METHOD = Schedule.ZBV
# CHUNK_NUM = 2
STAGE_PLACEMENT = Placement.INTERLEAVED
STAGE_PLACEMENT = Placement.SEARCHED
# STAGE_PLACEMENT = Placement.WAVELIKE
if SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
    STAGE_PLACEMENT = Placement.INTERLEAVED

# --------------------- Solver config ---------------------
test_upp = True if SCHEDULE_METHOD == Schedule.UnifiedPP else False
OVERLAP_AWARE_SCHEDULE = True
HETER_DEVICE = False
# --------------------- Simulator config ---------------------
FIND_OPTIMAL_RECOMP = True
TIME_LIMIT = 10000

# [1, 2, 100, None]
OVERLAP_DEGREE = None
MEMORY_CONSTRAIN = 0.99
MEMORY_REDUCATION = 0.0

EMB_F_TIME = 0
HEAD_F_TIME = 2
HEAD_B_TIME = 2
HEAD_W_TIME = 2
CE_F_TIME = 2
CE_B_TIME = 2
CE_W_TIME = 0
F_TIME = 4
B_TIME = 4
W_TIME = 4
COMM_TIME = 0

# No backward splitting
EMB_F_TIME = 0
EMB_B_TIME = 4
EMB_W_TIME = 0
HEAD_F_TIME = 34
HEAD_B_TIME = 32
HEAD_W_TIME = 0
CE_F_TIME = 0
CE_B_TIME = 0
CE_W_TIME = 0
F_TIME = 12
B_TIME = 24
W_TIME = 0
COMM_TIME = 0

SPLIT_BACKPROP = test_upp
if SCHEDULE_METHOD in (Schedule.STANDARD_ZBH1, Schedule.ZBV):
    SPLIT_BACKPROP = True
    if SCHEDULE_METHOD == Schedule.ZBV:
        STAGE_PLACEMENT = Placement.WAVELIKE
elif SCHEDULE_METHOD in (Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED):
    SPLIT_BACKPROP = False

if SPLIT_BACKPROP:
    EMB_F_TIME = 0
    EMB_B_TIME = 2
    EMB_W_TIME = 2
    HEAD_F_TIME = 34
    HEAD_B_TIME = 16
    HEAD_W_TIME = 16
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0
    F_TIME = 12
    B_TIME = 14
    W_TIME = 10
    COMM_TIME = 0

LAYERWISE = False
RECOMP = False
AUTO_RECOMP_SEARCH = RECOMP
RUN_SCHEDULE = False
RUN_STANDARD_ZBV = False
if not RUN_SCHEDULE and RUN_STANDARD_ZBV:
    print("WARN, overlook non-transformer layers")
    EMB_F_TIME = 0
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_F_TIME = 0
    HEAD_B_TIME = 0
    HEAD_W_TIME = 0
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0
    F_TIME = 12
    B_TIME = 12
    W_TIME = 12

SCHEDULE_UNIT = MICRO_BATCH_NUM // 1
REVERSE_LAST_STAGES = False
REVERSE_FIRST_STAGES = False
# Run standard ZBV ---------------------
# SCHEDULE_METHOD = Schedule.ZBV
# RUN_SCHEDULE = False
# RUN_STANDARD_ZBV = True
# Run standard ZBV ---------------------
DENSITY_MAX = 1
DENSITY_MIN = 1
# --------------------- Simulator config ---------------------


# Memory overhead calculation
GPU_MAX_MEM = 80 * G / G
FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes
MIX_TRAINING = True
DATA_TYPE: int = FP16 if MIX_TRAINING else FP32
b = MICRO_BATCH_SIZE
s = SEQ_LEN
h = HIDDEN_SIZE
a = NUM_ATTENTION_HEAD
l = LAYER_NUM
v = VOCAB_SIZE
i = INTER_SIZE

LAYER_PARA_NUM = 4 * h * h + 3 * h * i + 2 * h if MODEL_TYPE in ("LLAMA", "Qwen") else 12 * h * h + 13 * h 
HEAD_PARA_NUM = v * h
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM

LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G

OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 1

@dataclass
class Parameter:
    EMB: int = b * v * h
    HEAD: int = b * v * h
    LAYER: int = LAYER_PARA_NUM

@dataclass
class StateMemory:
    EMB: int = DATA_TYPE * Parameter.EMB / G / TP_SIZE
    HEAD: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    LAYER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    # Optimizer M + V, gradients * 1, model * 1
    OPTIMIZER: int = FP32 * 4 * (Parameter.LAYER * l + Parameter.EMB + Parameter.HEAD) / G / (TP_SIZE * PP_SIZE) / ZERO_SIZE

ACT_OPT_COE = 0.18298 # adjust by profiling results
ACT_B_RATIO = 0.5669
ACT_W_RATIO = 1 - ACT_B_RATIO
ACT_HEAD_B = 2/3
ACT_HEAD_W = 1 - ACT_HEAD_B
@dataclass
class Activation:
    INPUT: int = (2*b*s*h) / G / TP_SIZE
    FULL: int = (34*b*s*h + 5*b*s*s*a) * ACT_OPT_COE / G / TP_SIZE
    LOSS: int = (2*FP32*b*s*v) / G / TP_SIZE
    HEAD: int = (3*FP16*b*s*h) / G / TP_SIZE
    EMB: int = (FP16*b*s*h) / G / TP_SIZE

GRAD_COE = 0.225
# 1.5->0.6
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE * GRAD_COE
    PARAMETER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    HEAD_INPUT: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    HEAD_PARA: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    # HEAD_INPUT: int = 0
    # HEAD_PARA: int = 0



# --------------------- Painter Config ---------------------
PIXEL_BASE = 2
PP_HEIGHT = 35
PP_ALIGN = 5
SHOW_WORKLOAD_TEXT = False
# --------------------- Painter Config ---------------------

# --------------------- Save File Config ---------------------
SAVE_RES_TO_FILE = True
SCH_FILE_PATH = f"schedule_results/schedules/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
PLA_FILE_PATH = f"schedule_results/placements/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
TEMP_PLA_PATH = f"schedule_results/placement.txt"
TEMP_RES_PATH = f"schedule_results/result.txt"

STAGE_NUM = int(DEVICE_NUM * CHUNK_NUM)
assert STAGE_NUM <= LAYER_NUM, f"Stage ({STAGE_NUM}) should be less than Layer ({LAYER_NUM}). "

WORKLOAD_TYPE_NUM = 3
if not SPLIT_BACKPROP:
    B_TIME += W_TIME
    WORKLOAD_TYPE_NUM = 2

MAX_ACTIVATION_COUNTS = int(STAGE_NUM * MAX_ACTIVATION_TIMES_OF_STAGE_NUM)