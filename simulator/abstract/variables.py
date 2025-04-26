from enum import Enum
class RecomputeType(Enum):
    FULL = 1
    SELECTIVE = 2

class WorkloadType(Enum):
    F = "F"
    B = "B"
    W = "W"

class Placement(Enum):
    WAVELIKE = 1
    VSHAPE = 2
    INTERLEAVED = 3
    NAIVE = 4
    RECURRENT = 5
    CROSS = 6
    SEARCHED = 7

class Schedule(Enum):
    GREEDY_v1 = 1
    INTERLEAVED = 2
    ONE_F_ONE_B = 3
    ZBV = 4
    ZBH1 = 5
    BFW = 6
    GREEDY_v2 = 7
    Layerwise = 8
    Chimera = 9
    UnifiedPP = 15

    STANDARD_1F1B = 10
    STANDARD_INTERLEAVED = 11
    STANDARD_ZBH1 = 12
    STANDARD_ZBV = 13
    STANDARD_AFAB = 14

class StageSearchOrder(Enum):
    Random = "Random"
    IncDec = "IncDec"

class RunMode(Enum):
    SEARCH_SCHEDULE = "search_schedule"
    Z3_SOLVE = "z3"
    GUROBI_SOLVE = "gurobi"
    SIM_SOLVE = "sim"
    LAYERWISE_GUROBI_SOLVE = "layer"
    CHIMERA = "chimera"

class WorkloadConstraint:

    def __init__(self, 
                 device_id: int,
                 microbatch_id: int, 
                 stage_id: int, 
                 workload_type: WorkloadType,
                ) -> None:
        self.did = device_id
        self.mid: int = microbatch_id  # 微批次编号
        self.sid: int = stage_id              # 阶段编号
        self.workload_type: WorkloadType = workload_type  # 工作负载类型

    def __eq__(self, other):
        if not isinstance(other, WorkloadConstraint):
            return NotImplemented
        return (
            self.mid == other.mid 
            and self.sid == other.sid
            and self.workload_type == other.workload_type
        )
    
    def __hash__(self):
        return hash((self.mid, self.sid, self.workload_type))
    
    def __repr__(self):
        return (f"did={self.did}, mid={self.mid}, sid={self.sid}, wtype={self.workload_type.name})")