from .mutils import *
from .context import global_context as gpc

class Workload:
    # 定义状态常量
    NOT_STARTED = 1
    IN_PROGRESS = 2
    COMPLETED = 3

    def __init__(self, device_id:int, microbatch_id: int, stage_id: int, duration: int, total_stages:int, wtype: WorkloadType, recomp:bool):
        self.did = device_id
        self.mid: int = microbatch_id  # 微批次编号
        self.sid: int = stage_id              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = Workload.NOT_STARTED      # 初始状态为未开始
        self.ready_time: int = -1
        self.total_stages: int = total_stages
        self.recomp:bool = recomp
        if self.mid == 0 and self.sid == 0:
            self.ready_time = 0
        self.wtype: WorkloadType = wtype  # 工作负载类型
        self.constraints: set = set()               # {(i1, j1, C1), ...}表示Stage i1 上的Microbatch j1 的 C1 操作是前置约束
        self._generate_constraints()

    def _generate_constraints(self):
        if self.wtype == WorkloadType.F:
            if self.sid > 0:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        microbatch_id = self.mid,
                        stage_id = self.sid - 1,
                        workload_type = WorkloadType.F)
                )
        elif self.wtype == WorkloadType.B:
            if self.sid + 1 < self.total_stages:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id = self.sid+1, 
                        microbatch_id= self.mid, 
                        workload_type = WorkloadType.B)
                )
            else:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id=self.total_stages - 1, 
                        microbatch_id=self.mid, 
                        workload_type = WorkloadType.F)
                )
        elif self.wtype == WorkloadType.W:
            self.constraints.add(
                WorkloadConstraint(
                    device_id = self.did,
                    stage_id=self.sid, 
                    microbatch_id=self.mid, 
                    workload_type=WorkloadType.B)
            )

    def _generate_communication(self, time, constraint: WorkloadConstraint):
        if constraint.did != self.did:
            self.ready_time = max(self.ready_time, time + gpc["COMM_TIME"])
        else:
            self.ready_time = max(self.ready_time, time)

    def update_constraints(self, time, constraint: WorkloadConstraint):
        origin_len = len(self.constraints)
        self.constraints.discard(constraint)
        now_len = len(self.constraints)

        if origin_len != now_len:
            self._generate_communication(time, constraint)

    def is_executable(self, time):
        return len(self.constraints) == 0 and self.ready_time <= time and self.state == Workload.NOT_STARTED
    
    def execute(self, time) -> bool:
        if self.state == Workload.NOT_STARTED:
            if self.is_executable(time=time):
                self.state = Workload.IN_PROGRESS
                self.start_time = time
                self.end_time = self.start_time + self.duration
                return True
        return False

    def complete(self, time) -> None:
        """完成任务并更新状态"""
        if self.state == Workload.IN_PROGRESS and self.end_time <= time:
            self.state = Workload.COMPLETED
        
    def is_head_w(self):
        if self.wtype == WorkloadType.F:
            if  gpc["LAYERWISE"] and self.sid == gpc["LAYER_NUM"] + 1:
                return True
        return False

    @property
    def is_w(self):
        return self.wtype == WorkloadType.W
    
    @property
    def is_b(self):
        return self.wtype == WorkloadType.B
    
    @property
    def is_f(self):
        return self.wtype == WorkloadType.F
    
    def __repr__(self):
        return (f"did={self.did}, mid={self.mid}, sid={self.sid}, wtype={self.wtype.name}, "
            f"duration={self.duration}, start_time={self.start_time}, end_time={self.end_time}, state={self.state}, "
            f"ready_time={self.ready_time}, total_stages={self.total_stages}, "
            f"constraints={self.constraints}")
