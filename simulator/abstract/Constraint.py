from .mutils import *
class Constraint:
    def __init__(self, stage_id: int, microbatch_id: int, workload_type: WorkloadType) -> None:
        self.stage_id = stage_id
        self.microbatch_id = microbatch_id
        self.workload_type = workload_type
    