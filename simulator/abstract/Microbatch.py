from .Workload import *
class MicrobatchWorkloadRecorder:
    def __init__(self, recorder_id: int):
        self.recorder_id: int = recorder_id
        self.workloads_constraints: list = list()
        self._generate_recorder_matrix()

    def _generate_constraints(self, sid, mid):
        if self.workload_type == WorkloadType.F:
            self.constraints.add((self.stage_id-1, self.microbatch_id, WorkloadType.F))
        elif self.workload_type == WorkloadType.B:
            self.constraints.add((self.stage_id+1, self.microbatch_id, WorkloadType.B))
        elif self.workload_type == WorkloadType.W:
            self.constraints.add((self.stage_id, self.microbatch_id, WorkloadType.B))

    def _generate_recorder_matrix(self):
        self.workloads_constraints = [
            [
                [
                    (self.stage_id-1, self.microbatch_id, WorkloadType.F),
                    (self.stage_id+1, self.microbatch_id, WorkloadType.B),
                    (self.stage_id, self.microbatch_id, WorkloadType.B)
                ] for mid in range(MICRO_BATCH_NUM)
            ] for sid in range(STAGE_NUM)
        ]