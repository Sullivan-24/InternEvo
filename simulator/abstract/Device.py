from .Stage import *
from ..utils import save_to_file

def get_required_memory_by_workload_old(workload:Workload):
        stage_id = workload.sid
        layer_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"] // gpc["CHUNK_NUM"]
        workload_type = workload.wtype
        workload_type_num = gpc["WORKLOAD_TYPE_NUM"]
        layer_wise=gpc["LAYERWISE"]
        recomp=workload.recomp
        # wrapper of get required memory
        return get_required_memory(
            stage_id=stage_id,
            layer_num=layer_num,
            workload_type=workload_type,
            workload_type_num=workload_type_num,
            layer_wise=layer_wise,
            recomp=recomp
        )

# TODO rethinking the head memory cost
def get_required_memory(stage_id, layer_num, workload_type, workload_type_num = 3, layer_wise=True, recomp=None):
        if workload_type ==WorkloadType.F:
            required_memory = Activation.FULL * layer_num
            if recomp:
                required_memory = layer_num * (Activation.FULL * (1 - recomp) + Activation.INPUT * recomp)
            if stage_id == gpc["STAGE_NUM"] - 1:
                required_memory += Activation.LOSS
        else:
            if gpc["SPLIT_BACKPROP"]:
                if workload_type == WorkloadType.B:
                    required_memory = (Gradient.INPUT - Activation.FULL * ACT_B_RATIO) * layer_num
                    if recomp:
                        required_memory += (Activation.FULL - Activation.INPUT) * recomp * layer_num
                elif workload_type == WorkloadType.W:
                    required_memory = Gradient.PARAMETER * layer_num #NOTE Memory Spike
                else:
                    raise ValueError(f"get_required_memory: Unsupported workload type {workload_type}!")
            else:
                if workload_type == WorkloadType.B:
                    required_memory = (Gradient.INPUT + Gradient.PARAMETER - Activation.FULL) * layer_num
                    if recomp:
                        required_memory += (Activation.FULL - Activation.INPUT) * recomp * layer_num
                else:
                    raise ValueError(f"get_required_memory: Unsupported workload type {workload_type}!")
        return required_memory

def get_required_memory_old(stage_id, layer_num, workload_type, workload_type_num = 3, layer_wise=True, recomp=None):
        assert workload_type_num == gpc["WORKLOAD_TYPE_NUM"], "Mismatch in number of workload type!"
        if workload_type ==WorkloadType.F:
            required_memory = Activation.FULL * layer_num 
            if recomp:
                required_memory = layer_num * (Activation.FULL * (1 - recomp) + Activation.INPUT * recomp)
            if layer_wise and stage_id == gpc["LAYER_NUM"] + 1:
                required_memory = Activation.LOSS
            elif not layer_wise and stage_id == gpc["STAGE_NUM"] - 1:
                    required_memory += Activation.LOSS
        elif workload_type == WorkloadType.B:
            required_memory = Gradient.INPUT * layer_num
            if recomp:
                required_memory += layer_num * (Activation.FULL - Activation.INPUT) * recomp 
            if workload_type_num == 2:
                required_memory += Gradient.PARAMETER * layer_num
            if layer_wise and stage_id == gpc["LAYER_NUM"] + 1 and gpc["SPLIT_BACKPROP"]:
                required_memory = Gradient.HEAD_INPUT
        elif workload_type == WorkloadType.W:
            assert workload_type_num == 3, "Workload number error!"
            required_memory = Gradient.PARAMETER * layer_num
            if layer_wise and stage_id == gpc["LAYER_NUM"] + 1:
                required_memory = Gradient.HEAD_PARA
        else:
            raise ValueError("get_required_memory: Unsupported workload type!")

        if layer_wise and (stage_id == 0 or stage_id == gpc["LAYER_NUM"] + 2):
            required_memory = 0
        return required_memory

class MemoryMonitor:
    def __init__(self, nmb:int, stages:dict, device_id:int, max_memory:float):
        self.did = device_id
        self.nmb = nmb
        self.stages:dict[int,Stage] = stages
        self.max_memory = max_memory
        self.tracing_workloads:list[Workload] = []
        self.workloads_reserved_mem:list[int] = [0 for _ in range(self.nmb)]
        self.safe_workload_mids = []

    def init_monitor(self):
        self.init_reserved_mem()

    
    def get_required_memory_by_workload(self, workload:Workload):
        # wrapper of get required memory
        return self.stages[workload.sid].update_memory_usage(workload=workload, sim=True)

    def init_reserved_mem(self):
        workload_type = WorkloadType.F
        for sid in self.stages:
            for mid in range(self.nmb):
                workload = self.stages[sid].workloads[mid][workload_type]
                required_mem, _ = self.get_required_memory_by_workload(workload)
                self.workloads_reserved_mem[mid] += required_mem * 1.0
        
        # for sid in self.stages:
        #     for mid in range(self.nmb):
        #         workload = self.stages[sid].workloads[mid][WorkloadType.B]
        #         required_mem = self.get_required_memory_by_workload(workload)
        #         self.workloads_reserved_mem[mid] += required_mem * 1.0
        #     break

    def init_reserved_mem_old(self):
        workload_type = WorkloadType.F
        for mid in range(self.nmb):
            for sid in self.stages:
                workload = self.stages[sid].workloads[mid][workload_type]
                required_mem = self.get_required_memory_by_workload(workload)
                self.workloads_reserved_mem[mid] += required_mem
            # TODO rethinking the head memory cost
            # 应该是是动态变化的，而非一成不变，memory monitor需要保证至少剩余的显存容量足够完成一个mid的所有workload
            # 每次发生memory usage减少的时候都应该重新修改reserved mem的值
            # 并且还要判断显存开销的最高峰，比如loss的显存开销大于 input+para时要以loss为准，反之要以input+para为准
            # self.workloads_reserved_mem[mid] += Gradient.INPUT + Gradient.PARAMETER + Gradient.HEAD_INPUT + Gradient.HEAD_PARA
            self.workloads_reserved_mem[mid] += Gradient.INPUT + Gradient.PARAMETER
            if self.have_head_layer() and Gradient.HEAD_INPUT + Gradient.HEAD_PARA > Gradient.INPUT + Gradient.PARAMETER:
                self.workloads_reserved_mem[mid] -= Gradient.INPUT + Gradient.PARAMETER
                self.workloads_reserved_mem[mid] += Gradient.HEAD_INPUT + Gradient.HEAD_PARA

            # print(f"Device {self.did} Reserve {self.workloads_reserved_mem[mid]}G for mid {mid}")

    def have_head_layer(self):
        if gpc["LAYERWISE"] and (gpc["LAYER_NUM"] + 1 in list(self.stages.keys())):
            return True
        elif not gpc["LAYERWISE"] and (gpc["STAGE_NUM"] - 1 in list(self.stages.keys())):
            return True
        return False
    
    def is_last_w(self, workload:Workload):
        if workload.wtype == WorkloadType.W:
            sorted_sids = sorted(list(self.stages.keys()))
            # Consider the embeding layer
            min_sid_idx = 0 if gpc["LAYERWISE"] and 0 not in sorted_sids else 1
            if workload.sid == sorted_sids[min_sid_idx]:
                return True
        return False
    
    def trace_workload(self,workload:Workload):
        self.tracing_workloads.append(workload)
        required_mem, _ = self.get_required_memory_by_workload(workload)
        self.workloads_reserved_mem[workload.mid] -= required_mem
        if self.workloads_reserved_mem[workload.mid] <= 0:
            self.safe_workload_mids.append(workload.mid)
    
    def is_safe(self, workload:Workload, current_mem:float):
        sid = workload.sid
        mid = workload.mid
        wtype = workload.wtype
        peak_delta, delta = self.stages[sid].update_memory_usage(workload=workload,sim=True)
        return peak_delta + current_mem < self.max_memory or wtype != WorkloadType.F
        # return peak_delta + current_mem < self.max_memory or delta < 0

    def is_executable_workload(self, workload:Workload, current_mem:float):
        required_mem = self.get_required_memory_by_workload(workload)
        safe_count = 0
        if workload.mid in self.safe_workload_mids:
            return True
        self.workloads_reserved_mem[workload.mid] -= required_mem
        for mid in range(self.nmb):
            if mid in self.safe_workload_mids:
                continue
            if self.workloads_reserved_mem[mid] + required_mem + current_mem <= self.max_memory:
                safe_count += 1
        # TODO what about F and B? in SPLIT or not?
        # memory friendly and critical
        if workload.is_w and required_mem + current_mem <= self.max_memory:
            safe_count += 1
        self.workloads_reserved_mem[workload.mid] += required_mem
        return safe_count > 0

class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int, max_activation_counts: int, nmb:int, static_schedule: list = None, memory_usage_constrain_rate: float = 1, max_mem: int = gpc["GPU_MAX_MEM"], comp_power: float = 1, layer_density: list = None):
        self.did = device_id
        self.warmup_end_flag = False
        self.begin_warmup_num = (gpc["CHUNK_NUM"] - 1) * gpc["PP_SIZE"] + 1 + gpc["PP_SIZE"] - 1 - self.did
        self.steady_start_flag = False
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.proc_workload: Workload = None
        self.current_mem_usage: int = 0
        self.peak_memory_usage: int = 0
        self.nmb: int = nmb
        self.max_activation_counts: int = max_activation_counts
        self.mem_usage_record: dict[int, int] = {}
        self.peak_mem_usage_record: dict[int, int] = {}
        self.static_schedule: list[str] = static_schedule
        self.next_workload_idx: int = 0
        self.workload_type_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
        self.last_workload_type = None
        self.total_layers = 0
        self.mid_traverse_order:list[int] = list(range(0,self.nmb))
        self.warmup_num_f = 0
        self.warmup_num_b = 0
        self.warmup_num_w = 0
        self.exe_num_f = 0
        self.exe_num_b = 0
        self.exe_num_w = 0
        self.overlap_flag = False
        self.order_balance = True
        self.next_workload_type = None
        self.memory_usage_constrain_rate = memory_usage_constrain_rate
        self.executable_workloads:list[Workload] = []

        self.workload_type_traverse_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
        self.next_mid = 0
        self.released_workloads = []
        self.available_f_num = 0
        self.executing_mid_idx = 0
        self.ok_flag = False

        self.memory_monitor = None
        self.wait_for_schedule = 0

        self.situations = 1
        self.mid_priority = [3 * gpc["CHUNK_NUM"] for _ in range(self.nmb)]
        self.max_memory = max_mem
        self.comp_power = comp_power
        if layer_density is None:
            self.layer_density = [1 for _ in range(gpc["LAYER_NUM"])]
        else:
            self.layer_density = layer_density

        self.workload_execute_record: list[list[Workload]] = [[] for _ in range(gpc["DEVICE_NUM"])]

    def get_required_memory_by_workload(self, workload:Workload):
        # wrapper of get required memory
        return self.stages[workload.sid].update_memory_usage(workload=workload, sim=True)
    
    def init_memory_monitor(self):
        self.memory_monitor = MemoryMonitor(self.nmb, self.stages, self.did, max_memory=self.max_memory * gpc["MEMORY_CONSTRAIN"])
        self.memory_monitor.init_monitor()

    def get_executable_workload(self, time)->list[Workload]:
        executable_workoads = []
        workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        memory_constrain = MEMORY_CONSTRAIN * self.max_memory
        if self.is_bottleneck_device():
            memory_constrain -= MEMORY_REDUCATION

        if self.current_mem_usage <= memory_constrain:
            if self.last_workload_type == WorkloadType.B:
                workload_type_order = [WorkloadType.F,WorkloadType.W,WorkloadType.B]
            elif self.last_workload_type == WorkloadType.F:
                workload_type_order = [WorkloadType.B,WorkloadType.W,WorkloadType.F]
            elif self.last_workload_type == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        else:
            if self.last_workload_type == WorkloadType.B:
                workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
            elif self.last_workload_type == WorkloadType.F:
                workload_type_order = [WorkloadType.W,WorkloadType.B,WorkloadType.F]
            elif self.last_workload_type == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]

        if self.is_bottleneck_device():
            if self.current_mem_usage <= memory_constrain:
                if self.last_workload_type == WorkloadType.B:
                    workload_type_order = [WorkloadType.F,WorkloadType.B]
                elif self.last_workload_type == WorkloadType.F:
                    workload_type_order = [WorkloadType.B,WorkloadType.F]
                if self.exe_num_f == gpc["CHUNK_NUM"] * gpc["MICRO_BATCH_NUM"]:
                    workload_type_order = [WorkloadType.B, WorkloadType.W]
            else:
                if self.last_workload_type == WorkloadType.B:
                    workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
                elif self.last_workload_type == WorkloadType.F:
                    workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
                elif self.last_workload_type == WorkloadType.W:
                    workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]

        # deal with long tail, advance more B when memory is sufficient
        # if self.exe_num_f >= gpc["CHUNK_NUM"] * (gpc["MICRO_BATCH_NUM"] - 0) - self.begin_warmup_num:
        #     workload_type_order = [WorkloadType.F, WorkloadType.B,WorkloadType.W]
        f_b_diff = self.exe_num_f - self.exe_num_b
        f_w_diff = self.exe_num_f - self.exe_num_w
        if f_b_diff >= self.begin_warmup_num or f_w_diff >= self.begin_warmup_num:
            if f_b_diff >= f_w_diff:
                workload_type_order = [WorkloadType.B, WorkloadType.W,WorkloadType.F]
            else:
                workload_type_order = [WorkloadType.W, WorkloadType.B,WorkloadType.F]
        else:
            workload_type_order = [WorkloadType.F, WorkloadType.B,WorkloadType.W]

        if self.exe_num_f >= (gpc["CHUNK_NUM"]) * (gpc["MICRO_BATCH_NUM"] - 0):
            workload_type_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]

        # if self.exe_num_f >= (gpc["CHUNK_NUM"]) * (gpc["MICRO_BATCH_NUM"] - 16):
        #     if self.last_workload_type != WorkloadType.F:
        #         workload_type_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
        #     else:
        #         workload_type_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]

        # raise priority of head and ce
        if gpc["LAYERWISE"]:
            head_ce_workloads = []
            for workload_type in [WorkloadType.W, WorkloadType.B,WorkloadType.F]:
                for mid in range(self.nmb):
                    for stage_id in self.stages:
                        if stage_id > gpc["LAYER_NUM"] and gpc["LAYERWISE"]:
                            workloads = self.stages[stage_id].workloads
                            if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable(time=time):
                                head_ce_workloads.append(workloads[mid][workload_type])
            # ensure head to be executed as quickly as possible
            executable_workoads += head_ce_workloads
            if len(executable_workoads) > 0:
                if self.current_mem_usage + Activation.LOSS >= self.max_memory:
                    workload_type_order = [WorkloadType.W,WorkloadType.B,WorkloadType.F]
        else:
            head_ce_workloads = []
            for mid in range(self.nmb):
                for workload_type in [WorkloadType.W, WorkloadType.B]:
                    for stage_id in self.stages:
                        # if stage_id == gpc["STAGE_NUM"] - 1:
                        if stage_id == gpc["STAGE_NUM"] - 1 or stage_id == 0:
                            workloads = self.stages[stage_id].workloads
                            if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable(time=time):
                                head_ce_workloads.append(workloads[mid][workload_type])                          
            # ensure head to be executed as quickly as possible
            executable_workoads += head_ce_workloads

        delayed_workload = []
        canceled_workload = []
        for workload_type in workload_type_order:
            for mid in range(self.nmb):
                for stage_id in self.stages:
                    if stage_id > gpc["LAYER_NUM"] and gpc["LAYERWISE"]:
                        continue
                    workloads = self.stages[stage_id].workloads
                    if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable(time=time):
                        workload = workloads[mid][workload_type]
                        if gpc["OVERLAP_AWARE_SCHEDULE"] and self.exe_num_b > 0 and self.should_delay_for_overlap(time=time, workload=workload):
                            if self.is_bottleneck_device():
                                delayed_workload.append(workloads[mid][workload_type])
                            else:
                                canceled_workload.append(workloads[mid][workload_type])
                        else:
                            executable_workoads.append(workloads[mid][workload_type])

        executable_workoads = executable_workoads + delayed_workload
        # if len(executable_workoads) == 0:
        #     executable_workoads += canceled_workload

        return executable_workoads

    def is_bottleneck_device(self):
        return gpc["STAGE_NUM"] - 1 in list(self.stages.keys())

    def should_delay_for_overlap(self, time, workload:Workload):
        for did,executed_workloads in enumerate(self.workload_execute_record):
            if did == self.did or len(executed_workloads) == 0:
                continue
            pivot_workload = executed_workloads[-1]
            if pivot_workload.mid == workload.mid: # Only micro-batches with same the mid have dependency
                if pivot_workload.sid == workload.sid - 1 and pivot_workload.wtype == workload.wtype == WorkloadType.F:
                    if pivot_workload.end_time <= self.workload_execute_record[self.did][-1].start_time:
                        continue
                    return True
                if pivot_workload.sid == workload.sid + 1 and pivot_workload.wtype == workload.wtype == WorkloadType.B:
                    if pivot_workload.end_time <= self.workload_execute_record[self.did][-1].start_time:
                        continue
                    return True
        return False
    
    def should_delay_for_overlap_old(self, time, workload:Workload):
        for did,executed_workloads in enumerate(self.workload_execute_record):
            if did == self.did or len(executed_workloads) == 0:
                continue
            pivot_workload = executed_workloads[-1]
            if pivot_workload.sid == workload.sid - 1 and pivot_workload.wtype == workload.wtype == WorkloadType.F:
                # print(f"Delay workload ({workload.did},{workload.sid},{workload.mid},{workload.wtype}) due to ({pivot_workload.did},{pivot_workload.sid},{pivot_workload.mid},{pivot_workload.wtype})")
                if not gpc["OVERLAP_DEGREE"]:
                    return True
                if gpc["OVERLAP_DEGREE"] and pivot_workload.end_time + pivot_workload.duration // gpc["OVERLAP_DEGREE"] >= time:
                    return True
            if pivot_workload.sid == workload.sid + 1 and pivot_workload.wtype == workload.wtype == WorkloadType.B:
                # print(f"Delay workload ({workload.did},{workload.sid},{workload.mid},{workload.wtype}) due to ({pivot_workload.did},{pivot_workload.sid},{pivot_workload.mid},{pivot_workload.wtype})")
                if not gpc["OVERLAP_DEGREE"]:
                    return True
                if gpc["OVERLAP_DEGREE"] and pivot_workload.end_time + pivot_workload.duration // gpc["OVERLAP_DEGREE"] >= time:
                    return True
        return False



    def overlap_aware_executable_workload_reorder(self, workload:Workload):
        if workload:
            if workload.wtype ==  WorkloadType.F:
                next_stage_id = workload.sid + 1
            elif workload.wtype == WorkloadType.B:
                next_stage_id = workload.sid - 1
            else:
                return
            
            if next_stage_id in self.stages.keys():
                head = []
                tail = []
                for wl in self.executable_workloads:
                    if wl.mid == workload.mid and wl.wtype == workload.wtype and wl.sid == next_stage_id:
                        tail.append(wl)
                    else:
                        head.append(wl)
                self.executable_workloads = head + tail
                # if head+tail != self.executable_workloads:
                #     print(head+tail)
                #     print(self.executable_workloads)
                #     input()

    def show_stages(self, detail_info=False):
        for sid in self.stages:
            # print("Stage {} recomp={} on Device {}".format(sid, self.stages[sid].recomp, self.device_id))
            if detail_info:
                for mid in self.stages[sid].workloads:
                    if mid == 10:
                        for wlt in self.stages[sid].workloads[mid]:
                            print(self.stages[sid].workloads[mid][wlt])

    def add_stage(self, stage_id: int, layer_num:int,
                  recomp:bool = False, 
                  layerwise:bool = gpc["LAYERWISE"],
                  basic_memory = 0) -> None:
        stage_type = StageType.LAYERS
        para_num = 0
        if layerwise:
            assert layer_num == 1 and gpc["CHUNK_NUM"] == gpc["LAYER_NUM"] // gpc["PP_SIZE"], f"LAYERWISE require 1 layer per stage (CHUNK_NUM == LAYER_NUM // PP_SIZE) but got {layer_num} per stage"
            if stage_id == 0:
                stage_type = StageType.EMBD
                basic_memory = StateMemory.EMB
                para_num = Parameter.EMB
            elif stage_id == gpc["LAYER_NUM"] + 1:
                stage_type = StageType.HEAD
                basic_memory = StateMemory.HEAD
                para_num = Parameter.HEAD
            elif stage_id == gpc["LAYER_NUM"] + 2:
                stage_type = StageType.CE
            else:
                stage_type = StageType.LAYER
                basic_memory = StateMemory.LAYER
                para_num = Parameter.LAYER
        else:
            basic_memory = StateMemory.LAYER * layer_num
            para_num = Parameter.LAYER * layer_num
            if stage_id == 0:
                basic_memory += StateMemory.EMB
                para_num += Parameter.EMB
            elif stage_id == gpc["STAGE_NUM"] - 1:
                basic_memory += StateMemory.HEAD
                para_num += Parameter.HEAD
        stage = Stage(
                device_id=self.did, 
                stage_id=stage_id,
                para_num=para_num,
                stage_type=stage_type,
                microbatch_num=self.nmb,
                recomp=recomp,
                layerwise=layerwise,
                layer_num=layer_num,
                comp_power=self.comp_power,
                layer_density=self.layer_density,
            )
        self.stages[stage.sid] = stage
        self.total_layers+=layer_num

    def count_wtype_num(self, did : int, wtype : WorkloadType):
        count = 0
        for w in self.workload_execute_record[did]:
            count += 1 if w.wtype == wtype else 0
        return count

    def update_constraints(self, time, constraint):
        for sid in self.stages:
            self.stages[sid].update_constraints(time, constraint=constraint)
    
    def update_mid_traverse_order(self,mid=None):
        if type(self.mid_traverse_order) is not list:
            self.mid_traverse_order = list(self.mid_traverse_order)
        self.mid_traverse_order.sort()
        if mid:
            self.mid_traverse_order.remove(mid)
            self.mid_traverse_order.append(mid)
    
    def get_finished_workload_num_by_type(self, wtype:WorkloadType):
        workload_num = 0
        for workloads in self.workload_execute_record:
            for workload in workloads:
                if workload.wtype == wtype:
                    workload_num += 1
        return workload_num

    def get_executable_workload_num_by_type(self, wtype:WorkloadType):
        workload_num = 0
        for workload in self.executable_workloads:
            if workload.wtype == wtype:
                workload_num += 1
        return workload_num

    def execute_workload(self, time, run_schedule=False) -> None:
        assert time >= 0, f"Time should be non-negative (but got {time})."
        if self.state == Device.IDLE:
            if gpc["SCHEDULE_METHOD"] == Schedule.UnifiedPP:
                self.executable_workloads = self.get_executable_workload(time=time)
                save_to_file(f"schedule_results/workload_statistics/device{self.did}.txt",f"{time},{len(self.executable_workloads)}\n", 'a')
                
                if self.get_executable_workload_num_by_type(wtype=WorkloadType.B):
                    self.warmup_end_flag = True

                for workload in self.executable_workloads:
                    workload_type = workload.wtype
                    sid = workload.sid
                    mid = workload.mid
                    did = workload.did
                    
                    if self.exe_num_f < self.begin_warmup_num:
                        if workload_type != WorkloadType.F:
                            continue
                    else:
                        if self.get_finished_workload_num_by_type(wtype=WorkloadType.B) == 0:
                            if workload.sid != gpc["STAGE_NUM"] - 1:
                                continue
                        else:
                            # Strict type order constrain
                            if not self.warmup_end_flag:
                                continue
                            else:
                                if not self.steady_start_flag and workload.wtype != WorkloadType.F:
                                    continue
                                else:
                                    if not self.is_bottleneck_device():
                                        if self.steady_start_flag:
                                            if self.exe_num_f - self.exe_num_b >= self.begin_warmup_num + 1 and self.last_workload_type == workload.wtype:
                                                continue

                    # if not self.memory_monitor.is_executable_workload(workload=workload, current_mem=self.current_mem_usage):
                    #     continue
                    if self.steady_start_flag:
                        if not self.memory_monitor.is_safe(workload=workload, current_mem=self.current_mem_usage+MEMORY_REDUCATION):
                            continue
                    
                    proc_workload = self.stages[sid].execute_workload(time, mid=mid,workload_type=workload_type)
                    if proc_workload:
                        self.last_workload_type = workload_type
                        if workload_type == WorkloadType.F:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_f += 1
                                if self.warmup_end_flag:
                                    self.steady_start_flag = True
                        elif workload_type == WorkloadType.B:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_b += 1
                        elif workload_type == WorkloadType.W:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_w += 1
                        else:
                            raise Exception("Error workload type.")
                        
                        self.mid_priority[proc_workload.mid] -= 1
                        self.proc_workload = proc_workload
                        self.update_memory_usage()
                        self.state = Device.BUSY
                        self.memory_monitor.trace_workload(workload=workload)
                        # if self.did == 0:
                        #     print(self.memory_monitor.workloads_reserved_mem)
                        return proc_workload
            elif gpc["SCHEDULE_METHOD"] == Schedule.STANDARD_INTERLEAVED:
                if self.next_workload_idx == len(self.stages) * self.nmb * gpc["WORKLOAD_TYPE_NUM"]:
                    return None
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx]
                
                if self.did < gpc["PP_SIZE"] - 1:
                    if self.exe_num_f > (gpc["CHUNK_NUM"] - 1) * gpc["PP_SIZE"] + (gpc["PP_SIZE"] - self.did - 1) * 2 - 1 and self.exe_num_f < gpc["CHUNK_NUM"] * gpc["MICRO_BATCH_NUM"]:
                        if workload_type == WorkloadType.F and self.count_wtype_num(self.did, WorkloadType.B) < self.count_wtype_num(self.did + 1, WorkloadType.B) and self.workload_execute_record[self.did + 1][-1].wtype == WorkloadType.B:
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        elif workload_type in (WorkloadType.B, WorkloadType.W):
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        else:
                            return None
                    else:
                        proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                else:
                    proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                
                if proc_workload:
                    self.proc_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.last_workload_type = workload_type
                    if workload_type == WorkloadType.F:
                        self.exe_num_f += 1
                    elif workload_type == WorkloadType.B:
                            self.exe_num_b += 1
                    elif workload_type == WorkloadType.W:
                            self.exe_num_w += 1
                    else:
                        raise Exception("Error workload type.")
                    self.next_workload_idx += 1
                    return proc_workload
            elif run_schedule or gpc["SCHEDULE_METHOD"] in (Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB, Schedule.STANDARD_ZBH1):
                if self.next_workload_idx == len(self.stages) * self.nmb * gpc["WORKLOAD_TYPE_NUM"]:
                    return None
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                try:
                    (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx][:3]
                except Exception as e:
                    print((workload_type, workload_mid, workload_sid))
                    input()
                proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                if proc_workload:
                    self.proc_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.next_workload_idx += 1
                    return proc_workload
            elif gpc["SCHEDULE_METHOD"] in (Schedule.STANDARD_ZBV, Schedule.ZBV):
                if self.last_workload_type == WorkloadType.F:
                    workload_type = WorkloadType.B
                elif self.last_workload_type == WorkloadType.B:
                    workload_type = WorkloadType.W
                elif self.last_workload_type == WorkloadType.W:
                    workload_type = WorkloadType.F
                else:
                    workload_type = WorkloadType.F

                if self.warmup_num_f < gpc["DEVICE_NUM"] * 2:
                    workload_type = WorkloadType.F
                if self.warmup_num_f == gpc["MICRO_BATCH_NUM"] * 2:
                    workload_type = WorkloadType.B
                if self.warmup_num_b == gpc["MICRO_BATCH_NUM"] * 2:
                    workload_type = WorkloadType.W

                for mid in range(gpc["MICRO_BATCH_NUM"]):
                    for sid in self.stages:
                        required_memory = get_required_memory_old(
                            stage_id=sid, 
                            layer_num=gpc["LAYER_NUM"]//gpc["STAGE_NUM"],
                            workload_type=workload_type,
                            workload_type_num=gpc["WORKLOAD_TYPE_NUM"], 
                            layer_wise=True,
                            recomp=self.stages[sid].recomp,
                        )

                        workload_type = self._reset_workload_type(
                            workload_type=workload_type,
                            required_memory=required_memory,
                            current_mem_usage=self.current_mem_usage,
                            max_memory=self.max_memory,
                        )

                        proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
                        if proc_workload:
                            if workload_type == WorkloadType.F:
                                self.warmup_num_f += 1
                            elif workload_type == WorkloadType.B:
                                self.warmup_num_b += 1
                            elif workload_type == WorkloadType.W:
                                self.warmup_num_w += 1
                            self.last_workload_type = workload_type
                            self.proc_workload = proc_workload
                            self.update_memory_usage()
                            self.state = Device.BUSY
                            return proc_workload
                
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                for workload_type in now_workload_priority_order:
                    for mid in range(gpc["MICRO_BATCH_NUM"]):
                        for sid in self.stages:
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
                            if proc_workload:
                                if workload_type == WorkloadType.F:
                                    self.warmup_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    self.warmup_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    self.warmup_num_w += 1
                                self.last_workload_type = workload_type
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload             
            else:
                print("Schedule Not Supported")
        return None

    def _reset_workload_type(self, workload_type, required_memory, current_mem_usage, max_memory):
        #TODO 不同个Wave情况下，处于Wave不同边上的memory预留的值应该不同
        if current_mem_usage + required_memory >= max_memory - Gradient.INPUT - Gradient.PARAMETER:
            workload_type = WorkloadType.B
            if not gpc["SPLIT_BACKPROP"]:
                return workload_type
        if current_mem_usage + required_memory >= max_memory - Gradient.PARAMETER:
            workload_type = WorkloadType.W
        return workload_type


    def _finish_proc_workload(self,time) -> bool: 
        if self.state == Device.BUSY:
            if self.proc_workload and time >= self.proc_workload.end_time:
                return True
        return False

    def update_memory_usage(self) -> int:
        if self.proc_workload.state == Workload.IN_PROGRESS and self.proc_workload.wtype in (WorkloadType.F, WorkloadType.B):
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
        elif self.proc_workload.state == Workload.COMPLETED and self.proc_workload.wtype == WorkloadType.W:
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
        self.peak_memory_usage = sum(stage.peak_memory_usage for stage in self.stages.values())    
        self.current_mem_usage = sum(stage.memory_usage for stage in self.stages.values())
        self.mem_usage_record[(self.proc_workload.start_time,self.proc_workload.end_time)] = self.current_mem_usage
        self.peak_mem_usage_record[(self.proc_workload.start_time,self.proc_workload.end_time)] = self.peak_memory_usage
        for stage in self.stages.values(): # recover peak memory usage to current memory usage
            if stage.sid != self.proc_workload.sid:
                stage.peak_memory_usage = stage.memory_usage

    def get_memory_usage(self) -> int:
        return self.current_mem_usage

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}),state={self.state}"