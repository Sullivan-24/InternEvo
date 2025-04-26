from .Device import Device, Schedule, get_required_memory
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..ChimeraLayerwisePainter import LayerwiseSchedulingPainter as LSP
from ..chimera_utils import print_to_file

class ChimeraPipelineScheduler:

    def __init__(self) -> None:
        self.chimera_stream_num = 2
        self.device_num = DEVICE_NUM * self.chimera_stream_num
        self.layer_num = LAYER_NUM
        self.stage_num = STAGE_NUM
        self.emb_head_ce = True
        self.results = {}
        self.dsa = [[] for _ in range(self.chimera_stream_num)]
        self.devices: list[Device] = []
        self.microbatch_schedule_range = range(0,min(MICRO_BATCH_NUM//1, MICRO_BATCH_NUM))
        self.num_finished_microbatch = 0
        self._init_stage()
        self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)
        self.schedule = [[] for _ in range(DEVICE_NUM)]
        self.set_schedule()
    
    # NOTE _reset_workload_type is efficient but 
    # lead to random order of W in some cases
    # which will break solver constraint (not affect the correctness)
    def resort_w(self):
        w_times = [[] for _ in range(LAYER_NUM + 3)]
        for res in self.results:
            if res.startswith("w"):
                w,mid,sid = res.split("_")
                sid = int(sid)
                w_times[sid].append(self.results[res])

        for sid in range(LAYER_NUM + 3):
            w_times_in_sid = sorted(w_times[sid])
            for mid in range(len(w_times_in_sid)):
                w_key = f"w_{mid}_{sid}"
                self.results[w_key] = w_times_in_sid[mid]

    def show_detail_info(self):
        for device in self.devices:
            print("Device ID:{}".format(device.did))
            if device.did == 7:
                device.show_stages(detail_info=True)

    def set_microbatch_schedule_range(self, microbatch_schedule_range):
        for device in self.devices:
            device.mid_traverse_order = microbatch_schedule_range

    def set_layer_recomp(self, settings:list):
        for idx, r in enumerate(settings):
            self.recomp_set[idx] = True if r else False
            self.results[f"theta_{idx}"] = r

    def _init_stage(self):
        for did in range(self.device_num):
            device = Device(device_id = did, 
                            max_activation_counts=MAX_ACTIVATION_COUNTS, 
                            nmb=MICRO_BATCH_NUM,
                            )
            self.devices.append(device)
        real_device_num = self.device_num // self.chimera_stream_num
        increment = self.layer_num // real_device_num
        for i in range(real_device_num):
            start = i * increment
            end = start + increment
            for lid in list(range(max(0, start), min(end, self.layer_num))):
                self.devices[i].add_stage(lid + 1, layerwise=self.emb_head_ce)
                self.devices[self.device_num - 1 - i].add_stage(lid + 1, layerwise=self.emb_head_ce)

        self.devices[real_device_num // 2 - 1].add_stage(0, layerwise=self.emb_head_ce)
        self.devices[0].add_stage(self.layer_num+1, layerwise=self.emb_head_ce)
        self.devices[1].add_stage(self.layer_num+2, layerwise=self.emb_head_ce)

        self.devices[real_device_num + real_device_num // 2].add_stage(0, layerwise=self.emb_head_ce)
        self.devices[-1].add_stage(self.layer_num+1, layerwise=self.emb_head_ce)
        self.devices[-2].add_stage(self.layer_num+2, layerwise=self.emb_head_ce)

        for did in range(self.device_num):
            self.dsa[did // DEVICE_NUM].append(self.devices[did].stages.keys())

        for did in range(self.device_num):
            self.devices[did].show_stages()

    def set_schedule(self):
        for did in range(DEVICE_NUM):
            self.devices[did].static_schedule = self.schedule[did]

    def show_record(self):
        for k in self.results:
            print(k, self.results[k])

    def update_constraints(self, constraint:Workload):
        for device in self.devices:
            # NOTE only update constraints within the same Chimera stream
            if constraint.did // DEVICE_NUM == device.did // DEVICE_NUM:
                device.update_constraints(constraint=constraint)
        
    def record_workload(self, workload: Workload):
        if workload:
            print("did={}({}), {}, sid={}, mid={}, s={}, e={}".format(
                workload.did%DEVICE_NUM, workload.did, 
                workload.wtype.value, workload.sid, workload.mid,
                workload.start_time, workload.end_time,
                )
            )
            input()
            wlt = workload.wtype.value.lower()
            mid = workload.mid
            sid = workload.sid
            # NOTE new key for Chimera results
            did = workload.did
            k = 'cwi_{}_{}_{}_{}'.format(did//DEVICE_NUM,wlt,mid,sid)

            self.results[k] = workload.start_time

    def check_workload_status(self):
        for device in self.devices:
            if device._finish_proc_workload():
                if device.proc_workload.wtype == WorkloadType.W:
                    self.num_finished_microbatch += 1
                device.proc_workload.complete()
                self.update_constraints(constraint=device.proc_workload)
                device.update_memory_usage()
                device.state = Device.IDLE
                # 0 1 2 3      4 5 6 7
                real_device_num = self.device_num // self.chimera_stream_num
                self.devices[(device.did + real_device_num) % self.device_num].state = Device.IDLE

        # if self.num_finished_microbatch == (1 + LAYER_NUM) * len(self.microbatch_schedule_range):
        #     self.num_finished_microbatch = 0
        #     self.microbatch_schedule_range = [n + len(self.microbatch_schedule_range) for n in self.microbatch_schedule_range if n + len(self.microbatch_schedule_range) < MICRO_BATCH_NUM]
        #     self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)

    def chimera_execute_workload(self, device: Device):
        if device.state == Device.IDLE:
            now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
            if device.last_workload_type == WorkloadType.B:
                now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
            elif device.last_workload_type == WorkloadType.F:
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
            
            for workload_type in now_workload_priority_order:
                for mid in device.mid_traverse_order:
                    for sid in device.stages:
                        required_memory = get_required_memory(
                            stage_id=sid, 
                            layer_num=1,
                            workload_type=workload_type,
                            workload_type_num=WORKLOAD_TYPE_NUM, 
                            layer_wise=True,
                            recomp=device.stages[sid].recomp,
                        )

                        workload_type = device._reset_workload_type(
                            workload_type=workload_type,
                            required_memory=required_memory,
                            current_mem_usage=device.current_mem_usage,
                            max_memory=GPU_MAX_MEM,
                        )

                        proc_workload = device.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                        if proc_workload:
                            device.last_workload_type = workload_type
                            device.proc_workload = proc_workload
                            device.update_memory_usage()
                            device.state = Device.BUSY
                            real_device_num = self.device_num // self.chimera_stream_num
                            self.devices[(device.did + real_device_num) % self.device_num].state = Device.BUSY
                            # for d in self.devices:
                            #     print(d)
                            # input()
                            return proc_workload
        return None

    def execute_workload(self):
        for device in self.devices:
            processing_workload = self.chimera_execute_workload(device)
            self.record_workload(processing_workload)
            
    def run_pipeline_parallelism(self, time_limit = 10000):
        while GET_TIME() <= 1000:
            self.check_workload_status()
            self.execute_workload()             
            UPDATE_TIME()

    def show_mem_usage(self, device_id=(0,)):
        max_mem_usage = -1
        for device in self.devices:
            if device.did in device_id:
                print("Device {} mem usage:".format(device.did))
                last_mem_record = 0
                for t, mem_record in device.mem_usage_record.items():
                    print("Time {}, mem = {}, {}.".format(t, round(mem_record,2), round((mem_record - last_mem_record), 2)))
                    last_mem_record = mem_record
                    max_mem_usage = max(max_mem_usage, mem_record)
        if max_mem_usage > GPU_MAX_MEM and LAYERWISE:
            raise ValueError("Error: Out of Memory.")
    
    def get_workloadload_duration(self):
        fwd_time = [[F_TIME for _ in range(LAYER_NUM+3)],[F_TIME for _ in range(LAYER_NUM+3)]]
        iwd_time = [[B_TIME for _ in range(LAYER_NUM+3)],[B_TIME for _ in range(LAYER_NUM+3)]]
        pwd_time = [[W_TIME for _ in range(LAYER_NUM+3)],[W_TIME for _ in range(LAYER_NUM+3)]]
        for device in self.devices:
            for sid in device.stages:
                for mid in range(MICRO_BATCH_NUM):
                    fwd_time[device.did // DEVICE_NUM][sid] = device.stages[sid].workloads[mid][WorkloadType.F].duration
                    if WorkloadType.B in device.stages[sid].workloads[mid]:
                        iwd_time[device.did // DEVICE_NUM][sid] = device.stages[sid].workloads[mid][WorkloadType.B].duration
                    if WorkloadType.W in device.stages[sid].workloads[mid]:
                        pwd_time[device.did // DEVICE_NUM][sid] = device.stages[sid].workloads[mid][WorkloadType.W].duration
        return fwd_time, iwd_time, pwd_time
    
    def write_fbw_to_file(self):
        for key in self.results:
            if key.startswith(("f_","b_","w_")):
                print_to_file(f"sim_chimera_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key},{self.results[key]}\n")

    def draw(self) -> None:
        # 绘制结果的逻辑
        self.resort_w()
        # self.write_fbw_to_file()
        fwd_time, iwd_time, pwd_time = self.get_workloadload_duration()
        painter_conf = {
            "devices": self.dsa,
            "emb_head_ce": self.emb_head_ce,
            "device_num": self.device_num,
            "num_layer": LAYER_NUM+3,
            "stage_num": LAYER_NUM+3,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": PIXEL_BASE,
            "nmb": MICRO_BATCH_NUM,
            "forward_length": fwd_time,
            "backward_length": iwd_time,
            "backward_length2": pwd_time,
            "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
        }
        LSP(painter_conf).draw(self.results)