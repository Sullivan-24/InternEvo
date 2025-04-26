import time
from gurobipy import Model, GRB, quicksum
from .painter import SchedulingPainter
from .abstract.mutils import *
from .utils import resort_microbatch_index, print_to_file
from .abstract import Pipeline
from .abstract.Device import get_required_memory

workload_type_mapping = {
    'f':WorkloadType.F,
    'b':WorkloadType.B,
    'w':WorkloadType.W,
}

class GSimulator:

    def __init__(self, config: dict, device_stage_alignments=None, new_comm_length=None) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["stage_num"]
        self._device_size = config["device_num"]
        self._model_layer_num = config["layer_num"]
        self._num_microbatches = config["nmb"]
        self._max_activation_counts = config["max_activation_counts"]
        self._num_device = config["device_num"]

        self.estimated_time_cost = self.estimate_time_cost()
        self.M = self.set_M_value(self.estimated_time_cost)

        self._mix_training = config["mix_training"]
        self._model_para_num = config["para_num"]
        self._device_mem = config["device_mem"]
        # obtained by profiling
        self._profiled_layer_f_length = config["f_time"]
        self._profiled_layer_b_length = config["b_time"]
        self._profiled_layer_w_length = config["w_time"]
        self._comm_length = config["comm_time"] if not new_comm_length else new_comm_length
        # 检查输入参数
        assert isinstance(self._profiled_layer_f_length, (list, tuple))
        assert isinstance(self._profiled_layer_b_length, (list, tuple))
        assert isinstance(self._profiled_layer_w_length, (list, tuple))

        # 创建 Gurobi 模型
        self.model = Model("SPSimulator")

        self.minimal_time_with_sync_update = (DEVICE_NUM - 1) * (F_TIME // CHUNK_NUM + COMM_TIME) + (F_TIME + B_TIME + W_TIME) * MICRO_BATCH_NUM
        print("MINIMAL TIME WITH SYNC UPDATE:{}".format(self.minimal_time_with_sync_update))

        # 变量初始化
        self._stage_f_length  = []
        self._stage_b_length = []
        self._stage_w_length = []

        self._stage_recomp_rate = []

        self._stage_f_offsets = [[] for _ in range(self._pp_size)]
        self._stage_b_offsets = [[] for _ in range(self._pp_size)]
        self._stage_w_offsets = [[] for _ in range(self._pp_size)]

        if device_stage_alignments:
            self._devices = device_stage_alignments
        else:
            self._devices = [[] for _ in range(self._device_size)]
            self._fix_stages()

        # baseline solution
        if self._base_solution:
            self.pipeline_scheduler = Pipeline.PipelineScheduler(placement=self._devices)
            self.pipeline_scheduler.run_pipeline_parallelism()
            print(self.pipeline_scheduler.results)
            # self.pipeline_scheduler.draw()
        self.model_result = None

    def estimate_time_cost(self):
        fbw_time = (MICRO_BATCH_NUM + DEVICE_NUM) * (F_TIME + B_TIME + W_TIME)
        emb_time = EMB_F_TIME
        head_time = MICRO_BATCH_NUM * (HEAD_F_TIME + HEAD_B_TIME + HEAD_W_TIME)
        ce_time = MICRO_BATCH_NUM * (CE_F_TIME + CE_B_TIME + CE_W_TIME)
        comm_time = COMM_TIME
        res = fbw_time + emb_time + head_time + ce_time + comm_time
        assert res > 0, "Time cost should be greater than 0"
        print_to_file(self._file_path, "Estimated time cost {}.\n".format(res))
        return res

    def set_M_value(self, estimated_cost):
        import math
        n = math.floor(math.log10(estimated_cost)) + 2  # 计算位数
        print_to_file(self._file_path, "Set M to {}.\n".format(10 ** (n + 1)))
        return 10 ** (n + 1)  # 返回 10 的 (n + 1) 次方


    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        prefixes = ('f_', 'b_', 'w_')
        for key in self.model_result:
            if str(key).startswith(prefixes):
                print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            # if not (str(key).startswith("Do") or str(key).startswith("act") or str(key).startswith("binary")):
            #     print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _fix_stages(self):
        if self._schedule_method in (Schedule.ZBV, Schedule.GREEDY_v1, Schedule.GREEDY_v2):
            for pid in range(self._pp_size):
                if (pid // self._device_size) % 2 == 0:
                    self._devices[pid % self._device_size].append(pid)
                else:
                    self._devices[self._device_size - 1 - pid % self._device_size].append(pid)
        elif self._schedule_method in (Schedule.ZBH1, Schedule.ONE_F_ONE_B, Schedule.INTERLEAVED):
            for pid in range(self._pp_size):
                self._devices[pid % self._device_size].append(pid)
        else:
            assert("Stage alignment is not set.")

    def _reset_comm_length(self, dsa):
        new_comm_length = [[COMM_TIME if i != j else 0 for j in range(self._pp_size)] for i in range(self._pp_size)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length

    def _build_constraints(self) -> None:
        for sid in range(self._pp_size):

            self._stage_recomp_rate.append(self.model.addVar(vtype=GRB.BINARY, name=f"theta_{sid}"))
            for mid in range(self._num_microbatches):
                
                self._stage_f_offsets[sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mid}_{sid}", lb=0))
                self._stage_b_offsets[sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mid}_{sid}", lb=0))
                if SPLIT_BACKPROP:
                    self._stage_w_offsets[sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mid}_{sid}", lb=0))
            
            # self._stage_f_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{sid}_f"))
            self._stage_b_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{sid}_b"))
            if SPLIT_BACKPROP:
                self._stage_w_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{sid}_w"))

            self._stage_f_length.append(self._profiled_layer_f_length[sid])
            self.model.addConstr(self._stage_b_length[sid] == self._profiled_layer_b_length[sid] + self._stage_recomp_rate[sid] * self._stage_f_length[sid], name=f"length_of_b_{sid}")
            if SPLIT_BACKPROP:
                self.model.addConstr(self._stage_w_length[sid] == self._profiled_layer_w_length[sid])

        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()
        self._add_memory_constraints()
        # self._pipeline_activation_accumulation_constraint()

    def freeze_schedule_by_mid(self, mid):
        if BASE_SOLUTION:
            self.model.update()
            for var in self.model.getVars():
                if not var.VarName.startswith(("f_","b_","w_")):
                    continue
                _mid = int(var.VarName.split("_")[1])
                if _mid == mid:
                    self.model.addConstr(var == self.pipeline_scheduler.results[var.VarName])

    def _add_memory_constraints(self):
        """
            self._orders = {
                device_id:{
                    lid:{
                        mid:{
                            lid1:{
                                f:[],
                                b:[],
                                w:[],
                            },
                            lid2:{
                                f:[],
                                b:[],
                                w:[],
                            },
                            ...
                        },
                        ...
                    }
                    ...
                }
                ...
            }
        """
        self._orders = {}
        for did in range(self._num_device):
            self._orders[did] = {}
            for lid in self._devices[did]:
                self._orders[did][lid] = {}
                for mid in range(self._num_microbatches):
                    self._orders[did][lid][mid] = {}
                    for o_lid in self._devices[did]:
                        self._orders[did][lid][mid][o_lid] = {
                            'f': [self.model.addVar(name=f'act_f_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)],
                            'b': [self.model.addVar(name=f'act_b_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)],
                        }
                        if SPLIT_BACKPROP:
                            self._orders[did][lid][mid][o_lid]['w'] = [self.model.addVar(name=f'act_w_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)]

        for did in range(self._num_device):
            for lid in self._devices[did]:
                for mid in range(self._num_microbatches):
                    if SPLIT_BACKPROP:
                        pivot = self._stage_w_offsets[lid][mid]
                    else:
                        pivot = self._stage_b_offsets[lid][mid]
                        
                    for o_lid in self._devices[did]:
                        for o_mid in range(self._num_microbatches):
                            if o_lid == lid and o_mid == mid: # necessary, or leads to solution finding failure
                                continue 
                            binary_f = self.model.addVar(vtype=GRB.BINARY, name=f'mem_binary_f_{lid}_{mid}_{o_lid}_{o_mid}')
                            binary_b = self.model.addVar(vtype=GRB.BINARY, name=f'mem_binary_b_{lid}_{mid}_{o_lid}_{o_mid}')
                            binary_w = self.model.addVar(vtype=GRB.BINARY, name=f'mem_binary_w_{lid}_{mid}_{o_lid}_{o_mid}')
                            eps = 0.1
                            # M = 10000
                            M = self.M
                            self.model.addConstr(pivot >= self._stage_f_offsets[o_lid][o_mid] + eps - M * (1 - binary_f) )
                            self.model.addConstr(pivot <= self._stage_f_offsets[o_lid][o_mid] + M * binary_f)

                            self.model.addConstr(pivot >= self._stage_b_offsets[o_lid][o_mid] + eps - M * (1 - binary_b) )
                            self.model.addConstr(pivot <= self._stage_b_offsets[o_lid][o_mid] + M * binary_b)

                            if SPLIT_BACKPROP:
                                self.model.addConstr(pivot >= self._stage_w_offsets[o_lid][o_mid] + eps - M * (1 - binary_w) )
                                self.model.addConstr(pivot <= self._stage_w_offsets[o_lid][o_mid] + M * binary_w)

                            self.model.addConstr((binary_f == 1) >> (self._orders[did][lid][mid][o_lid]['f'][o_mid] == 1))
                            self.model.addConstr((binary_f == 0) >> (self._orders[did][lid][mid][o_lid]['f'][o_mid] == 0))
                            
                            self.model.addConstr((binary_b == 1) >> (self._orders[did][lid][mid][o_lid]['b'][o_mid] == 1))
                            self.model.addConstr((binary_b == 0) >> (self._orders[did][lid][mid][o_lid]['b'][o_mid] == 0))

                            if SPLIT_BACKPROP:                                
                                self.model.addConstr((binary_w == 1) >> (self._orders[did][lid][mid][o_lid]['w'][o_mid] == 1))
                                self.model.addConstr((binary_w == 0) >> (self._orders[did][lid][mid][o_lid]['w'][o_mid] == 0))
                    
                    required_memory = get_required_memory(
                        stage_id=lid,
                        layer_num=LAYER_NUM//DEVICE_NUM//CHUNK_NUM,
                        workload_type=workload_type_mapping['w' if SPLIT_BACKPROP else 'b'],
                        workload_type_num=WORKLOAD_TYPE_NUM,
                        layer_wise=False,
                        recomp=self._stage_recomp_rate[lid],
                    )
                    contain_head = LAYER_NUM + 1 in self._devices[did]
                    layer_per_stage = LAYER_NUM//DEVICE_NUM//CHUNK_NUM
                    base_memory = StateMemory.OPTIMIZER + StateMemory.LAYER * layer_per_stage + contain_head * StateMemory.HEAD
                    accumulated_activations = self._get_accumulated_activations(did=did, lid=lid, mid=mid) * layer_per_stage
                    accumulated_input_gradients = self._get_accumulated_input_gradients(did=did, lid=lid, mid=mid) * layer_per_stage
                    released_memory = self._get_released_memory(did=did, lid=lid, mid=mid) * layer_per_stage
                    # mw = self.model.addVar(name=f"mem_w_{lid}_{mid}",vtype=GRB.CONTINUOUS)

                    # self.model.addConstr(
                    #     (accumulated_activations
                    #     + accumulated_input_gradients
                    #     - released_memory
                    #     + base_memory
                    #     + required_memory) == mw
                    # )   
                    # self.model.addConstr(
                    #     mw <= GPU_MAX_MEM
                    # )

                    self.model.addConstr(
                        (accumulated_activations
                        + accumulated_input_gradients
                        - released_memory
                        + base_memory
                        + required_memory) <= GPU_MAX_MEM,name=f"mem_w_{lid}_{mid}"
                    )

    def _get_accumulated_activations(self, did, lid, mid):
        accumulated_activations = 0
        orders = self._orders[did][lid][mid]
        for o_lid in self._devices[did]:
            if o_lid == 0:
                continue                
            for o_mid in range(self._num_microbatches):
                if o_lid == lid and o_mid == mid: # necessary
                    continue
                if o_lid == STAGE_NUM - 1:
                    accumulated_activations += orders[o_lid]['f'][o_mid] * Activation.LOSS
                else:
                    accumulated_activations += orders[o_lid]['f'][o_mid] * (Activation.FULL * (1 - self._stage_recomp_rate[o_lid]) + Activation.INPUT * self._stage_recomp_rate[o_lid])
        return accumulated_activations

    def _get_accumulated_input_gradients(self, did, lid, mid):
        accumulated_input_gradients = 0
        orders = self._orders[did][lid][mid]
        for o_lid in self._devices[did]:
            if o_lid == 0:
                continue 
            for o_mid in range(self._num_microbatches):
                if o_lid == lid and o_mid == mid: # necessary
                    continue
                if o_lid == STAGE_NUM - 1:
                    accumulated_input_gradients -= orders[o_lid]['b'][o_mid] * Activation.LOSS
                else:
                    accumulated_input_gradients += orders[o_lid]['b'][o_mid] * (Gradient.INPUT + (Activation.FULL - Activation.INPUT) * self._stage_recomp_rate[o_lid])
        return accumulated_input_gradients
    
    def _get_released_memory(self, did, lid, mid):
        released_memory = 0
        if SPLIT_BACKPROP:
            orders = self._orders[did][lid][mid]
            for o_lid in self._devices[did]:
                if o_lid == 0:
                    continue 
                for o_mid in range(self._num_microbatches):
                    if o_lid == lid and o_mid == mid: # necessary
                        continue
                    released_memory += orders[o_lid]['w'][o_mid] * (Gradient.INPUT + Activation.FULL)
        return released_memory
    
    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            for i in range(1, self._pp_size):
                self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i - 1][mb] +
                                     self._stage_f_length[i - 1] + self._comm_length[i - 1][i])

            for i in range(self._pp_size - 1, 0, -1):
                self.model.addConstr(self._stage_b_offsets[i - 1][mb] >= self._stage_b_offsets[i][mb] +
                                     self._stage_b_length[i] + self._comm_length[i][i - 1])

            self.model.addConstr(self._stage_b_offsets[self._pp_size - 1][mb] >= self._stage_f_offsets[self._pp_size - 1][mb] +
                                 self._stage_f_length[self._pp_size - 1])

            if SPLIT_BACKPROP:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_w_offsets[i][mb] >= self._stage_b_offsets[i][mb] +
                                        self._stage_b_length[i])

            if mb > 0:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i][mb - 1] +
                                         self._stage_f_length[i])
                    self.model.addConstr(self._stage_b_offsets[i][mb] >= self._stage_b_offsets[i][mb - 1] +
                                         self._stage_b_length[i])
                    if SPLIT_BACKPROP:
                        self.model.addConstr(self._stage_w_offsets[i][mb] >= self._stage_w_offsets[i][mb - 1] +
                                            self._stage_w_length[i])
            # else:
            #     self.model.addConstr(self._forward_f_offsets[0][0] == 0)
        self.model.addConstr(self._stage_f_offsets[0][0] == 0)

    def _serial_computation_within_device_constraint(self):
        print_to_file(self._file_path, "Stage alignment:{}.\n".format(self._devices))
        total_constraints = 0
        same_mb_redundant_constraints = 0
        for did in range(self._num_device):
            # 加入对w的判断，同时修改_length的判断
            layers_within_device = self._devices[did]
            _pp_vars = []
            for pp in layers_within_device:
                _pp_vars += self._stage_f_offsets[pp] + self._stage_b_offsets[pp]
                if SPLIT_BACKPROP:
                    _pp_vars += self._stage_w_offsets[pp]
            type_of_workload = WORKLOAD_TYPE_NUM
            group_size = self._num_microbatches * type_of_workload
            for i, _ in enumerate(_pp_vars):
                i_pp = layers_within_device[i // group_size]
                _i_length = (
                    self._stage_f_length[i_pp]
                    if (i % group_size) // self._num_microbatches == 0 
                    else(
                        self._stage_b_length[i_pp] 
                        if (i % group_size) // self._num_microbatches == 1 
                        else self._stage_w_length[i_pp]
                    )
                )
                for j in range(i + 1, len(_pp_vars)):
                    total_constraints += 1
                    if j // (self._num_microbatches * type_of_workload) == i // (self._num_microbatches * type_of_workload):
                        if j % self._num_microbatches == i % self._num_microbatches:
                            same_mb_redundant_constraints += 1
                            continue
                    j_pp = layers_within_device[j // group_size]
                    _j_length = (
                        self._stage_f_length[j_pp]
                        if (j % group_size) // self._num_microbatches == 0
                        else(
                            self._stage_b_length[j_pp] 
                            if (j % group_size) // self._num_microbatches == 1 
                            else self._stage_w_length[j_pp]
                        )
                    )
                    y = self.model.addVar(vtype=GRB.BINARY, name=f"Do{did}_{i}_{j}")
                    # when time increses, M also increases to ensure right answer
                    # M = 1e4
                    M = self.M
                    self.model.addConstr(_pp_vars[j] >= _pp_vars[i] + _i_length - (1 - y) * M, name=f"Do{did}_{i}_{j}1") 
                    self.model.addConstr(_pp_vars[j] + _j_length <= _pp_vars[i] + y * M, name=f"Do{did}_{i}_{j}2")
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for pp in range(self._pp_size):
            if SPLIT_BACKPROP:
                self.model.addConstr(max_var >= self._stage_w_offsets[pp][-1] + self._stage_w_length[pp])
            else:
                self.model.addConstr(max_var >= self._stage_b_offsets[pp][-1] + self._stage_b_length[pp])

        self.model.setObjective(max_var, GRB.MINIMIZE)

    def set_baseline_solution(self):
        self.model.update()
        for var in self.model.getVars():
            if var.VarName in self.pipeline_scheduler.results.keys():
                var.Start = self.pipeline_scheduler.results[var.VarName]

        self.model.write("schedule_results/model.lp")

    def run(self, draw=False) -> None:
        """run simulation"""
        self._build_constraints()        
        self._build_optimize_objectives()

        self.model.setParam('TimeLimit', self._time_limit)
        # self.model.setParam('MIPGap', 0.00)

        if self._base_solution:
            self.set_baseline_solution()

        for mid in range(MICRO_BATCH_NUM):
            self.freeze_schedule_by_mid(mid=mid)

        start_time = time.time()
        print_to_file(self._file_path, "Gurobi Solver Solving...\n")
        self.model.optimize()
        end_time = time.time()
        if self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")

            # 计算不一致子集
            self.model.computeIIS()

            # 输出导致不可行的约束
            print("The following constraints are infeasible:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"{c.ConstrName}")
                    input()

            # 可选：输出变量
            # print("The following variables are part of the IIS:")
            # for v in self.model.getVars():
            #     if v.IISVar:
            #         print(f"{v.VarName}")

        if self.model.status == GRB.OPTIMAL:
            print_to_file(self._file_path, f"Optimal Result Cost: {end_time - start_time:.2f}.\n")
            # tranforms the result to a dictionary.
        elif self.model.status != GRB.INFEASIBLE:
            print_to_file(self._file_path, f"Result Found Cost: {end_time - start_time:.2f}.\n")
        else:
            print_to_file(self._file_path, "No Solution Found.\n")
            return {"max_start_offset": 999999999999}
        
        results = {var.varName: var.x for var in self.model.getVars()}
        self.model_result = results
        print_to_file(self._file_path, "MinExeTime:{}.\n".format(results["max_start_offset"]))
        for i in range(self._pp_size):
            # self._stage_f_length[i] = self.model_result[self._stage_f_length[i].varName]
            self._stage_b_length[i] = self.model_result[self._stage_b_length[i].varName] 
            print(self._stage_b_length[i])
            # input()
            if SPLIT_BACKPROP:
                self._stage_w_length[i] = self.model_result[self._stage_w_length[i].varName]
        
        if draw:
            # 4. draws the result.
            results = {str(key) : self.model_result[key] for key in self.model_result if str(key)[0:2] in ["f_","b_","w_"]}
            self._draw(resort_microbatch_index(self._num_microbatches ,results))

        # for s in self._de_st_mb.keys():
        #     print(self._de_st_mb[s])
        return results
    
    def get_workload_len(self, key):
        workload_type, mid, lid = key.split("_")
        mid = int(mid)
        lid = int(lid)
        if SCHEDULE_METHOD == Schedule.Layerwise:
            layers = 1
        else:
            layers = LAYER_NUM // STAGE_NUM

        if workload_type == "f":
            workload_len = F_TIME * layers
            if SCHEDULE_METHOD == Schedule.Layerwise:
                if lid == 0:
                    workload_len = EMB_F_TIME
                elif lid == LAYER_NUM - 1:
                    workload_len = CE_F_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_F_TIME
            else:
                if lid == 0:
                    workload_len += EMB_F_TIME
                elif lid == STAGE_NUM - 1:
                    workload_len += CE_F_TIME + HEAD_F_TIME
        elif workload_type == "b":
            workload_len = B_TIME * layers
            if SCHEDULE_METHOD == Schedule.Layerwise:
                if lid == LAYER_NUM - 1:
                    workload_len = CE_B_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_B_TIME
            else:
                if lid == STAGE_NUM - 1:
                    workload_len += CE_B_TIME + HEAD_B_TIME
        elif workload_type == "w":
            workload_len = W_TIME * layers
            if SCHEDULE_METHOD == Schedule.Layerwise:
                if lid == LAYER_NUM - 1:
                    workload_len = CE_W_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_W_TIME
            else:
                if lid == STAGE_NUM - 1:
                    workload_len += CE_W_TIME + HEAD_W_TIME
        return workload_len 
    
    def write_fbw_to_file(self):
        for key in self.model_result:
            if key.startswith(("f_","b_","w_")):
                print_to_file(f"gurobi_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key},{self.model_result[key]}\n")

                workload_len = self.get_workload_len(key=key)
                print_to_file(f"gurobi_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key}_e,{self.model_result[key] + workload_len}\n")

    def _draw(self, results: dict) -> None:
        # 绘制结果的逻辑
        # self.write_fbw_to_file()
        painter_conf = {
            "device_num": self._device_size,
            "devices": self._devices,
            "stage_num": self._pp_size,
            "pp_height": PP_HEIGHT,
            "pp_align": PP_ALIGN,
            "pixel_base": PIXEL_BASE,
            "nmb": self._num_microbatches,
            "forward_length": self._stage_f_length,
            "backward_length": self._stage_b_length,
            "backward_length2": self._stage_w_length,
            "comm_length": self._comm_length,
            "file_path": self._file_path,
            "max_time": self.model_result['max_start_offset'],
        }

        SchedulingPainter(painter_conf).draw(results)