import time
from gurobipy import Model, GRB, quicksum
from .ChimeraLayerwisePainter import LayerwiseSchedulingPainter
from .abstract.mutils import *
from .chimera_utils import resort_microbatch_index, print_to_file
from .abstract import Pipeline
from .abstract.Device import get_required_memory

workload_type_mapping = {
    'f':WorkloadType.F,
    'b':WorkloadType.B,
    'w':WorkloadType.W,
}

class ChimeraSimulator:

    def __init__(self, config: dict) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["stage_num"]
        self._num_device = config["device_num"]
        self._num_layer = config["layer_num"]
        self.total_stream = 2
        self._num_microbatches = config["nmb"]
        self._emb_head_ce = config["emb_head_ce"]
        if self._emb_head_ce:
            self._num_layer += 3

        # 估算总时间限定大M取值（须大于等于估算值）
        self.estimated_time_cost = self.estimate_time_cost()
        self.M = self.set_M_value(self.estimated_time_cost)

        self.model = Model("ChimeraSimulator")
        self.streams = [
            ChimeraStream(
                config=config, 
                order=True,
                model=self.model,
                chimera_way_idx=idx,
                total_stream=self.total_stream,
            ) if idx % 2 else ChimeraStream(
                config=config, 
                order=False,
                model=self.model,
                chimera_way_idx=idx,
                total_stream=self.total_stream,
            )
            for idx in range(2)
        ]
        self.model_result = None

    def _add_memory_constraints(self):
        self._orders = {}
        for stream_idx, stream in enumerate(self.streams):
            self._orders[stream_idx] = {}
            for did in range(self._num_device):
                self._orders[stream_idx][did] = {}
                for lid in stream._devices[did]:
                    self._orders[stream_idx][did][lid] = {}
                    for mid in range(self._num_microbatches):
                        self._orders[stream_idx][did][lid][mid] = {}
                        for o_stream_idx, o_stream in enumerate(self.streams):
                            for o_lid in o_stream._devices[did]:
                                self._orders[stream_idx][did][lid][mid][o_lid] = {
                                    'f': [self.model.addVar(name=f'cwi_{stream._chimera_way_idx}_act_f_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)],
                                    'b': [self.model.addVar(name=f'cwi_{stream._chimera_way_idx}_act_b_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)],
                                }
                                if SPLIT_BACKPROP:
                                    self._orders[stream_idx][did][lid][mid][o_lid]['w'] = [self.model.addVar(name=f'cwi_{stream._chimera_way_idx}_act_w_{did}_{lid}_{mid}_{o_lid}_{time.time()}', vtype=GRB.BINARY) for mid in range(self._num_microbatches)]

        for stream_idx, stream in enumerate(self.streams):
            for did in range(self._num_device):
                for lid in stream._devices[did]:
                    for mid in range(self._num_microbatches):
                        pivot = stream._layer_b_offsets[lid][mid]
                        if SPLIT_BACKPROP:
                            pivot = stream._layer_w_offsets[lid][mid]
                        for o_stream_idx, o_stream in enumerate(self.streams):
                            for o_lid in o_stream._devices[did]:
                                for o_mid in range(o_stream._num_microbatches):
                                    if o_lid == lid and o_mid == mid: # necessary, or leads to solution finding failure
                                        continue 
                                    binary_f = self.model.addVar(vtype=GRB.BINARY, name=f'cwi_{stream._chimera_way_idx}_binary_f_{lid}_{mid}_{o_lid}_{o_mid}')
                                    binary_b = self.model.addVar(vtype=GRB.BINARY, name=f'cwi_{stream._chimera_way_idx}_binary_b_{lid}_{mid}_{o_lid}_{o_mid}')
                                    if SPLIT_BACKPROP:
                                        binary_w = self.model.addVar(vtype=GRB.BINARY, name=f'cwi_{stream._chimera_way_idx}_binary_w_{lid}_{mid}_{o_lid}_{o_mid}')

                                    eps = 0.1
                                    # M = 10000
                                    M = self.M
                                    self.model.addConstr(pivot >= o_stream._layer_f_offsets[o_lid][o_mid] + eps - M * (1 - binary_f) )
                                    self.model.addConstr(pivot <= o_stream._layer_f_offsets[o_lid][o_mid] + M * binary_f)

                                    self.model.addConstr(pivot >= o_stream._layer_b_offsets[o_lid][o_mid] + eps - M * (1 - binary_b) )
                                    self.model.addConstr(pivot <= o_stream._layer_b_offsets[o_lid][o_mid] + M * binary_b)

                                    if SPLIT_BACKPROP:
                                        self.model.addConstr(pivot >= o_stream._layer_w_offsets[o_lid][o_mid] + eps - M * (1 - binary_w) )
                                        self.model.addConstr(pivot <= o_stream._layer_w_offsets[o_lid][o_mid] + M * binary_w)

                                    self.model.addConstr((binary_f == 1) >> (self._orders[stream_idx][did][lid][mid][o_lid]['f'][o_mid] == 1))
                                    self.model.addConstr((binary_f == 0) >> (self._orders[stream_idx][did][lid][mid][o_lid]['f'][o_mid] == 0))
                                    
                                    self.model.addConstr((binary_b == 1) >> (self._orders[stream_idx][did][lid][mid][o_lid]['b'][o_mid] == 1))
                                    self.model.addConstr((binary_b == 0) >> (self._orders[stream_idx][did][lid][mid][o_lid]['b'][o_mid] == 0))

                                    if SPLIT_BACKPROP:                                
                                        self.model.addConstr((binary_w == 1) >> (self._orders[stream_idx][did][lid][mid][o_lid]['w'][o_mid] == 1))
                                        self.model.addConstr((binary_w == 0) >> (self._orders[stream_idx][did][lid][mid][o_lid]['w'][o_mid] == 0))
                            
                        required_memory = get_required_memory(
                            stage_id=lid,
                            layer_num=1,
                            workload_type=workload_type_mapping['w' if SPLIT_BACKPROP else 'b'],
                            workload_type_num=WORKLOAD_TYPE_NUM,
                            layer_wise=True,
                            recomp=stream._layer_recomp_rate[lid],
                        )
                        base_memory = 0
                        for s in self.streams:
                            base_memory += StateMemory.OPTIMIZER + StateMemory.LAYER * len([l for l in s._devices[did] if l not in (0, LAYER_NUM - 1, LAYER_NUM - 2)])
                        accumulated_activations = self._get_accumulated_activations(stream_idx=stream_idx, did=did, lid=lid, mid=mid)
                        accumulated_input_gradients = self._get_accumulated_input_gradients(stream_idx=stream_idx, did=did, lid=lid, mid=mid)
                        released_memory = self._get_released_memory(stream_idx=stream_idx, did=did, lid=lid, mid=mid)

                        self.model.addConstr(
                            (accumulated_activations
                            + accumulated_input_gradients
                            - released_memory
                            + base_memory
                            + required_memory) <= GPU_MAX_MEM,name=f"cwi_{stream._chimera_way_idx}_mem_w_{lid}_{mid}_{time.time()}"
                        )

    def _get_accumulated_activations(self, stream_idx, did, lid, mid):
        accumulated_activations = 0
        orders = self._orders[stream_idx][did][lid][mid]
        for idx, stream in enumerate(self.streams):
            for o_lid in stream._devices[did]:
                if o_lid == 0 or o_lid >= LAYER_NUM - 2:
                    continue
                for o_mid in range(self._num_microbatches):
                    if o_lid == lid and o_mid == mid: # necessary
                        continue
                    accumulated_activations += orders[o_lid]['f'][o_mid] * (Activation.FULL * (1 - stream._layer_recomp_rate[o_lid]) + Activation.INPUT * stream._layer_recomp_rate[o_lid])
        return accumulated_activations

    def _get_accumulated_input_gradients(self, stream_idx, did, lid, mid):
        accumulated_input_gradients = 0
        if not SPLIT_BACKPROP:
            return accumulated_input_gradients
        orders = self._orders[stream_idx][did][lid][mid]
        for idx, stream in enumerate(self.streams):
            for o_lid in stream._devices[did]:
                if o_lid == 0 or o_lid >= LAYER_NUM - 2:
                    continue
                for o_mid in range(self._num_microbatches):
                    if o_lid == lid and o_mid == mid: # necessary
                        continue
                    accumulated_input_gradients += orders[o_lid]['b'][o_mid] * (Gradient.INPUT + (Activation.FULL - Activation.INPUT)* stream._layer_recomp_rate[o_lid])
        return accumulated_input_gradients
    
    def _get_released_memory(self, stream_idx, did, lid, mid):
        released_memory = 0
        orders = self._orders[stream_idx][did][lid][mid]
        for idx, stream in enumerate(self.streams):
            for o_lid in stream._devices[did]:
                if o_lid == 0 or o_lid >= LAYER_NUM - 2:
                    continue
                for o_mid in range(self._num_microbatches):
                    if o_lid == lid and o_mid == mid: # necessary
                        continue
                    if SPLIT_BACKPROP:
                        released_memory += orders[o_lid]['w'][o_mid] * (Gradient.INPUT + Activation.FULL)
                    else:
                        released_memory += orders[o_lid]['b'][o_mid] * (Gradient.INPUT)
        return released_memory

    def _serial_computation_within_device_constraint(self):
        for stream in self.streams:
            print_to_file(self._file_path, "Chimera idx:{}, Stage alignment:{}.\n".format(stream._chimera_way_idx, stream._devices))
        
        total_constraints = 0
        same_mb_redundant_constraints = 0
        for did in range(self._num_device):

            layers_within_device = [[] for _ in range(len(self.streams))]
            for idx,stream in enumerate(self.streams):
                layers_within_device[idx] += stream._devices[did]
            
            _pp_vars = [[] for _ in range(len(self.streams))]
            for idx,stream in enumerate(self.streams):
                for pp in layers_within_device[idx]:            
                    _pp_vars[idx] += stream._layer_f_offsets[pp] + stream._layer_b_offsets[pp]
                    if SPLIT_BACKPROP:
                        _pp_vars[idx] += stream._layer_w_offsets[pp]

            type_of_workload = WORKLOAD_TYPE_NUM
            group_size = self._num_microbatches * type_of_workload
            for idx,stream in enumerate(self.streams):
                for i, _ in enumerate(_pp_vars[idx]):
                    i_pp = layers_within_device[idx][i // group_size]
                    _i_length = (
                        self.streams[idx]._layer_f_length[i_pp]
                        if (i % group_size) // self._num_microbatches == 0 
                        else(
                            self.streams[idx]._layer_b_length[i_pp] 
                            if (i % group_size) // self._num_microbatches == 1 
                            else self.streams[idx]._layer_w_length[i_pp]
                        )
                    )
                    for jdx,stream in enumerate(self.streams):
                        if jdx == idx:
                            for j in range(i + 1, len(_pp_vars[idx])):
                                total_constraints += 1
                                if j // (self._num_microbatches * type_of_workload) == i // (self._num_microbatches * type_of_workload):
                                    if j % self._num_microbatches == i % self._num_microbatches:
                                        same_mb_redundant_constraints += 1
                                        continue
                                j_pp = layers_within_device[jdx][j // group_size]
                                _j_length = (
                                    self.streams[jdx]._layer_f_length[j_pp]
                                    if (j % group_size) // self._num_microbatches == 0
                                    else(
                                        self.streams[jdx]._layer_b_length[j_pp] 
                                        if (j % group_size) // self._num_microbatches == 1 
                                        else self.streams[jdx]._layer_w_length[j_pp]
                                    )
                                )
                                y = self.model.addVar(vtype=GRB.BINARY, name=f"cwi_{self.streams[idx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}")
                                # when time increses, M also increases to ensure right answer
                                # M = 1e4
                                M = self.M
                                self.model.addConstr(_pp_vars[jdx][j] >= _pp_vars[idx][i] + _i_length - (1 - y) * M, name=f"cwi_{self.streams[idx]._chimera_way_idx}_{self.streams[jdx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}") 
                                self.model.addConstr(_pp_vars[jdx][j] + _j_length <= _pp_vars[idx][i] + y * M, name=f"cwi_{self.streams[idx]._chimera_way_idx}_{self.streams[jdx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}")
                        else:
                            for j in range(len(_pp_vars[jdx])):
                                total_constraints += 1
                                j_pp = layers_within_device[jdx][j // group_size]
                                _j_length = (
                                    self.streams[jdx]._layer_f_length[j_pp]
                                    if (j % group_size) // self._num_microbatches == 0
                                    else(
                                        self.streams[jdx]._layer_b_length[j_pp] 
                                        if (j % group_size) // self._num_microbatches == 1 
                                        else self.streams[jdx]._layer_w_length[j_pp]
                                    )
                                )
                                y = self.model.addVar(vtype=GRB.BINARY, name=f"cwi_{self.streams[idx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}")
                                # when time increses, M also increases to ensure right answer
                                # M = 1e4
                                M = self.M
                                self.model.addConstr(_pp_vars[jdx][j] >= _pp_vars[idx][i] + _i_length - (1 - y) * M, name=f"cwi_{self.streams[idx]._chimera_way_idx}_{self.streams[jdx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}") 
                                self.model.addConstr(_pp_vars[jdx][j] + _j_length <= _pp_vars[idx][i] + y * M, name=f"cwi_{self.streams[idx]._chimera_way_idx}_{self.streams[jdx]._chimera_way_idx}_Do{did}_{i_pp}_{j_pp}_{time.time()}")
                        
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for stream in self.streams:
            for lid in range(self._num_layer):
                if SPLIT_BACKPROP:
                    self.model.addConstr(max_var >= stream._layer_w_offsets[lid][-1] + stream._layer_w_length[lid])
                else:
                    self.model.addConstr(max_var >= stream._layer_b_offsets[lid][-1] + stream._layer_b_length[lid])
        self.model.setObjective(max_var, GRB.MINIMIZE)

    def run(self, draw=False) -> None:
        for stream in self.streams:
            stream._build_constraints()
        self._serial_computation_within_device_constraint()
        self._add_memory_constraints()
        self._build_optimize_objectives()

        self.model.setParam('TimeLimit', self._time_limit)

        start_time = time.time()
        print_to_file(self._file_path, "Gurobi Solver Solving...\n")
        self.model.optimize()
        end_time = time.time()
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
        for i in range(self._num_layer):
            self.streams[0]._layer_f_length[i] = self.model_result[f"cwi_0_s{i}_f"]
            self.streams[0]._layer_b_length[i] = self.model_result[f"cwi_0_s{i}_b"] 
            self.streams[1]._layer_f_length[i] = self.model_result[f"cwi_1_s{i}_f"]
            self.streams[1]._layer_b_length[i] = self.model_result[f"cwi_1_s{i}_b"] 
            if SPLIT_BACKPROP:
                self.streams[0]._layer_w_length[i] = self.model_result[f"cwi_0_s{i}_w"] 
                self.streams[1]._layer_w_length[i] = self.model_result[f"cwi_1_s{i}_w"] 
        
        # print(self.model_result)
        if draw:
            # 4. draws the result.
            results = {str(key) : self.model_result[key] for key in self.model_result if str(key).startswith((
                "cwi_0_f_","cwi_0_b_","cwi_0_w_",
                "cwi_1_f_","cwi_1_b_","cwi_1_w_",
                ))}
            self._draw(resort_microbatch_index(self._num_microbatches ,results))
        self.model.write("schedule_results/model.lp")
        return results
    

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
        n = math.floor(math.log10(estimated_cost)) + 1  # 计算位数
        print_to_file(self._file_path, "Set M to {}.\n".format(10 ** (n + 1)))
        return 10 ** (n + 1)  # 返回 10 的 (n + 1) 次方


    def _draw(self, results: dict) -> None:
        # 绘制结果的逻辑
        painter_conf = {
            "device_num": self.streams[0]._num_device,
            "devices": [stream._devices for stream in self.streams],
            "stage_num": self.streams[0]._pp_size,
            "pp_height": PP_HEIGHT,
            "pp_align": PP_ALIGN,
            "pixel_base": PIXEL_BASE,
            "nmb": self.streams[0]._num_microbatches,
            "forward_length": [stream._layer_f_length for stream in self.streams],
            "backward_length": [stream._layer_b_length for stream in self.streams],
            "backward_length2": [stream._layer_w_length for stream in self.streams],
            "comm_length": [stream._comm_length for stream in self.streams],
            "file_path": self.streams[0]._file_path,
            "max_time": self.model_result['max_start_offset'],
            "num_layer": self.streams[0]._num_layer,
            "emb_head_ce": self._emb_head_ce,
        }

        LayerwiseSchedulingPainter(painter_conf).draw(results)


class ChimeraStream:
    def __init__(self, config: dict, model:Model, chimera_way_idx:int, total_stream:int, order=None, new_comm_length=None) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["stage_num"]
        self._num_device = config["device_num"]
        self._num_layer = config["layer_num"]
        self._num_microbatches = config["nmb"]
        self._emb_head_ce = config["emb_head_ce"]
        self._chimera_way_idx = chimera_way_idx
        self._total_stream = total_stream

        self._profiled_layer_f_length = config["f_time"]
        self._profiled_layer_b_length = config["b_time"]
        self._profiled_layer_w_length = config["w_time"]

        if self._emb_head_ce:
            self._profiled_layer_f_length = [EMB_F_TIME] + [F_TIME for _ in range(LAYER_NUM)] + [HEAD_F_TIME, CE_F_TIME]
            self._profiled_layer_b_length = [0] + [B_TIME for _ in range(LAYER_NUM)] + [HEAD_B_TIME, CE_B_TIME]
            self._profiled_layer_w_length = [0] + [W_TIME for _ in range(LAYER_NUM)] + [HEAD_W_TIME, CE_W_TIME]
            self._num_layer += 3
        else:        
            self._profiled_layer_f_length[0] += EMB_F_TIME
            self._profiled_layer_f_length[-1] += CE_F_TIME + HEAD_F_TIME
            self._profiled_layer_b_length[-1] += CE_B_TIME + HEAD_B_TIME
            self._profiled_layer_w_length[-1] += CE_W_TIME + HEAD_W_TIME

        self._comm_length = config["comm_time"] if not new_comm_length else new_comm_length
        
        # 检查输入参数
        assert isinstance(self._profiled_layer_f_length, (list, tuple))
        assert isinstance(self._profiled_layer_b_length, (list, tuple))
        assert isinstance(self._profiled_layer_w_length, (list, tuple))

        self.model = model

        # 变量初始化
        self._layer_f_length  = []
        self._layer_b_length = []
        self._layer_w_length = []

        self._layer_recomp_rate = []
        self._device_layers_mapping = [[] for _ in range(self._num_device)]

        self._layer_f_offsets = [[] for _ in range(self._num_layer)]
        self._layer_b_offsets = [[] for _ in range(self._num_layer)]
        self._layer_w_offsets = [[] for _ in range(self._num_layer)]
        self._devices = self.split_layers_into_stages(layer_num=self._num_layer if not self._emb_head_ce else self._num_layer-3, stage_num=self._pp_size, reverse=order)
        print(self._devices)

    def split_layers_into_stages(self, layer_num, stage_num, reverse=False):
        result = []
        increment = layer_num // stage_num
        for i in range(stage_num):
            start = i * increment
            if reverse:
                start = layer_num - (i + 1) * increment
            end = start + increment
            result.append(list(range(max(0, start), min(end, layer_num))))
        
        if self._emb_head_ce:
            for did, _ in enumerate(result):
                for lid, _ in enumerate(result[did]):
                    result[did][lid] += 1
            if reverse:
                result[self._num_device // 2].append(0)
                result[-1].append(self._num_layer-2)
                result[-2].append(self._num_layer-1)
            else:
                result[self._num_device // 2 - 1].append(0)
                result[0].append(self._num_layer-2)
                result[1].append(self._num_layer-1)
        return result

    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def _reset_comm_length(self, dsa):
        new_comm_length = [[COMM_TIME if i != j else 0 for j in range(self._num_layer)] for i in range(self._num_layer)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length
    
    def _generate_variables(self):
        for lid in range(self._num_layer):
            self._layer_recomp_rate.append(self.model.addVar(vtype=GRB.BINARY, name=f"cwi_{self._chimera_way_idx}_theta_{lid}"))
            for mid in range(self._num_microbatches):
                
                self._layer_f_offsets[lid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_f_{mid}_{lid}", lb=0))
                self._layer_b_offsets[lid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_b_{mid}_{lid}", lb=0))
                if SPLIT_BACKPROP:
                    self._layer_w_offsets[lid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_w_{mid}_{lid}", lb=0))
            
            self._layer_f_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_s{lid}_f"))
            self._layer_b_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_s{lid}_b"))
            if SPLIT_BACKPROP:
                self._layer_w_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"cwi_{self._chimera_way_idx}_s{lid}_w"))
            
            self.model.addConstr(self._layer_f_length[lid] == self._profiled_layer_f_length[lid])
            self.model.addConstr(self._layer_b_length[lid] == self._profiled_layer_b_length[lid] + self._layer_recomp_rate[lid] * self._layer_f_length[lid])
            if SPLIT_BACKPROP:
                self.model.addConstr(self._layer_w_length[lid] == self._profiled_layer_w_length[lid])

    def _build_constraints(self) -> None:
        self._generate_variables()
        
        if SEQ_LEN > 4*K:
            print("Set recomputing to 1")
            self.model.update()  # 更新模型以确保所有变量都被添加
            for var in self._layer_recomp_rate:
                var.start = 1  # 设置初始值为 1

        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict(split=self._emb_head_ce)

    def _real_pipeline_modeling_constraint_strict(self, split=False):
        if split:
            for mb in range(self._num_microbatches):
                for i in range(1, self._num_layer):
                    self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i - 1][mb] +
                                        self._layer_f_length[i - 1] + self._comm_length[i - 1][i])

                for i in range(self._num_layer - 1, 1, -1):
                    self.model.addConstr(self._layer_b_offsets[i - 1][mb] >= self._layer_b_offsets[i][mb] +
                                        self._layer_b_length[i] + self._comm_length[i][i - 1])
                # NOTE: Embedding layer has no B
                self.model.addConstr(self._layer_b_offsets[0][mb] == self._layer_f_offsets[0][mb] + self._layer_f_length[0])
                

                self.model.addConstr(self._layer_b_offsets[self._num_layer - 1][mb] >= self._layer_f_offsets[self._num_layer - 1][mb] +
                                    self._layer_f_length[self._num_layer - 1])

                if SPLIT_BACKPROP:
                    # NOTE: Embedding layer has no W
                    self.model.addConstr(self._layer_w_offsets[0][mb] == self._layer_f_offsets[0][mb] + self._layer_f_length[0])
                    for i in range(1, self._num_layer - 1):
                        self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_b_offsets[i][mb] +
                                            self._layer_b_length[i])
                    # NOTE: CE layer has no W
                    self.model.addConstr(self._layer_w_offsets[self._num_layer - 1][mb] == self._layer_b_offsets[self._num_layer - 1][mb] +
                                            self._layer_b_length[self._num_layer - 1])

                if mb > 0:
                    # NOTE: Embedding layer has no B and W
                    self.model.addConstr(self._layer_f_offsets[0][mb] >= self._layer_f_offsets[0][mb - 1] +
                                            self._layer_f_length[0])
                    # Transformer layer + Head layer
                    for i in range(1, self._num_layer - 1):
                        self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i][mb - 1] +
                                            self._layer_f_length[i])
                        self.model.addConstr(self._layer_b_offsets[i][mb] >= self._layer_b_offsets[i][mb - 1] +
                                            self._layer_b_length[i])
                        if SPLIT_BACKPROP:
                            self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_w_offsets[i][mb - 1] +
                                                self._layer_w_length[i])
                    # NOTE: CE layer has no W
                    self.model.addConstr(self._layer_f_offsets[self._num_layer - 1][mb] >= self._layer_f_offsets[self._num_layer - 1][mb - 1] +
                                            self._layer_f_length[self._num_layer - 1])
                    self.model.addConstr(self._layer_b_offsets[self._num_layer - 1][mb] >= self._layer_b_offsets[self._num_layer - 1][mb - 1] +
                                            self._layer_b_length[self._num_layer - 1])
        else:
            for mb in range(self._num_microbatches):
                for i in range(1, self._num_layer):
                    self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i - 1][mb] +
                                        self._layer_f_length[i - 1] + self._comm_length[i - 1][i])

                for i in range(self._num_layer - 1, 0, -1):
                    self.model.addConstr(self._layer_b_offsets[i - 1][mb] >= self._layer_b_offsets[i][mb] +
                                        self._layer_b_length[i] + self._comm_length[i][i - 1])

                self.model.addConstr(self._layer_b_offsets[self._num_layer - 1][mb] >= self._layer_f_offsets[self._num_layer - 1][mb] +
                                    self._layer_f_length[self._num_layer - 1])

                if SPLIT_BACKPROP:
                    for i in range(0, self._num_layer):
                        self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_b_offsets[i][mb] +
                                            self._layer_b_length[i])
                    
                if mb > 0:
                    for i in range(0, self._num_layer):
                        self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i][mb - 1] +
                                            self._layer_f_length[i])
                        self.model.addConstr(self._layer_b_offsets[i][mb] >= self._layer_b_offsets[i][mb - 1] +
                                            self._layer_b_length[i])
                        if SPLIT_BACKPROP:
                            self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_w_offsets[i][mb - 1] +
                                                self._layer_w_length[i])
        
        self.model.addConstr(self._layer_f_offsets[0][0] == 0)

