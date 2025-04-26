import os
import time
import copy
import itertools
from .GSimulator import GSimulator
from .SPSimulator import SPSimulator
from .utils import resort_microbatch_index, print_to_file
from .abstract.mutils import F_TIME, W_TIME, B_TIME, LAYER_NUM

class DSASimulator:
    def __init__(self, config, solver_type="gurobi") -> None:

        self._pp_size                   = config["stage_num"]
        self._device_size               = config["device_num"]
        self.config                     = config
        self._basic_comm_length         = config["comm_time"]
        self._device_stage_alignments   = []
        self._dsa_hash                  = set()
        self._file_path                 = config["file_path"]
        self._solver_type               = solver_type
        self._stage_order_search        = config["stage_order_search"]
    def _prune_result(self, device_stage_alignment):
        for dsa in device_stage_alignment:
            if len(dsa) != self._pp_size // self._device_size:
                return False
            if len(dsa) == 0:
                return False
            # if len(dsa) < self._pp_size // self._device_size - 1:
            #     return False
            # if len(dsa) > self._pp_size // self._device_size + 1:
            #     return False
        return True

    def _reset_comm_length(self, dsa):
        new_comm_length = self._basic_comm_length
        for d in dsa:
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length

    def _transpose(self, matrix):
        return [row for row in zip(*matrix)]
    
    # def _traverse_01_limited_stage_alignment(self):
    #     segment_size = self._pp_size // self._device_size
    #     segments = [[] for _ in range(segment_size)]
    #     for i in range(segment_size):
    #         lb = self._device_size * i
    #         ub = self._device_size * (i + 1)
    #         increasing_stage = tuple(range(lb, ub))
    #         decreasing_stage = tuple(reversed(increasing_stage))
    #         segments[i] = [increasing_stage, decreasing_stage]    
    #     for combination in itertools.product(*segments):
    #         dsa = self._transpose(combination)
    #         self._dsa_hash.add(frozenset(dsa))
    
    def _generate_stage_constraint(self, lb, ub):
        if self._stage_order_search == "01":
            increasing_stage = tuple(range(lb, ub))
            decreasing_stage = tuple(reversed(increasing_stage))
            return [increasing_stage, decreasing_stage]
        else:
            return [l for l in list(itertools.permutations(range(lb, ub)))]
    def _traverse_limited_stage_alignment(self):
        segment_size = self._pp_size // self._device_size
        segments = [[] for _ in range(segment_size)]
        for i in range(segment_size):
            lb = self._device_size * i
            ub = self._device_size * (i + 1)
            # segments[i] = [l for l in list(itertools.permutations(range(lb, ub)))]        
            segments[i] = self._generate_stage_constraint(lb, ub)
        # Slightly Faster
        # segments = [[list(l) for l in list(itertools.permutations(range(self._device_size * i, self._device_size * (i + 1))))] for i in range(segment_size)]
        
        t1 = time.time()
        for combination in itertools.product(*segments):
            dsa = self._transpose(combination)
            # device size = 6, speedup from 0.71s → 0.57s
            # device size = 7, speedup from 38.3 → 30.2s
            self._dsa_hash.add(frozenset(dsa))
        t2 = time.time()
        print_to_file(self._file_path, "DSA search cost:{}\n".format(t2-t1))

    def _generate_dsa_results(self, file_path):
        if os.path.exists(file_path):
            print_to_file(self._file_path, "DSA results exists.\n")
            with open(file_path, 'r') as file:
                self._dsa_hash = eval(file.read())
        else:
            print_to_file(self._file_path, "Searching DSA results.\n")
            self._traverse_limited_stage_alignment()
            with open(file_path, 'w') as file:
                file.write(str(self._dsa_hash))

    def traverse_run(self) -> None:

        print_to_file(self._file_path, "Traversing every stage alignment...\n")
        # device_stage_alignments = [[] for _ in range(self._device_size)]
        # self._traverse_every_stage_alignment(0, device_stage_alignment=device_stage_alignments)
        dsa_file_name = './dsa/{}-{}-{}.txt'.format(self._device_size, self._pp_size, self._stage_order_search.name)
        self._generate_dsa_results(dsa_file_name)

        # self._traverse_limited_stage_alignment()
        print_to_file(self._file_path, "Traversing over. {} situations found.\n".format(len(self._dsa_hash)))
        print_to_file(self._file_path, "D={},S={},M={},F={},B={},W={},C={},SOS={}.\n".format(
            self._device_size, self._pp_size, LAYER_NUM, 
            F_TIME, B_TIME, 
            W_TIME, self._basic_comm_length,
            self._stage_order_search,
        ))
        
        best_result = None
        minimal_time = 999999999999
        # simulators = []
        for dsa in self._dsa_hash:
            dsa = list(dsa)
            if self._solver_type == "z3":
                temp_simulator = SPSimulator(self.config, device_stage_alignments=dsa)
            else:
                temp_simulator = GSimulator(self.config, device_stage_alignments=dsa)
            # simulators.append(temp_simulator)
            result = temp_simulator.run()

            if self._solver_type == "z3":
                result_time = result["max_start_offset"].as_long()
            else:
                result_time = result["max_start_offset"]

            if result_time < minimal_time:
                minimal_time = result_time
                best_result = temp_simulator
        
        if self._solver_type == "z3":
            # z3 way
            result = {str(key) : best_result.model_result[key].as_long() for key in best_result.model_result if str(key)[0:2] in ["f_","b_","w_"]}
        else:
            # gurobi way
            result = {str(key) : best_result.model_result[key] for key in best_result.model_result if str(key)[0:2] in ["f_","b_","w_"]}
        end_time = time.time()
        best_result.show_device_stage_mapping()
        best_result.show_solution_detail()
        best_result._draw(resort_microbatch_index(best_result._num_microbatches ,result))

        return end_time
    
    # very slow down up to 30x
    # device = 6, 518400 → 720 cases
    def _unique_result(self, device_stage_alignment):    
        for existing_result in self._device_stage_alignments:
            acc = 0
            for stage_alignment in device_stage_alignment:
                if stage_alignment not in existing_result:
                    break
                else:
                    acc += 1
            if acc == self._device_size:
                return False
        return True
    
    def _traverse_every_stage_alignment(self, sid, device_stage_alignment):
        if sid == self._pp_size:
            if self._prune_result(device_stage_alignment) and self._unique_result(device_stage_alignment):
                self._device_stage_alignments.append(copy.deepcopy(device_stage_alignment))
        else:
            for did in range(self._device_size):
                device_stage_alignment[did].append(sid)
                self._traverse_every_stage_alignment(sid + 1, device_stage_alignment)
                device_stage_alignment[did].pop()
                