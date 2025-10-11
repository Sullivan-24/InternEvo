import copy
import json
import os
from enum import Enum, IntEnum
import os
import torch
import pdb
import ast
class WorkloadType(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'
    RECOMPUTE = 'r'
class ModuleType(Enum):
    CHIMERA = 'Chimera'
    INTERLEAVED = 'Interleaved'
    VSHAPE = 'Vshape'
    ONEFONEB = '1f1b'
    ZBH1 = 'Zbh1'
    HET = 'Het'
    
def busy_wait_kernel(time):
    for _ in range(time):
        N = 1024
        a = torch.rand(N, N, device='cuda')
        b = torch.rand(N, N, device='cuda')
        # GPU矩阵乘法
        for __ in range(25):
            torch.matmul(a, b)
    return

def interval_distance(a, b):
    """
    计算区间[a1, a2]和[b1, b2]之间的距离
    参数: 
        a, b: 元组表示的区间(a1, a2)和(b1, b2)
    返回: 
        距离(重叠时为0)
    """
    a1, a2 = a
    b1, b2 = b
    return min(abs(b1 - a2), abs(a1 - b2))

def distence(point,begin,end):
    if point < begin:
        return begin - point
    elif point > end:
        return point - end
    else:
        return 0

def _get_chunk_by_stage(stage_id: int,stage_placement:list) -> int:
    for pp_rank_stage in stage_placement:
        for chunk_id in range(len(pp_rank_stage)):
            if pp_rank_stage[chunk_id] == stage_id:
                return chunk_id

def _get_pp_rank_by_placement(stage_id: int, stage_placement:list) -> int:
    for pp_rank in range(len(stage_placement)):
        for stage_in_pp_rank in stage_placement[pp_rank]:
            if stage_in_pp_rank == stage_id:
                return pp_rank

def recvnum(comm_graph):
    #print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(comm_graph):
        recvF = 0
        recvB = 0
        for comm_op in comm_stage:
            for recvlistB in comm_op[WorkloadType.BACKWARD.value]:
                if recvlistB[0] == WorkloadType.BACKWARD.value:
                    recvB += 1
                elif recvlistB[0] == WorkloadType.FORWARD.value:
                    recvF += 1
            for recvlistA in comm_op['A']:
                if recvlistA[0] == WorkloadType.BACKWARD.value:
                    recvB += 1
                elif recvlistA[0] == WorkloadType.FORWARD.value:
                    recvF += 1
        #print(f"rank_id {rank_id}: recvF {recvF}, recvB {recvB}")
        ##print(f'{comm_stage},')
    #print(']')

def count_workloads(workloads):
    f_num = 0
    b_num = 0
    w_num = 0
    r_num = 0
    r_stages = set()
    for s in workloads:
        workload_type = s["workload_type"]
        if workload_type == WorkloadType.FORWARD.value:
            f_num += 1
            continue
        elif workload_type == WorkloadType.BACKWARD.value:
            b_num += 1
        elif workload_type == WorkloadType.FORWARD.value:
            w_num += 1
        elif workload_type == WorkloadType.RECOMPUTE.value:
            r_num += 1
            r_stages.add(s["stage_id"])
    return f_num, b_num, w_num, r_num , r_stages

def judge_scheduler_type(stage_placement):
    ranks = len(stage_placement)
    if ranks <= 1:
        return None
    num_chunks_per_pp_rank = set()
    sum_stageId_per_pp_rank = set()

    for i in range(ranks):
        num_chunks_per_pp_rank.add(len(stage_placement[i]))
        sum_stageId = sum(stage_placement[i])
        sum_stageId_per_pp_rank.add(sum_stageId)
    sum_stageId_per_pp_rank = sorted(list(sum_stageId_per_pp_rank))
    num_chunks_per_pp_rank = sorted(list(num_chunks_per_pp_rank))
    if len(num_chunks_per_pp_rank) > 1:
        return ModuleType.HET.value
    else:
        if num_chunks_per_pp_rank[0] == 1:
            return ModuleType.ONEFONEB.value
        elif len(sum_stageId_per_pp_rank) == 1:
            if sum_stageId_per_pp_rank[0] == ranks-1:
                return ModuleType.CHIMERA.value
            else:
                return ModuleType.VSHAPE.value
        elif 1< len(sum_stageId_per_pp_rank) < ranks :
            return ModuleType.HET.value
        else:# len(sum_stageId_per_pp_rank) == ranks
            num_chunks = num_chunks_per_pp_rank[0]
            for i in range(1,ranks):
                if sum_stageId_per_pp_rank[i] - sum_stageId_per_pp_rank[i-1] != num_chunks:
                    return ModuleType.HET.value
            return ModuleType.INTERLEAVED.value

def judge_split_backward(unified_scheduler):
    if unified_scheduler[0][0][-1]["workload_type"] == WorkloadType.WEIGHT.value:
        return True
    else:
        return False

def write_json(jsonpath, content):
    with open(jsonpath, 'a',encoding='utf-8') as f:
        json.dump(content, f)
        f.write('\n')

def order_result_mutichunk(input: str, stage_placement: list, num_microbatches:int, dp_size:int, pp_size:int) -> None:
    all_rank_workloads = [[[] for _ in range(pp_size)] for _ in range(dp_size)]
    all_workload = input.split('\n')
    ##print(all_workload)
    max_end_time = 0
    for workload in all_workload:
        if workload == '':
            continue
        start_time = float(workload.split(',')[-2])
        end_time = float(workload.split(',')[-1])
        if end_time > max_end_time:
            max_end_time = end_time
        infor = workload.split(',')[0]
        if infor is None or infor == '' or infor[0] not in [WorkloadType.BACKWARD.value, WorkloadType.WEIGHT.value, WorkloadType.FORWARD.value, WorkloadType.RECOMPUTE.value]:
            continue
        workload_type, microbatch_id, stage_id, dp_rank = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        dp_rank = int(dp_rank)
        pp_rank = _get_pp_rank_by_placement(stage_id, stage_placement)
        chunk_id = _get_chunk_by_stage(stage_id, stage_placement)
        workload_infor = {"workload_type":workload_type, "microbatch_id":microbatch_id, "stage_id":stage_id, "chunk_id":chunk_id, \
                               "start_time":start_time, "end_time":end_time, "source_dp_rank":int(microbatch_id/num_microbatches)}
        # pp_rank_workloads[dp_rank][pp_rank].append((workload_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
        all_rank_workloads[dp_rank][pp_rank].append(workload_infor)
    
    recomp_stages = set()
    if dp_size == 1 :
        for d in range(pp_size):
            f_num, b_num, w_num, r_num, r_stages = count_workloads(all_rank_workloads[0][d])
            each_workloads_num = len(stage_placement[d])*num_microbatches
            assert r_num == len(r_stages)*num_microbatches, f'r_num:{r_num} must be equal to r_stages:{r_stages}*num_microbatces:{num_microbatches}'
            assert f_num == each_workloads_num and b_num == each_workloads_num and (w_num ==0 or w_num == each_workloads_num), f'rank: {d}, right_num: {each_workloads_num}, f_num: {f_num}, b_num: {b_num}, w_num: {w_num}'
            all_rank_workloads[0][d].sort(key=lambda x: x["start_time"])
            recomp_stages = recomp_stages.union(r_stages)
        #     #print(f'{pp_rank_workloads[d]},')
        #print(']')
    #elif dp_size>1:#TODO,sum by stage id 
    recomp_stages = list(recomp_stages)
    # print(all_rank_workloads)
    return all_rank_workloads,recomp_stages, max_end_time

def find_mismatch(matrix):

    
    # 定义指令对应关系
    pair = {
        "SA": "RA",
        "RA": "SA",
        "SG": "RG",
        "RG": "SG"
    }
    n = len(matrix)
    mismatches = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            list_ij = matrix[i][j]
            list_ji = matrix[j][i]
            if len(list_ij) != len(list_ji):
                print(f" num_comm are differrent between device_{i}:{len(list_ij)}, and device_{j}:{len(list_ji)}")
            # print(f"matrix[{i}][{j}]:{list_ij}")
            # print(f"matrix[{j}][{i}]:{list_ji}")
            for k in range(max(len(list_ij), len(list_ji))):
                if k >= len(list_ij) or k >= len(list_ji):
                    continue
                cmd = list_ij[k]
                expected = pair.get(cmd)
                # if expected is None:
                #     continue  # 不是要检查的指令
                if list_ji[k] != expected:
                    mismatches.append({
                        "i": i,
                        "j": j,
                        "k": k,
                        "cmd": cmd,
                        "expected": expected,
                        "actual": list_ji[k]
                    })
    return mismatches

def search_by_infor(workloads,op,microbatch_id,stage_id,source_dp_rank):
    for s_index, workload in enumerate(workloads):
        s_op, s_microbatch_id, s_stage_id, s_source_dp_rank = workload["workload_type"], workload["microbatch_id"], workload["stage_id"], workload["source_dp_rank"]
        if s_source_dp_rank == source_dp_rank and s_op == op and s_microbatch_id == microbatch_id and stage_id == s_stage_id:
            return s_index

def search_by_time(workloads, time):
    for s_index,workload in enumerate(workloads):
        s_end_time = workload["end_time"]
        if s_end_time >= time:
            return s_index
    return len(workloads)

def generate_comm_martix_(comm_graph, comp_graph, dp_size, pp_size):
    # dir = 'InternEvo/pp_ranks_operations/'
    # os.makedirs(dir, exist_ok=True)
    device_num = dp_size*pp_size
    comm_graph_martix = [[[] for __ in range(device_num)] for _ in range(device_num)]
    for dp_rank,comm_graph_ in enumerate(comm_graph):
        for pp_rank, ops in enumerate(comm_graph_):
            source_device_id =  dp_rank*pp_size+pp_rank
            # comps = comp_graph[dp_rank][pp_rank]
            # jsonpath = dir+"pp"+str(pp_rank)+"_ops.json"
            for workload_index, comms in enumerate(ops):
                for comm in comms:
                    op_type, _, match_dp_rank, match_pp_rank, stage_id, chunk_id, microbatch_id, source_dp_rank, match_workload_index  = comm
                    match_device_id = match_dp_rank*pp_size+match_pp_rank
                    comm_graph_martix[source_device_id][match_device_id].append((op_type)) 
            #     write_json(jsonpath,{"operation":op_type, "local_rank":pp_rank, "workload_index":workload_index, "match_rank":match_pp_rank,"match_workload_index":match_workload_index, "source_stage_id":stage_id,"microbatch_id":microbatch_id}) 
            # if workload_index< len(comps):
            #     op, microbatch_id, stage_id, chunk_id, _, _ = comps[workload_index]
            #     json_content = {"workload_type": op, "local_rank": pp_rank, "workload_id": workload_index, "chunk_id": chunk_id, "stage_id": stage_id, "microbatch_id": microbatch_id, "operation": "compute"}
            #     write_json(jsonpath,json_content)
    return comm_graph_martix

def generate_comm_graph(comp_graph, stage_placement, max_end_time, send_immediately, dp_size, pp_size, transfer_info):
    #comp_graph 是之前生成的计算图
    # 初始化通信图,comm_graph[pp_rank][workload]表示计算操作前需要进行的通信操作list
    comm_graph = [[[[] for _ in range(len(comp_graph[i][j])+1)]for j in range(pp_size)] for i in range(dp_size)]
    processed_comp_graph = [[[False for _ in range(len(comp_graph[i][j]))] for j in range(pp_size)] for i in range(dp_size)]
    #R表示在comp op前需要接收的，S表示comp后要发送的，所以每次可以进行合并
    max_stage_id = max([stage_id for row in stage_placement for stage_id in row])
    min_stage_id = min([stage_id for row in stage_placement for stage_id in row])
    time = 0
    while(time <= max_end_time):
        time += 1
        for dp_rank,comp_graph_ in enumerate(comp_graph):
            for pp_rank, workloads in enumerate(comp_graph_):
                for current_workload_index, current_workload in enumerate(workloads):
                    if not processed_comp_graph[dp_rank][pp_rank][current_workload_index]:
                        workload_type, microbatch_id, stage_id, chunk_id, end_time, source_dp_rank = current_workload["workload_type"], current_workload["microbatch_id"], \
                            current_workload["stage_id"], current_workload["chunk_id"], current_workload["end_time"], current_workload["source_dp_rank"]
                        if time >= end_time:
                            processed_comp_graph[dp_rank][pp_rank][current_workload_index] = True
                            dst_pp_rank = None
                            if (workload_type == WorkloadType.FORWARD.value and stage_id<max_stage_id) or (workload_type == WorkloadType.BACKWARD.value and stage_id>min_stage_id):
                                #search which workload before
                                send_location = current_workload_index #因为要在下一个操作前发送
                                send_interval = None
                                recv_location = None
                                recv_interval = None
                                wait_time = float('inf')
                                recv_start_index = None
                                recv_end_index = None
                                send_end_index = None

                                recv_dp_rank = None
                                
                                if workload_type == WorkloadType.FORWARD.value:
                                    dst_pp_rank = _get_pp_rank_by_placement(stage_id+1,stage_placement)
                                    if dst_pp_rank is not None:
                                        if dst_pp_rank != pp_rank:
                                            recv_dp_rank = dp_rank
                                        if source_dp_rank==dp_rank and len(transfer_info[dp_rank])>0:#发送到代算的dp
                                            for transfer_info_ in transfer_info[dp_rank]:
                                                if transfer_info_["stage_id"] == stage_id+1 and microbatch_id in transfer_info_["microbatch_ids"]:
                                                    recv_dp_rank = transfer_info_["dst_dp_rank"]

                                        elif source_dp_rank!=dp_rank: #代算的workload算完后 发送到原始dp
                                            for transfer_info_ in transfer_info[source_dp_rank]:
                                                if transfer_info_["dst_dp_rank"] == dp_rank and transfer_info_["stage_id"] == stage_id and microbatch_id in transfer_info_["microbatch_ids"]:
                                                    recv_dp_rank = source_dp_rank

                                elif workload_type == WorkloadType.BACKWARD.value:
                                    dst_pp_rank = _get_pp_rank_by_placement(stage_id-1,stage_placement)
                                    if dst_pp_rank is not None:
                                        if dst_pp_rank != pp_rank:
                                            recv_dp_rank = dp_rank
                                        if source_dp_rank==dp_rank and len(transfer_info[dp_rank])>0:#发送到代算的dp
                                            for transfer_info_ in transfer_info[dp_rank]:
                                                if transfer_info_["stage_id"] == stage_id-1 and microbatch_id in transfer_info_["microbatch_ids"]:
                                                    recv_dp_rank = transfer_info_["dst_dp_rank"]
                                        elif source_dp_rank!=dp_rank:#代算的workload算完后 发送到原始dp
                                            for transfer_info_ in transfer_info[source_dp_rank]:
                                                if transfer_info_["dst_dp_rank"] == dp_rank and transfer_info_["stage_id"] == stage_id and microbatch_id in transfer_info_["microbatch_ids"]:
                                                    recv_dp_rank = source_dp_rank                                  

                                if recv_dp_rank is not None and dst_pp_rank is not None:
                                    recv_workloads=comp_graph[recv_dp_rank][dst_pp_rank]
                                    if workload_type == WorkloadType.FORWARD.value:
                                        recv_end_index = search_by_infor(recv_workloads,workload_type,microbatch_id,stage_id+1,source_dp_rank)
                                    elif workload_type == WorkloadType.BACKWARD.value:
                                        recv_end_index = search_by_infor(recv_workloads,workload_type,microbatch_id,stage_id-1,source_dp_rank)
                                    assert recv_end_index is not None, print(f"recv_workloads:{recv_workloads},current_workload:{current_workload}, recv_dp_rank:{recv_dp_rank}, dst_pp_rank{dst_pp_rank}")
                                    recv_end_index += 1
                                    #确定接收的pp_rank有哪些接收区间
                                    recv_start_index = search_by_time(recv_workloads, end_time) #因为是要检索comp 前的区间
                                    #确定发送的pp_rank有哪些
                                    recv_op_start_time =None
                                    if recv_end_index >= len(recv_workloads):
                                        recv_op_start_time = recv_workloads[-1]["start_time"]
                                    else:
                                        recv_op_start_time = recv_workloads[recv_end_index]["start_time"]
                                    send_end_index = current_workload_index+1
                                    if not send_immediately:
                                        send_end_index = search_by_time(workloads, recv_op_start_time)+1

                                    # print(f"send_location:{send_location}, send_end_index:{send_end_index}, recv_start_index:{recv_start_index}, recv_end_index:{recv_end_index}")
                                    for s_index in range(current_workload_index, send_end_index):
                                        if s_index < len(workloads)-1:
                                            send_interval = (workloads[s_index]["end_time"],workloads[s_index+1]["start_time"])
                                        else:
                                            send_interval = (workloads[-1]["end_time"],float('inf'))

                                        for r_index in range(recv_start_index, recv_end_index):
                                            r_start_time =  recv_workloads[r_index]["start_time"]
                                            if r_index == 0 :
                                                recv_interval = (float('-inf'),r_start_time)
                                            else:
                                                recv_interval = (recv_workloads[r_index-1]["end_time"],r_start_time)
                                            wait_time_ = interval_distance(send_interval, recv_interval)
                                            # print(f"wait_time_:{wait_time_}")
                                            if wait_time_ < wait_time:
                                                wait_time = wait_time_
                                                recv_location = r_index
                                                send_location = s_index
                                            if wait_time == 0:
                                                break
                                            # if r_op == op and r_microbatch_id == microbatch_id :
                                            #     if (op == WorkloadType.FORWARD.value and r_stage_id == stage_id+1) or (op == WorkloadType.BACKWARD.value and r_stage_id == stage_id-1):
                                            #         break
                                        if wait_time == 0:
                                            break
                                    assert send_interval is not None
                                    assert recv_location is not None
                                    assert dst_pp_rank is not None
                                    # comm_graph[dst_pp_rank][recv_location]['R'].append((op, _, pp_rank, stage_id, chunk_id, microbatch_id,_))               
                                    # comm_graph[dst_pp_rank][recv_location][WorkloadType.BACKWARD.value].append((op, _, pp_rank, stage_id, chunk_id, microbatch_id,_))
                                    if workload_type == WorkloadType.FORWARD.value:
                                        comm_graph[recv_dp_rank][dst_pp_rank][recv_location].append(('RA', end_time, dp_rank, pp_rank, stage_id, chunk_id, microbatch_id, source_dp_rank, send_location+1))
                                        comm_graph[dp_rank][pp_rank][send_location+1].append(('SA', end_time,recv_dp_rank, dst_pp_rank, stage_id, chunk_id, microbatch_id,source_dp_rank, recv_location))   
                                    # comm_graph[pp_rank][send_location]['S'].append((op, _, dst_pp_rank, stage_id, chunk_id, microbatch_id,_))
                                    elif workload_type == WorkloadType.BACKWARD.value:
                                        comm_graph[recv_dp_rank][dst_pp_rank][recv_location].append(('RG', end_time, dp_rank, pp_rank, stage_id, chunk_id, microbatch_id,source_dp_rank, send_location+1))
                                        comm_graph[dp_rank][pp_rank][send_location+1].append(('SG', end_time, recv_dp_rank, dst_pp_rank, stage_id, chunk_id, microbatch_id,source_dp_rank,recv_location)) 
                        break
                            # recv_info = {}
                            # recv_info['recv_op'] = op
                            # recv_info['recv_pp_rank'] = pp_rank
                            # recv_info['recv_stage_id'] = stage_id
                            # recv_info['recv_chunk_id'] = chunk_id
                            # recv_info['recv_microbatch_id'] = microbatch_id
                            # comm_graph[dst_pp_rank][recv_location]['R'].append(recv_info)

                            # send_info = {}
                            # send_info['send_op'] = op
                            # send_info['send_pp_rank'] = pp_rank
                            # send_info['send_stage_id'] = stage_id
                            # send_info['send_chunk_id'] = chunk_id
                            # send_info['send_microbatch_id'] = microbatch_id
                            # comm_graph[pp_rank][send_location]['S'].append(recv_info)
    # comm_matrix = generate_comm_martix_(comm_graph,comp_graph,dp_size,pp_size)
    # print(f"wrong comm order:{find_mismatch(comm_matrix)}")
    return comm_graph

def generate_():
    stage_placement = ""
    input_str=""
    file_path = 'InternEvo/executor_config'
    os.makedirs(file_path,exist_ok=True)
    with open(file_path+'/placement.txt', 'r', encoding='utf-8') as file:
        stage_placement = file.read()
    with open(file_path+'/result.txt', 'r', encoding='utf-8') as file:
        input_str = file.read()
    stage_placement = json.loads(stage_placement)
    dp_size=2
    pp_size = len(stage_placement)

    transfer_info = None 
    with open(file_path+'/transfer_info.txt', 'r', encoding='utf-8') as file:
        transfer_info = file.read()
    transfer_info = ast.literal_eval(transfer_info)
    DP_Transfer = False
    for index,info in enumerate(transfer_info):
        if len(info) > 0:
            DP_Transfer = True
            break
    num_microbatches = pp_size*2
    send_immediately = False
    unified_scheduler, recomp_stages, max_end_time = order_result_mutichunk(input_str, stage_placement, num_microbatches, dp_size, pp_size)
    comm_graph = generate_comm_graph(unified_scheduler,stage_placement,max_end_time, send_immediately, dp_size, pp_size, transfer_info)
    scheduler_type = judge_scheduler_type(stage_placement)
    split_backward = judge_split_backward(unified_scheduler)
    last_stage = max(max(row) for row in stage_placement)
    first_stage = min(min(row) for row in stage_placement)
    pp_ranks_containing_last_stage = [i for i, row in enumerate(stage_placement) if last_stage in row]
    # self.Thepp_ranks_containg_first_stage = [i for i, row in enumerate(self.stage_placement) if self.first_stage in row]
    # result = {'num_microbatches':num_microbatches, 'pp_size':pp_size, \
    #           'stage_placement':stage_placement, 'scheduler_type': scheduler_type, 'split_backward':split_backward, \
    #           'first_stage':first_stage, 'last_stage':last_stage, 'pp_ranks_containing_last_stage':pp_ranks_containing_last_stage,\
    #            'unified_scheduler':unified_scheduler, 'comm_graph':comm_graph, 'recomp_stages':recomp_stages}
    # with open(file_path+'/runtime.json',WorkloadType.WEIGHT.value) as file:
    #     json.dump(result,file)
    # print(f'num_microbatches:{num_microbatches*dp_size}, pp_size:{pp_size}, stage_placement:{stage_placement}, scheduler_type:{scheduler_type}, split_backward:{split_backward}, \
    #       recomp_stages:{recomp_stages}')
    if DP_Transfer:
        return num_microbatches*dp_size, pp_size, stage_placement, scheduler_type ,\
                split_backward, unified_scheduler, comm_graph, first_stage ,\
                last_stage, pp_ranks_containing_last_stage, recomp_stages, dp_size, \
                DP_Transfer, num_microbatches
    else:
        return num_microbatches, pp_size, stage_placement, scheduler_type ,\
                split_backward, unified_scheduler, comm_graph, first_stage ,\
                last_stage, pp_ranks_containing_last_stage, recomp_stages, dp_size, \
                DP_Transfer, num_microbatches

if __name__ == '__main__':
    generate_()