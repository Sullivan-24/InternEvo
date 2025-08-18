import copy
import json
import os
from enum import Enum, IntEnum
import os
class Step(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'
class ModuleType(Enum):
    CHIMERA = 'Chimera'
    INTERLEAVED = 'Interleaved'
    VSHAPE = 'Vshape'
    ONEFONEB = '1f1b'
    ZBH1 = 'Zbh1'
    HET = 'Het'

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
    for device_stage in stage_placement:
        for chunk_id in range(len(device_stage)):
            if device_stage[chunk_id] == stage_id:
                return chunk_id

def _get_deviceid_by_alignment(stage_id: int, stage_placement:list) -> int:
    for device_id in range(len(stage_placement)):
        for stage_in_device in stage_placement[device_id]:
            if stage_in_device == stage_id:
                return device_id

def recvnum(comm_graph):
    #print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(comm_graph):
        recvF = 0
        recvB = 0
        for comm_op in comm_stage:
            for recvlistB in comm_op['B']:
                if recvlistB[0] == 'b':
                    recvB += 1
                elif recvlistB[0] == 'f':
                    recvF += 1
            for recvlistA in comm_op['A']:
                if recvlistA[0] == 'b':
                    recvB += 1
                elif recvlistA[0] == 'f':
                    recvF += 1
        #print(f"rank_id {rank_id}: recvF {recvF}, recvB {recvB}")
        ##print(f'{comm_stage},')
    #print(']')

def count_steps(steps):
    f_num = 0
    b_num = 0
    w_num = 0
    r_num = 0
    r_stages = set()
    for s in steps:
        step_type = s[0]
        if step_type == 'f':
            f_num += 1
            continue
        elif step_type == 'b':
            b_num += 1
        elif step_type == 'w':
            w_num += 1
        elif step_type == 'r':
            r_num += 1
            r_stages.add(s[2])
    return f_num, b_num, w_num, r_num , r_stages

def judge_placement_strategy(stage_placement):
    ranks = len(stage_placement)
    if ranks <= 1:
        return None
    num_chunks_per_device = set()
    sum_stageId_per_device = set()

    for i in range(ranks):
        num_chunks_per_device.add(len(stage_placement[i]))
        sum_stageId = sum(stage_placement[i])
        sum_stageId_per_device.add(sum_stageId)
    sum_stageId_per_device = sorted(list(sum_stageId_per_device))
    num_chunks_per_device = sorted(list(num_chunks_per_device))
    if len(num_chunks_per_device) > 1:
        return ModuleType.HET.value
    else:
        if num_chunks_per_device[0] == 1:
            return ModuleType.ONEFONEB.value
        elif len(sum_stageId_per_device) == 1:
            if sum_stageId_per_device[0] == ranks-1:
                return ModuleType.CHIMERA.value
            else:
                return ModuleType.VSHAPE.value
        elif 1< len(sum_stageId_per_device) < ranks :
            return ModuleType.HET.value
        else:# len(sum_stageId_per_device) == ranks
            num_chunks = num_chunks_per_device[0]
            for i in range(1,ranks):
                if sum_stageId_per_device[i] - sum_stageId_per_device[i-1] != num_chunks:
                    return ModuleType.HET.value
            return ModuleType.INTERLEAVED.value

def judge_split_backward(unified_scheduler):
    if unified_scheduler[0][-1][0] == Step.WEIGHT.value:
        return True
    else:
        return False

def write_json(jsonpath, content):
    with open(jsonpath, 'a',encoding='utf-8') as f:
        json.dump(content, f)
        f.write('\n')

def order_result_mutichunk(input: str, stage_placement: list, num_microbatches:int) -> None:
    device_steps = [[] for _ in range(len(stage_placement))]
    all_step = input.split('\n')
    ##print(all_step)
    max_end_time = 0
    for step in all_step:
        if step == '':
            continue
        start_time = float(step.split(',')[-2])
        end_time = float(step.split(',')[-1])
        if end_time > max_end_time:
            max_end_time = end_time
        infor = step.split(',')[0]
        if infor is None or infor == '' or infor[0] not in ['b','w','f','r']:
            continue
        step_type, microbatch_id, stage_id = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        device_id = _get_deviceid_by_alignment(stage_id, stage_placement)
        chunk_id = _get_chunk_by_stage(stage_id, stage_placement)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    #print('[')
    recomp_stages = set()
    for d in range(len(stage_placement)):
        f_num, b_num, w_num, r_num, r_stages = count_steps(device_steps[d])
        each_steps_num = len(stage_placement[d])*num_microbatches
        assert r_num == len(r_stages)*num_microbatches, f'r_num:{r_num} must be equal to r_stages:{r_stages}*num_microbatces:{num_microbatches}'
        assert f_num == each_steps_num and b_num == each_steps_num and (w_num ==0 or w_num == each_steps_num), f'rank: {d}, right_num: {each_steps_num}, f_num: {f_num}, b_num: {b_num}, w_num: {w_num}'
        device_steps[d].sort(key=lambda x: x[-2])
        recomp_stages = recomp_stages.union(r_stages)
    #     #print(f'{device_steps[d]},')
    #print(']')
    recomp_stages = list(recomp_stages)
    return device_steps, recomp_stages, max_end_time

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
                print(f" num_comm are differrent between device{i}:{len(list_ij)}, and device{j}:{len(list_ji)}")
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

def search_by_infor(steps,op,microbatch_id,stage_id):
    for s_index,step in enumerate(steps):
        s_op, s_microbatch_id, s_stage_id, _, s_start_time, s_end_time = step
        if s_op == op and s_microbatch_id == microbatch_id and stage_id == s_stage_id:
            return s_index

def search_by_time(steps, time):
    for s_index,step in enumerate(steps):
        s_op, s_microbatch_id, s_stage_id, _, s_start_time, s_end_time = step
        if s_end_time >= time:
            return s_index
    return len(steps)

def generate_comm_martix_(comm_graph, comp_graph):
    dir = 'InternEvo/devices_operations/'
    os.makedirs(dir, exist_ok=True)
    comm_graph_martix  = [[[] for __ in range(len(comm_graph))] for _ in range(len(comm_graph))]
    for device_id,ops in enumerate(comm_graph):
        comps = comp_graph[device_id]
        jsonpath = dir+"pp"+str(device_id)+"_ops.json"
        for step_index, comms in enumerate(ops):
            for comm in comms:
                op_type, _, match_device_id, stage_id, chunk_id, microbatch_id, match_step_index = comm
                comm_graph_martix[device_id][match_device_id].append((op_type)) 
            #     write_json(jsonpath,{"operation":op_type, "local_rank":device_id, "step_index":step_index, "match_rank":match_device_id,"match_step_index":match_step_index, "source_stage_id":stage_id,"microbatch_id":microbatch_id}) 
            # if step_index< len(comps):
            #     op, microbatch_id, stage_id, chunk_id, _, _ = comps[step_index]
            #     json_content = {"step_type": op, "local_rank": device_id, "step_id": step_index, "chunk_id": chunk_id, "stage_id": stage_id, "microbatch_id": microbatch_id, "operation": "compute"}
            #     write_json(jsonpath,json_content)
    return comm_graph_martix

def generate_comm_graph(comp_graph, stage_placement, max_end_time, send_immediately=False):
    #comp_graph 是之前生成的计算图
    # 初始化通信图,comm_graph[deviceid][step]表示计算操作前需要进行的通信操作list
    comm_graph = [[[] for __ in range(len(comp_graph[0])+1)] for _ in range(len(comp_graph))]
    processed_comp_graph = [[False for __ in range(len(comp_graph[i]))] for i in range(len(comp_graph))]
    #R表示在comp op前需要接收的，S表示comp后要发送的，所以每次可以进行合并
    max_stage_id = max([stage_id for row in stage_placement for stage_id in row])
    min_stage_id = min([stage_id for row in stage_placement for stage_id in row])
    time = 0
    while(time <= max_end_time):
        time += 1
        for device_id, stage_ops in enumerate(comp_graph):
            for current_op_index, current_op in enumerate(stage_ops):
                if not processed_comp_graph[device_id][current_op_index]:
                    op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
                    if time >= end_time:
                        processed_comp_graph[device_id][current_op_index] = True
                        dst_device_id = None
                        if (op == 'f' and stage_id<max_stage_id) or (op == 'b' and stage_id>min_stage_id):
                            if op == 'f':
                                dst_device_id = _get_deviceid_by_alignment(stage_id+1,stage_placement)
                            else:
                                dst_device_id = _get_deviceid_by_alignment(stage_id-1,stage_placement)
                            if dst_device_id is not None and  dst_device_id != device_id:
                                recv_steps = comp_graph[dst_device_id]
                                #search which step before
                                send_location = current_op_index #因为要在下一个操作前发送
                                send_interval = None
                                recv_location = None
                                recv_interval = None
                                wait_time = float('inf')
                                recv_start_index = None
                                recv_end_index = None
                                send_end_index = None

                                #确定接收的device有哪些接收区间
                                recv_start_index = search_by_time(recv_steps, end_time) #因为是要检索comp 前的区间
                                if op == 'f':
                                    recv_end_index = search_by_infor(recv_steps,op,microbatch_id,stage_id+1)+1
                                else:
                                    recv_end_index = search_by_infor(recv_steps,op,microbatch_id,stage_id-1)+1
                                #确定发送的device有哪些
                                recv_op_start_time = recv_steps[recv_end_index][-2]
                                send_end_index = current_op_index+1
                                if not send_immediately:
                                    send_end_index = search_by_time(stage_ops, recv_op_start_time)+1
                                if recv_start_index is None or recv_start_index>=recv_end_index:
                                    recv_start_index = recv_end_index-1
                                #print(f"device_id:{device_id}, stage_id:{stage_id}, chunk_id:{chunk_id}, microbatch_id:{microbatch_id}, op:{op}, end_time:{end_time}")
                                # print(f"send_location:{send_location}, send_end_index:{send_end_index}, recv_start_index:{recv_start_index}, recv_end_index:{recv_end_index}")
                                #做完就发
                                for s_index in range(current_op_index, send_end_index):
                                    if s_index < len(stage_ops)-1:
                                        send_interval = (stage_ops[s_index][-1],stage_ops[s_index+1][-2])
                                    else:
                                        send_interval = (stage_ops[s_index][-1],float('inf'))

                                    for r_index in range(recv_start_index, recv_end_index):
                                        r_start_time =  recv_steps[r_index][-2]
                                        if r_index == 0 :
                                            recv_interval = (float('-inf'),r_start_time)
                                        else:
                                            recv_interval = (recv_steps[r_index-1][-1],r_start_time)
                                        wait_time_ = interval_distance(send_interval, recv_interval)
                                        # print(f"wait_time:{wait_time} ; wait_time_:{wait_time_}")
                                        if wait_time_ < wait_time:
                                            wait_time = wait_time_
                                            recv_location = r_index
                                            send_location = s_index
                                        if wait_time == 0:
                                            break
                                        # if r_op == op and r_microbatch_id == microbatch_id :
                                        #     if (op == 'f' and r_stage_id == stage_id+1) or (op == 'b' and r_stage_id == stage_id-1):
                                        #         break
                                    if wait_time == 0:
                                        break
                                assert send_interval is not None
                                assert recv_location is not None
                                assert dst_device_id is not None
                                # comm_graph[dst_device_id][recv_location]['R'].append((op, _, device_id, stage_id, chunk_id, microbatch_id,_))               
                                # comm_graph[dst_device_id][recv_location]['B'].append((op, _, device_id, stage_id, chunk_id, microbatch_id,_))
                                if op == 'f':
                                    comm_graph[dst_device_id][recv_location].append(('RA', end_time, device_id, stage_id, chunk_id, microbatch_id, send_location+1))
                                    comm_graph[device_id][send_location+1].append(('SA', end_time, dst_device_id, stage_id, chunk_id, microbatch_id,recv_location))   
                                # comm_graph[device_id][send_location]['S'].append((op, _, dst_device_id, stage_id, chunk_id, microbatch_id,_))
                                else:
                                    comm_graph[dst_device_id][recv_location].append(('RG', end_time, device_id, stage_id, chunk_id, microbatch_id,send_location+1))
                                    comm_graph[device_id][send_location+1].append(('SG', end_time, dst_device_id, stage_id, chunk_id, microbatch_id,recv_location)) 
                    break
                        # recv_info = {}
                        # recv_info['recv_op'] = op
                        # recv_info['recv_pp_rank'] = device_id
                        # recv_info['recv_stage_id'] = stage_id
                        # recv_info['recv_chunk_id'] = chunk_id
                        # recv_info['recv_microbatch_id'] = microbatch_id
                        # comm_graph[dst_device_id][recv_location]['R'].append(recv_info)

                        # send_info = {}
                        # send_info['send_op'] = op
                        # send_info['send_pp_rank'] = device_id
                        # send_info['send_stage_id'] = stage_id
                        # send_info['send_chunk_id'] = chunk_id
                        # send_info['send_microbatch_id'] = microbatch_id
                        # comm_graph[device_id][send_location]['S'].append(recv_info)
    # comm_matrix = generate_comm_martix_(comm_graph,comp_graph)
    # print(f"wrong comm order:{find_mismatch(comm_matrix)}") 
    return comm_graph

def search_step(steps,now_index):
    pre_fetch_w_index = []
    end_index = len(steps)-1
    for step_index in range(now_index):
        step_type, microbatch_id, stage_id, chunk_id, start_time, end_time = steps[step_index]
        if step_type == 'b':
            for find_w_index in range(now_index+1, end_index+1):# not include self
                step_type_w, microbatch_id_w, stage_id_w, chunk_id_w, start_time_w, end_time_w = steps[find_w_index]
                if step_type_w == 'w' and microbatch_id_w == microbatch_id and stage_id == stage_id_w:
                    pre_fetch_w_index.append(find_w_index)
    return pre_fetch_w_index

def pre_fetch_w(unified_scheduler):
    all_pre_fetch_w = []
    for pipe_rank in range(len(unified_scheduler)):
        pipe_rank_pre = []
        pipe_rank_steps = unified_scheduler[pipe_rank]
        for step_index in range(len(pipe_rank_steps)):
            pre_fetch_w_index = search_step(pipe_rank_steps,step_index)
            pipe_rank_pre.append(pre_fetch_w_index)
        all_pre_fetch_w.append(pipe_rank_pre)
    return all_pre_fetch_w

def generate_():
    stage_placement = ""
    input_str=""
    file_path = '/cpfs01/user/matenghui/InternEvo'
    with open(file_path+'/placement.txt', 'r', encoding='utf-8') as file:
        stage_placement = file.read()
    with open(file_path+'/result.txt', 'r', encoding='utf-8') as file:
        input_str = file.read()
    stage_placement = json.loads(stage_placement)

    pp_size = len(stage_placement)
    num_microbatches = 12
    unified_scheduler, recomp_stages, max_end_time = order_result_mutichunk(input_str,stage_placement,num_microbatches)
    send_immediately = True#
    comm_graph = generate_comm_graph(unified_scheduler,stage_placement,max_end_time,send_immediately)
    all_pre_fetch_w = pre_fetch_w(unified_scheduler)    
    placement_strategy = judge_placement_strategy(stage_placement)
    split_backward = judge_split_backward(unified_scheduler)
    last_stage = max(max(row) for row in stage_placement)
    first_stage = min(min(row) for row in stage_placement)
    Devices_containing_last_stage = [i for i, row in enumerate(stage_placement) if last_stage in row]
    # self.TheDevices_containg_first_stage = [i for i, row in enumerate(self.stage_placement) if self.first_stage in row]
    result = {'num_microbatches':num_microbatches, 'pp_size':pp_size, \
              'stage_placement':stage_placement, 'placement_strategy': placement_strategy, 'split_backward':split_backward, \
              'first_stage':first_stage, 'last_stage':last_stage, 'Devices_containing_last_stage':Devices_containing_last_stage,\
               'unified_scheduler':unified_scheduler, 'comm_graph':comm_graph}
    # with open(file_path+'/runtime.json','w') as file:
    #     json.dump(result,file)
    # print(f'num_microbatches:{num_microbatches}, pp_size:{pp_size}, stage_placement:{stage_placement}, placement_strategy:{placement_strategy}, split_backward:{split_backward}, \
    #       recomp_stages:{recomp_stages},placement_strategy:{placement_strategy},all_pre_fetch_w:{all_pre_fetch_w}')
    return num_microbatches, pp_size, stage_placement, placement_strategy ,\
            split_backward, unified_scheduler, comm_graph, first_stage ,\
            last_stage, Devices_containing_last_stage, recomp_stages, all_pre_fetch_w

if __name__ == '__main__':
    generate_()