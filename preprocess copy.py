import copy
import json
from enum import Enum, IntEnum
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

def distence(point,begin,end):
    if point < begin:
        return begin - point
    elif point > end:
        return point - end
    else:
        return 0
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
                return
            for k in range(len(list_ij)):
                cmd = list_ij[k][0]
                expected = pair.get(cmd)
                if expected is None:
                    continue  # 不是要检查的指令
                if list_ji[k][0] != expected:
                    mismatches.append({
                        "i": i,
                        "j": j,
                        "k": k,
                        "cmd": cmd,
                        "expected": expected,
                        "actual": list_ji[k][0]
                    })
    return mismatches

def interval_distance(interval1, interval2):
    """
    计算两个区间之间的距离。如果区间重叠返回0。
    区间格式为 (start, end)，假设 start <= end。
    """
    a_start, a_end = interval1
    b_start, b_end = interval2

    # 如果区间有重叠，距离为 0
    if a_end >= b_start and b_end >= a_start:
        return 0
    else:
        # 计算两个区间之间的距离
        return max(b_start - a_end, a_start - b_end)
    
def next_sa_sg_idx(lst, start):
    """返回lst中从start起第一个SA/SG的位置不含start找不到则返回len(lst)"""
    for idx in range(start+1, len(lst)):
        if lst[idx][0] in ("SA", "SG"):
            return idx
    return len(lst)


def _get_chunk_by_stage(stage_id: int,stage_alignment:list) -> int:
    for device_stage in stage_alignment:
        for chunk_id in range(len(device_stage)):
            if device_stage[chunk_id] == stage_id:
                return chunk_id
def _get_deviceid_by_alignment(stage_id: int, stage_alignment:list) -> int:
    for device_id in range(len(stage_alignment)):
        for stage_in_device in stage_alignment[device_id]:
            if stage_in_device == stage_id:
                return device_id
def recvFromSameDevice(stage_alignment):
    recvFfromSameDevice = []
    recvBfromSameDevice = []
    for stages in stage_alignment:
        for stage in stages:
            if stage-1 in stages:
                recvFfromSameDevice.append(stage)
            if stage+1 in stages:
                recvBfromSameDevice.append(stage)
    return recvFfromSameDevice,recvBfromSameDevice
def SendToSameDevice(stage_alignment):
    sendFtoSameDevice = []
    sendBtoSameDevice = []
    for stages in stage_alignment:
        for stage in stages:
            if stage+1 in stages:
                sendFtoSameDevice.append(stage)
            if stage-1 in stages:
                sendBtoSameDevice.append(stage)
    return sendFtoSameDevice,sendBtoSameDevice

def recvnum(communication_graph):
    print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(communication_graph):
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
        print(f"rank_id {rank_id}: recvF {recvF}, recvB {recvB}")
        #print(f'{comm_stage},')
    print(']')

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

def dfs(op, rec_stack,visited,communication_graph):
    key = op['Infor']
    if key in rec_stack:
        return True, -1  # 发现环，返回 True 和无效的 index
    if key in visited:
        return False, -1  # 已经访问过，无需继续

    visited.add(key)
    rec_stack.add(key)

    for idx, a_task in enumerate(op['A']):
        _, _, recv_device_id, recv_stage_id, _, recv_microbatch_id, recv_index = a_task
        # 找到对应的通信任务
        has_cycle, cycle_index = dfs(communication_graph[recv_device_id][recv_index],rec_stack, visited, communication_graph,)
        if has_cycle:
            # 如果发现环，返回 True 和当前任务的 index
            return True, idx
    rec_stack.remove(key)
    return False, -1  # 不存在环

def judge_scheduler_type(stage_placement):
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
        if len(sum_stageId_per_device) == 1:
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
def order_result_mutichunk(input: str, stage_alignment: list, num_microbatches:int) -> None:
    device_steps = [[] for _ in range(len(stage_alignment))]
    all_step = input.split('\n')
    #print(all_step)
    for step in all_step:
        if step == '':
            continue
        start_time = float(step.split(',')[-2])
        end_time = float(step.split(',')[-1])
        infor = step.split(',')[0]
        if infor is None or infor == '' or infor[0] not in ['b','w','f','r']:
            continue
        step_type, microbatch_id, stage_id = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        device_id = _get_deviceid_by_alignment(stage_id, stage_alignment)
        chunk_id = _get_chunk_by_stage(stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    # print('[')
    recomp_stages = set()
    for d in range(len(stage_alignment)):
        f_num, b_num, w_num, r_num, r_stages = count_steps(device_steps[d])
        each_steps_num = len(stage_alignment[d])*num_microbatches
        assert r_num == len(r_stages)*num_microbatches, f'r_num:{r_num} must be equal to r_stages:{r_stages}*num_microbatces:{num_microbatches}'
        assert f_num == each_steps_num and b_num == each_steps_num and (w_num ==0 or w_num == each_steps_num), f'rank: {d}, right_num: {each_steps_num}, f_num: {f_num}, b_num: {b_num}, w_num: {w_num}'
        device_steps[d].sort(key=lambda x: x[-2])
        recomp_stages = recomp_stages.union(r_stages)
    #     print(f'{device_steps[d]},')
    # print(']')
    return device_steps,recomp_stages
def comm_graph_muti_chunk(grouped_data, stage_alignment):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []
    # 构建邻接矩阵，表示两两rank之间的通信list
    communication_matrix = [[[] for __ in range(len(stage_alignment))] for _ in range(len(stage_alignment))]
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    min_stage_id = min([stage_id for row in stage_alignment for stage_id in row])
    for device_id, stage_ops in enumerate(grouped_data):
        stages = stage_alignment[device_id]
        needrecv = {}
        needrecv['F_stage'] = [s-1 for s in stages if s > min_stage_id and s-1 not in stages]
        needrecv['F_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['F_stage']]
        needrecv['B_stage'] = [s+1 for s in stages if s < max_stage_id and s+1 not in stages]
        needrecv['B_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['B_stage']]
        #print(needrecv)
        communication_stage = []
        # 标记已经接收的操作
        received_prev_stage = set()  # 记录每个 stage 中已经接收的 f 操作
        received_next_stage = set()  # 记录每个 stage 中已经接收的 b 操作

        current_op_comm_add_in_next = ''
        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, stage_id, microbatch_id)
            comm_op['B'] = []#computation前的通信instructions list
            comm_op['A'] = []#computation后的通信instructions list弃用，只有最后一个使用
            recv_interval_B_start = stage_ops[m - 1][-1] if m > 0 else 0
            recv_interval_B_end = start_time
            recv_interval_B = [recv_interval_B_start, recv_interval_B_end]
            recv_interval_A_start = end_time
            recv_interval_A_end = stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else float('inf')
            recv_interval_A = [recv_interval_A_start, recv_interval_A_end]

            for i in range(len(needrecv['F_stage'])):
                recvFstage_id = needrecv['F_stage'][i]
                recvFdevice_id = needrecv['F_device'][i]
                prev_stage_ops = grouped_data[recvFdevice_id]
                for n, prev_op in enumerate(prev_stage_ops):
                    prev_op_name, prev_microbatch_id, prev_stage_id, prev_chunk_id, prev_start_time, prev_end_time = prev_op
                    # if prev_start_time > end_time:
                    #  break
                    # 跳过已经接收的 f 操作
                    if prev_op in received_prev_stage:
                        continue
                    if prev_op_name != 'f' or prev_stage_id != recvFstage_id:
                        continue
                    # 如果prev_op操作为f且microbatch_id 相同
                    if op == 'f' and prev_microbatch_id == microbatch_id and prev_stage_id == stage_id-1:           
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))#n表示prev_op在grouped_data[recvFdevice_id]中的index
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    # 计算时间区间（发送方的通信区间）
                    send_interval_start = prev_end_time
                    send_interval_end = prev_stage_ops[n + 1][-2] if n + 1 < len(prev_stage_ops) else float('inf')
                    send_interval = [send_interval_start, send_interval_end]
                    
                    #计算接收op的前后时间与区间的距离
                    B_distance = interval_distance(recv_interval_B, send_interval)
                    A_distance = interval_distance(recv_interval_A, send_interval)

                    # 找到最小距离
                    if B_distance < A_distance:
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    else:
                        #最后一个step
                        if m == len(stage_ops)-1:
                            comm_op['A'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))
                            received_prev_stage.add(prev_op)
                        break

            # 处理下一个 stage (stage+1)
            for j in range(len(needrecv['B_stage'])):
                recvBstage_id = needrecv['B_stage'][j]
                recvBdevice_id = needrecv['B_device'][j]
                next_stage_ops = grouped_data[recvBdevice_id]

                for n, next_op in enumerate(next_stage_ops):
                    next_op_name, next_microbatch_id, next_stage_id, next_chunk_id, next_start_time, next_end_time = next_op
                    # if next_start_time > end_time:
                    #     break
                    if next_op_name != 'b' or recvBstage_id != next_stage_id:
                        continue
                    # 跳过已经接收的 f 操作
                    if next_op in received_next_stage :
                        continue
                    # 如果本次操作为b 且 microbatch_id 相同
                    if op == 'b' and next_microbatch_id == microbatch_id and next_stage_id == stage_id+1:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
                        received_next_stage.add(next_op)  # 标记为已接收
                        continue

                    # 计算时间区间（发送方的通信区间）
                    send_interval_start = next_end_time
                    send_interval_end = next_stage_ops[n + 1][-2] if n + 1 < len(next_stage_ops) else float('inf')
                    send_interval = [send_interval_start, send_interval_end]

                    B_distance = interval_distance(recv_interval_B, send_interval)
                    A_distance = interval_distance(recv_interval_A, send_interval)
                    if B_distance < A_distance:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
                        received_next_stage.add(next_op)
                        continue
                    else:
                        if m == len(stage_ops)-1:
                            comm_op['A'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
                            received_next_stage.add(next_op)
                        break
            current_op_sendcomm = ''
            dst_device = None
            if op == 'f' and stage_id < max_stage_id and stage_id+1 not in stages:
                current_op_sendcomm = 'SA'
                dst_device = _get_deviceid_by_alignment(stage_id+1, stage_alignment)
            elif op == 'b' and stage_id > min_stage_id and stage_id-1 not in stages:
                current_op_sendcomm = 'SG'
                dst_device = _get_deviceid_by_alignment(stage_id-1, stage_alignment)

            if len(comm_op['B'])>0: 
                comm_op['B'].sort(key=lambda x:x[1])
                for comm in comm_op['B']:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
                    if recv_op_type == 'f':
                        communication_matrix[device_id][recv_device_id].append(('RA',m,comm,False))
                    elif recv_op_type == 'b':
                        communication_matrix[device_id][recv_device_id].append(('RG',m,comm,False))
            #add op
            # if current_op_sendcomm != '':
            #     communication_matrix[device_id][dst_device].append((op,m,"_","_"))
                        
            if device_id%2 == 0:
                if current_op_sendcomm != '':
                    communication_matrix[device_id][dst_device].append((current_op_sendcomm,m, "_","_"))
                if len(comm_op['A'])>0:
                    comm_op['A'].sort(key=lambda x:x[1])
                    for comm in comm_op['A']:#for last op
                        recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
                        if recv_op_type == 'f':
                            communication_matrix[device_id][recv_device_id].append(('RA',m,comm,True))
                        elif recv_op_type == 'b':
                            communication_matrix[device_id][recv_device_id].append(('RG',m,comm,True))
            else:
                if len(comm_op['A'])>0:
                    comm_op['A'].sort(key=lambda x:x[1])
                    for comm in comm_op['A']:#for last op
                        recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
                        if recv_op_type == 'f':
                            communication_matrix[device_id][recv_device_id].append(('RA',m,comm,True))
                        elif recv_op_type == 'b':
                            communication_matrix[device_id][recv_device_id].append(('RG',m,comm,True))
                    # if current_op_sendcomm != '':
                    #     communication_matrix[device_id][dst_device].append((current_op_sendcomm,m, "_","_"))

                if current_op_comm_add_in_next != '':
                    device_id_, dst_device_, current_op_sendcomm_, current_op_index = current_op_comm_add_in_next
                    communication_matrix[device_id_][dst_device_].append((current_op_sendcomm_, current_op_index, "_","_"))
                    current_op_comm_add_in_next = ''
                if current_op_sendcomm != '':
                    current_op_comm_add_in_next = (device_id,dst_device, current_op_sendcomm, m)

            communication_stage.append(comm_op)
        communication_graph.append(communication_stage)
    print(find_mismatch(communication_matrix))
    communication_matrix,fix_communication_graph, actions = fix_matrix_keep_sa_sg_order(communication_matrix,grouped_data)
    print(actions)
    print(find_mismatch(communication_matrix))

    # communication_graph = detect_cross_deadlock_mutichunk(communication_graph,stage_alignment)
    recvnum(fix_communication_graph)
    #print(communication_graph)
    return fix_communication_graph

def fix_matrix_keep_sa_sg_order(matrix, grouped_data):
    pair = {"SA": "RA", "RA": "SA", "SG": "RG", "RG": "SG"}
    n = len(matrix)
    actions = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            li = matrix[i][j]
            lj = matrix[j][i]
            # index_i = 0
            # index_j = 0
            # while(index_i < len(li) and index_j< len(lj)):
            #     li_k_comm_ins = li[index_i][0]
            #     if li_k_comm_ins not in ("SA","SG","RA","RG"):
            #         index_i += 1
            #         continue
            #     lj_k_comm_ins = lj[index_j][0]
            #     if lj_k_comm_ins not in ("SA","SG","RA","RG"):
            #         index_j += 1
            #     li_expected_inlj = pair.get(li_k_comm_ins)
            #     lj_expected_inli = pair.get(lj_k_comm_ins)
            #     if lj_k_comm_ins != li_expected_inlj:
            #         next_block_i = len(li)
            #         # found_i = None
            #         if li_k_comm_ins in ("SA","SG") or lj_expected_inli in ("SA","SG"): #S开头的instructions要保持原有的顺序,所以不能交换
            #             next_block_i = next_sa_sg_idx(li, k)

            #         next_block_j = len(lj)
            #         if lj_k_comm_ins in ("SA","SG") or li_expected_inlj in ("SA","SG"): #S开头的instructions要保持原有的顺序   
            #             next_block_j = next_sa_sg_idx(lj, k)
            #         end_index = max(next_block_i,next_block_j)
            #         swap_index = k+1
            #         have_swap = False           
            #         while(swap_index <= end_index):

            for k in range(len(li)):
                li_k_comm_ins = li[k][0]#
                if li_k_comm_ins not in ("SA","SG","RA","RG"):
                    continue
                li_expected_inlj = pair.get(li_k_comm_ins)
                lj_k_comm_ins = lj[k][0]
                lj_expected_inli = pair.get(lj_k_comm_ins)
                if lj_k_comm_ins != li_expected_inlj:
                    # 查找li[k+1:next_sa_sg_i]区间
                    next_block_i = len(li)
                    if li_k_comm_ins in ("SA","SG"):
                        if lj_expected_inli in ("SA","SG"): #S开头的instructions要保持原有的顺序
                            next_block_i = k #同为S通信，无法交换
                        else:
                            next_block_i = next_sa_sg_idx(li, k)-1
                    else:
                        if lj_expected_inli in ("SA","SG"):
                            next_block_i = next_sa_sg_idx(li, k)
                    
                    next_block_j = len(lj)
                    if lj_k_comm_ins in ("SA","SG"):
                        if li_expected_inlj in ("SA","SG"): #S开头的instructions要保持原有的顺序
                            next_block_j = k #同为S通信，无法交换
                        else:
                            next_block_j = next_sa_sg_idx(lj, k)-1
                    else:
                        if li_expected_inlj in ("SA","SG"):
                            next_block_j = next_sa_sg_idx(lj, k)

                    end_index = max(next_block_i,next_block_j)
                    swap_index = k+1
                    have_swap = False
                    # print(f"swap_index:{swap_index}, next_block_i:{next_block_i}, next_block_j:{next_block_j}")
                    # print(f"li_k_comm_ins:{li_k_comm_ins}, li_expected_inlj:{li_expected_inlj}, lj_k_comm_ins:{lj_k_comm_ins}, lj_expected_inli:{lj_expected_inli}")
                    while(swap_index <= end_index):
                        if swap_index <= next_block_i and pair.get(li[swap_index][0]) == lj_k_comm_ins:
                            li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA = li[k]
                            li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA = li[swap_index]
                            if li_k_comm_ins in ("SA","SG") or li_swap_comm in ("SA","SG"):
                                if li_k_comm_ins in ("SA","SG"):#前面是S的comm instrcution，后面要交换的是R的comm instruction
                                    if i%2 == 0: #偶数rank，算发收
                                        for adapt_index in range(k+1, swap_index):#不包含swap_index
                                            li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA = li[adapt_index]
                                            li[adapt_index] = (li_adapt_index_comm_ins, li_k_op_step, li_adapt_index_comm,li_adapt_index_endA)
                                        li[k] = (li_swap_comm_ins, li_k_op_step, li_swap_comm, li_swap_endA)
                                        li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA)
                                    else:
                                        for adapt_index in range(k+1, swap_index):
                                            li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA = li[adapt_index]
                                            li[adapt_index] = (li_adapt_index_comm_ins, li_k_op_step+1, li_adapt_index_comm, li_adapt_index_endA)
                                        li[k] = (li_swap_comm_ins, li_k_op_step+1, li_swap_comm, li_swap_endA)
                                        li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA)
                                else:#前面是R的comm instrcution，后面要交换的是S的comm instruction
                                    if i%2 == 0:
                                        for adapt_index in range(k+1, swap_index):
                                            li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA = li[adapt_index]
                                            li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step+1, li_adapt_index_comm,li_adapt_index_endA)
                                        li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA)
                                        li[swap_index] = (li_k_comm_ins, li_swap_op_step+1, li_k_comm, li_k_endA)
                                    else:
                                        for adapt_index in range(k+1, swap_index):
                                            li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA = li[adapt_index]
                                            li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step+1, li_adapt_index_comm, li_adapt_index_endA)
                                        li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA)
                                        li[swap_index] = (li_k_comm_ins, li_swap_op_step+1, li_k_comm, li_k_endA)
                            else:#前后要交换的都是R的comm instruction
                                li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA)
                                li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA)
                            # matrix[i][j][k], matrix[i][j][swap_index] = matrix[i][j][swap_index], matrix[i][j][k]
                            actions.append(f"swap matrix[{i}][{j}][{k}] <-> matrix[{i}][{j}][{swap_index}]")
                            have_swap = True
                            break
                        if next_block_j <= next_block_j and pair.get(lj[swap_index][0]) == li_k_comm_ins:
                            # lj[k], lj[swap_index] = lj[swap_index], lj[k]
                            # matrix[j][i][k], matrix[j][i][swap_index] = matrix[j][i][swap_index], matrix[j][i][k]
                            lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA = lj[k]
                            lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA = lj[swap_index]
                            if lj_k_comm_ins in ("SA","SG") or lj_swap_comm in ("SA","SG"):
                                if lj_k_comm_ins in ("SA","SG"):#前面是S的comm instrcution，后面要交换的是R的comm instruction
                                    if i%2 == 0: #偶数rank，算发收
                                        for adapt_index in range(k+1, swap_index):#不包含swap_index
                                            lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA = lj[adapt_index]
                                            lj[adapt_index] = (lj_adapt_index_comm_ins, lj_k_op_step, lj_adapt_index_comm,lj_adapt_index_endA)
                                        lj[k] = (lj_swap_comm_ins, lj_k_op_step, lj_swap_comm, lj_swap_endA)
                                        lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA)
                                    else:
                                        for adapt_index in range(k+1, swap_index):
                                            lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA = lj[adapt_index]
                                            lj[adapt_index] = (lj_adapt_index_comm_ins, lj_k_op_step+1, lj_adapt_index_comm, lj_adapt_index_endA)
                                        lj[k] = (lj_swap_comm_ins, lj_k_op_step+1, lj_swap_comm, lj_swap_endA)
                                        lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA)
                                else:#前面是R的comm instrcution，后面要交换的是S的comm instruction
                                    if i%2 == 0:
                                        for adapt_index in range(k+1, swap_index):
                                            lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA = lj[adapt_index]
                                            lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step+1, lj_adapt_index_comm,lj_adapt_index_endA)
                                        lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA)
                                        lj[swap_index] = (lj_k_comm_ins, lj_swap_op_step+1, lj_k_comm, lj_k_endA)
                                    else:
                                        for adapt_index in range(k+1, swap_index):
                                            lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA = lj[adapt_index]
                                            lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step+1, lj_adapt_index_comm, lj_adapt_index_endA)
                                        lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA)
                                        lj[swap_index] = (lj_k_comm_ins, lj_swap_op_step+1, lj_k_comm, lj_k_endA)
                            else:#前后要交换的都是R的comm instruction
                                lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA)
                                lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA)                          
                            
                            actions.append(f"swap matrix[{j}][{i}][{k}] <-> matrix[{j}][{i}][{swap_index}]")
                            have_swap = True
                            break
                        swap_index += 1
                    if not have_swap:
                        actions.append(f"no match found for matrix[{i}][{j}][{k}] and matrix[{j}][{i}][{k}]")
    #根据调整好顺序的邻接矩阵 ，按照step重新生成通信队列
    fix_communication_graph = []
    for device_id, stage_ops in enumerate(grouped_data):
        device_comms = []
        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, stage_id, microbatch_id)
            comm_op['B'] = []#computation前的通信instructions list
            comm_op['A'] = []
            device_comms.append(comm_op)
        fix_communication_graph.append(device_comms)

    for device_id in range(n):
        recv_list = fix_communication_graph[device_id]
        num_step = len(recv_list)
        for match_device in range(n):
            if device_id == match_device :
                continue
            comms = matrix[device_id][match_device]
            if device_id%2 ==1 :
                assert comms[-1][0] in ('SG','SA')
            for recv_comm_ in comms:
                recv_comm_ins, recv_comm_op_step, recv_comm, recv_comm_endA = recv_comm_
                if recv_comm_ins in ('RG','RA'):
                    if recv_comm_endA:
                        assert recv_comm_op_step == num_step
                        recv_list[recv_comm_op_step]['A'].append(recv_comm)
                    else:
                        recv_list[recv_comm_op_step]['B'].append(recv_comm)                 
    return matrix, fix_communication_graph, actions

def detect_cycle_deadlock_mutichunk(communication_graph, stage_alignment):
    sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    communication_graph_copy = copy.deepcopy(communication_graph)
    for rank_id, rank_ops in enumerate(communication_graph_copy):
        if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-收-发）
            continue
        len_rank_ops = len(rank_ops)#rank_ops = communication_graph[rank_id]
        rank_ops_copy = copy.deepcopy(rank_ops)
        for current_index, op in enumerate(rank_ops_copy):
            #op = rank_ops[current_index]
            op_type, stage_id, microbatch_id = op['Infor']
            #没有发送需求就不会有死锁
            if op_type == 'f':
                if stage_id in sendFtoSameDevice or stage_id == max_stage_id:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
            elif op_type == 'b':
                if stage_id in sendBtoSameDevice or stage_id == 0:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
            else:
                continue

            #判断环
            visited = set()  # 用于记录已经访问过的任务
            rec_stack = set()  # 用于记录当前递归栈中的任务
            has_cycle, cycle_index = dfs(op,rec_stack, visited,communication_graph_copy,)
            if has_cycle and cycle_index != -1 and current_index < len_rank_ops - 1:
                communication_graph[rank_id][current_index + 1]['B'].append(communication_graph[rank_id][current_index]['A'].pop(cycle_index))
                #print(f"cycle dead lock:{op['Infor']}")
    return communication_graph

def detect_cross_deadlock_mutichunk(communication_graph, stage_alignment):
    sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    communication_graph_copy = copy.deepcopy(communication_graph)
    for rank_id, rank_ops in enumerate(communication_graph_copy):

        len_rank_ops = len(rank_ops)
        rank_ops_copy = copy.deepcopy(rank_ops)
        for current_index, op in enumerate(rank_ops_copy):
            op_type, stage_id, microbatch_id = op['Infor']
            #没有发送需求就不会有死锁
            if op_type == 'f':
                if stage_id in sendFtoSameDevice or stage_id == max_stage_id:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
            elif op_type == 'b':
                if stage_id in sendBtoSameDevice or stage_id == 0:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
            else:
                continue

            if rank_id % 2 == 0: #判断偶数rank（算-发-收）
                #死锁场景
                #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
                needjude = [op['A']]
                if current_index + 1 < len_rank_ops:
                    next_op = rank_ops_copy[current_index + 1]
                    if len(next_op['B'])>0:
                        needjude += next_op['B']
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    recvDevice_op = copy.deepcopy(communication_graph_copy[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']+recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index = rc
                        if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                            goonjudge = False
                            break
                    judgedistance = 0
                    if not goonjudge:
                        break
                    while(goonjudge is True):
                        judgedistance += 1
                        if index+judgedistance < len(communication_graph_copy[recv_device_id]):
                            recvDevice_nextop_n = copy.deepcopy(communication_graph[recv_device_id][index+judgedistance])
                            recvDevice_nextop_nb = recvDevice_nextop_n['B']
                            for rnb, recvDevice_nextop_n_b in enumerate(recvDevice_nextop_nb):
                                next_recv_op_type_B, next_recv_end_time_B, next_recv_device_id_B, next_recv_stage_id_B, next_recv_chunk_id_B, next_recv_microbatch_id_B, next_index_B = recvDevice_nextop_n_b
                                if (next_recv_op_type_B,next_recv_stage_id_B,next_recv_microbatch_id_B) == op['Infor']:
                                    rnbresult = communication_graph[recv_device_id][index+judgedistance]['B'].pop(rnb)
                                    communication_graph[recv_device_id][index]['A'].append(rnbresult)
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            recvDevice_nextop_na = recvDevice_nextop_n['A']
                            for rna, recvDevice_nextop_n_a in enumerate(recvDevice_nextop_na):
                                next_recv_op_type_A, next_recv_end_time_A, next_recv_device_id_A, next_recv_stage_id_A, next_recv_chunk_id_A, next_recv_microbatch_id_A, next_index_A = recvDevice_nextop_n_a
                                if (next_recv_op_type_A,next_recv_stage_id_A,next_recv_microbatch_id_A) == op['Infor']:
                                    rnaresult = communication_graph[recv_device_id][index+judgedistance]['A'].pop(rna)
                                    communication_graph[recv_device_id][index]['A'].append(rnaresult)
                                    goonjudge = False
                                    break
                        if not goonjudge:
                            break
                        if index-judgedistance >= 0 :
                            recvDevice_nextop_bab = communication_graph[recv_device_id][index-judgedistance]
                            for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_chunk_id_bab, next_recv_microbatch_id_bab, next_index_bab= rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    if not goonjudge:
                        break
            else:
                needjude = list(reversed(op['A']))+op['B']#TODO
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_stream_id, recv_microbatch_id, index= judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    if recv_op_type == op_type and microbatch_id == recv_microbatch_id and ((op_type == "f" and recv_stage_id>stage_id) or (op_type == "b" and recv_stage_id<stage_id)):
                        communication_graph[rank_id][current_index+1]['B'].insert(0,judgeop)
                        if judgeop in op['A']:
                            communication_graph[rank_id][current_index]['A'].remove(judgeop)
                        else:
                            communication_graph[rank_id][current_index]['B'].remove(judgeop)
                        continue

                    recvDevice_op = copy.deepcopy(communication_graph[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index = rc
                        if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                            goonjudge = False
                            break
                    judgedistance = 0
                    for rv_,rv in enumerate(recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index = rv
                        if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                            communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index]['B'].pop(rv_))
                            goonjudge = False
                            break
                    if not goonjudge:
                        break
                    while(goonjudge is True):
                        judgedistance += 1
                        if index-judgedistance >= 0 :
                            recvDevice_nextop_bab = copy.deepcopy(communication_graph[recv_device_id][index-judgedistance])
                            for rnbab_id,rnbab in enumerate(recvDevice_nextop_bab['A']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['A'].pop(rnbab_id))
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            for rnbab__id,rnbab_ in enumerate(recvDevice_nextop_bab['B']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab_
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['B'].pop(rnbab__id))
                                    goonjudge = False
                                    break
                        if not goonjudge:
                            break
                        if index+judgedistance < len(communication_graph_copy[recv_device_id]) :
                            recvDevice_nextop_bab = communication_graph[recv_device_id][index+judgedistance]
                            for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    if not goonjudge:
                        break
    return communication_graph

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
    num_microbatches = 16
    unified_scheduler, recomp_stages = order_result_mutichunk(input_str,stage_placement,num_microbatches)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_placement)
    scheduler_type = judge_scheduler_type(stage_placement)
    split_backward = judge_split_backward(unified_scheduler)
    last_stage = max(max(row) for row in stage_placement)
    first_stage = min(min(row) for row in stage_placement)
    Devices_containing_last_stage = [i for i, row in enumerate(stage_placement) if last_stage in row]
    # self.TheDevices_containg_first_stage = [i for i, row in enumerate(self.stage_placement) if self.first_stage in row]
    result = {'num_microbatches':num_microbatches, 'pp_size':pp_size, \
              'stage_placement':stage_placement, 'scheduler_type': scheduler_type, 'split_backward':split_backward, \
              'first_stage':first_stage, 'last_stage':last_stage, 'Devices_containing_last_stage':Devices_containing_last_stage,\
               'unified_scheduler':unified_scheduler, 'comm_graph':comm_graph}
    with open(file_path+'/runtime.json','w') as file:
        json.dump(result,file)
    print(f'num_microbatches:{num_microbatches}, pp_size:{pp_size}, stage_placement:{stage_placement}, scheduler_type:{scheduler_type}, split_backward:{split_backward}, \
          recomp_stages:{recomp_stages}')
    return num_microbatches, pp_size, stage_placement, scheduler_type ,\
            split_backward, unified_scheduler, comm_graph, first_stage ,\
            last_stage, Devices_containing_last_stage, recomp_stages

if __name__ == '__main__':
    generate_()