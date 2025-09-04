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
    
def busy_wait_kernel(time):
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

# def recvFromSameDevice(stage_alignment):
#     recvFfromSameDevice = []
#     recvBfromSameDevice = []
#     for stages in stage_alignment:
#         for stage in stages:
#             if stage-1 in stages:
#                 recvFfromSameDevice.append(stage)
#             if stage+1 in stages:
#                 recvBfromSameDevice.append(stage)
#     return recvFfromSameDevice,recvBfromSameDevice

# def SendToSameDevice(stage_alignment):
#     sendFtoSameDevice = []
#     sendBtoSameDevice = []
#     for stages in stage_alignment:
#         for stage in stages:
#             if stage+1 in stages:
#                 sendFtoSameDevice.append(stage)
#             if stage-1 in stages:
#                 sendBtoSameDevice.append(stage)
#     return sendFtoSameDevice,sendBtoSameDevice

# def dfs(op, rec_stack, visited, comm_graph):
#     key = op['Infor']
#     if key in rec_stack:
#         return True, -1  # 发现环，返回 True 和无效的 index
#     if key in visited:
#         return False, -1  # 已经访问过，无需继续

#     visited.add(key)
#     rec_stack.add(key)

#     for idx, a_task in enumerate(op['A']):
#         _, _, recv_device_id, recv_stage_id, _, recv_microbatch_id, recv_index = a_task
#         # 找到对应的通信任务
#         has_cycle, cycle_index = dfs(comm_graph[recv_device_id][recv_index],rec_stack, visited, comm_graph)
#         if has_cycle:
#             # 如果发现环，返回 True 和当前任务的 index
#             return True, idx
#     rec_stack.remove(key)
#     return False, -1  # 不存在环

# def comm_graph_muti_chunk(comp_graph, stage_alignment):   # 假设 comp_graph 是之前生成的计算图
#     # 初始化通信图
#     comm_graph = []
#     # 构建邻接矩阵，表示两两rank之间的通信list

#     max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
#     min_stage_id = min([stage_id for row in stage_alignment for stage_id in row])
#     for device_id, stage_ops in enumerate(comp_graph):
#         stages = stage_alignment[device_id]
#         needrecv = {}
#         needrecv['F_stage'] = [s-1 for s in stages if s > min_stage_id and s-1 not in stages]
#         needrecv['F_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['F_stage']]
#         needrecv['B_stage'] = [s+1 for s in stages if s < max_stage_id and s+1 not in stages]
#         needrecv['B_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['B_stage']]
#         ##print(needrecv)
#         comm_stage = []
#         # 标记已经接收的操作
#         received_prev_stage = set()  # 记录每个 stage 中已经接收的 f 操作
#         received_next_stage = set()  # 记录每个 stage 中已经接收的 b 操作
        
#         for m, current_op in enumerate(stage_ops):
#             op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
#             comm_op = {}
#             comm_op['Infor'] = (op, stage_id, microbatch_id)
#             comm_op['B'] = []
#             comm_op['A'] = []
#             # if op != 'f' and op !='b':
#             #     communication_stage.append(comm_op)
#             #     continue
#             # 处理上一个 stage (stage-1)
#             for i in range(len(needrecv['F_stage'])):
#                 recvFstage_id = needrecv['F_stage'][i]
#                 recvFdevice_id = needrecv['F_device'][i]
#                 prev_stage_ops = comp_graph[recvFdevice_id]
#                 for n, prev_op in enumerate(prev_stage_ops):
#                     prev_op_name, prev_microbatch_id, prev_stage_id, prev_chunk_id, prev_start_time, prev_end_time = prev_op
#                     # if prev_start_time > end_time:
#                     #     break
#                     # 跳过已经接收的 f 操作
#                     if prev_op in received_prev_stage:
#                         continue
#                     if prev_op_name != 'f' or prev_stage_id != recvFstage_id:
#                         continue
#                     # 如果本次操作为f 且microbatch_id 相同
#                     if op == 'f' and prev_microbatch_id == microbatch_id and prev_stage_id == stage_id-1:           
#                         comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))
#                         received_prev_stage.add(prev_op)  # 标记为已接收
#                         continue
                    
#                     # 计算时间区间
#                     interval_start = prev_end_time
#                     interval_end = prev_stage_ops[n + 1][-2] if n + 1 < len(prev_stage_ops) else prev_end_time

#                     # 计算四个个时间点与区间的距离
#                     current_start_dist = distence(start_time, interval_start, interval_end)
#                     current_end_dist = distence(end_time, interval_start, interval_end)
#                     next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
#                     next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
#                     # 找到最小距离
#                     min_dist = min(current_start_dist, current_end_dist, next_start_dist,next_end_dist)
#                     #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收

#                     if min_dist == current_start_dist:
#                         comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))
#                         received_prev_stage.add(prev_op)  # 标记为已接收
#                         continue
#                     elif min_dist == current_end_dist:
#                         comm_op['A'].append(('f',prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n))
#                         received_prev_stage.add(prev_op)  # 标记为已接收
#                         continue
#                     elif min_dist == next_start_dist:
#                         break
#                     elif min_dist == next_end_dist:
#                         break

#             # 处理下一个 stage (stage+1)
#             for j in range(len(needrecv['B_stage'])):
#                 recvBstage_id = needrecv['B_stage'][j]
#                 recvBdevice_id = needrecv['B_device'][j]
#                 next_stage_ops = comp_graph[recvBdevice_id]

#                 for n, next_op in enumerate(next_stage_ops):
#                     next_op_name, next_microbatch_id, next_stage_id, next_chunk_id, next_start_time, next_end_time = next_op
#                     # if next_start_time > end_time:
#                     #     break
#                     if next_op_name != 'b' or recvBstage_id != next_stage_id:
#                         continue
#                     # 跳过已经接收的 f 操作
#                     if next_op in received_next_stage :
#                         continue
#                     # 如果本次操作为b 且 microbatch_id 相同
#                     if op == 'b' and next_microbatch_id == microbatch_id and next_stage_id == stage_id+1:
#                         comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
#                         received_next_stage.add(next_op)  # 标记为已接收
#                         continue

#                     # 计算时间区间
#                     interval_start = next_end_time
#                     interval_end = next_stage_ops[n + 1][-2] if n + 1 < len(next_stage_ops) else next_end_time

#                     # 计算四个时间点与区间的距离
#                     current_start_dist = distence(start_time, interval_start, interval_end)
#                     current_end_dist = distence(end_time, interval_start, interval_end)
#                     next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
#                     next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
#                     # 找到最小距离
#                     min_dist = min(current_start_dist, current_end_dist, next_start_dist, next_end_dist)
#                     #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收
                    
#                     if min_dist == current_start_dist:
#                         comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
#                         received_next_stage.add(next_op)
#                         continue
#                     elif min_dist == current_end_dist:
#                         comm_op['A'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n))
#                         received_next_stage.add(next_op)
#                         continue
#                     elif min_dist == next_start_dist:
#                         break
#                     elif min_dist == next_end_dist:
#                         break
#             comm_op['B'].sort(key=lambda x:x[1])
#             comm_op['A'].sort(key=lambda x:x[1])
#             comm_stage.append(comm_op)
#         comm_graph.append(comm_stage)
#     # 
#     comm_matrix = generate_comm_martix(comm_graph,stage_alignment,comp_graph)
#     #print(f"wrong comm order:{find_mismatch(comm_matrix)}")
#     # comm_graph, actions = fix_matrix_keep_sa_sg_order(comm_matrix,comm_graph,)
#     comm_graph,actions = fix_matrix_keep_sa_sg_order(comm_matrix,comm_graph)
#     # comm_graph = detect_cross_deadlock_mutichunk(comm_graph,stage_alignment)
#     # generate_ops_josn(comm_graph,stage_alignment,comp_graph)
#     # comm_graph = detect_cycle_deadlock_mutichunk(comm_graph,stage_alignment)
#     # print(f"fix actions:{actions}")
#     print(f"test fixed comm graph:{find_mismatch(generate_comm_martix(comm_graph, stage_alignment, comp_graph))}")
#     return comm_graph

# def fix_matrix_keep_sa_sg_order(matrix, comm_graph):
#     pair = {"SA": "RA", "RA": "SA", "SG": "RG", "RG": "SG"}
#     n = len(matrix)
#     actions = []
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             li = matrix[i][j]
#             lj = matrix[j][i]
#             device_comms_li = comm_graph[i]
#             device_comms_lj = comm_graph[j]
#             for k in range(len(li)):
#                 # li_k_comm_ins = li[k][0]
#                 li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA, li_k_before_op_recv = li[k]
#                 li_expected_inlj = pair.get(li_k_comm_ins)
                
#                 # lj_k_comm_ins = lj[k][0]
#                 lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv = lj[k]
#                 lj_expected_inli = pair.get(lj_k_comm_ins)
#                 if lj_k_comm_ins != li_expected_inlj:
#                     # 查找li[k+1:next_sa_sg_i]区间
#                     next_block_i = len(li)
#                     if li_k_comm_ins in ("SA","SG"):
#                         if lj_expected_inli in ("SA","SG"): #S开头的instructions要保持原有的顺序
#                             next_block_i = k #同为S通信，无法交换
#                         else:
#                             next_block_i = next_sa_sg_idx(li, k)-1
#                     else:
#                         if lj_expected_inli in ("SA","SG"):
#                             next_block_i = next_sa_sg_idx(li, k)
#                     # if li_k_op_step == li_k_before_op_recv:
#                     #     next_block_i = k
#                     next_block_j = len(lj)
#                     if lj_k_comm_ins in ("SA","SG"):
#                         if li_expected_inlj in ("SA","SG"): #S开头的instructions要保持原有的顺序
#                             next_block_j = k #同为S通信，无法交换
#                         else:
#                             next_block_j = next_sa_sg_idx(lj, k)-1
#                     else:
#                         if li_expected_inlj in ("SA","SG"):
#                             next_block_j = next_sa_sg_idx(lj, k)
#                     # if lj_k_op_step == lj_k_before_op_recv:
#                     #     next_block_j = k
#                     end_index = max(next_block_i,next_block_j)
#                     swap_index = k+1
#                     have_swap = False
#                     #print(f"--------------------------------------------")
#                     while(swap_index <= end_index):
#                         if swap_index <= next_block_i and pair.get(li[swap_index][0]) == lj_k_comm_ins:
#                             have_swap = True
#                             li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv = li[swap_index]
#                             if li_k_comm_ins in ("SA","SG"):#前面是S的comm instrcution，后面要交换的是R的comm instruction
#                                 if i%2 == 0: #偶数rank，算发收
#                                     device_comms_li[li_k_op_step]['B'].append(li_swap_comm)
#                                     if li_swap_endA:
#                                         device_comms_li[li_swap_op_step]['A'].remove(li_swap_comm)
#                                     else:
#                                         device_comms_li[li_swap_op_step]['B'].remove(li_swap_comm)
#                                     for adapt_index in range(k+1, swap_index):#不包含swap_index
#                                         li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_before_op_recv= li[adapt_index]
#                                         device_comms_li[li_k_op_step]['B'].append(li_adapt_index_comm)
#                                         if li_adapt_index_endA:
#                                             device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                         else:
#                                             device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)
#                                         li[adapt_index] = (li_adapt_index_comm_ins, li_k_op_step, li_adapt_index_comm, False, li_adapt_index_before_op_recv)
#                                     li[k] = (li_swap_comm_ins, li_k_op_step, li_swap_comm, False,li_swap_before_op_recv)
#                                     li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA, li_k_before_op_recv)
#                                 else:
#                                     device_comms_li[li_k_op_step]['A'].append(li_swap_comm)
#                                     if li_swap_endA:
#                                         device_comms_li[li_swap_op_step]['A'].remove(li_swap_comm)
#                                     else:
#                                         device_comms_li[li_swap_op_step]['B'].remove(li_swap_comm)
#                                     for adapt_index in range(k+1, swap_index):
#                                         li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_befor_op_recv = li[adapt_index]
#                                         device_comms_li[li_k_op_step]['A'].append(li_adapt_index_comm)
#                                         if li_adapt_index_endA:
#                                             device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                         else:
#                                             device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)
#                                         li[adapt_index] = (li_adapt_index_comm_ins, li_k_op_step, li_adapt_index_comm, True, li_adapt_index_befor_op_recv)
#                                     li[k] = (li_swap_comm_ins, li_k_op_step, li_swap_comm, True, li_swap_before_op_recv)
#                                     li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA, li_k_before_op_recv)
#                             elif li_swap_comm_ins in ("SA","SG"):#前面是R的comm instrcution，后面要交换的是S的comm instruction
#                                 if i%2 == 0:
#                                     if li_swap_op_step >= li_k_before_op_recv:
#                                         have_swap = False
#                                     if have_swap:
#                                         device_comms_li[li_swap_op_step]['A'].insert(0,li_k_comm)
#                                         if li_k_endA:
#                                             device_comms_li[li_k_op_step]['A'].remove(li_k_comm)
#                                         else:
#                                             device_comms_li[li_k_op_step]['B'].remove(li_k_comm)

#                                         for adapt_index in range(swap_index-1,k,-1):#倒序保持中间顺序
#                                             li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_befor_op_recv= li[adapt_index]
#                                             device_comms_li[li_swap_op_step]['A'].insert(0,li_adapt_index_comm)
#                                             if li_adapt_index_endA:
#                                                 device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                             else:
#                                                 device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)                                        
#                                             li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step, li_adapt_index_comm,True, li_adapt_index_befor_op_recv)
#                                         li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, True, li_swap_before_op_recv)
#                                         li[swap_index] = (li_k_comm_ins, li_swap_op_step, li_k_comm, li_k_endA, li_k_before_op_recv)
#                                 else:
#                                     if li_swap_op_step+1 > li_k_before_op_recv:
#                                         have_swap = False
#                                     if have_swap: 
#                                         device_comms_li[li_swap_op_step+1]['B'].insert(0,li_k_comm)
#                                         if li_k_endA:
#                                             device_comms_li[li_k_op_step]['A'].remove(li_k_comm)
#                                         else:
#                                             device_comms_li[li_k_op_step]['B'].remove(li_k_comm)
#                                         for adapt_index in range(swap_index-1,k,-1):
#                                             li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_befor_op_recv = li[adapt_index]
#                                             device_comms_li[li_swap_op_step+1]['B'].insert(0,li_adapt_index_comm)
#                                             if li_adapt_index_endA:
#                                                 device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                             else:
#                                                 device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)   
#                                             li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step+1, li_adapt_index_comm, False, li_adapt_index_befor_op_recv)
#                                         li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv)
#                                         li[swap_index] = (li_k_comm_ins, li_swap_op_step+1, li_k_comm, False, li_k_before_op_recv)
#                             else:#前后要交换的都是R的comm instruction
#                                 if li_swap_op_step > li_k_before_op_recv or (li_k_before_op_recv == li_swap_op_step and li_swap_endA):
#                                     have_swap = False
#                                 if have_swap:
#                                     # li_k_comm_index = device_comms_li[li_k_op_step]['B'].index(li_k_comm)
#                                     # device_comms_li[li_k_op_step]['B'].insert(li_k_comm_index,li_swap_comm)
#                                     # device_comms_li[li_k_op_step]['B'].remove(li_k_comm)
#                                     # for adapt_index in range(k+1,swap_index):
#                                     #     li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_befor_op_recv = li[adapt_index]
#                                     #     if li_adapt_index_comm not in device_comms_li[li_k_op_step]['B']:
#                                     #         device_comms_li[li_k_op_step]['B'].append(li_adapt_index_comm)
#                                     #     if li_adapt_index_endA:
#                                     #         device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                     #     else:
#                                     #         device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)   
#                                     #     li[adapt_index] = (li_adapt_index_comm_ins, li_k_op_step, li_adapt_index_comm, False, li_adapt_index_befor_op_recv)                                           
#                                     # device_comms_li[li_k_op_step]['B'].append(li_k_comm)
#                                     # li[k] = (li_swap_comm_ins, li_k_op_step, li_swap_comm, False, li_swap_before_op_recv)
#                                     # li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, False, li_k_before_op_recv)
#                                 # else:
#                                     if li_k_endA:
#                                         li_k_in_comm_index = device_comms_li[li_k_op_step]['A'].index(li_k_comm)
#                                         device_comms_li[li_k_op_step]['A'][li_k_in_comm_index] = li_swap_comm
#                                     else:
#                                         li_k_in_comm_index = device_comms_li[li_k_op_step]['B'].index(li_k_comm)
#                                         device_comms_li[li_k_op_step]['B'][li_k_in_comm_index] = li_swap_comm
#                                     if li_swap_endA:
#                                         li_swap_in_comm_index = device_comms_li[li_swap_op_step]['A'].index(li_swap_comm)
#                                         device_comms_li[li_swap_op_step]['A'][li_swap_in_comm_index] = li_k_comm
#                                     else:
#                                         li_swap_in_comm_index = device_comms_li[li_swap_op_step]['B'].index(li_swap_comm)
#                                         device_comms_li[li_swap_op_step]['B'][li_swap_in_comm_index] = li_k_comm
#                                     li[k] = (li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv)
#                                     li[swap_index] = (li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA, li_k_before_op_recv)
#                             #print(f"af_comms_li:{device_comms_li[li_k_op_step:li_swap_op_step+1]}")
#                         if have_swap:
#                             actions.append(f"swap matrix[{i}][{j}][{k}] <-> matrix[{i}][{j}][{swap_index}]")
#                             break
#                         else:
#                             if swap_index <= next_block_j and pair.get(lj[swap_index][0]) == li_k_comm_ins:
#                                 have_swap = True
#                                 lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv = lj[swap_index]
#                                 if lj_k_comm_ins in ("SA","SG"):#前面是S的comm instrcution，后面要交换的是R的comm instruction
#                                     if j%2 == 0: #偶数rank，算发收
#                                         device_comms_lj[lj_k_op_step]['B'].append(lj_swap_comm)
#                                         if lj_swap_endA:
#                                             device_comms_lj[lj_swap_op_step]['A'].remove(lj_swap_comm)
#                                         else:
#                                             device_comms_lj[lj_swap_op_step]['B'].remove(lj_swap_comm)
#                                         for adapt_index in range(k+1, swap_index):#不包含swap_index
#                                             lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_before_op_recv= lj[adapt_index]
#                                             device_comms_lj[lj_k_op_step]['B'].append(lj_adapt_index_comm)
#                                             if lj_adapt_index_endA:
#                                                 device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                             else:
#                                                 device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)
#                                             lj[adapt_index] = (lj_adapt_index_comm_ins, lj_k_op_step, lj_adapt_index_comm, False, lj_adapt_index_before_op_recv)
#                                         lj[k] = (lj_swap_comm_ins, lj_k_op_step, lj_swap_comm, False,lj_swap_before_op_recv)
#                                         lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv)
#                                     else:
#                                         device_comms_lj[lj_k_op_step]['A'].append(lj_swap_comm)
#                                         if lj_swap_endA:
#                                             device_comms_lj[lj_swap_op_step]['A'].remove(lj_swap_comm)
#                                         else:
#                                             device_comms_lj[lj_swap_op_step]['B'].remove(lj_swap_comm)
#                                         for adapt_index in range(k+1, swap_index):
#                                             lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_befor_op_recv = lj[adapt_index]
#                                             device_comms_lj[lj_k_op_step]['A'].append(lj_adapt_index_comm)
#                                             if lj_adapt_index_endA:
#                                                 device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                             else:
#                                                 device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)
#                                             lj[adapt_index] = (lj_adapt_index_comm_ins, lj_k_op_step, lj_adapt_index_comm, True, lj_adapt_index_befor_op_recv)
#                                         lj[k] = (lj_swap_comm_ins, lj_k_op_step, lj_swap_comm, True, lj_swap_before_op_recv)
#                                         lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv)
#                                 elif lj_swap_comm_ins in ("SA","SG"):#前面是R的comm instrcution，后面要交换的是S的comm instruction
#                                     if j%2 == 0:
#                                         if lj_swap_op_step >= lj_k_before_op_recv:
#                                             have_swap = False
#                                         if have_swap:
#                                             device_comms_lj[lj_swap_op_step]['A'].insert(0,lj_k_comm)
#                                             if lj_k_endA:
#                                                 device_comms_lj[lj_k_op_step]['A'].remove(lj_k_comm)
#                                             else:
#                                                 device_comms_lj[lj_k_op_step]['B'].remove(lj_k_comm)

#                                             for adapt_index in range(swap_index-1,k,-1):#倒序保持中间顺序
#                                                 lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_befor_op_recv= lj[adapt_index]
#                                                 device_comms_lj[lj_swap_op_step]['A'].insert(0,lj_adapt_index_comm)
#                                                 if lj_adapt_index_endA:
#                                                     device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                                 else:
#                                                     device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)                                        
#                                                 lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step, lj_adapt_index_comm,True, lj_adapt_index_befor_op_recv)
#                                             lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, True, lj_swap_before_op_recv)
#                                             lj[swap_index] = (lj_k_comm_ins, lj_swap_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv)
#                                     else:
#                                         if lj_swap_op_step+1 > lj_k_before_op_recv:
#                                             have_swap = False
#                                         if have_swap: 
#                                             device_comms_lj[lj_swap_op_step+1]['B'].insert(0,lj_k_comm)
#                                             if lj_k_endA:
#                                                 device_comms_lj[lj_k_op_step]['A'].remove(lj_k_comm)
#                                             else:
#                                                 device_comms_lj[lj_k_op_step]['B'].remove(lj_k_comm)
#                                             for adapt_index in range(swap_index-1,k,-1):
#                                                 lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_befor_op_recv = lj[adapt_index]
#                                                 device_comms_lj[lj_swap_op_step+1]['B'].insert(0,lj_adapt_index_comm)
#                                                 if lj_adapt_index_endA:
#                                                     device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                                 else:
#                                                     device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)   
#                                                 lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step+1, lj_adapt_index_comm, False, lj_adapt_index_befor_op_recv)
#                                             lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv)
#                                             lj[swap_index] = (lj_k_comm_ins, lj_swap_op_step+1, lj_k_comm, False, lj_k_before_op_recv)
#                                 else:#前后要交换的都是R的comm instruction
#                                     if lj_swap_op_step > lj_k_before_op_recv or (lj_k_before_op_recv == lj_swap_op_step and lj_swap_endA):
#                                         have_swap = False
#                                     if have_swap:
#                                         # lj_k_comm_index = device_comms_lj[lj_k_op_step]['B'].index(lj_k_comm)
#                                         # device_comms_lj[lj_k_op_step]['B'].insert(lj_k_comm_index,lj_swap_comm)
#                                         # device_comms_lj[lj_k_op_step]['B'].remove(lj_k_comm)
#                                         # for adapt_index in range(k+1,swap_index):
#                                         #     lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_befor_op_recv = lj[adapt_index]
#                                         #     if lj_adapt_index_comm not in device_comms_lj[lj_k_op_step]['B']:
#                                         #         device_comms_lj[lj_k_op_step]['B'].append(lj_adapt_index_comm)
#                                         #     if lj_adapt_index_endA:
#                                         #         device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                         #     else:
#                                         #         device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)   
#                                         #     lj[adapt_index] = (lj_adapt_index_comm_ins, lj_k_op_step, lj_adapt_index_comm, False, lj_adapt_index_befor_op_recv)                                           
#                                         # device_comms_lj[lj_k_op_step]['B'].append(lj_k_comm)
#                                         # lj[k] = (lj_swap_comm_ins, lj_k_op_step, lj_swap_comm, False, lj_swap_before_op_recv)
#                                         # lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, False, lj_k_before_op_recv)
#                                     # else:
#                                         if lj_k_endA:
#                                             lj_k_in_comm_index = device_comms_lj[lj_k_op_step]['A'].index(lj_k_comm)
#                                             device_comms_lj[lj_k_op_step]['A'][lj_k_in_comm_index] = lj_swap_comm
#                                         else:
#                                             lj_k_in_comm_index = device_comms_lj[lj_k_op_step]['B'].index(lj_k_comm)
#                                             device_comms_lj[lj_k_op_step]['B'][lj_k_in_comm_index] = lj_swap_comm
#                                         if lj_swap_endA:
#                                             lj_swap_in_comm_index = device_comms_lj[lj_swap_op_step]['A'].index(lj_swap_comm)
#                                             device_comms_lj[lj_swap_op_step]['A'][lj_swap_in_comm_index] = lj_k_comm
#                                         else:
#                                             lj_swap_in_comm_index = device_comms_lj[lj_swap_op_step]['B'].index(lj_swap_comm)
#                                             device_comms_lj[lj_swap_op_step]['B'][lj_swap_in_comm_index] = lj_k_comm
#                                         lj[k] = (lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv)
#                                         lj[swap_index] = (lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv)                               
#                         if have_swap:
#                             #print(f"af_comms_lj:{device_comms_lj[lj_k_op_step:lj_swap_op_step+1]}")
#                             actions.append(f"swap matrix[{j}][{i}][{k}] <-> matrix[{j}][{i}][{swap_index}]")
#                             break
#                         else:
#                             swap_index += 1
#                             continue
#                     if not have_swap:
#                         print(f"no match found for matrix[{i}][{j}][{k}] and matrix[{j}][{i}][{k}]")
#                         actions.append(f"no match found for matrix[{i}][{j}][{k}] and matrix[{j}][{i}][{k}]")
#     return comm_graph, actions

# def fix_matrix_keep_sa_sg_order_(matrix, comm_graph):
#     pair = {"SA": "RA", "RA": "SA", "SG": "RG", "RG": "SG"}
#     n = len(matrix)
#     actions = []
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             li = matrix[i][j]
#             lj = matrix[j][i]
#             device_comms_li = comm_graph[i]
#             device_comms_lj = comm_graph[j]
#             comms_len_between_ranks = len(li)
#             for k in range(comms_len_between_ranks):
#                 li_k_comm_ins, li_k_op_step, li_k_comm, li_k_endA, li_k_before_op_recv = li[k]
#                 li_expected_inlj = pair.get(li_k_comm_ins)
                
#                 lj_k_comm_ins, lj_k_op_step, lj_k_comm, lj_k_endA, lj_k_before_op_recv = lj[k]
#                 lj_expected_inli = pair.get(lj_k_comm_ins)
#                 if lj_k_comm_ins != li_expected_inlj:
#                     if li_k_comm_ins in ("SA","SG"):#将lj中对应的("RA","RG")提前
#                         for change_index in range(k+1,comms_len_between_ranks):
#                             lj_change_index_comm_ins, lj_change_index_op_step, lj_change_index_comm, lj_change_index_endA, lj_change_index_before_op_recv = lj[change_index]
#                             if li_expected_inlj == lj_change_index_comm_ins:#将正确的通信插入在lj[k]对应的comm_graph位置的前面
#                                 #判断原来lj_k_comm在什么位置，如果是S开头则，需要判断是rank编号是奇数或者偶数
#                                 if lj_change_index_endA:
#                                     device_comms_lj[lj_change_index_op_step]['A'].remove(lj_change_index_comm)
#                                 else:
#                                     device_comms_lj[lj_change_index_op_step]['B'].remove(lj_change_index_comm)
#                                 if lj_k_comm_ins in ("RA","RG"):
#                                     if lj_k_endA:
#                                         lj_k_index = device_comms_lj[lj_k_op_step]['A'].index(lj_k_comm)
#                                         device_comms_lj[lj_k_op_step]['A'].insert(lj_k_index,lj_change_index_comm)
#                                         lj_change_index_endA = True
#                                     else:
#                                         lj_k_index = device_comms_lj[lj_k_op_step]['B'].index(lj_k_comm)
#                                         device_comms_lj[lj_k_op_step]['B'].insert(lj_k_index,lj_change_index_comm)
#                                         lj_change_index_endA = False
#                                 else:
#                                     if j%2 == 0:
#                                         device_comms_lj[lj_k_op_step]['B'].append(lj_change_index_comm)
#                                         lj_change_index_endA = False
#                                     else:
#                                         device_comms_lj[lj_k_op_step]['A'].append(lj_change_index_comm)
#                                         lj_change_index_endA = True

#                                 lj.pop(change_index)
#                                 lj.insert(k,(lj_change_index_comm_ins, lj_k_op_step, lj_change_index_comm, lj_change_index_endA, lj_change_index_before_op_recv))
#                     elif lj_k_comm_ins in ("SA","SG"):#将li中对应的("RA","RG")提前
#                         for change_index in range(k+1,comms_len_between_ranks):
#                             li_change_index_comm_ins, li_change_index_op_step, li_change_index_comm, li_change_index_endA, li_change_index_before_op_recv = li[change_index]
#                             if lj_expected_inli == li_change_index_comm_ins:#将正确的通信插入在li[k]对应的comm_graph位置的前面
#                                 #判断原来li_k_comm在什么位置，如果是S开头则，需要判断是rank编号是奇数或者偶数
#                                 if li_change_index_endA:
#                                     device_comms_li[li_change_index_op_step]['A'].remove(li_change_index_comm)
#                                 else:
#                                     device_comms_li[li_change_index_op_step]['B'].remove(li_change_index_comm)
#                                 if li_k_comm_ins in ("RA","RG"):
#                                     if li_k_endA:
#                                         li_k_index = device_comms_li[li_k_op_step]['A'].index(li_k_comm)
#                                         device_comms_li[li_k_op_step]['A'].insert(li_k_index,li_change_index_comm)
#                                         li_change_index_endA = True
#                                     else:
#                                         li_k_index = device_comms_li[li_k_op_step]['B'].index(li_k_comm)
#                                         device_comms_li[li_k_op_step]['B'].insert(li_k_index,li_change_index_comm)
#                                         li_change_index_endA = False
#                                 else:
#                                     if j%2 == 0:
#                                         device_comms_li[li_k_op_step]['B'].append(li_change_index_comm)
#                                         li_change_index_endA = False
#                                     else:
#                                         device_comms_li[li_k_op_step]['A'].append(li_change_index_comm)
#                                         li_change_index_endA = True

#                                 li.pop(change_index)
#                                 li.insert(k,(li_change_index_comm_ins, li_k_op_step, li_change_index_comm, li_change_index_endA, li_change_index_before_op_recv))
#                     else:
#                         swap_index_li = next_sa_sg_idx_(li,k,lj_expected_inli)
#                         swap_index_lj = next_sa_sg_idx_(lj,k,li_expected_inlj)
#                         have_swap = False
#                         if swap_index_li is not None:
#                             li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv = li[swap_index_li]
#                             if i % 2 == 0 and li_k_before_op_recv > li_swap_op_step:
#                                 for adapt_index in range(k, swap_index_li):#不包含swap_index_li
#                                     li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_before_op_recv= li[adapt_index]
#                                     device_comms_li[li_swap_op_step]['A'].append(li_adapt_index_comm)
#                                     if li_adapt_index_endA:
#                                         device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                     else:
#                                         device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)
#                                     li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step, li_adapt_index_comm, True, li_adapt_index_before_op_recv)
#                                 li.pop(swap_index_li)
#                                 li.insert(k,(li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv))                               
#                                 have_swap = True
#                             elif i % 2 == 1 and li_k_before_op_recv >= li_swap_op_step+1:
#                                 for adapt_index in range(k, swap_index_li):#不包含swap_index_li
#                                     li_adapt_index_comm_ins, li_adapt_index_op_step, li_adapt_index_comm, li_adapt_index_endA, li_adapt_index_before_op_recv= li[adapt_index]
#                                     device_comms_li[li_swap_op_step+1]['B'].append(li_adapt_index_comm)
#                                     if li_adapt_index_endA:
#                                         device_comms_li[li_adapt_index_op_step]['A'].remove(li_adapt_index_comm)
#                                     else:
#                                         device_comms_li[li_adapt_index_op_step]['B'].remove(li_adapt_index_comm)
#                                     li[adapt_index] = (li_adapt_index_comm_ins, li_swap_op_step+1, li_adapt_index_comm, False, li_adapt_index_before_op_recv)
#                                 li.pop(swap_index_li)
#                                 li.insert(k,(li_swap_comm_ins, li_swap_op_step, li_swap_comm, li_swap_endA, li_swap_before_op_recv))                                  
#                                 have_swap = True
#                             else:
#                                 have_swap = False
#                         elif swap_index_lj is not None and not have_swap:
#                             lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv = lj[swap_index_lj]
#                             if j % 2 == 0 and lj_k_before_op_recv > lj_swap_op_step:
#                                 for adapt_index in range(k, swap_index_lj):#不包含swap_index_lj
#                                     lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_before_op_recv= lj[adapt_index]
#                                     device_comms_lj[lj_swap_op_step]['A'].append(lj_adapt_index_comm)
#                                     if lj_adapt_index_endA:
#                                         device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                     else:
#                                         device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)
#                                     lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step, lj_adapt_index_comm, True, lj_adapt_index_before_op_recv)
#                                 lj.pop(swap_index_lj)
#                                 lj.insert(k,(lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv))                               
#                                 have_swap = True
#                             elif j % 2 == 1 and lj_k_before_op_recv >= lj_swap_op_step+1:
#                                 for adapt_index in range(k, swap_index_lj):#不包含swap_index_lj
#                                     lj_adapt_index_comm_ins, lj_adapt_index_op_step, lj_adapt_index_comm, lj_adapt_index_endA, lj_adapt_index_before_op_recv= lj[adapt_index]
#                                     device_comms_lj[lj_swap_op_step+1]['B'].append(lj_adapt_index_comm)
#                                     if lj_adapt_index_endA:
#                                         device_comms_lj[lj_adapt_index_op_step]['A'].remove(lj_adapt_index_comm)
#                                     else:
#                                         device_comms_lj[lj_adapt_index_op_step]['B'].remove(lj_adapt_index_comm)
#                                     lj[adapt_index] = (lj_adapt_index_comm_ins, lj_swap_op_step+1, lj_adapt_index_comm, False, lj_adapt_index_before_op_recv)
#                                 lj.pop(swap_index_lj)
#                                 lj.insert(k,(lj_swap_comm_ins, lj_swap_op_step, lj_swap_comm, lj_swap_endA, lj_swap_before_op_recv))                                  
#                                 have_swap = True
#                             else:
#                                 have_swap = False                                           
#                         else:
#                             print("wrong")
#     return comm_graph

# def generate_ops_josn(comm_graph, stage_alignment, comp_graph):
#     dir = '/cpfs01/user/matenghui/InternEvo/devices_operations/'
#     os.makedirs(dir, exist_ok=True)
#     max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
#     min_stage_id = min([stage_id for row in stage_alignment for stage_id in row])
#     comm_graph_martix  = [[[] for __ in range(len(stage_alignment))] for _ in range(len(stage_alignment))]
#     for device_id in range(len(comp_graph)):
#         jsonpath = dir+"pp"+str(device_id)+"_ops.json"
#         stages = stage_alignment[device_id]
#         stage_ops = comp_graph[device_id]
#         device_comms = comm_graph[device_id]
#         assert len(stage_ops) == len(device_comms)
#         for step_index in range(len(stage_ops)):
#             op, microbatch_id, stage_id, chunk_id, start_time, end_time = stage_ops[step_index]
#             comm_per_step = device_comms[step_index]
#             current_op_sendcomm = ''
#             dst_device = None
#             if op == 'f' and stage_id < max_stage_id and stage_id+1 not in stages:
#                 current_op_sendcomm = 'SA'
#                 dst_device = _get_deviceid_by_alignment(stage_id+1, stage_alignment)
#             elif op == 'b' and stage_id > min_stage_id and stage_id-1 not in stages:
#                 current_op_sendcomm = 'SG'
#                 dst_device = _get_deviceid_by_alignment(stage_id-1, stage_alignment)

#             if len(comm_per_step['B'])>0:
#                 for comm in comm_per_step['B']:
#                     recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                     if recv_op_type == 'f':
#                         json_content={"operation": "RA", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                         write_json(jsonpath,json_content)
#                     elif recv_op_type == 'b':
#                         json_content={"operation": "RG", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                         write_json(jsonpath,json_content)
#             json_content = {"step_type": op, "local_rank": device_id, "step_id": step_index, "chunk_id": chunk_id, "stage_id": stage_id, "microbatch_id": microbatch_id, "operation": "compute"}
#             write_json(jsonpath,json_content)
#             if device_id%2 == 0:
#                 if current_op_sendcomm != '':
#                     json_content={"operation": current_op_sendcomm, "local_rank":device_id ,"step_id": step_index, "match_rank": dst_device}
#                     write_json(jsonpath,json_content)
#                 if len(comm_per_step['A'])>0:
#                     for comm in comm_per_step['A']:
#                         recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                         if recv_op_type == 'f':
#                             json_content={"operation": "RA", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                             write_json(jsonpath,json_content)
#                         elif recv_op_type == 'b':
#                             json_content={"operation": "RG", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                             write_json(jsonpath,json_content)                       
#             else:
#                 if len(comm_per_step['A'])>0:
#                     for comm in comm_per_step['A']:
#                         recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                         if recv_op_type == 'f':
#                             json_content={"operation": "RA", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                             write_json(jsonpath,json_content)
#                         elif recv_op_type == 'b':
#                             json_content={"operation": "RG", "local_rank":device_id ,"step_id": step_index, "match_rank": recv_device_id, "comm":comm}
#                             write_json(jsonpath,json_content)    
#                 if current_op_sendcomm != '':
#                     json_content={"operation": current_op_sendcomm, "local_rank":device_id ,"step_id": step_index, "match_rank": dst_device}
#                     write_json(jsonpath,json_content)                    

# def _get_comp_recv_index(ops,recv_op_type,recv_stage_id,recv_microbatch_id):
#     for i in range(len(ops)):
#         op, microbatch_id, stage_id, chunk_id, start_time, end_time = ops[i]
#         if microbatch_id == recv_microbatch_id and recv_op_type == op :
#             if (op == 'f' and stage_id == recv_stage_id+1) or (op == 'b' and stage_id == recv_stage_id-1):
#                 return i

# def generate_comm_martix(comm_graph, stage_alignment, comp_graph):
#     max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
#     min_stage_id = min([stage_id for row in stage_alignment for stage_id in row])
#     comm_graph_martix  = [[[] for __ in range(len(stage_alignment))] for _ in range(len(stage_alignment))]
#     for device_id in range(len(comp_graph)):
#         stages = stage_alignment[device_id]
#         stage_ops = comp_graph[device_id]
#         device_comms = comm_graph[device_id]
#         assert len(stage_ops) == len(device_comms)
#         for step_index in range(len(stage_ops)):
#             op, microbatch_id, stage_id, chunk_id, start_time, end_time = stage_ops[step_index]
#             comm_per_step = device_comms[step_index]
#             current_op_sendcomm = ''
#             dst_device = None
#             if op == 'f' and stage_id < max_stage_id and stage_id+1 not in stages:
#                 current_op_sendcomm = 'SA'
#                 dst_device = _get_deviceid_by_alignment(stage_id+1, stage_alignment)
#             elif op == 'b' and stage_id > min_stage_id and stage_id-1 not in stages:
#                 current_op_sendcomm = 'SG'
#                 dst_device = _get_deviceid_by_alignment(stage_id-1, stage_alignment)

#             if len(comm_per_step['B'])>0:
#                 for comm in comm_per_step['B']:
#                     recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                     before_op_recv_indedx = _get_comp_recv_index(stage_ops,recv_op_type,recv_stage_id,recv_microbatch_id)
#                     assert before_op_recv_indedx is not None
#                     if recv_op_type == 'f':
#                         comm_graph_martix[device_id][recv_device_id].append(('RA',step_index,comm,False,before_op_recv_indedx))
#                     elif recv_op_type == 'b':
#                         comm_graph_martix[device_id][recv_device_id].append(('RG',step_index,comm,False,before_op_recv_indedx))

#             if device_id%2 == 0:
#                 if current_op_sendcomm != '':
#                     comm_graph_martix[device_id][dst_device].append((current_op_sendcomm,step_index, "_","_","_"))
#                 if len(comm_per_step['A'])>0:
#                     for comm in comm_per_step['A']:
#                         recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                         before_op_recv_indedx = _get_comp_recv_index(stage_ops,recv_op_type,recv_stage_id,recv_microbatch_id)
#                         assert before_op_recv_indedx is not None                          
#                         if recv_op_type == 'f':
#                             comm_graph_martix[device_id][recv_device_id].append(('RA',step_index,comm,True,before_op_recv_indedx))
#                         elif recv_op_type == 'b':
#                             comm_graph_martix[device_id][recv_device_id].append(('RG',step_index,comm,True,before_op_recv_indedx))            
#             else:
#                 if len(comm_per_step['A'])>0:
#                     for comm in comm_per_step['A']:
#                         recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, _= comm
#                         before_op_recv_indedx = _get_comp_recv_index(stage_ops,recv_op_type,recv_stage_id,recv_microbatch_id)
#                         assert before_op_recv_indedx is not None
#                         if recv_op_type == 'f':
#                             comm_graph_martix[device_id][recv_device_id].append(('RA',step_index,comm,True,before_op_recv_indedx))
#                         elif recv_op_type == 'b':
#                             comm_graph_martix[device_id][recv_device_id].append(('RG',step_index,comm,True,before_op_recv_indedx))
#                 if current_op_sendcomm != '':
#                     comm_graph_martix[device_id][dst_device].append((current_op_sendcomm, step_index, "_","_","_"))
#     return comm_graph_martix
 
# def next_sa_sg_idx(lst, start):
#     """返回lst中从start起第一个SA/SG的位置不含start找不到则返回None"""
#     for idx in range(start+1, len(lst)):
#         if lst[idx][0] in ("SA", "SG"):
#             return idx

# def next_sa_sg_idx_(lst, start, send_comms):
#     """返回lst中从start起第一个SA/SG的位置不含start找不到则返回None"""
#     for idx in range(start+1, len(lst)):
#         if lst[idx][0] in ("SA", "SG"):
#             if lst[idx][0] == send_comms:
#                 return idx
#             else:
#                 return None

# def detect_cycle_deadlock_mutichunk(comm_graph, stage_alignment):
#     sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
#     max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
#     # comm_graph_copy = copy.deepcopy(comm_graph)
#     for rank_id, rank_ops in enumerate(comm_graph):
#         if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-收-发）
#             continue
#         len_rank_ops = len(rank_ops)#rank_ops = comm_graph[rank_id]
#         # rank_ops_copy = copy.deepcopy(rank_ops)
#         for current_index, op in enumerate(rank_ops):
#             #op = rank_ops[current_index]
#             op_type, stage_id, microbatch_id = op['Infor']
#             #没有发送需求就不会有死锁
#             if op_type == 'f':
#                 if stage_id in sendFtoSameDevice or stage_id == max_stage_id:
#                     continue
#                 dst_rank_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
#             elif op_type == 'b':
#                 if stage_id in sendBtoSameDevice or stage_id == 0:
#                     continue
#                 dst_rank_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
#             else:
#                 continue

#             #判断环
#             visited = set()  # 用于记录已经访问过的任务
#             rec_stack = set()  # 用于记录当前递归栈中的任务
#             has_cycle, cycle_index = dfs(op,rec_stack, visited, comm_graph)
#             if has_cycle and cycle_index != -1 and current_index < len_rank_ops - 1:
#                 comm_graph[rank_id][current_index + 1]['B'].append(comm_graph[rank_id][current_index]['A'].pop(cycle_index))
#                 # print(f"cycle dead lock:{op['Infor']}")
#     return comm_graph

# def detect_cross_deadlock_mutichunk(comm_graph, stage_alignment):
#     sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
#     max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
#     comm_graph_copy = copy.deepcopy(comm_graph)
#     for rank_id, rank_ops in enumerate(comm_graph_copy):
#         #判断偶数rank（收-算-发-收）
#         len_rank_ops = len(rank_ops)
#         rank_ops_copy = copy.deepcopy(rank_ops)
#         for current_index, op in enumerate(rank_ops_copy):
#             assert len(rank_ops) == len_rank_ops
#             op_type, stage_id, microbatch_id = op['Infor']
#             #没有发送需求就不会有死锁
#             if op_type == 'f':
#                 if stage_id in sendFtoSameDevice or stage_id == max_stage_id:
#                     continue
#                 dst_rank_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
#             elif op_type == 'b':
#                 if stage_id in sendBtoSameDevice or stage_id == 0:
#                     continue
#                 dst_rank_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
#             else:
#                 continue

#             if rank_id % 2 == 0:
#                 #死锁场景
#                 #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
#                 needjude = op['A']
#                 if current_index + 1 < len_rank_ops:
#                     next_op = rank_ops_copy[current_index + 1]
#                     if len(next_op['B'])>0:
#                         needjude += next_op['B']
#                 for judgeop in needjude:
#                     recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index = judgeop
#                     if rank_id == recv_device_id or recv_device_id != dst_rank_id:
#                         continue
#                     recvDevice_op = copy.deepcopy(comm_graph[recv_device_id][index])
#                     goonjudge = True
#                     for rc in (recvDevice_op['A']+recvDevice_op['B']):
#                         next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index = rc
#                         if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
#                             goonjudge = False
#                             break
#                     judgedistance = 0
#                     if not goonjudge:
#                         break
#                     while(goonjudge is True):
#                         judgedistance += 1
#                         if index+judgedistance < len(comm_graph_copy[recv_device_id]):
#                             recvDevice_nextop_n = copy.deepcopy(comm_graph[recv_device_id][index+judgedistance])
#                             recvDevice_nextop_nb = recvDevice_nextop_n['B']
#                             for rnb, recvDevice_nextop_n_b in enumerate(recvDevice_nextop_nb):
#                                 next_recv_op_type_B, next_recv_end_time_B, next_recv_device_id_B, next_recv_stage_id_B, next_recv_chunk_id_B, next_recv_microbatch_id_B, next_index_B = recvDevice_nextop_n_b
#                                 if (next_recv_op_type_B,next_recv_stage_id_B,next_recv_microbatch_id_B) == op['Infor']:
#                                     rnbresult = comm_graph[recv_device_id][index+judgedistance]['B'].pop(rnb)
#                                     comm_graph[recv_device_id][index]['A'].append(rnbresult)
#                                     goonjudge = False
#                                     break
#                             if not goonjudge:
#                                 break
#                             recvDevice_nextop_na = recvDevice_nextop_n['A']
#                             for rna, recvDevice_nextop_n_a in enumerate(recvDevice_nextop_na):
#                                 next_recv_op_type_A, next_recv_end_time_A, next_recv_device_id_A, next_recv_stage_id_A, next_recv_chunk_id_A, next_recv_microbatch_id_A, next_index_A = recvDevice_nextop_n_a
#                                 if (next_recv_op_type_A,next_recv_stage_id_A,next_recv_microbatch_id_A) == op['Infor']:
#                                     rnaresult = comm_graph[recv_device_id][index+judgedistance]['A'].pop(rna)
#                                     comm_graph[recv_device_id][index]['A'].append(rnaresult)
#                                     goonjudge = False
#                                     break
#                         if not goonjudge:
#                             break
#                         if index-judgedistance >= 0 :
#                             recvDevice_nextop_bab = comm_graph[recv_device_id][index-judgedistance]
#                             for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
#                                 next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_chunk_id_bab, next_recv_microbatch_id_bab, next_index_bab= rnbab
#                                 if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
#                                     goonjudge = False
#                                     break
#                     if not goonjudge:
#                         break
#             else:
#                 needjude = list(reversed(op['A']))+op['B']#TODO
#                 for judgeop in needjude:
#                     recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_stream_id, recv_microbatch_id, index= judgeop
#                     if rank_id == recv_device_id or recv_device_id != dst_rank_id:
#                         continue
#                     if recv_op_type == op_type and microbatch_id == recv_microbatch_id and ((op_type == "f" and recv_stage_id>stage_id) or (op_type == "b" and recv_stage_id<stage_id)):
#                         comm_graph[rank_id][current_index+1]['B'].insert(0,judgeop)
#                         if judgeop in op['A']:
#                             comm_graph[rank_id][current_index]['A'].remove(judgeop)
#                         else:
#                             comm_graph[rank_id][current_index]['B'].remove(judgeop)
#                         continue

#                     recvDevice_op = copy.deepcopy(comm_graph[recv_device_id][index])
#                     goonjudge = True
#                     for rc in (recvDevice_op['A']):
#                         next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index = rc
#                         if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
#                             goonjudge = False
#                             break
#                     judgedistance = 0
#                     for rv_,rv in enumerate(recvDevice_op['B']):
#                         next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index = rv
#                         if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
#                             comm_graph[recv_device_id][index]['A'].append(comm_graph[recv_device_id][index]['B'].pop(rv_))
#                             goonjudge = False
#                             break
#                     if not goonjudge:
#                         break
#                     # opInfor = op['Infor']
#                     # print(f'opInfor:{opInfor}')
#                     # print(f'needjude:{judgeop}')
#                     # print(f'recvDevice_op:{recvDevice_op}')
#                     while(goonjudge is True):
#                         judgedistance += 1
#                         if index-judgedistance >= 0 :
#                             recvDevice_nextop_bab = copy.deepcopy(comm_graph[recv_device_id][index-judgedistance])
#                             for rnbab_id,rnbab in enumerate(recvDevice_nextop_bab['A']):
#                                 next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab
#                                 if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
#                                     comm_graph[recv_device_id][index]['A'].append(comm_graph[recv_device_id][index-judgedistance]['A'].pop(rnbab_id))
#                                     goonjudge = False
#                                     break
#                             if not goonjudge:
#                                 break
#                             for rnbab__id,rnbab_ in enumerate(recvDevice_nextop_bab['B']):
#                                 next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab_
#                                 if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
#                                     comm_graph[recv_device_id][index]['A'].append(comm_graph[recv_device_id][index-judgedistance]['B'].pop(rnbab__id))
#                                     goonjudge = False
#                                     break
#                         if not goonjudge:
#                             break
#                         if index+judgedistance < len(comm_graph_copy[recv_device_id]) :
#                             recvDevice_nextop_bab = comm_graph[recv_device_id][index+judgedistance]
#                             for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
#                                 next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab = rnbab
#                                 if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
#                                     goonjudge = False
#                                     break
#                     #print(judgedistance)
#                     # recvnum(communication_graph)
#                     if not goonjudge:
#                         break
#     return comm_graph

def order_result_mutichunk(input: str, stage_alignment: list, num_microbatches:int) -> None:
    device_steps = [[] for _ in range(len(stage_alignment))]
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
        device_id = _get_deviceid_by_alignment(stage_id, stage_alignment)
        chunk_id = _get_chunk_by_stage(stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    #print('[')
    recomp_stages = set()
    for d in range(len(stage_alignment)):
        f_num, b_num, w_num, r_num, r_stages = count_steps(device_steps[d])
        each_steps_num = len(stage_alignment[d])*num_microbatches
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

def generate_comm_graph(comp_graph, stage_alignment, max_end_time):
    #comp_graph 是之前生成的计算图
    # 初始化通信图,comm_graph[deviceid][step]表示计算操作前需要进行的通信操作list
    comm_graph = [[[] for __ in range(len(comp_graph[0])+1)] for _ in range(len(comp_graph))]
    processed_comp_graph = [[False for __ in range(len(comp_graph[i]))] for i in range(len(comp_graph))]
    #R表示在comp op前需要接收的，S表示comp后要发送的，所以每次可以进行合并
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    min_stage_id = min([stage_id for row in stage_alignment for stage_id in row])
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
                                dst_device_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
                            else:
                                dst_device_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
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
                                send_end_index = search_by_time(stage_ops, recv_op_start_time)+1

                                # print(f"send_location:{send_location}, send_end_index:{send_end_index}, recv_start_index:{recv_start_index}, recv_end_index:{recv_end_index}")
                                if current_op_index + 1 > send_end_index:
                                    send_end_index = current_op_index + 1
                                for s_index in range(current_op_index, min(send_end_index, len(stage_ops))):
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
                                        # print(f"wait_time_:{wait_time_}")
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

def generate_(num_microbatches=None):
    stage_placement = ""
    input_str=""
    file_path = 'InternEvo'
    with open(file_path+'/placement.txt', 'r', encoding='utf-8') as file:
        stage_placement = file.read()
    with open(file_path+'/result.txt', 'r', encoding='utf-8') as file:
        input_str = file.read()
    stage_placement = json.loads(stage_placement)

    pp_size = len(stage_placement)
    if not num_microbatches:
        num_microbatches = pp_size*4
    unified_scheduler, recomp_stages, max_end_time = order_result_mutichunk(input_str,stage_placement,num_microbatches)
    comm_graph = generate_comm_graph(unified_scheduler,stage_placement, max_end_time)
    scheduler_type = judge_scheduler_type(stage_placement)
    split_backward = judge_split_backward(unified_scheduler)
    last_stage = max(max(row) for row in stage_placement)
    first_stage = min(min(row) for row in stage_placement)
    Devices_containing_last_stage = [i for i, row in enumerate(stage_placement) if last_stage in row]
    # self.TheDevices_containg_first_stage = [i for i, row in enumerate(self.stage_placement) if self.first_stage in row]
    # result = {'num_microbatches':num_microbatches, 'pp_size':pp_size, \
    #           'stage_placement':stage_placement, 'scheduler_type': scheduler_type, 'split_backward':split_backward, \
    #           'first_stage':first_stage, 'last_stage':last_stage, 'Devices_containing_last_stage':Devices_containing_last_stage,\
    #            'unified_scheduler':unified_scheduler, 'comm_graph':comm_graph, 'recomp_stages':recomp_stages}
    # with open(file_path+'/runtime.json','w') as file:
    #     json.dump(result,file)
    # print(f'num_microbatches:{num_microbatches}, pp_size:{pp_size}, stage_placement:{stage_placement}, scheduler_type:{scheduler_type}, split_backward:{split_backward}, \
    #       recomp_stages:{recomp_stages}')
    return num_microbatches, pp_size, stage_placement, scheduler_type ,\
            split_backward, unified_scheduler, comm_graph, first_stage ,\
            last_stage, Devices_containing_last_stage, recomp_stages

if __name__ == '__main__':
    generate_(256)