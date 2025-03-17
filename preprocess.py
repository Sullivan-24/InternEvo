import copy
import json
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
    for s in steps:
        step_type = s[0]
        if step_type == 'f':
            f_num += 1
            continue
        elif step_type == 'b':
            b_num += 1
        elif step_type == 'w':
            w_num += 1
    return f_num, b_num, w_num

def dfs(op, rec_stack,visited,communication_graph):
    key = op['Infor']
    if key in rec_stack:
        return True, -1  # 发现环，返回 True 和无效的 index
    if key in visited:
        return False, -1  # 已经访问过，无需继续

    visited.add(key)
    rec_stack.add(key)

    for idx, a_task in enumerate(op['A']):
        _, _, recv_device_id, recv_stage_id, _, recv_microbatch_id, recv_index, _ = a_task
        # 找到对应的通信任务
        has_cycle, cycle_index = dfs(communication_graph[recv_device_id][recv_index],rec_stack, visited, communication_graph,)
        if has_cycle:
            # 如果发现环，返回 True 和当前任务的 index
            return True, idx
    rec_stack.remove(key)
    return False, -1  # 不存在环

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
        if infor is None or infor == '' or infor[0] not in ['b','w','f']:
            continue
        step_type, microbatch_id, stage_id = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        device_id = _get_deviceid_by_alignment(stage_id, stage_alignment)
        chunk_id = _get_chunk_by_stage(stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    # print('[')
    for d in range(len(stage_alignment)):
        each_steps_num = len(stage_alignment[d])*num_microbatches
        f_num, b_num, w_num = count_steps(device_steps[d])
        each_steps_num = len(stage_alignment[d])*num_microbatches
        assert f_num == each_steps_num and b_num == each_steps_num and (w_num ==0 or w_num == each_steps_num), f'rank: {d}, right_num: {each_steps_num}, f_num: {f_num}, b_num: {b_num}, w_num: {w_num}'
        device_steps[d].sort(key=lambda x: x[-2])
    #     print(f'{device_steps[d]},')
    # print(']')
    return device_steps
def comm_graph_muti_chunk(grouped_data, stage_alignment):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []

    # 找到最大值
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
        
        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, stage_id, microbatch_id)
            comm_op['B'] = []
            comm_op['A'] = []
            # if op != 'f' and op !='b':
            #     communication_stage.append(comm_op)
            #     continue
            
            # 处理上一个 stage (stage-1)
            for i in range(len(needrecv['F_stage'])):
                recvFstage_id = needrecv['F_stage'][i]
                recvFdevice_id = needrecv['F_device'][i]
                prev_stage_ops = grouped_data[recvFdevice_id]
                for n, prev_op in enumerate(prev_stage_ops):
                    prev_op_name, prev_microbatch_id, prev_stage_id, prev_chunk_id, prev_start_time, prev_end_time = prev_op
                    # if prev_start_time > end_time:
                    #     break
                    # 跳过已经接收的 f 操作
                    if prev_op in received_prev_stage:
                        continue
                    if prev_op_name != 'f' or prev_stage_id != recvFstage_id:
                        continue
                    # 如果本次操作为f 且microbatch_id 相同
                    if op == 'f' and prev_microbatch_id == microbatch_id and prev_stage_id == stage_id-1:           
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = prev_end_time
                    interval_end = prev_stage_ops[n + 1][-2] if n + 1 < len(prev_stage_ops) else prev_end_time

                    # 计算四个个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist,next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收

                    if min_dist == current_start_dist:
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('f',prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
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
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = next_end_time
                    interval_end = next_stage_ops[n + 1][-2] if n + 1 < len(next_stage_ops) else next_end_time

                    # 计算四个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist, next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收
                    
                    if min_dist == current_start_dist:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break
            comm_op['B'].sort(key=lambda x:x[1])
            comm_op['A'].sort(key=lambda x:x[1])

            # 将通信元组添加到当前 stage 的通信图中
            communication_stage.append(comm_op)
        # 将当前 stage 的通信图添加到总的通信图中
        communication_graph.append(communication_stage)
        #print(communication_stage)
    #communication_graph = detect_cycle_deadlock_mutichunk(communication_graph,stage_alignment)
    communication_graph = detect_cross_deadlock_mutichunk(communication_graph,stage_alignment)
    recvnum(communication_graph)
    # print('[')
    # # # 输出通信图
    # for rank_id, comm_stage in enumerate(communication_graph):
    #     print(f'{comm_stage},')
    # print(']')
    return communication_graph

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
        # if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-发-收）
        #     continue
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

            if rank_id % 2 == 0:
                #死锁场景
                #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
                needjude = op['A']
                if current_index + 1 < len_rank_ops:
                    next_op = rank_ops_copy[current_index + 1]
                    if len(next_op['B'])>0:
                        needjude += next_op['B']
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index, _ = judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    recvDevice_op = copy.deepcopy(communication_graph[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']+recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index,_ = rc
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
                                next_recv_op_type_B, next_recv_end_time_B, next_recv_device_id_B, next_recv_stage_id_B, next_recv_chunk_id_B, next_recv_microbatch_id_B, next_index_B,_ = recvDevice_nextop_n_b
                                if (next_recv_op_type_B,next_recv_stage_id_B,next_recv_microbatch_id_B) == op['Infor']:
                                    rnbresult = communication_graph[recv_device_id][index+judgedistance]['B'].pop(rnb)
                                    communication_graph[recv_device_id][index]['A'].append(rnbresult)
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            recvDevice_nextop_na = recvDevice_nextop_n['A']
                            for rna, recvDevice_nextop_n_a in enumerate(recvDevice_nextop_na):
                                next_recv_op_type_A, next_recv_end_time_A, next_recv_device_id_A, next_recv_stage_id_A, next_recv_chunk_id_A, next_recv_microbatch_id_A, next_index_A,_ = recvDevice_nextop_n_a
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
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_chunk_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    if not goonjudge:
                        break
            else:
                needjude = list(reversed(op['A']))+op['B']#TODO
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_stream_id, recv_microbatch_id, index, _ = judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    if recv_op_type == op_type and microbatch_id == recv_microbatch_id and ((op_type == "f" and recv_stage_id>stage_id) or (op_type == "b" and recv_stage_id<stage_id)):
                        communication_graph[rank_id][current_index+1]['B'].append(judgeop)
                        if judgeop in op['A']:
                            communication_graph[rank_id][current_index]['A'].remove(judgeop)
                        else:
                            communication_graph[rank_id][current_index]['B'].remove(judgeop)
                        continue

                    recvDevice_op = copy.deepcopy(communication_graph[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index,_ = rc
                        if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                            goonjudge = False
                            break
                    judgedistance = 0
                    for rv_,rv in enumerate(recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index,_ = rv
                        if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                            communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index]['B'].pop(rv_))
                            goonjudge = False
                            break
                    if not goonjudge:
                        break
                    # opInfor = op['Infor']
                    # print(f'opInfor:{opInfor}')
                    # print(f'needjude:{judgeop}')
                    # print(f'recvDevice_op:{recvDevice_op}')
                    while(goonjudge is True):
                        judgedistance += 1
                        if index-judgedistance >= 0 :
                            recvDevice_nextop_bab = copy.deepcopy(communication_graph[recv_device_id][index-judgedistance])
                            for rnbab_id,rnbab in enumerate(recvDevice_nextop_bab['A']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['A'].pop(rnbab_id))
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            for rnbab__id,rnbab_ in enumerate(recvDevice_nextop_bab['B']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab_
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['B'].pop(rnbab__id))
                                    goonjudge = False
                                    break
                        if not goonjudge:
                            break
                        if index+judgedistance < len(communication_graph_copy[recv_device_id]) :
                            recvDevice_nextop_bab = communication_graph[recv_device_id][index+judgedistance]
                            for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    #print(judgedistance)
                    # recvnum(communication_graph)
                    if not goonjudge:
                        break
    return communication_graph

def _get_deviceid_mutistream(stream: int, stage_id: int, stage_alignment:list) -> int:
    for device_id in range(len(stage_alignment)):
        if stage_id == stage_alignment[device_id][stream]:
            return device_id
    return -1
def order_result_mutistream(input: str, stage_alignment: list) -> None:
#[[0,3],[1,2],[2,1],[3,0]]
#0_w_7_2,902.0,914.0
    # input = input.replace("", "")
    # input = input.replace(" ", "")
    device_steps = [[] for _ in range(len(stage_alignment))]
    all_step = input.split('\n')
    #print(all_step)
    for step in all_step:
        if step == '':
            continue
        start_time = float(step.split(',')[-2])
        end_time = float(step.split(',')[-1])
        infor = step.split(',')[0]
        stream, step_type, microbatch_id, stage_id = infor.split('_')
        if step_type is None or step_type == '' or step_type not in ['b','w','f']:
            continue
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        stream = int(stream)
        device_id = _get_deviceid_mutistream(stream, stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, stream, start_time, end_time))
    print('[')
    for d in range(len(stage_alignment)):
        device_steps[d].sort(key=lambda x: x[-2])
        print(f'{device_steps[d]},')
    print(']')
    return device_steps
def comm_graph_mutistream(grouped_data, stage_alignment):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []
    # 找到最大值
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    # 遍历每个 stage
    for device_id, stage_ops in enumerate(grouped_data):
        stages = stage_alignment[device_id]
        needrecv = {}

        needrecv['F_stage'] = [s-1 for s in stages]
        needrecv['F_device'] = [_get_deviceid_mutistream(stream,s,stage_alignment) for stream,s in enumerate(needrecv['F_stage'])]
        needrecv['B_stage'] = [s+1 for s in stages]
        needrecv['B_device'] = [_get_deviceid_mutistream(stream,s,stage_alignment) for stream,s in enumerate(needrecv['B_stage'])]
        print(needrecv)
        communication_stage = []
        # 标记已经接收的操作
        received_prev_stage = set()  # 记录每个 stage 中已经接收的 f 操作
        received_next_stage = set()  # 记录每个 stage 中已经接收的 b 操作
        
        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, stage_id, stream_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, stage_id, stream_id,microbatch_id)
            comm_op['B'] = []
            comm_op['A'] = []
            # if op != 'f' and op !='b':
            #     communication_stage.append(comm_op)
            #     continue

            # 处理上一个 stage (stage-1)
            for i in range(len(needrecv['F_stage'])):
                recvFstage_id = needrecv['F_stage'][i]
                recvFdevice_id = needrecv['F_device'][i]
                if recvFdevice_id < 0:
                    continue
                prev_stage_ops = grouped_data[recvFdevice_id]
                for n, prev_op in enumerate(prev_stage_ops):
                    prev_op_name, prev_microbatch_id, prev_stage_id, prev_stream_id, prev_start_time, prev_end_time = prev_op
                    # if prev_start_time > end_time:
                    #     break
                    # 跳过已经接收的 f 操作
                    if prev_op in received_prev_stage:
                        continue
                    if prev_op_name != 'f' or prev_stage_id != recvFstage_id:
                        continue

                    # 如果本次操作为f 且microbatch_id 相同
                    if op == 'f' and prev_microbatch_id == microbatch_id and prev_stage_id == stage_id-1:           
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_stream_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = prev_end_time
                    interval_end = prev_stage_ops[n + 1][-2] if n + 1 < len(prev_stage_ops) else prev_end_time

                    # 计算四个个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist,next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收

                    if min_dist == current_start_dist:
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_stream_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('f',prev_end_time, recvFdevice_id, prev_stage_id,prev_stream_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break

            # 处理下一个 stage (stage+1)
            for j in range(len(needrecv['B_stage'])):
                recvBstage_id = needrecv['B_stage'][j]
                recvBdevice_id = needrecv['B_device'][j]
                if recvBdevice_id < 0:####!!!!粗心写成了recvFdevice_id
                    continue
                next_stage_ops = grouped_data[recvBdevice_id]
                for n, next_op in enumerate(next_stage_ops):
                    next_op_name, next_microbatch_id, next_stage_id, next_stream_id, next_start_time, next_end_time = next_op
                    # if next_start_time > end_time:
                    #     break
                    if next_op_name != 'b' or recvBstage_id != next_stage_id:
                        continue
                    # 跳过已经接收的 f 操作
                    if next_op in received_next_stage :
                        continue
                    # 如果本次操作为b 且 microbatch_id 相同
                    if op == 'b' and next_microbatch_id == microbatch_id and next_stage_id == stage_id+1:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_stream_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = next_end_time
                    interval_end = next_stage_ops[n + 1][-2] if n + 1 < len(next_stage_ops) else next_end_time

                    # 计算四个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist, next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收
                    
                    if min_dist == current_start_dist:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_stream_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_stream_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break

   
            comm_op['B'].sort(key=lambda x:x[1])
            comm_op['A'].sort(key=lambda x:x[1])

            # 将通信元组添加到当前 stage 的通信图中
            communication_stage.append(comm_op)
        # 将当前 stage 的通信图添加到总的通信图中
        communication_graph.append(communication_stage)
        #print(communication_stage)
    #communication_graph = detect_cycle_deadlock_mutistream(communication_graph,stage_alignment)
    communication_graph = detect_cross_deadlock_mutistream(communication_graph,stage_alignment)
    recvnum(communication_graph)
    print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(communication_graph):
        print(f'{comm_stage},')
    print(']')
    #print(communication_graph)
def detect_cycle_deadlock_mutistream(communication_graph, stage_alignment):
    #sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    communication_graph_copy = copy.deepcopy(communication_graph)
    for rank_id, rank_ops in enumerate(communication_graph_copy):
        if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-收-发）
            continue
        len_rank_ops = len(rank_ops)#rank_ops = communication_graph[rank_id]
        rank_ops_copy = copy.deepcopy(rank_ops)
        for current_index, op in enumerate(rank_ops_copy):
            #op = rank_ops[current_index]
            op_type, stage_id, stream_id, microbatch_id = op['Infor']
            #没有发送需求就不会有死锁
            if op_type == 'f':
                if stage_id == max_stage_id:
                    continue
                dst_rank_id = _get_deviceid_mutistream(stream_id,stage_id+1,stage_alignment)
            elif op_type == 'b':
                if stage_id == 0:
                    continue
                dst_rank_id = _get_deviceid_mutistream(stream_id, stage_id-1,stage_alignment)
            else:
                continue

            #判断环
            visited = set()  # 用于记录已经访问过的任务
            rec_stack = set()  # 用于记录当前递归栈中的任务
            has_cycle, cycle_index = dfs(op,rec_stack, visited,communication_graph_copy,)
            if has_cycle and cycle_index != -1 and current_index < len_rank_ops - 1:
                communication_graph[rank_id][current_index + 1]['B'].append(communication_graph[rank_id][current_index]['A'].pop(cycle_index))
                print(f"cycle dead lock:{op['Infor']}")
    return communication_graph
def detect_cross_deadlock_mutistream(communication_graph, stage_alignment):
    #sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    communication_graph_copy = copy.deepcopy(communication_graph)
    for rank_id, rank_ops in enumerate(communication_graph_copy):
        # if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-发-收）
        #     continue
        len_rank_ops = len(rank_ops)
        rank_ops_copy = copy.deepcopy(rank_ops)
        for current_index, op in enumerate(rank_ops_copy):
            op_type, stage_id, stream_id, microbatch_id = op['Infor']
            #没有发送需求就不会有死锁
            if op_type == 'f':
                if stage_id == max_stage_id:
                    continue
                dst_rank_id = _get_deviceid_mutistream(stream_id,stage_id+1,stage_alignment)
            elif op_type == 'b':
                if stage_id == 0:
                    continue
                dst_rank_id = _get_deviceid_mutistream(stream_id,stage_id-1,stage_alignment)
            else:
                continue
            if rank_id%2 == 0 : #因为修改了通信，只判断偶数rank（收-算-发-收）
                # #死锁场景1
                #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
                needjude = op['A']
                if current_index + 1 < len_rank_ops:
                    next_op = rank_ops_copy[current_index + 1]
                    if len(next_op['B'])>0:
                        needjude += next_op['B']
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_stream_id, recv_microbatch_id, index, _ = judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    recvDevice_op = copy.deepcopy(communication_graph[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']+recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index,_ = rc
                        if (next_recv_op_type,next_recv_stage_id,next_recv_stream_id,next_recv_microbatch_id) == op['Infor']:
                            goonjudge = False
                            break
                    judgedistance = 0

                    if not goonjudge:
                        break
                    opInfor = op['Infor']
                    print(f'opInfor:{opInfor}')
                    print(f'needjude:{judgeop}')
                    print(f'recvDevice_op:{recvDevice_op}')
                    while(goonjudge is True):
                        judgedistance += 1
                        if index+judgedistance < len_rank_ops :
                            recvDevice_nextop_n = copy.deepcopy(communication_graph[recv_device_id][index+judgedistance])
                            recvDevice_nextop_nb = recvDevice_nextop_n['B']
                            for rnb, recvDevice_nextop_n_b in enumerate(recvDevice_nextop_nb):
                                next_recv_op_type_B, next_recv_end_time_B, next_recv_device_id_B, next_recv_stage_id_B, next_recv_stream_id_B, next_recv_microbatch_id_B, next_index_B,_ = recvDevice_nextop_n_b
                                if (next_recv_op_type_B,next_recv_stage_id_B,next_recv_stream_id_B,next_recv_microbatch_id_B) == op['Infor']:
                                    recvDevice_nextop_n_numb = len(recvDevice_nextop_n['B'])
                                    need_addnum = len(recvDevice_op['A'])
                                    print(f'pop_op:{recvDevice_nextop_n}')
                                    rnbresult = communication_graph[recv_device_id][index+judgedistance]['B'].pop(rnb)
                                    communication_graph[recv_device_id][index]['A'].append(rnbresult)
                                    print(f'pop:{rnbresult}')
                                    print(f'after_pop_op:{recvDevice_nextop_n}')
                                    print(f'after_recvDevice_op:{recvDevice_op}')
                                    assert len(communication_graph[recv_device_id][index+judgedistance]['B']) == recvDevice_nextop_n_numb-1
                                    assert len(communication_graph[recv_device_id][index]['A']) == need_addnum+1
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            recvDevice_nextop_na = recvDevice_nextop_n['A']
                            for rna, recvDevice_nextop_n_a in enumerate(recvDevice_nextop_na):
                                next_recv_op_type_A, next_recv_end_time_A, next_recv_device_id_A, next_recv_stage_id_A, next_recv_stream_id_A, next_recv_microbatch_id_A, next_index_A,_ = recvDevice_nextop_n_a
                                if (next_recv_op_type_A,next_recv_stage_id_A,next_recv_stream_id_A,next_recv_microbatch_id_A) == op['Infor']:
                                    recvDevice_nextop_n_numa = len(recvDevice_nextop_n['A'])
                                    need_addnum = len(recvDevice_op['A'])
                                    print(f'popop:{recvDevice_nextop_n}')
                                    rnaresult = communication_graph[recv_device_id][index+judgedistance]['A'].pop(rna)
                                    communication_graph[recv_device_id][index]['A'].append(rnaresult)
                                    print(f'pop:{rnaresult}')
                                    print(f'after_pop_op:{recvDevice_nextop_n}')
                                    print(f'after_recvDevice_op:{recvDevice_op}')
                                    assert len(communication_graph[recv_device_id][index+judgedistance]['A']) == recvDevice_nextop_n_numa-1
                                    assert len(communication_graph[recv_device_id][index]['A']) == need_addnum+1
                                    goonjudge = False
                                    break
                        if not goonjudge:
                            break
                        if index-judgedistance >= 0 :
                            recvDevice_nextop_bab = communication_graph[recv_device_id][index-judgedistance]
                            for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_stream_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    print(judgedistance)
                    recvnum(communication_graph)
                    if not goonjudge:
                        break
            else:
                needjude = op['A']+op['B']
                # if current_index + 1 < len_rank_ops:
                #     next_op = rank_ops[current_index + 1]
                #     if len(next_op['B'])>0:
                #         needjude += next_op['B']
                for judgeop in needjude:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_stream_id, recv_microbatch_id, index, _ = judgeop
                    if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                        continue
                    recvDevice_op = copy.deepcopy(communication_graph[recv_device_id][index])
                    goonjudge = True
                    for rc in (recvDevice_op['A']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index,_ = rc
                        if (next_recv_op_type,next_recv_stage_id,next_recv_stream_id,next_recv_microbatch_id) == op['Infor']:
                            goonjudge = False
                            break
                    judgedistance = 0
                    for rc_,rv in enumerate(recvDevice_op['B']):
                        next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_stream_id, next_recv_microbatch_id, next_index,_ = rv
                        if (next_recv_op_type,next_recv_stage_id,next_recv_stream_id,next_recv_microbatch_id) == op['Infor']:
                            communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index]['B'].pop(rc_))
                            goonjudge = False
                            break
                    if not goonjudge:
                        break
                    opInfor = op['Infor']
                    print(f'opInfor:{opInfor}')
                    print(f'needjude:{judgeop}')
                    print(f'recvDevice_op:{recvDevice_op}')
                    while(goonjudge is True):
                        judgedistance += 1
                        if index-judgedistance >= 0 :
                            recvDevice_nextop_bab = copy.deepcopy(communication_graph[recv_device_id][index-judgedistance])
                            for rnbab_id,rnbab in enumerate(recvDevice_nextop_bab['A']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_stream_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['A'].pop(rnbab_id))
                                    goonjudge = False
                                    break
                            if not goonjudge:
                                break
                            for rnbab__id,rnbab_ in enumerate(recvDevice_nextop_bab['B']):
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab_
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_stream_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    communication_graph[recv_device_id][index]['A'].append(communication_graph[recv_device_id][index-judgedistance]['B'].pop(rnbab__id))
                                    goonjudge = False
                                    break
                        if not goonjudge:
                            break
                        if index+judgedistance < len_rank_ops :
                            recvDevice_nextop_bab = communication_graph[recv_device_id][index+judgedistance]
                            for rnbab in recvDevice_nextop_bab['A']+recvDevice_nextop_bab['B']:
                                next_recv_op_type_bab, next_recv_end_time_bab, next_recv_device_id_bab, next_recv_stage_id_bab, next_recv_stream_id_bab, next_recv_microbatch_id_bab, next_index_bab,_ = rnbab
                                if (next_recv_op_type_bab,next_recv_stage_id_bab,next_recv_stream_id_bab,next_recv_microbatch_id_bab) == op['Infor']:
                                    goonjudge = False
                                    break
                    print(judgedistance)
                    recvnum(communication_graph)
                    if not goonjudge:
                        break         
    return communication_graph

def generate_Interleaved_4pp_20chunk_16mb():
    pp_size = 4
    chunk_size = 20
    num_microbatches = 16
    stage_alignment = []
    for i in range(pp_size):
        stages_in_ranks = []
        for j in range(chunk_size):
            stages_in_ranks.append(i+j*pp_size)
        stage_alignment.append(stages_in_ranks)
    input_str = """"""
    unified_scheduler = order_result_mutichunk(input_str,stage_alignment,num_microbatches)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)
    return stage_alignment, unified_scheduler, comm_graph
def generate_Wavelike_4pp_20chunk_16mb():
    pp_size = 4
    chunk_size = 20
    num_microbatches = 16
    stage_alignment = [[0 for _ in range(chunk_size)] for _ in range(pp_size)]
    
    #修改 stage_alignment
    for i in range(chunk_size):
        if i % 2 == 0:
            stage_alignment[0][i] = pp_size * i
            for j in range(1, pp_size):
                stage_alignment[j][i] = stage_alignment[j - 1][i] + 1
        else:
            stage_alignment[pp_size - 1][i] = pp_size * i
            for j in range(pp_size - 2, -1, -1):  # 修正循环范围
                stage_alignment[j][i] = stage_alignment[j + 1][i] + 1
    input_str =""""""
    unified_scheduler = order_result_mutichunk(input_str,stage_alignment,num_microbatches)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)
    return stage_alignment, unified_scheduler, comm_graph
def generate_():
    stage_placement = ""
    input_str=""
    with open('/mnt/petrelfs/matenghui/InternEvo/placement.txt', 'r', encoding='utf-8') as file:
        stage_placement = file.read()
    with open('/mnt/petrelfs/matenghui/InternEvo/result.txt', 'r', encoding='utf-8') as file:
        input_str = file.read()
    stage_placement = json.loads(stage_placement)
    num_microbatches = 16

    unified_scheduler = order_result_mutichunk(input_str,stage_placement,num_microbatches)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_placement)
    result = {'stage_placement':stage_placement, 'unified_scheduler':unified_scheduler, 'comm_graph':comm_graph}
    with open('/mnt/petrelfs/matenghui/InternEvo/runtime.json','w') as file:
        json.dump(result,file)

if __name__ == '__main__':
    generate_()
