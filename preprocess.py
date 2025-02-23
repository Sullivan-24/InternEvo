import copy
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

def order_result_mutichunk(input: str, stage_alignment: list) -> None:
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
    # for d in range(len(stage_alignment)):
    #     device_steps[d].sort(key=lambda x: x[-2])
    #     print(f'{device_steps[d]},')
    # print(']')
    return device_steps
def comm_graph_muti_chunk(grouped_data, stage_alignment):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []

    # 找到最大值
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])

    # import pdb;pdb.set_trace()
    # 遍历每个 stage
    for device_id, stage_ops in enumerate(grouped_data):

        stages = stage_alignment[device_id]
        needrecv = {}
        needrecv['F_stage'] = [s-1 for s in stages if s > 0 and s-1 not in stages]
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
                print(f"cycle dead lock:{op['Infor']}")
    return communication_graph

def detect_cross_deadlock_mutichunk(communication_graph, stage_alignment):
    sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for row in stage_alignment for stage_id in row])
    communication_graph_copy = copy.deepcopy(communication_graph)
    for rank_id, rank_ops in enumerate(communication_graph_copy):
        if rank_id%2 == 1 : #因为修改了通信，只判断偶数rank（收-算-发-收）
            continue
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

            #死锁场景
            #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
            needjude = op['A']
            if current_index + 1 < len_rank_ops:
                next_op = rank_ops[current_index + 1]
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
                    if index+judgedistance < len_rank_ops :
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
                next_stage_ops = grouped_data[recvBdevice_id]
                if recvFdevice_id < 0:
                    continue
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
                    next_op = rank_ops[current_index + 1]
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
def generate_():
#     stage_alignment = [[0,4],[1,5],[2,6],[3,7]]
#     input_str = '''f_0_0,0.0,12.0
# f_1_0,12.0,24.0
# f_0_1,12.0,24.0
# f_2_0,24.0,36.0
# f_1_1,24.0,36.0
# f_0_2,24.0,36.0
# f_3_0,36.0,48.0
# f_2_1,36.0,48.0
# f_1_2,36.0,48.0
# f_0_3,36.0,48.0
# f_0_4,48.0,60.0
# f_3_1,48.0,60.0
# f_2_2,48.0,60.0
# f_1_3,48.0,60.0
# f_1_4,60.0,72.0
# f_0_5,60.0,72.0
# f_3_2,60.0,72.0
# f_2_3,60.0,72.0
# f_2_4,72.0,84.0
# f_1_5,72.0,84.0
# f_0_6,72.0,84.0
# f_3_3,72.0,84.0
# f_3_4,84.0,96.0
# f_2_5,84.0,96.0
# f_1_6,84.0,96.0
# f_0_7,84.0,96.0
# f_4_0,96.0,108.0
# f_3_5,96.0,108.0
# f_2_6,96.0,108.0
# b_0_7,96.0,132.0
# f_5_0,108.0,120.0
# f_4_1,108.0,120.0
# f_6_0,120.0,132.0
# b_0_6,132.0,168.0
# f_1_7,132.0,144.0
# b_1_7,144.0,180.0
# b_0_5,168.0,204.0
# f_3_6,168.0,180.0
# b_1_6,180.0,216.0
# f_2_7,180.0,192.0
# b_2_7,192.0,228.0
# b_0_4,204.0,240.0
# f_5_1,204.0,216.0
# b_1_5,216.0,252.0
# f_4_2,216.0,228.0
# b_2_6,228.0,264.0
# f_3_7,228.0,240.0
# f_7_0,240.0,252.0
# b_0_3,240.0,276.0
# b_1_4,252.0,288.0
# f_6_1,252.0,264.0
# b_2_5,264.0,300.0
# f_5_2,264.0,276.0
# b_0_2,276.0,312.0
# f_4_3,276.0,288.0
# f_4_4,288.0,300.0
# b_1_3,288.0,324.0
# b_2_4,300.0,336.0
# f_4_5,300.0,312.0
# b_0_1,312.0,348.0
# f_4_6,312.0,324.0
# b_1_2,324.0,360.0
# f_4_7,324.0,336.0
# f_8_0,336.0,348.0
# b_2_3,336.0,372.0
# b_0_0,348.0,384.0
# f_7_1,348.0,360.0
# b_1_1,360.0,396.0
# f_6_2,360.0,372.0
# b_2_2,372.0,408.0
# f_5_3,372.0,384.0
# f_5_4,384.0,396.0
# b_3_7,384.0,420.0
# b_1_0,396.0,432.0
# f_5_5,396.0,408.0
# b_2_1,408.0,444.0
# f_5_6,408.0,420.0
# b_3_6,420.0,456.0
# f_5_7,420.0,432.0
# f_9_0,432.0,444.0
# b_4_7,432.0,468.0
# b_2_0,444.0,480.0
# f_8_1,444.0,456.0
# b_3_5,456.0,492.0
# f_7_2,456.0,468.0
# b_4_6,468.0,504.0
# f_6_3,468.0,480.0
# f_6_4,480.0,492.0
# b_5_7,480.0,516.0
# b_3_4,492.0,528.0
# f_6_5,492.0,504.0
# b_4_5,504.0,540.0
# f_6_6,504.0,516.0
# b_5_6,516.0,552.0
# f_6_7,516.0,528.0
# f_10_0,528.0,540.0
# b_3_3,528.0,564.0
# b_4_4,540.0,576.0
# f_9_1,540.0,552.0
# b_5_5,552.0,588.0
# f_8_2,552.0,564.0
# b_3_2,564.0,600.0
# f_7_3,564.0,576.0
# f_7_4,576.0,588.0
# b_4_3,576.0,612.0
# b_5_4,588.0,624.0
# f_7_5,588.0,600.0
# b_3_1,600.0,636.0
# f_7_6,600.0,612.0
# b_4_2,612.0,648.0
# f_7_7,612.0,624.0
# f_11_0,624.0,636.0
# b_5_3,624.0,660.0
# b_3_0,636.0,672.0
# f_10_1,636.0,648.0
# b_4_1,648.0,684.0
# f_9_2,648.0,660.0
# b_5_2,660.0,696.0
# f_8_3,660.0,672.0
# f_8_4,672.0,684.0
# b_6_7,672.0,708.0
# b_4_0,684.0,720.0
# f_8_5,684.0,696.0
# b_5_1,696.0,732.0
# f_8_6,696.0,708.0
# b_6_6,708.0,744.0
# f_8_7,708.0,720.0
# f_12_0,720.0,732.0
# b_7_7,720.0,756.0
# b_5_0,732.0,768.0
# f_11_1,732.0,744.0
# b_6_5,744.0,780.0
# f_10_2,744.0,756.0
# b_7_6,756.0,792.0
# f_9_3,756.0,768.0
# f_9_4,768.0,780.0
# b_8_7,768.0,804.0
# b_6_4,780.0,816.0
# f_9_5,780.0,792.0
# b_7_5,792.0,828.0
# f_9_6,792.0,804.0
# b_8_6,804.0,840.0
# f_9_7,804.0,816.0
# f_13_0,816.0,828.0
# b_6_3,816.0,852.0
# b_7_4,828.0,864.0
# f_12_1,828.0,840.0
# b_8_5,840.0,876.0
# f_11_2,840.0,852.0
# b_6_2,852.0,888.0
# f_10_3,852.0,864.0
# f_10_4,864.0,876.0
# b_7_3,864.0,900.0
# b_8_4,876.0,912.0
# f_10_5,876.0,888.0
# b_6_1,888.0,924.0
# f_10_6,888.0,900.0
# b_7_2,900.0,936.0
# f_10_7,900.0,912.0
# f_14_0,912.0,924.0
# b_8_3,912.0,948.0
# b_6_0,924.0,960.0
# f_13_1,924.0,936.0
# b_7_1,936.0,972.0
# f_12_2,936.0,948.0
# b_8_2,948.0,984.0
# f_11_3,948.0,960.0
# f_11_4,960.0,972.0
# b_9_7,960.0,996.0
# b_7_0,972.0,1008.0
# f_11_5,972.0,984.0
# b_8_1,984.0,1020.0
# f_11_6,984.0,996.0
# b_9_6,996.0,1032.0
# f_11_7,996.0,1008.0
# f_15_0,1008.0,1020.0
# b_10_7,1008.0,1044.0
# b_8_0,1020.0,1056.0
# f_14_1,1020.0,1032.0
# b_9_5,1032.0,1068.0
# f_13_2,1032.0,1044.0
# b_10_6,1044.0,1080.0
# f_12_3,1044.0,1056.0
# f_12_4,1056.0,1068.0
# b_11_7,1056.0,1092.0
# b_9_4,1068.0,1104.0
# f_12_5,1068.0,1080.0
# b_10_5,1080.0,1116.0
# f_12_6,1080.0,1092.0
# b_11_6,1092.0,1128.0
# f_12_7,1092.0,1104.0
# b_9_3,1104.0,1140.0
# b_10_4,1116.0,1152.0
# f_15_1,1116.0,1128.0
# b_11_5,1128.0,1164.0
# f_14_2,1128.0,1140.0
# b_9_2,1140.0,1176.0
# f_13_3,1140.0,1152.0
# f_13_4,1152.0,1164.0
# b_10_3,1152.0,1188.0
# b_11_4,1164.0,1200.0
# f_13_5,1164.0,1176.0
# b_9_1,1176.0,1212.0
# f_13_6,1176.0,1188.0
# b_10_2,1188.0,1224.0
# f_13_7,1188.0,1200.0
# b_11_3,1200.0,1236.0
# b_9_0,1212.0,1248.0
# b_10_1,1224.0,1260.0
# f_15_2,1224.0,1236.0
# b_11_2,1236.0,1272.0
# f_14_3,1236.0,1248.0
# f_14_4,1248.0,1260.0
# b_12_7,1248.0,1284.0
# b_10_0,1260.0,1296.0
# f_14_5,1260.0,1272.0
# b_11_1,1272.0,1308.0
# f_14_6,1272.0,1284.0
# b_12_6,1284.0,1320.0
# f_14_7,1284.0,1296.0
# b_13_7,1296.0,1332.0
# b_11_0,1308.0,1344.0
# b_12_5,1320.0,1356.0
# b_13_6,1332.0,1368.0
# f_15_3,1332.0,1344.0
# f_15_4,1344.0,1356.0
# b_14_7,1344.0,1380.0
# b_12_4,1356.0,1392.0
# f_15_5,1356.0,1368.0
# b_13_5,1368.0,1404.0
# f_15_6,1368.0,1380.0
# b_14_6,1380.0,1416.0
# f_15_7,1380.0,1392.0
# b_12_3,1392.0,1428.0
# b_13_4,1404.0,1440.0
# b_14_5,1416.0,1452.0
# b_12_2,1428.0,1464.0
# b_15_7,1428.0,1464.0
# b_14_4,1452.0,1488.0
# b_12_1,1464.0,1500.0
# b_15_6,1464.0,1500.0
# b_13_3,1464.0,1500.0
# b_12_0,1500.0,1536.0
# b_15_5,1500.0,1536.0
# b_13_2,1500.0,1536.0
# b_14_3,1500.0,1536.0
# b_15_4,1536.0,1572.0
# b_13_1,1536.0,1572.0
# b_14_2,1536.0,1572.0
# b_13_0,1572.0,1608.0
# b_14_1,1572.0,1608.0
# b_15_3,1572.0,1608.0
# b_14_0,1608.0,1644.0
# b_15_2,1608.0,1644.0
# b_15_1,1644.0,1680.0
# b_15_0,1680.0,1716.0'''
#     unified_scheduler = order_result_mutichunk(input_str,stage_alignment)
#     comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)
#     return stage_alignment,unified_scheduler,comm_graph
    stage_alignment = [[0, 8, 16, 24, 32, 40, 48, 56, 64, 72], [1, 9, 17, 25, 33, 41, 49, 57, 65, 73], [2, 10, 18, 26, 34, 42, 50, 58, 66, 74], [3, 11, 19, 27, 35, 43, 51, 59, 67, 75], [4, 12, 20, 28, 36, 44, 52, 60, 68, 76], [5, 13, 21, 29, 37, 45, 53, 61, 69, 77], [6, 14, 22, 30, 38, 46, 54, 62, 70, 78], [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]]#[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]#[[0,15],[1,14],[2,13],[3,12],[4,11],[5,10],[6,9],[7,8]]##[[0,4],[1,5],[2,6],[3,7]]##[[0,7],[1,6],[2,5],[3,4]]
#     # 输入字符串
    input_str = """f_0_0,0,4
f_1_0,4,8
f_0_1,4,8
f_2_0,8,12
f_1_1,8,12
f_0_2,8,12
f_3_0,12,16
f_2_1,12,16
f_1_2,12,16
f_0_3,12,16
f_4_0,16,20
f_3_1,16,20
f_2_2,16,20
f_1_3,16,20
f_0_4,16,20
f_5_0,20,24
f_4_1,20,24
f_3_2,20,24
f_2_3,20,24
f_1_4,20,24
f_0_5,20,24
f_6_0,24,28
f_5_1,24,28
f_4_2,24,28
f_3_3,24,28
f_2_4,24,28
f_1_5,24,28
f_0_6,24,28
f_7_0,28,32
f_6_1,28,32
f_5_2,28,32
f_4_3,28,32
f_3_4,28,32
f_2_5,28,32
f_1_6,28,32
f_0_7,28,32
f_0_8,32,36
f_7_1,32,36
f_6_2,32,36
f_5_3,32,36
f_4_4,32,36
f_3_5,32,36
f_2_6,32,36
f_1_7,32,36
f_1_8,36,40
f_0_9,36,40
f_7_2,36,40
f_6_3,36,40
f_5_4,36,40
f_4_5,36,40
f_3_6,36,40
f_2_7,36,40
f_2_8,40,44
f_1_9,40,44
f_0_10,40,44
f_7_3,40,44
f_6_4,40,44
f_5_5,40,44
f_4_6,40,44
f_3_7,40,44
f_3_8,44,48
f_2_9,44,48
f_1_10,44,48
f_0_11,44,48
f_7_4,44,48
f_6_5,44,48
f_5_6,44,48
f_4_7,44,48
f_4_8,48,52
f_3_9,48,52
f_2_10,48,52
f_1_11,48,52
f_0_12,48,52
f_7_5,48,52
f_6_6,48,52
f_5_7,48,52
f_5_8,52,56
f_4_9,52,56
f_3_10,52,56
f_2_11,52,56
f_1_12,52,56
f_0_13,52,56
f_7_6,52,56
f_6_7,52,56
f_6_8,56,60
f_5_9,56,60
f_4_10,56,60
f_3_11,56,60
f_2_12,56,60
f_1_13,56,60
f_0_14,56,60
f_7_7,56,60
f_7_8,60,64
f_6_9,60,64
f_5_10,60,64
f_4_11,60,64
f_3_12,60,64
f_2_13,60,64
f_1_14,60,64
f_0_15,60,64
f_0_16,64,68
f_7_9,64,68
f_6_10,64,68
f_5_11,64,68
f_4_12,64,68
f_3_13,64,68
f_2_14,64,68
f_1_15,64,68
f_1_16,68,72
f_0_17,68,72
f_7_10,68,72
f_6_11,68,72
f_5_12,68,72
f_4_13,68,72
f_3_14,68,72
f_2_15,68,72
f_2_16,72,76
f_1_17,72,76
f_0_18,72,76
f_7_11,72,76
f_6_12,72,76
f_5_13,72,76
f_4_14,72,76
f_3_15,72,76
f_3_16,76,80
f_2_17,76,80
f_1_18,76,80
f_0_19,76,80
f_7_12,76,80
f_6_13,76,80
f_5_14,76,80
f_4_15,76,80
f_4_16,80,84
f_3_17,80,84
f_2_18,80,84
f_1_19,80,84
f_0_20,80,84
f_7_13,80,84
f_6_14,80,84
f_5_15,80,84
f_5_16,84,88
f_4_17,84,88
f_3_18,84,88
f_2_19,84,88
f_1_20,84,88
f_0_21,84,88
f_7_14,84,88
f_6_15,84,88
f_6_16,88,92
f_5_17,88,92
f_4_18,88,92
f_3_19,88,92
f_2_20,88,92
f_1_21,88,92
f_0_22,88,92
f_7_15,88,92
f_7_16,92,96
f_6_17,92,96
f_5_18,92,96
f_4_19,92,96
f_3_20,92,96
f_2_21,92,96
f_1_22,92,96
f_0_23,92,96
f_0_24,96,100
f_7_17,96,100
f_6_18,96,100
f_5_19,96,100
f_4_20,96,100
f_3_21,96,100
f_2_22,96,100
f_1_23,96,100
f_1_24,100,104
f_0_25,100,104
f_7_18,100,104
f_6_19,100,104
f_5_20,100,104
f_4_21,100,104
f_3_22,100,104
f_2_23,100,104
f_2_24,104,108
f_1_25,104,108
f_0_26,104,108
f_7_19,104,108
f_6_20,104,108
f_5_21,104,108
f_4_22,104,108
f_3_23,104,108
f_3_24,108,112
f_2_25,108,112
f_1_26,108,112
f_0_27,108,112
f_7_20,108,112
f_6_21,108,112
f_5_22,108,112
f_4_23,108,112
f_4_24,112,116
f_3_25,112,116
f_2_26,112,116
f_1_27,112,116
f_0_28,112,116
f_7_21,112,116
f_6_22,112,116
f_5_23,112,116
f_5_24,116,120
f_4_25,116,120
f_3_26,116,120
f_2_27,116,120
f_1_28,116,120
f_0_29,116,120
f_7_22,116,120
f_6_23,116,120
f_6_24,120,124
f_5_25,120,124
f_4_26,120,124
f_3_27,120,124
f_2_28,120,124
f_1_29,120,124
f_0_30,120,124
f_7_23,120,124
f_7_24,124,128
f_6_25,124,128
f_5_26,124,128
f_4_27,124,128
f_3_28,124,128
f_2_29,124,128
f_1_30,124,128
f_0_31,124,128
f_0_32,128,132
f_7_25,128,132
f_6_26,128,132
f_5_27,128,132
f_4_28,128,132
f_3_29,128,132
f_2_30,128,132
f_1_31,128,132
f_1_32,132,136
f_0_33,132,136
f_7_26,132,136
f_6_27,132,136
f_5_28,132,136
f_4_29,132,136
f_3_30,132,136
f_2_31,132,136
f_2_32,136,140
f_1_33,136,140
f_0_34,136,140
f_7_27,136,140
f_6_28,136,140
f_5_29,136,140
f_4_30,136,140
f_3_31,136,140
f_3_32,140,144
f_2_33,140,144
f_1_34,140,144
f_0_35,140,144
f_7_28,140,144
f_6_29,140,144
f_5_30,140,144
f_4_31,140,144
f_4_32,144,148
f_3_33,144,148
f_2_34,144,148
f_1_35,144,148
f_0_36,144,148
f_7_29,144,148
f_6_30,144,148
f_5_31,144,148
f_5_32,148,152
f_4_33,148,152
f_3_34,148,152
f_2_35,148,152
f_1_36,148,152
f_0_37,148,152
f_7_30,148,152
f_6_31,148,152
f_6_32,152,156
f_5_33,152,156
f_4_34,152,156
f_3_35,152,156
f_2_36,152,156
f_1_37,152,156
f_0_38,152,156
f_7_31,152,156
f_7_32,156,160
f_6_33,156,160
f_5_34,156,160
f_4_35,156,160
f_3_36,156,160
f_2_37,156,160
f_1_38,156,160
f_0_39,156,160
f_0_40,160,164
f_7_33,160,164
f_6_34,160,164
f_5_35,160,164
f_4_36,160,164
f_3_37,160,164
f_2_38,160,164
f_1_39,160,164
f_1_40,164,168
f_0_41,164,168
f_7_34,164,168
f_6_35,164,168
f_5_36,164,168
f_4_37,164,168
f_3_38,164,168
f_2_39,164,168
f_2_40,168,172
f_1_41,168,172
f_0_42,168,172
f_7_35,168,172
f_6_36,168,172
f_5_37,168,172
f_4_38,168,172
f_3_39,168,172
f_3_40,172,176
f_2_41,172,176
f_1_42,172,176
f_0_43,172,176
f_7_36,172,176
f_6_37,172,176
f_5_38,172,176
f_4_39,172,176
f_4_40,176,180
f_3_41,176,180
f_2_42,176,180
f_1_43,176,180
f_0_44,176,180
f_7_37,176,180
f_6_38,176,180
f_5_39,176,180
f_5_40,180,184
f_4_41,180,184
f_3_42,180,184
f_2_43,180,184
f_1_44,180,184
f_0_45,180,184
f_7_38,180,184
f_6_39,180,184
f_6_40,184,188
f_5_41,184,188
f_4_42,184,188
f_3_43,184,188
f_2_44,184,188
f_1_45,184,188
f_0_46,184,188
f_7_39,184,188
f_7_40,188,192
f_6_41,188,192
f_5_42,188,192
f_4_43,188,192
f_3_44,188,192
f_2_45,188,192
f_1_46,188,192
f_0_47,188,192
f_0_48,192,196
f_7_41,192,196
f_6_42,192,196
f_5_43,192,196
f_4_44,192,196
f_3_45,192,196
f_2_46,192,196
f_1_47,192,196
f_1_48,196,200
f_0_49,196,200
f_7_42,196,200
f_6_43,196,200
f_5_44,196,200
f_4_45,196,200
f_3_46,196,200
f_2_47,196,200
f_2_48,200,204
f_1_49,200,204
f_0_50,200,204
f_7_43,200,204
f_6_44,200,204
f_5_45,200,204
f_4_46,200,204
f_3_47,200,204
f_3_48,204,208
f_2_49,204,208
f_1_50,204,208
f_0_51,204,208
f_7_44,204,208
f_6_45,204,208
f_5_46,204,208
f_4_47,204,208
f_4_48,208,212
f_3_49,208,212
f_2_50,208,212
f_1_51,208,212
f_0_52,208,212
f_7_45,208,212
f_6_46,208,212
f_5_47,208,212
f_5_48,212,216
f_4_49,212,216
f_3_50,212,216
f_2_51,212,216
f_1_52,212,216
f_0_53,212,216
f_7_46,212,216
f_6_47,212,216
f_6_48,216,220
f_5_49,216,220
f_4_50,216,220
f_3_51,216,220
f_2_52,216,220
f_1_53,216,220
f_0_54,216,220
f_7_47,216,220
f_7_48,220,224
f_6_49,220,224
f_5_50,220,224
f_4_51,220,224
f_3_52,220,224
f_2_53,220,224
f_1_54,220,224
f_0_55,220,224
f_0_56,224,228
f_7_49,224,228
f_6_50,224,228
f_5_51,224,228
f_4_52,224,228
f_3_53,224,228
f_2_54,224,228
f_1_55,224,228
f_1_56,228,232
f_0_57,228,232
f_7_50,228,232
f_6_51,228,232
f_5_52,228,232
f_4_53,228,232
f_3_54,228,232
f_2_55,228,232
f_2_56,232,236
f_1_57,232,236
f_0_58,232,236
f_7_51,232,236
f_6_52,232,236
f_5_53,232,236
f_4_54,232,236
f_3_55,232,236
f_3_56,236,240
f_2_57,236,240
f_1_58,236,240
f_0_59,236,240
f_7_52,236,240
f_6_53,236,240
f_5_54,236,240
f_4_55,236,240
f_4_56,240,244
f_3_57,240,244
f_2_58,240,244
f_1_59,240,244
f_0_60,240,244
f_7_53,240,244
f_6_54,240,244
f_5_55,240,244
f_5_56,244,248
f_4_57,244,248
f_3_58,244,248
f_2_59,244,248
f_1_60,244,248
f_0_61,244,248
f_7_54,244,248
f_6_55,244,248
f_6_56,248,252
f_5_57,248,252
f_4_58,248,252
f_3_59,248,252
f_2_60,248,252
f_1_61,248,252
f_0_62,248,252
f_7_55,248,252
f_7_56,252,256
f_6_57,252,256
f_5_58,252,256
f_4_59,252,256
f_3_60,252,256
f_2_61,252,256
f_1_62,252,256
f_0_63,252,256
f_0_64,256,260
f_7_57,256,260
f_6_58,256,260
f_5_59,256,260
f_4_60,256,260
f_3_61,256,260
f_2_62,256,260
f_1_63,256,260
f_1_64,260,264
f_0_65,260,264
f_7_58,260,264
f_6_59,260,264
f_5_60,260,264
f_4_61,260,264
f_3_62,260,264
f_2_63,260,264
f_2_64,264,268
f_1_65,264,268
f_0_66,264,268
f_7_59,264,268
f_6_60,264,268
f_5_61,264,268
f_4_62,264,268
f_3_63,264,268
f_3_64,268,272
f_2_65,268,272
f_1_66,268,272
f_0_67,268,272
f_7_60,268,272
f_6_61,268,272
f_5_62,268,272
f_4_63,268,272
f_4_64,272,276
f_3_65,272,276
f_2_66,272,276
f_1_67,272,276
f_0_68,272,276
f_7_61,272,276
f_6_62,272,276
f_5_63,272,276
f_5_64,276,280
f_4_65,276,280
f_3_66,276,280
f_2_67,276,280
f_1_68,276,280
f_0_69,276,280
f_7_62,276,280
f_6_63,276,280
f_6_64,280,284
f_5_65,280,284
f_4_66,280,284
f_3_67,280,284
f_2_68,280,284
f_1_69,280,284
f_0_70,280,284
f_7_63,280,284
f_7_64,284,288
f_6_65,284,288
f_5_66,284,288
f_4_67,284,288
f_3_68,284,288
f_2_69,284,288
f_1_70,284,288
f_0_71,284,288
f_0_72,288,292
f_7_65,288,292
f_6_66,288,292
f_5_67,288,292
f_4_68,288,292
f_3_69,288,292
f_2_70,288,292
f_1_71,288,292
f_1_72,292,296
f_0_73,292,296
f_7_66,292,296
f_6_67,292,296
f_5_68,292,296
f_4_69,292,296
f_3_70,292,296
f_2_71,292,296
f_2_72,296,300
f_1_73,296,300
f_0_74,296,300
f_7_67,296,300
f_6_68,296,300
f_5_69,296,300
f_4_70,296,300
f_3_71,296,300
f_3_72,300,304
f_2_73,300,304
f_1_74,300,304
f_0_75,300,304
f_7_68,300,304
f_6_69,300,304
f_5_70,300,304
f_4_71,300,304
f_4_72,304,308
f_3_73,304,308
f_2_74,304,308
f_1_75,304,308
f_0_76,304,308
f_7_69,304,308
f_6_70,304,308
f_5_71,304,308
f_5_72,308,312
f_4_73,308,312
f_3_74,308,312
f_2_75,308,312
f_1_76,308,312
f_0_77,308,312
f_7_70,308,312
f_6_71,308,312
f_6_72,312,316
f_5_73,312,316
f_4_74,312,316
f_3_75,312,316
f_2_76,312,316
f_1_77,312,316
f_0_78,312,316
f_7_71,312,316
f_7_72,316,320
f_6_73,316,320
f_5_74,316,320
f_4_75,316,320
f_3_76,316,320
f_2_77,316,320
f_1_78,316,320
f_0_79,316,320
f_8_0,320,324
f_7_73,320,324
f_6_74,320,324
f_5_75,320,324
f_4_76,320,324
f_3_77,320,324
f_2_78,320,324
b_0_79,320,328
f_9_0,324,328
f_8_1,324,328
f_7_74,324,328
f_6_75,324,328
f_5_76,324,328
f_4_77,324,328
f_3_78,324,328
f_10_0,328,332
f_9_1,328,332
f_8_2,328,332
f_7_75,328,332
f_6_76,328,332
f_5_77,328,332
b_0_78,328,336
f_1_79,328,332
f_10_1,332,336
f_9_2,332,336
f_8_3,332,336
f_7_76,332,336
f_6_77,332,336
b_1_79,332,340
f_10_2,336,340
f_9_3,336,340
f_8_4,336,340
b_0_77,336,344
f_4_78,336,340
f_10_3,340,344
f_9_4,340,344
b_1_78,340,348
f_2_79,340,344
b_0_76,344,352
f_7_77,344,348
b_2_79,344,352
b_1_77,348,356
f_5_78,348,352
b_0_75,352,360
f_10_4,352,356
b_2_78,352,360
f_3_79,352,356
b_1_76,356,364
f_8_5,356,360
b_3_79,356,364
b_0_74,360,368
b_2_77,360,368
f_6_78,360,364
b_1_75,364,372
b_3_78,364,372
f_4_79,364,368
b_0_73,368,376
b_2_76,368,376
f_9_5,368,372
b_4_79,368,376
b_1_74,372,380
b_3_77,372,380
f_7_78,372,376
b_0_72,376,384
b_2_75,376,384
b_4_78,376,384
f_5_79,376,380
b_1_73,380,388
b_3_76,380,388
f_10_5,380,384
b_5_79,380,388
f_11_0,384,388
b_2_74,384,392
b_4_77,384,392
f_8_6,384,388
b_1_72,388,396
f_11_1,388,392
b_3_75,388,396
b_5_78,388,396
b_0_71,388,396
b_2_73,392,400
f_11_2,392,396
b_4_76,392,400
f_12_0,396,400
b_3_74,396,404
f_11_3,396,400
b_5_77,396,404
b_0_70,396,404
f_6_79,396,400
b_2_72,400,408
f_12_1,400,404
b_4_75,400,408
f_11_4,400,404
b_1_71,400,408
b_3_73,404,412
f_12_2,404,408
b_5_76,404,412
b_0_69,404,412
f_9_6,404,408
f_13_0,408,412
b_4_74,408,416
f_12_3,408,412
b_1_70,408,416
b_2_71,408,416
b_3_72,412,420
f_13_1,412,416
b_5_75,412,420
b_0_68,412,420
f_11_5,412,416
b_4_73,416,424
f_13_2,416,420
b_1_69,416,424
b_2_70,416,424
b_6_79,416,424
f_14_0,420,424
b_5_74,420,428
b_0_67,420,428
f_12_4,420,424
b_4_72,424,432
f_14_1,424,428
b_1_68,424,432
b_2_69,424,432
b_6_78,424,432
f_7_79,424,428
b_5_73,428,436
b_0_66,428,436
f_13_3,428,432
b_3_71,428,436
f_15_0,432,436
b_1_67,432,440
b_2_68,432,440
b_6_77,432,440
f_10_6,432,436
b_5_72,436,444
b_0_65,436,444
f_14_2,436,440
b_3_70,436,444
b_4_71,436,444
b_1_66,440,448
b_2_67,440,448
b_6_76,440,448
f_12_5,440,444
b_0_64,444,452
f_15_1,444,448
b_3_69,444,452
f_11_6,444,448
b_5_71,444,452
b_1_65,448,456
b_2_66,448,456
b_6_75,448,456
f_13_4,448,452
b_4_70,448,456
f_16_0,452,456
b_3_68,452,460
f_13_5,452,456
b_0_63,452,460
b_1_64,456,464
b_2_65,456,464
b_6_74,456,464
f_14_3,456,460
b_4_69,456,464
f_12_6,456,460
b_3_67,460,468
f_14_4,460,464
b_0_62,460,468
b_7_79,460,468
b_2_64,464,472
b_6_73,464,472
f_15_2,464,468
b_4_68,464,472
f_14_5,464,468
b_3_66,468,476
f_15_3,468,472
b_0_61,468,476
f_13_6,468,472
f_8_7,468,472
b_6_72,472,480
f_16_1,472,476
b_4_67,472,480
f_15_4,472,476
b_5_70,472,480
b_1_63,472,480
b_3_65,476,484
f_16_2,476,480
b_0_60,476,484
f_15_5,476,480
f_8_8,480,484
b_4_66,480,488
f_16_3,480,484
b_5_69,480,488
f_14_6,480,484
f_9_7,480,484
b_3_64,484,492
f_8_9,484,488
b_0_59,484,492
f_16_4,484,488
b_1_62,484,492
b_2_63,484,492
b_4_65,488,496
f_8_10,488,492
b_5_68,488,496
f_16_5,488,492
f_9_8,492,496
b_0_58,492,500
f_8_11,492,496
b_1_61,492,500
f_15_6,492,496
f_10_7,492,496
b_4_64,496,504
f_9_9,496,500
b_5_67,496,504
f_8_12,496,500
b_2_62,496,504
b_3_63,496,504
b_0_57,500,508
f_9_10,500,504
b_1_60,500,508
f_8_13,500,504
f_10_8,504,508
b_5_66,504,512
f_9_11,504,508
b_2_61,504,512
f_8_14,504,508
f_11_7,504,508
b_0_56,508,516
f_10_9,508,512
b_1_59,508,516
f_9_12,508,512
b_3_62,508,516
b_4_63,508,516
b_5_65,512,520
f_10_10,512,516
b_2_60,512,520
f_9_13,512,516
f_11_8,516,520
b_1_58,516,524
f_10_11,516,520
b_3_61,516,524
f_9_14,516,520
f_8_15,516,520
b_5_64,520,528
f_11_9,520,524
b_2_59,520,528
f_10_12,520,524
b_4_62,520,528
b_0_55,520,528
b_1_57,524,532
f_11_10,524,528
b_3_60,524,532
f_10_13,524,528
f_8_16,528,532
b_2_58,528,536
f_11_11,528,532
b_4_61,528,536
f_10_14,528,532
f_9_15,528,532
b_1_56,532,540
f_8_17,532,536
b_3_59,532,540
f_11_12,532,536
b_0_54,532,540
b_5_63,532,540
b_2_57,536,544
f_8_18,536,540
b_4_60,536,544
f_11_13,536,540
f_9_16,540,544
b_3_58,540,548
f_8_19,540,544
b_0_53,540,548
f_11_14,540,544
f_10_15,540,544
b_2_56,544,552
f_9_17,544,548
b_4_59,544,552
f_8_20,544,548
b_5_62,544,552
b_1_55,544,552
b_3_57,548,556
f_9_18,548,552
b_0_52,548,556
f_8_21,548,552
f_10_16,552,556
b_4_58,552,560
f_9_19,552,556
b_5_61,552,560
f_8_22,552,556
f_11_15,552,556
b_3_56,556,564
f_10_17,556,560
b_0_51,556,564
f_9_20,556,560
b_1_54,556,564
b_2_55,556,564
b_4_57,560,568
f_10_18,560,564
b_5_60,560,568
f_9_21,560,564
f_11_16,564,568
b_0_50,564,572
f_10_19,564,568
b_1_53,564,572
f_9_22,564,568
f_8_23,564,568
b_4_56,568,576
f_11_17,568,572
b_5_59,568,576
f_10_20,568,572
b_2_54,568,576
b_3_55,568,576
b_0_49,572,580
f_11_18,572,576
b_1_52,572,580
f_10_21,572,576
f_8_24,576,580
b_5_58,576,584
f_11_19,576,580
b_2_53,576,584
f_10_22,576,580
f_9_23,576,580
b_0_48,580,588
f_8_25,580,584
b_1_51,580,588
f_11_20,580,584
b_3_54,580,588
b_4_55,580,588
b_5_57,584,592
f_8_26,584,588
b_2_52,584,592
f_11_21,584,588
f_9_24,588,592
b_1_50,588,596
f_8_27,588,592
b_3_53,588,596
f_11_22,588,592
f_10_23,588,592
b_5_56,592,600
f_9_25,592,596
b_2_51,592,600
f_8_28,592,596
b_4_54,592,600
b_0_47,592,600
b_1_49,596,604
f_9_26,596,600
b_3_52,596,604
f_8_29,596,600
f_10_24,600,604
b_2_50,600,608
f_9_27,600,604
b_4_53,600,608
f_8_30,600,604
f_11_23,600,604
b_1_48,604,612
f_10_25,604,608
b_3_51,604,612
f_9_28,604,608
b_0_46,604,612
b_5_55,604,612
b_2_49,608,616
f_10_26,608,612
b_4_52,608,616
f_9_29,608,612
f_11_24,612,616
b_3_50,612,620
f_10_27,612,616
b_0_45,612,620
f_9_30,612,616
f_8_31,612,616
b_2_48,616,624
f_11_25,616,620
b_4_51,616,624
f_10_28,616,620
b_5_54,616,624
b_1_47,616,624
b_3_49,620,628
f_11_26,620,624
b_0_44,620,628
f_10_29,620,624
f_8_32,624,628
b_4_50,624,632
f_11_27,624,628
b_5_53,624,632
f_10_30,624,628
f_9_31,624,628
b_3_48,628,636
f_8_33,628,632
b_0_43,628,636
f_11_28,628,632
b_1_46,628,636
b_2_47,628,636
b_4_49,632,640
f_8_34,632,636
b_5_52,632,640
f_11_29,632,636
f_9_32,636,640
b_0_42,636,644
f_8_35,636,640
b_1_45,636,644
f_11_30,636,640
f_10_31,636,640
b_4_48,640,648
f_9_33,640,644
b_5_51,640,648
f_8_36,640,644
b_2_46,640,648
b_3_47,640,648
b_0_41,644,652
f_9_34,644,648
b_1_44,644,652
f_8_37,644,648
f_10_32,648,652
b_5_50,648,656
f_9_35,648,652
b_2_45,648,656
f_8_38,648,652
f_11_31,648,652
b_0_40,652,660
f_10_33,652,656
b_1_43,652,660
f_9_36,652,656
b_3_46,652,660
b_4_47,652,660
b_5_49,656,664
f_10_34,656,660
b_2_44,656,664
f_9_37,656,660
f_11_32,660,664
b_1_42,660,668
f_10_35,660,664
b_3_45,660,668
f_9_38,660,664
f_8_39,660,664
b_5_48,664,672
f_11_33,664,668
b_2_43,664,672
f_10_36,664,668
b_4_46,664,672
b_0_39,664,672
b_1_41,668,676
f_11_34,668,672
b_3_44,668,676
f_10_37,668,672
f_8_40,672,676
b_2_42,672,680
f_11_35,672,676
b_4_45,672,680
f_10_38,672,676
f_9_39,672,676
b_1_40,676,684
f_8_41,676,680
b_3_43,676,684
f_11_36,676,680
b_0_38,676,684
b_5_47,676,684
b_2_41,680,688
f_8_42,680,684
b_4_44,680,688
f_11_37,680,684
f_9_40,684,688
b_3_42,684,692
f_8_43,684,688
b_0_37,684,692
f_11_38,684,688
f_10_39,684,688
b_2_40,688,696
f_9_41,688,692
b_4_43,688,696
f_8_44,688,692
b_5_46,688,696
b_1_39,688,696
b_3_41,692,700
f_9_42,692,696
b_0_36,692,700
f_8_45,692,696
f_10_40,696,700
b_4_42,696,704
f_9_43,696,700
b_5_45,696,704
f_8_46,696,700
f_11_39,696,700
b_3_40,700,708
f_10_41,700,704
b_0_35,700,708
f_9_44,700,704
b_1_38,700,708
b_2_39,700,708
b_4_41,704,712
f_10_42,704,708
b_5_44,704,712
f_9_45,704,708
f_11_40,708,712
b_0_34,708,716
f_10_43,708,712
b_1_37,708,716
f_9_46,708,712
f_8_47,708,712
b_4_40,712,720
f_11_41,712,716
b_5_43,712,720
f_10_44,712,716
b_2_38,712,720
b_3_39,712,720
b_0_33,716,724
f_11_42,716,720
b_1_36,716,724
f_10_45,716,720
f_8_48,720,724
b_5_42,720,728
f_11_43,720,724
b_2_37,720,728
f_10_46,720,724
f_9_47,720,724
b_0_32,724,732
f_8_49,724,728
b_1_35,724,732
f_11_44,724,728
b_3_38,724,732
b_4_39,724,732
b_5_41,728,736
f_8_50,728,732
b_2_36,728,736
f_11_45,728,732
f_9_48,732,736
b_1_34,732,740
f_8_51,732,736
b_3_37,732,740
f_11_46,732,736
f_10_47,732,736
b_5_40,736,744
f_9_49,736,740
b_2_35,736,744
f_8_52,736,740
b_4_38,736,744
b_0_31,736,744
b_1_33,740,748
f_9_50,740,744
b_3_36,740,748
f_8_53,740,744
f_10_48,744,748
b_2_34,744,752
f_9_51,744,748
b_4_37,744,752
f_8_54,744,748
f_11_47,744,748
b_1_32,748,756
f_10_49,748,752
b_3_35,748,756
f_9_52,748,752
b_0_30,748,756
b_5_39,748,756
b_2_33,752,760
f_10_50,752,756
b_4_36,752,760
f_9_53,752,756
f_11_48,756,760
b_3_34,756,764
f_10_51,756,760
b_0_29,756,764
f_9_54,756,760
f_8_55,756,760
b_2_32,760,768
f_11_49,760,764
b_4_35,760,768
f_10_52,760,764
b_5_38,760,768
b_1_31,760,768
b_3_33,764,772
f_11_50,764,768
b_0_28,764,772
f_10_53,764,768
f_8_56,768,772
b_4_34,768,776
f_11_51,768,772
b_5_37,768,776
f_10_54,768,772
f_9_55,768,772
b_3_32,772,780
f_8_57,772,776
b_0_27,772,780
f_11_52,772,776
b_1_30,772,780
b_2_31,772,780
b_4_33,776,784
f_8_58,776,780
b_5_36,776,784
f_11_53,776,780
f_9_56,780,784
b_0_26,780,788
f_8_59,780,784
b_1_29,780,788
f_11_54,780,784
f_10_55,780,784
b_4_32,784,792
f_9_57,784,788
b_5_35,784,792
f_8_60,784,788
b_2_30,784,792
b_3_31,784,792
b_0_25,788,796
f_9_58,788,792
b_1_28,788,796
f_8_61,788,792
f_10_56,792,796
b_5_34,792,800
f_9_59,792,796
b_2_29,792,800
f_8_62,792,796
f_11_55,792,796
b_0_24,796,804
f_10_57,796,800
b_1_27,796,804
f_9_60,796,800
b_3_30,796,804
b_4_31,796,804
b_5_33,800,808
f_10_58,800,804
b_2_28,800,808
f_9_61,800,804
f_11_56,804,808
b_1_26,804,812
f_10_59,804,808
b_3_29,804,812
f_9_62,804,808
f_8_63,804,808
b_5_32,808,816
f_11_57,808,812
b_2_27,808,816
f_10_60,808,812
b_4_30,808,816
b_0_23,808,816
b_1_25,812,820
f_11_58,812,816
b_3_28,812,820
f_10_61,812,816
f_8_64,816,820
b_2_26,816,824
f_11_59,816,820
b_4_29,816,824
f_10_62,816,820
f_9_63,816,820
b_1_24,820,828
f_8_65,820,824
b_3_27,820,828
f_11_60,820,824
b_0_22,820,828
b_5_31,820,828
b_2_25,824,832
f_8_66,824,828
b_4_28,824,832
f_11_61,824,828
f_9_64,828,832
b_3_26,828,836
f_8_67,828,832
b_0_21,828,836
f_11_62,828,832
f_10_63,828,832
b_2_24,832,840
f_9_65,832,836
b_4_27,832,840
f_8_68,832,836
b_5_30,832,840
b_1_23,832,840
b_3_25,836,844
f_9_66,836,840
b_0_20,836,844
f_8_69,836,840
f_10_64,840,844
b_4_26,840,848
f_9_67,840,844
b_5_29,840,848
f_8_70,840,844
f_11_63,840,844
b_3_24,844,852
f_10_65,844,848
b_0_19,844,852
f_9_68,844,848
b_1_22,844,852
b_2_23,844,852
b_4_25,848,856
f_10_66,848,852
b_5_28,848,856
f_9_69,848,852
f_11_64,852,856
b_0_18,852,860
f_10_67,852,856
b_1_21,852,860
f_9_70,852,856
f_8_71,852,856
b_4_24,856,864
f_11_65,856,860
b_5_27,856,864
f_10_68,856,860
b_2_22,856,864
b_3_23,856,864
b_0_17,860,868
f_11_66,860,864
b_1_20,860,868
f_10_69,860,864
f_8_72,864,868
b_5_26,864,872
f_11_67,864,868
b_2_21,864,872
f_10_70,864,868
f_9_71,864,868
b_0_16,868,876
f_8_73,868,872
b_1_19,868,876
f_11_68,868,872
b_3_22,868,876
b_4_23,868,876
b_5_25,872,880
f_8_74,872,876
b_2_20,872,880
f_11_69,872,876
f_9_72,876,880
b_1_18,876,884
f_8_75,876,880
b_3_21,876,884
f_11_70,876,880
f_10_71,876,880
b_5_24,880,888
f_9_73,880,884
b_2_19,880,888
f_8_76,880,884
b_4_22,880,888
b_0_15,880,888
b_1_17,884,892
f_9_74,884,888
b_3_20,884,892
f_8_77,884,888
f_10_72,888,892
b_2_18,888,896
f_9_75,888,892
b_4_21,888,896
f_8_78,888,892
f_11_71,888,892
b_1_16,892,900
f_10_73,892,896
b_3_19,892,900
f_9_76,892,896
b_0_14,892,900
b_5_23,892,900
b_2_17,896,904
f_10_74,896,900
b_4_20,896,904
f_9_77,896,900
f_11_72,900,904
b_3_18,900,908
f_10_75,900,904
b_0_13,900,908
f_9_78,900,904
f_8_79,900,904
b_2_16,904,912
f_11_73,904,908
b_4_19,904,912
f_10_76,904,908
b_5_22,904,912
b_1_15,904,912
b_3_17,908,916
f_11_74,908,912
b_0_12,908,916
f_10_77,908,912
f_17_0,912,916
b_4_18,912,920
f_11_75,912,916
b_5_21,912,920
f_10_78,912,916
f_9_79,912,916
b_3_16,916,924
f_17_1,916,920
b_0_11,916,924
f_11_76,916,920
b_1_14,916,924
b_2_15,916,924
b_4_17,920,928
f_17_2,920,924
b_5_20,920,928
f_11_77,920,924
f_18_0,924,928
b_0_10,924,932
f_17_3,924,928
b_1_13,924,932
f_11_78,924,928
b_3_15,924,932
b_4_16,928,936
f_18_1,928,932
b_5_19,928,936
f_17_4,928,932
b_2_14,928,936
b_0_9,932,940
f_18_2,932,936
b_1_12,932,940
f_17_5,932,936
b_6_71,932,940
f_19_0,936,940
b_5_18,936,944
f_18_3,936,940
b_2_13,936,944
f_16_6,936,940
b_0_8,940,948
f_19_1,940,944
b_1_11,940,948
f_18_4,940,944
b_3_14,940,948
b_4_15,940,948
b_5_17,944,952
f_19_2,944,948
b_2_12,944,952
f_18_5,944,948
f_20_0,948,952
b_1_10,948,956
f_19_3,948,952
b_3_13,948,956
f_17_6,948,952
b_0_7,948,956
b_5_16,952,960
f_20_1,952,956
b_2_11,952,960
f_19_4,952,956
b_4_14,952,960
b_1_9,956,964
f_20_2,956,960
b_3_12,956,964
f_19_5,956,960
b_8_79,956,964
f_21_0,960,964
b_2_10,960,968
f_20_3,960,964
b_4_13,960,968
f_18_6,960,964
b_1_8,964,972
f_21_1,964,968
b_3_11,964,972
f_20_4,964,968
b_0_6,964,972
f_10_79,964,968
b_2_9,968,976
f_21_2,968,972
b_4_12,968,976
f_20_5,968,972
b_5_15,968,976
f_22_0,972,976
b_3_10,972,980
f_21_3,972,976
b_0_5,972,980
f_19_6,972,976
b_2_8,976,984
f_22_1,976,980
b_4_11,976,984
f_21_4,976,980
b_5_14,976,984
b_1_7,976,984
b_3_9,980,988
f_22_2,980,984
b_0_4,980,988
f_21_5,980,984
f_23_0,984,988
b_4_10,984,992
f_22_3,984,988
b_5_13,984,992
f_20_6,984,988
f_11_79,984,988
b_3_8,988,996
f_23_1,988,992
b_0_3,988,996
f_22_4,988,992
b_1_6,988,996
b_2_7,988,996
b_4_9,992,1000
f_23_2,992,996
b_5_12,992,1000
f_22_5,992,996
f_24_0,996,1000
b_0_2,996,1004
f_23_3,996,1000
b_1_5,996,1004
f_21_6,996,1000
b_3_7,996,1004
b_4_8,1000,1008
f_24_1,1000,1004
b_5_11,1000,1008
f_23_4,1000,1004
b_2_6,1000,1008
b_0_1,1004,1012
f_24_2,1004,1008
b_1_4,1004,1012
f_23_5,1004,1008
b_9_79,1004,1012
f_25_0,1008,1012
b_5_10,1008,1016
f_24_3,1008,1012
b_2_5,1008,1016
f_22_6,1008,1012
b_0_0,1012,1020
f_25_1,1012,1016
b_1_3,1012,1020
f_24_4,1012,1016
b_3_6,1012,1020
f_12_7,1012,1016
b_5_9,1016,1024
f_25_2,1016,1020
b_2_4,1016,1024
f_24_5,1016,1020
b_4_7,1016,1024
f_12_8,1020,1024
b_1_2,1020,1028
f_25_3,1020,1024
b_3_5,1020,1028
f_23_6,1020,1024
b_5_8,1024,1032
f_12_9,1024,1028
b_2_3,1024,1032
f_25_4,1024,1028
b_4_6,1024,1032
f_13_7,1024,1028
b_1_1,1028,1036
f_12_10,1028,1032
b_3_4,1028,1036
f_25_5,1028,1032
b_10_79,1028,1036
f_13_8,1032,1036
b_2_2,1032,1040
f_12_11,1032,1036
b_4_5,1032,1040
f_24_6,1032,1036
b_1_0,1036,1044
f_13_9,1036,1040
b_3_3,1036,1044
f_12_12,1036,1040
b_6_70,1036,1044
f_14_7,1036,1040
b_2_1,1040,1048
f_13_10,1040,1044
b_4_4,1040,1048
f_12_13,1040,1044
b_5_7,1040,1048
f_14_8,1044,1048
b_3_2,1044,1052
f_13_11,1044,1048
b_6_69,1044,1052
f_12_14,1044,1048
b_2_0,1048,1056
f_14_9,1048,1052
b_4_3,1048,1056
f_13_12,1048,1052
b_5_6,1048,1056
f_12_15,1048,1052
b_3_1,1052,1060
f_14_10,1052,1056
b_6_68,1052,1060
f_13_13,1052,1056
b_11_79,1052,1060
f_12_16,1056,1060
b_4_2,1056,1064
f_14_11,1056,1060
b_5_5,1056,1064
f_13_14,1056,1060
b_3_0,1060,1068
f_12_17,1060,1064
b_6_67,1060,1068
f_14_12,1060,1064
b_7_78,1060,1068
f_13_15,1060,1064
b_4_1,1064,1072
f_12_18,1064,1068
b_5_4,1064,1072
f_14_13,1064,1068
f_15_7,1064,1068
f_13_16,1068,1072
b_6_66,1068,1076
f_12_19,1068,1072
b_7_77,1068,1076
f_14_14,1068,1072
f_16_7,1068,1072
b_4_0,1072,1080
f_13_17,1072,1076
b_5_3,1072,1080
f_12_20,1072,1076
b_8_78,1072,1080
f_14_15,1072,1076
b_6_65,1076,1084
f_13_18,1076,1080
b_7_76,1076,1084
f_12_21,1076,1080
f_17_7,1076,1080
f_14_16,1080,1084
b_5_2,1080,1088
f_13_19,1080,1084
b_8_77,1080,1088
f_12_22,1080,1084
f_18_7,1080,1084
b_6_64,1084,1092
f_14_17,1084,1088
b_7_75,1084,1092
f_13_20,1084,1088
b_9_78,1084,1092
f_12_23,1084,1088
b_5_1,1088,1096
f_14_18,1088,1092
b_8_76,1088,1096
f_13_21,1088,1092
f_19_7,1088,1092
f_12_24,1092,1096
b_7_74,1092,1100
f_14_19,1092,1096
b_9_77,1092,1100
f_13_22,1092,1096
b_6_63,1092,1100
b_5_0,1096,1104
f_12_25,1096,1100
b_8_75,1096,1104
f_14_20,1096,1100
b_10_78,1096,1104
b_7_73,1100,1108
f_12_26,1100,1104
b_9_76,1100,1108
f_14_21,1100,1104
f_13_23,1100,1104
f_13_24,1104,1108
b_8_74,1104,1112
f_12_27,1104,1108
b_10_77,1104,1112
f_14_22,1104,1108
f_20_7,1104,1108
b_7_72,1108,1116
f_13_25,1108,1112
b_9_75,1108,1116
f_12_28,1108,1112
b_6_62,1108,1116
f_14_23,1108,1112
b_8_73,1112,1120
f_13_26,1112,1116
b_10_76,1112,1120
f_12_29,1112,1116
f_21_7,1112,1116
f_14_24,1116,1120
b_9_74,1116,1124
f_13_27,1116,1120
b_6_61,1116,1124
f_12_30,1116,1120
b_7_71,1116,1124
b_8_72,1120,1128
f_14_25,1120,1124
b_10_75,1120,1128
f_13_28,1120,1124
b_11_78,1120,1128
b_9_73,1124,1132
f_14_26,1124,1128
b_6_60,1124,1132
f_13_29,1124,1128
f_12_31,1124,1128
f_12_32,1128,1132
b_10_74,1128,1136
f_14_27,1128,1132
b_11_77,1128,1136
f_13_30,1128,1132
b_8_71,1128,1136
b_9_72,1132,1140
f_12_33,1132,1136
b_6_59,1132,1140
f_14_28,1132,1136
b_7_70,1132,1140
b_10_73,1136,1144
f_12_34,1136,1140
b_11_76,1136,1144
f_14_29,1136,1140
f_13_31,1136,1140
f_13_32,1140,1144
b_6_58,1140,1148
f_12_35,1140,1144
b_7_69,1140,1148
f_14_30,1140,1144
b_9_71,1140,1148
b_10_72,1144,1152
f_13_33,1144,1148
b_11_75,1144,1152
f_12_36,1144,1148
b_8_70,1144,1152
b_6_57,1148,1156
f_13_34,1148,1152
b_7_68,1148,1156
f_12_37,1148,1152
f_14_31,1148,1152
f_14_32,1152,1156
b_11_74,1152,1160
f_13_35,1152,1156
b_8_69,1152,1160
f_12_38,1152,1156
b_10_71,1152,1160
b_6_56,1156,1164
f_14_33,1156,1160
b_7_67,1156,1164
f_13_36,1156,1160
b_9_70,1156,1164
b_11_73,1160,1168
f_14_34,1160,1164
b_8_68,1160,1168
f_13_37,1160,1164
f_12_39,1160,1164
f_12_40,1164,1168
b_7_66,1164,1172
f_14_35,1164,1168
b_9_69,1164,1172
f_13_38,1164,1168
b_6_55,1164,1172
b_11_72,1168,1176
f_12_41,1168,1172
b_8_67,1168,1176
f_14_36,1168,1172
b_10_70,1168,1176
b_7_65,1172,1180
f_12_42,1172,1176
b_9_68,1172,1180
f_14_37,1172,1176
f_13_39,1172,1176
f_13_40,1176,1180
b_8_66,1176,1184
f_12_43,1176,1180
b_10_69,1176,1184
f_14_38,1176,1180
b_11_71,1176,1184
b_7_64,1180,1188
f_13_41,1180,1184
b_9_67,1180,1188
f_12_44,1180,1184
b_6_54,1180,1188
b_8_65,1184,1192
f_13_42,1184,1188
b_10_68,1184,1192
f_12_45,1184,1188
f_14_39,1184,1188
f_14_40,1188,1192
b_9_66,1188,1196
f_13_43,1188,1192
b_6_53,1188,1196
f_12_46,1188,1192
b_7_63,1188,1196
b_8_64,1192,1200
f_14_41,1192,1196
b_10_67,1192,1200
f_13_44,1192,1196
b_11_70,1192,1200
b_9_65,1196,1204
f_14_42,1196,1200
b_6_52,1196,1204
f_13_45,1196,1200
f_12_47,1196,1200
f_12_48,1200,1204
b_10_66,1200,1208
f_14_43,1200,1204
b_11_69,1200,1208
f_13_46,1200,1204
b_8_63,1200,1208
b_9_64,1204,1212
f_12_49,1204,1208
b_6_51,1204,1212
f_14_44,1204,1208
b_7_62,1204,1212
b_10_65,1208,1216
f_12_50,1208,1212
b_11_68,1208,1216
f_14_45,1208,1212
f_13_47,1208,1212
f_13_48,1212,1216
b_6_50,1212,1220
f_12_51,1212,1216
b_7_61,1212,1220
f_14_46,1212,1216
b_9_63,1212,1220
b_10_64,1216,1224
f_13_49,1216,1220
b_11_67,1216,1224
f_12_52,1216,1220
b_8_62,1216,1224
b_6_49,1220,1228
f_13_50,1220,1224
b_7_60,1220,1228
f_12_53,1220,1224
f_14_47,1220,1224
f_14_48,1224,1228
b_11_66,1224,1232
f_13_51,1224,1228
b_8_61,1224,1232
f_12_54,1224,1228
b_10_63,1224,1232
b_6_48,1228,1236
f_14_49,1228,1232
b_7_59,1228,1236
f_13_52,1228,1232
b_9_62,1228,1236
b_11_65,1232,1240
f_14_50,1232,1236
b_8_60,1232,1240
f_13_53,1232,1236
f_12_55,1232,1236
f_12_56,1236,1240
b_7_58,1236,1244
f_14_51,1236,1240
b_9_61,1236,1244
f_13_54,1236,1240
b_6_47,1236,1244
b_11_64,1240,1248
f_12_57,1240,1244
b_8_59,1240,1248
f_14_52,1240,1244
b_10_62,1240,1248
b_7_57,1244,1252
f_12_58,1244,1248
b_9_60,1244,1252
f_14_53,1244,1248
f_13_55,1244,1248
f_13_56,1248,1252
b_8_58,1248,1256
f_12_59,1248,1252
b_10_61,1248,1256
f_14_54,1248,1252
b_11_63,1248,1256
b_7_56,1252,1260
f_13_57,1252,1256
b_9_59,1252,1260
f_12_60,1252,1256
b_6_46,1252,1260
b_8_57,1256,1264
f_13_58,1256,1260
b_10_60,1256,1264
f_12_61,1256,1260
f_14_55,1256,1260
f_14_56,1260,1264
b_9_58,1260,1268
f_13_59,1260,1264
b_6_45,1260,1268
f_12_62,1260,1264
b_7_55,1260,1268
b_8_56,1264,1272
f_14_57,1264,1268
b_10_59,1264,1272
f_13_60,1264,1268
b_11_62,1264,1272
b_9_57,1268,1276
f_14_58,1268,1272
b_6_44,1268,1276
f_13_61,1268,1272
f_12_63,1268,1272
f_12_64,1272,1276
b_10_58,1272,1280
f_14_59,1272,1276
b_11_61,1272,1280
f_13_62,1272,1276
b_8_55,1272,1280
b_9_56,1276,1284
f_12_65,1276,1280
b_6_43,1276,1284
f_14_60,1276,1280
b_7_54,1276,1284
b_10_57,1280,1288
f_12_66,1280,1284
b_11_60,1280,1288
f_14_61,1280,1284
f_13_63,1280,1284
f_13_64,1284,1288
b_6_42,1284,1292
f_12_67,1284,1288
b_7_53,1284,1292
f_14_62,1284,1288
b_9_55,1284,1292
b_10_56,1288,1296
f_13_65,1288,1292
b_11_59,1288,1296
f_12_68,1288,1292
b_8_54,1288,1296
b_6_41,1292,1300
f_13_66,1292,1296
b_7_52,1292,1300
f_12_69,1292,1296
f_14_63,1292,1296
f_14_64,1296,1300
b_11_58,1296,1304
f_13_67,1296,1300
b_8_53,1296,1304
f_12_70,1296,1300
b_10_55,1296,1304
b_6_40,1300,1308
f_14_65,1300,1304
b_7_51,1300,1308
f_13_68,1300,1304
b_9_54,1300,1308
b_11_57,1304,1312
f_14_66,1304,1308
b_8_52,1304,1312
f_13_69,1304,1308
f_12_71,1304,1308
f_12_72,1308,1312
b_7_50,1308,1316
f_14_67,1308,1312
b_9_53,1308,1316
f_13_70,1308,1312
b_6_39,1308,1316
b_11_56,1312,1320
f_12_73,1312,1316
b_8_51,1312,1320
f_14_68,1312,1316
b_10_54,1312,1320
b_7_49,1316,1324
f_12_74,1316,1320
b_9_52,1316,1324
f_14_69,1316,1320
f_13_71,1316,1320
f_13_72,1320,1324
b_8_50,1320,1328
f_12_75,1320,1324
b_10_53,1320,1328
f_14_70,1320,1324
b_11_55,1320,1328
b_7_48,1324,1332
f_13_73,1324,1328
b_9_51,1324,1332
f_12_76,1324,1328
b_6_38,1324,1332
b_8_49,1328,1336
f_13_74,1328,1332
b_10_52,1328,1336
f_12_77,1328,1332
f_14_71,1328,1332
f_14_72,1332,1336
b_9_50,1332,1340
f_13_75,1332,1336
b_6_37,1332,1340
f_12_78,1332,1336
b_7_47,1332,1340
b_8_48,1336,1344
f_14_73,1336,1340
b_10_51,1336,1344
f_13_76,1336,1340
b_11_54,1336,1344
b_9_49,1340,1348
f_14_74,1340,1344
b_6_36,1340,1348
f_13_77,1340,1344
f_12_79,1340,1344
f_15_8,1344,1348
b_10_50,1344,1352
f_14_75,1344,1348
b_11_53,1344,1352
f_13_78,1344,1348
b_8_47,1344,1352
b_9_48,1348,1356
f_15_9,1348,1352
b_6_35,1348,1356
f_14_76,1348,1352
b_7_46,1348,1356
b_10_49,1352,1360
f_15_10,1352,1356
b_11_52,1352,1360
f_14_77,1352,1356
b_12_79,1352,1360
f_16_8,1356,1360
b_6_34,1356,1364
f_15_11,1356,1360
b_7_45,1356,1364
f_14_78,1356,1360
b_10_48,1360,1368
f_16_9,1360,1364
b_11_51,1360,1368
f_15_12,1360,1364
b_8_46,1360,1368
f_13_79,1360,1364
b_6_33,1364,1372
f_16_10,1364,1368
b_7_44,1364,1372
f_15_13,1364,1368
b_9_47,1364,1372
f_17_8,1368,1372
b_11_50,1368,1376
f_16_11,1368,1372
b_8_45,1368,1376
f_15_14,1368,1372
b_6_32,1372,1380
f_17_9,1372,1376
b_7_43,1372,1380
f_16_12,1372,1376
b_9_46,1372,1380
b_10_47,1372,1380
b_11_49,1376,1384
f_17_10,1376,1380
b_8_44,1376,1384
f_16_13,1376,1380
f_18_8,1380,1384
b_7_42,1380,1388
f_17_11,1380,1384
b_9_45,1380,1388
f_16_14,1380,1384
f_14_79,1380,1384
b_11_48,1384,1392
f_18_9,1384,1388
b_8_43,1384,1392
f_17_12,1384,1388
b_10_46,1384,1392
b_6_31,1384,1392
b_7_41,1388,1396
f_18_10,1388,1392
b_9_44,1388,1396
f_17_13,1388,1392
f_19_8,1392,1396
b_8_42,1392,1400
f_18_11,1392,1396
b_10_45,1392,1400
f_17_14,1392,1396
b_11_47,1392,1400
b_7_40,1396,1404
f_19_9,1396,1400
b_9_43,1396,1404
f_18_12,1396,1400
b_6_30,1396,1404
b_8_41,1400,1408
f_19_10,1400,1404
b_10_44,1400,1408
f_18_13,1400,1404
b_13_79,1400,1408
f_20_8,1404,1408
b_9_42,1404,1412
f_19_11,1404,1408
b_6_29,1404,1412
f_18_14,1404,1408
b_8_40,1408,1416
f_20_9,1408,1412
b_10_43,1408,1416
f_19_12,1408,1412
b_11_46,1408,1416
f_15_15,1408,1412
b_9_41,1412,1420
f_20_10,1412,1416
b_6_28,1412,1420
f_19_13,1412,1416
b_7_39,1412,1420
f_15_16,1416,1420
b_10_42,1416,1424
f_20_11,1416,1420
b_11_45,1416,1424
f_19_14,1416,1420
b_9_40,1420,1428
f_15_17,1420,1424
b_6_27,1420,1428
f_20_12,1420,1424
b_7_38,1420,1428
f_16_15,1420,1424
b_10_41,1424,1432
f_15_18,1424,1428
b_11_44,1424,1432
f_20_13,1424,1428
b_8_39,1424,1432
f_16_16,1428,1432
b_6_26,1428,1436
f_15_19,1428,1432
b_7_37,1428,1436
f_20_14,1428,1432
b_10_40,1432,1440
f_16_17,1432,1436
b_11_43,1432,1440
f_15_20,1432,1436
b_8_38,1432,1440
f_17_15,1432,1436
b_6_25,1436,1444
f_16_18,1436,1440
b_7_36,1436,1444
f_15_21,1436,1440
b_9_39,1436,1444
f_17_16,1440,1444
b_11_42,1440,1448
f_16_19,1440,1444
b_8_37,1440,1448
f_15_22,1440,1444
b_6_24,1444,1452
f_17_17,1444,1448
b_7_35,1444,1452
f_16_20,1444,1448
b_9_38,1444,1452
f_15_23,1444,1448
b_11_41,1448,1456
f_17_18,1448,1452
b_8_36,1448,1456
f_16_21,1448,1452
b_10_39,1448,1456
f_15_24,1452,1456
b_7_34,1452,1460
f_17_19,1452,1456
b_9_37,1452,1460
f_16_22,1452,1456
b_11_40,1456,1464
f_15_25,1456,1460
b_8_35,1456,1464
f_17_20,1456,1460
b_10_38,1456,1464
f_16_23,1456,1460
b_7_33,1460,1468
f_15_26,1460,1464
b_9_36,1460,1468
f_17_21,1460,1464
b_6_23,1460,1468
f_16_24,1464,1468
b_8_34,1464,1472
f_15_27,1464,1468
b_10_37,1464,1472
f_17_22,1464,1468
b_7_32,1468,1476
f_16_25,1468,1472
b_9_35,1468,1476
f_15_28,1468,1472
b_6_22,1468,1476
f_17_23,1468,1472
b_8_33,1472,1480
f_16_26,1472,1476
b_10_36,1472,1480
f_15_29,1472,1476
b_11_39,1472,1480
f_17_24,1476,1480
b_9_34,1476,1484
f_16_27,1476,1480
b_6_21,1476,1484
f_15_30,1476,1480
b_8_32,1480,1488
f_17_25,1480,1484
b_10_35,1480,1488
f_16_28,1480,1484
b_11_38,1480,1488
f_15_31,1480,1484
b_9_33,1484,1492
f_17_26,1484,1488
b_6_20,1484,1492
f_16_29,1484,1488
b_7_31,1484,1492
f_15_32,1488,1492
b_10_34,1488,1496
f_17_27,1488,1492
b_11_37,1488,1496
f_16_30,1488,1492
b_9_32,1492,1500
f_15_33,1492,1496
b_6_19,1492,1500
f_17_28,1492,1496
b_7_30,1492,1500
f_16_31,1492,1496
b_10_33,1496,1504
f_15_34,1496,1500
b_11_36,1496,1504
f_17_29,1496,1500
b_8_31,1496,1504
f_16_32,1500,1504
b_6_18,1500,1508
f_15_35,1500,1504
b_7_29,1500,1508
f_17_30,1500,1504
b_10_32,1504,1512
f_16_33,1504,1508
b_11_35,1504,1512
f_15_36,1504,1508
b_8_30,1504,1512
f_17_31,1504,1508
b_6_17,1508,1516
f_16_34,1508,1512
b_7_28,1508,1516
f_15_37,1508,1512
b_9_31,1508,1516
f_17_32,1512,1516
b_11_34,1512,1520
f_16_35,1512,1516
b_8_29,1512,1520
f_15_38,1512,1516
b_6_16,1516,1524
f_17_33,1516,1520
b_7_27,1516,1524
f_16_36,1516,1520
b_9_30,1516,1524
f_15_39,1516,1520
b_11_33,1520,1528
f_17_34,1520,1524
b_8_28,1520,1528
f_16_37,1520,1524
b_10_31,1520,1528
f_15_40,1524,1528
b_7_26,1524,1532
f_17_35,1524,1528
b_9_29,1524,1532
f_16_38,1524,1528
b_11_32,1528,1536
f_15_41,1528,1532
b_8_27,1528,1536
f_17_36,1528,1532
b_10_30,1528,1536
f_16_39,1528,1532
b_7_25,1532,1540
f_15_42,1532,1536
b_9_28,1532,1540
f_17_37,1532,1536
b_6_15,1532,1540
f_16_40,1536,1540
b_8_26,1536,1544
f_15_43,1536,1540
b_10_29,1536,1544
f_17_38,1536,1540
b_7_24,1540,1548
f_16_41,1540,1544
b_9_27,1540,1548
f_15_44,1540,1544
b_6_14,1540,1548
f_17_39,1540,1544
b_8_25,1544,1552
f_16_42,1544,1548
b_10_28,1544,1552
f_15_45,1544,1548
b_11_31,1544,1552
f_17_40,1548,1552
b_9_26,1548,1556
f_16_43,1548,1552
b_6_13,1548,1556
f_15_46,1548,1552
b_8_24,1552,1560
f_17_41,1552,1556
b_10_27,1552,1560
f_16_44,1552,1556
b_11_30,1552,1560
f_15_47,1552,1556
b_9_25,1556,1564
f_17_42,1556,1560
b_6_12,1556,1564
f_16_45,1556,1560
b_7_23,1556,1564
f_15_48,1560,1564
b_10_26,1560,1568
f_17_43,1560,1564
b_11_29,1560,1568
f_16_46,1560,1564
b_9_24,1564,1572
f_15_49,1564,1568
b_6_11,1564,1572
f_17_44,1564,1568
b_7_22,1564,1572
f_16_47,1564,1568
b_10_25,1568,1576
f_15_50,1568,1572
b_11_28,1568,1576
f_17_45,1568,1572
b_8_23,1568,1576
f_16_48,1572,1576
b_6_10,1572,1580
f_15_51,1572,1576
b_7_21,1572,1580
f_17_46,1572,1576
b_10_24,1576,1584
f_16_49,1576,1580
b_11_27,1576,1584
f_15_52,1576,1580
b_8_22,1576,1584
f_17_47,1576,1580
b_6_9,1580,1588
f_16_50,1580,1584
b_7_20,1580,1588
f_15_53,1580,1584
b_9_23,1580,1588
f_17_48,1584,1588
b_11_26,1584,1592
f_16_51,1584,1588
b_8_21,1584,1592
f_15_54,1584,1588
b_6_8,1588,1596
f_17_49,1588,1592
b_7_19,1588,1596
f_16_52,1588,1592
b_9_22,1588,1596
f_15_55,1588,1592
b_11_25,1592,1600
f_17_50,1592,1596
b_8_20,1592,1600
f_16_53,1592,1596
b_10_23,1592,1600
f_15_56,1596,1600
b_7_18,1596,1604
f_17_51,1596,1600
b_9_21,1596,1604
f_16_54,1596,1600
b_11_24,1600,1608
f_15_57,1600,1604
b_8_19,1600,1608
f_17_52,1600,1604
b_10_22,1600,1608
f_16_55,1600,1604
b_7_17,1604,1612
f_15_58,1604,1608
b_9_20,1604,1612
f_17_53,1604,1608
b_6_7,1604,1612
f_16_56,1608,1612
b_8_18,1608,1616
f_15_59,1608,1612
b_10_21,1608,1616
f_17_54,1608,1612
b_7_16,1612,1620
f_16_57,1612,1616
b_9_19,1612,1620
f_15_60,1612,1616
b_6_6,1612,1620
f_17_55,1612,1616
b_8_17,1616,1624
f_16_58,1616,1620
b_10_20,1616,1624
f_15_61,1616,1620
b_11_23,1616,1624
f_17_56,1620,1624
b_9_18,1620,1628
f_16_59,1620,1624
b_6_5,1620,1628
f_15_62,1620,1624
b_8_16,1624,1632
f_17_57,1624,1628
b_10_19,1624,1632
f_16_60,1624,1628
b_11_22,1624,1632
f_15_63,1624,1628
b_9_17,1628,1636
f_17_58,1628,1632
b_6_4,1628,1636
f_16_61,1628,1632
b_7_15,1628,1636
f_15_64,1632,1636
b_10_18,1632,1640
f_17_59,1632,1636
b_11_21,1632,1640
f_16_62,1632,1636
b_9_16,1636,1644
f_15_65,1636,1640
b_6_3,1636,1644
f_17_60,1636,1640
b_7_14,1636,1644
f_16_63,1636,1640
b_10_17,1640,1648
f_15_66,1640,1644
b_11_20,1640,1648
f_17_61,1640,1644
b_8_15,1640,1648
f_16_64,1644,1648
b_6_2,1644,1652
f_15_67,1644,1648
b_7_13,1644,1652
f_17_62,1644,1648
b_10_16,1648,1656
f_16_65,1648,1652
b_11_19,1648,1656
f_15_68,1648,1652
b_8_14,1648,1656
f_17_63,1648,1652
b_6_1,1652,1660
f_16_66,1652,1656
b_7_12,1652,1660
f_15_69,1652,1656
b_9_15,1652,1660
f_17_64,1656,1660
b_11_18,1656,1664
f_16_67,1656,1660
b_8_13,1656,1664
f_15_70,1656,1660
b_6_0,1660,1668
f_17_65,1660,1664
b_7_11,1660,1668
f_16_68,1660,1664
b_9_14,1660,1668
f_15_71,1660,1664
b_11_17,1664,1672
f_17_66,1664,1668
b_8_12,1664,1672
f_16_69,1664,1668
b_10_15,1664,1672
f_15_72,1668,1672
b_7_10,1668,1676
f_17_67,1668,1672
b_9_13,1668,1676
f_16_70,1668,1672
b_11_16,1672,1680
f_15_73,1672,1676
b_8_11,1672,1680
f_17_68,1672,1676
b_10_14,1672,1680
f_16_71,1672,1676
b_7_9,1676,1684
f_15_74,1676,1680
b_9_12,1676,1684
f_17_69,1676,1680
b_14_79,1676,1684
f_16_72,1680,1684
b_8_10,1680,1688
f_15_75,1680,1684
b_10_13,1680,1688
f_17_70,1680,1684
b_7_8,1684,1692
f_16_73,1684,1688
b_9_11,1684,1692
f_15_76,1684,1688
b_12_78,1684,1692
f_17_71,1684,1688
b_8_9,1688,1696
f_16_74,1688,1692
b_10_12,1688,1696
f_15_77,1688,1692
b_11_15,1688,1696
f_17_72,1692,1696
b_9_10,1692,1700
f_16_75,1692,1696
b_12_77,1692,1700
f_15_78,1692,1696
b_8_8,1696,1704
f_17_73,1696,1700
b_10_11,1696,1704
f_16_76,1696,1700
b_11_14,1696,1704
f_15_79,1696,1700
b_9_9,1700,1708
f_17_74,1700,1704
b_12_76,1700,1708
f_16_77,1700,1704
b_7_7,1700,1708
f_21_8,1704,1708
b_10_10,1704,1712
f_17_75,1704,1708
b_11_13,1704,1712
f_16_78,1704,1708
b_9_8,1708,1716
f_21_9,1708,1712
b_12_75,1708,1716
f_17_76,1708,1712
b_7_6,1708,1716
f_16_79,1708,1712
b_10_9,1712,1720
f_21_10,1712,1716
b_11_12,1712,1720
f_17_77,1712,1716
b_8_7,1712,1720
f_26_0,1716,1720
b_12_74,1716,1724
f_21_11,1716,1720
b_7_5,1716,1724
f_17_78,1716,1720
b_10_8,1720,1728
f_26_1,1720,1724
b_11_11,1720,1728
f_21_12,1720,1724
b_8_6,1720,1728
b_9_7,1720,1728
b_12_73,1724,1732
f_26_2,1724,1728
b_7_4,1724,1732
f_21_13,1724,1728
f_27_0,1728,1732
b_11_10,1728,1736
f_26_3,1728,1732
b_8_5,1728,1736
f_21_14,1728,1732
b_10_7,1728,1736
b_12_72,1732,1740
f_27_1,1732,1736
b_7_3,1732,1740
f_26_4,1732,1736
b_9_6,1732,1740
b_11_9,1736,1744
f_27_2,1736,1740
b_8_4,1736,1744
f_26_5,1736,1740
b_15_79,1736,1744
f_28_0,1740,1744
b_7_2,1740,1748
f_27_3,1740,1744
b_9_5,1740,1748
f_25_6,1740,1744
b_11_8,1744,1752
f_28_1,1744,1748
b_8_3,1744,1752
f_27_4,1744,1748
b_10_6,1744,1752
f_17_79,1744,1748
b_7_1,1748,1756
f_28_2,1748,1752
b_9_4,1748,1756
f_27_5,1748,1752
b_12_71,1748,1756
f_29_0,1752,1756
b_8_2,1752,1760
f_28_3,1752,1756
b_10_5,1752,1760
f_26_6,1752,1756
b_7_0,1756,1764
f_29_1,1756,1760
b_9_3,1756,1764
f_28_4,1756,1760
b_12_70,1756,1764
b_11_7,1756,1764
b_8_1,1760,1768
f_29_2,1760,1764
b_10_4,1760,1768
f_28_5,1760,1764
f_30_0,1764,1768
b_9_2,1764,1772
f_29_3,1764,1768
b_12_69,1764,1772
f_27_6,1764,1768
f_18_15,1764,1768
b_8_0,1768,1776
f_30_1,1768,1772
b_10_3,1768,1776
f_29_4,1768,1772
b_11_6,1768,1776
b_16_79,1768,1776
b_9_1,1772,1780
f_30_2,1772,1776
b_12_68,1772,1780
f_29_5,1772,1776
f_18_16,1776,1780
b_10_2,1776,1784
f_30_3,1776,1780
b_11_5,1776,1784
f_28_6,1776,1780
f_19_15,1776,1780
b_9_0,1780,1788
f_18_17,1780,1784
b_12_67,1780,1788
f_30_4,1780,1784
b_13_78,1780,1788
b_17_79,1780,1788
b_10_1,1784,1792
f_18_18,1784,1788
b_11_4,1784,1792
f_30_5,1784,1788
f_19_16,1788,1792
b_12_66,1788,1796
f_18_19,1788,1792
b_13_77,1788,1796
f_29_6,1788,1792
f_20_15,1788,1792
b_10_0,1792,1800
f_19_17,1792,1796
b_11_3,1792,1800
f_18_20,1792,1796
b_14_78,1792,1800
f_21_15,1792,1796
b_12_65,1796,1804
f_19_18,1796,1800
b_13_76,1796,1804
f_18_21,1796,1800
f_22_7,1796,1800
f_20_16,1800,1804
b_11_2,1800,1808
f_19_19,1800,1804
b_14_77,1800,1808
f_18_22,1800,1804
f_23_7,1800,1804
b_12_64,1804,1812
f_20_17,1804,1808
b_13_75,1804,1812
f_19_20,1804,1808
b_15_78,1804,1812
f_18_23,1804,1808
b_11_1,1808,1816
f_20_18,1808,1812
b_14_76,1808,1816
f_19_21,1808,1812
f_24_7,1808,1812
f_18_24,1812,1816
b_13_74,1812,1820
f_20_19,1812,1816
b_15_77,1812,1820
f_19_22,1812,1816
b_12_63,1812,1820
b_11_0,1816,1824
f_18_25,1816,1820
b_14_75,1816,1824
f_20_20,1816,1820
b_16_78,1816,1824
b_13_73,1820,1828
f_18_26,1820,1824
b_15_76,1820,1828
f_20_21,1820,1824
f_19_23,1820,1824
f_19_24,1824,1828
b_14_74,1824,1832
f_18_27,1824,1828
b_16_77,1824,1832
f_20_22,1824,1828
f_25_7,1824,1828
b_13_72,1828,1836
f_19_25,1828,1832
b_15_75,1828,1836
f_18_28,1828,1832
b_12_62,1828,1836
f_20_23,1828,1832
b_14_73,1832,1840
f_19_26,1832,1836
b_16_76,1832,1840
f_18_29,1832,1836
f_26_7,1832,1836
f_20_24,1836,1840
b_15_74,1836,1844
f_19_27,1836,1840
b_12_61,1836,1844
f_18_30,1836,1840
b_13_71,1836,1844
b_14_72,1840,1848
f_20_25,1840,1844
b_16_75,1840,1848
f_19_28,1840,1844
b_17_78,1840,1848
b_15_73,1844,1852
f_20_26,1844,1848
b_12_60,1844,1852
f_19_29,1844,1848
f_18_31,1844,1848
f_18_32,1848,1852
b_16_74,1848,1856
f_20_27,1848,1852
b_17_77,1848,1856
f_19_30,1848,1852
b_14_71,1848,1856
b_15_72,1852,1860
f_18_33,1852,1856
b_12_59,1852,1860
f_20_28,1852,1856
b_13_70,1852,1860
b_16_73,1856,1864
f_18_34,1856,1860
b_17_76,1856,1864
f_20_29,1856,1860
f_19_31,1856,1860
f_19_32,1860,1864
b_12_58,1860,1868
f_18_35,1860,1864
b_13_69,1860,1868
f_20_30,1860,1864
b_15_71,1860,1868
b_16_72,1864,1872
f_19_33,1864,1868
b_17_75,1864,1872
f_18_36,1864,1868
b_14_70,1864,1872
b_12_57,1868,1876
f_19_34,1868,1872
b_13_68,1868,1876
f_18_37,1868,1872
f_20_31,1868,1872
f_20_32,1872,1876
b_17_74,1872,1880
f_19_35,1872,1876
b_14_69,1872,1880
f_18_38,1872,1876
b_16_71,1872,1880
b_12_56,1876,1884
f_20_33,1876,1880
b_13_67,1876,1884
f_19_36,1876,1880
b_15_70,1876,1884
b_17_73,1880,1888
f_20_34,1880,1884
b_14_68,1880,1888
f_19_37,1880,1884
f_18_39,1880,1884
f_18_40,1884,1888
b_13_66,1884,1892
f_20_35,1884,1888
b_15_69,1884,1892
f_19_38,1884,1888
b_12_55,1884,1892
b_17_72,1888,1896
f_18_41,1888,1892
b_14_67,1888,1896
f_20_36,1888,1892
b_16_70,1888,1896
b_13_65,1892,1900
f_18_42,1892,1896
b_15_68,1892,1900
f_20_37,1892,1896
f_19_39,1892,1896
f_19_40,1896,1900
b_14_66,1896,1904
f_18_43,1896,1900
b_16_69,1896,1904
f_20_38,1896,1900
b_17_71,1896,1904
b_13_64,1900,1908
f_19_41,1900,1904
b_15_67,1900,1908
f_18_44,1900,1904
b_12_54,1900,1908
b_14_65,1904,1912
f_19_42,1904,1908
b_16_68,1904,1912
f_18_45,1904,1908
f_20_39,1904,1908
f_20_40,1908,1912
b_15_66,1908,1916
f_19_43,1908,1912
b_12_53,1908,1916
f_18_46,1908,1912
b_13_63,1908,1916
b_14_64,1912,1920
f_20_41,1912,1916
b_16_67,1912,1920
f_19_44,1912,1916
b_17_70,1912,1920
b_15_65,1916,1924
f_20_42,1916,1920
b_12_52,1916,1924
f_19_45,1916,1920
f_18_47,1916,1920
f_18_48,1920,1924
b_16_66,1920,1928
f_20_43,1920,1924
b_17_69,1920,1928
f_19_46,1920,1924
b_14_63,1920,1928
b_15_64,1924,1932
f_18_49,1924,1928
b_12_51,1924,1932
f_20_44,1924,1928
b_13_62,1924,1932
b_16_65,1928,1936
f_18_50,1928,1932
b_17_68,1928,1936
f_20_45,1928,1932
f_19_47,1928,1932
f_19_48,1932,1936
b_12_50,1932,1940
f_18_51,1932,1936
b_13_61,1932,1940
f_20_46,1932,1936
b_15_63,1932,1940
b_16_64,1936,1944
f_19_49,1936,1940
b_17_67,1936,1944
f_18_52,1936,1940
b_14_62,1936,1944
b_12_49,1940,1948
f_19_50,1940,1944
b_13_60,1940,1948
f_18_53,1940,1944
f_20_47,1940,1944
f_20_48,1944,1948
b_17_66,1944,1952
f_19_51,1944,1948
b_14_61,1944,1952
f_18_54,1944,1948
b_16_63,1944,1952
b_12_48,1948,1956
f_20_49,1948,1952
b_13_59,1948,1956
f_19_52,1948,1952
b_15_62,1948,1956
b_17_65,1952,1960
f_20_50,1952,1956
b_14_60,1952,1960
f_19_53,1952,1956
f_18_55,1952,1956
f_18_56,1956,1960
b_13_58,1956,1964
f_20_51,1956,1960
b_15_61,1956,1964
f_19_54,1956,1960
b_12_47,1956,1964
b_17_64,1960,1968
f_18_57,1960,1964
b_14_59,1960,1968
f_20_52,1960,1964
b_16_62,1960,1968
b_13_57,1964,1972
f_18_58,1964,1968
b_15_60,1964,1972
f_20_53,1964,1968
f_19_55,1964,1968
f_19_56,1968,1972
b_14_58,1968,1976
f_18_59,1968,1972
b_16_61,1968,1976
f_20_54,1968,1972
b_17_63,1968,1976
b_13_56,1972,1980
f_19_57,1972,1976
b_15_59,1972,1980
f_18_60,1972,1976
b_12_46,1972,1980
b_14_57,1976,1984
f_19_58,1976,1980
b_16_60,1976,1984
f_18_61,1976,1980
f_20_55,1976,1980
f_20_56,1980,1984
b_15_58,1980,1988
f_19_59,1980,1984
b_12_45,1980,1988
f_18_62,1980,1984
b_13_55,1980,1988
b_14_56,1984,1992
f_20_57,1984,1988
b_16_59,1984,1992
f_19_60,1984,1988
b_17_62,1984,1992
b_15_57,1988,1996
f_20_58,1988,1992
b_12_44,1988,1996
f_19_61,1988,1992
f_18_63,1988,1992
f_18_64,1992,1996
b_16_58,1992,2000
f_20_59,1992,1996
b_17_61,1992,2000
f_19_62,1992,1996
b_14_55,1992,2000
b_15_56,1996,2004
f_18_65,1996,2000
b_12_43,1996,2004
f_20_60,1996,2000
b_13_54,1996,2004
b_16_57,2000,2008
f_18_66,2000,2004
b_17_60,2000,2008
f_20_61,2000,2004
f_19_63,2000,2004
f_19_64,2004,2008
b_12_42,2004,2012
f_18_67,2004,2008
b_13_53,2004,2012
f_20_62,2004,2008
b_15_55,2004,2012
b_16_56,2008,2016
f_19_65,2008,2012
b_17_59,2008,2016
f_18_68,2008,2012
b_14_54,2008,2016
b_12_41,2012,2020
f_19_66,2012,2016
b_13_52,2012,2020
f_18_69,2012,2016
f_20_63,2012,2016
f_20_64,2016,2020
b_17_58,2016,2024
f_19_67,2016,2020
b_14_53,2016,2024
f_18_70,2016,2020
b_16_55,2016,2024
b_12_40,2020,2028
f_20_65,2020,2024
b_13_51,2020,2028
f_19_68,2020,2024
b_15_54,2020,2028
b_17_57,2024,2032
f_20_66,2024,2028
b_14_52,2024,2032
f_19_69,2024,2028
f_18_71,2024,2028
f_18_72,2028,2032
b_13_50,2028,2036
f_20_67,2028,2032
b_15_53,2028,2036
f_19_70,2028,2032
b_12_39,2028,2036
b_17_56,2032,2040
f_18_73,2032,2036
b_14_51,2032,2040
f_20_68,2032,2036
b_16_54,2032,2040
b_13_49,2036,2044
f_18_74,2036,2040
b_15_52,2036,2044
f_20_69,2036,2040
f_19_71,2036,2040
f_19_72,2040,2044
b_14_50,2040,2048
f_18_75,2040,2044
b_16_53,2040,2048
f_20_70,2040,2044
b_17_55,2040,2048
b_13_48,2044,2052
f_19_73,2044,2048
b_15_51,2044,2052
f_18_76,2044,2048
b_12_38,2044,2052
b_14_49,2048,2056
f_19_74,2048,2052
b_16_52,2048,2056
f_18_77,2048,2052
f_20_71,2048,2052
f_20_72,2052,2056
b_15_50,2052,2060
f_19_75,2052,2056
b_12_37,2052,2060
f_18_78,2052,2056
b_13_47,2052,2060
b_14_48,2056,2064
f_20_73,2056,2060
b_16_51,2056,2064
f_19_76,2056,2060
b_17_54,2056,2064
b_15_49,2060,2068
f_20_74,2060,2064
b_12_36,2060,2068
f_19_77,2060,2064
f_18_79,2060,2064
f_21_16,2064,2068
b_16_50,2064,2072
f_20_75,2064,2068
b_17_53,2064,2072
f_19_78,2064,2068
b_14_47,2064,2072
b_15_48,2068,2076
f_21_17,2068,2072
b_12_35,2068,2076
f_20_76,2068,2072
b_13_46,2068,2076
b_16_49,2072,2080
f_21_18,2072,2076
b_17_52,2072,2080
f_20_77,2072,2076
b_18_79,2072,2080
f_22_8,2076,2080
b_12_34,2076,2084
f_21_19,2076,2080
b_13_45,2076,2084
f_20_78,2076,2080
b_16_48,2080,2088
f_22_9,2080,2084
b_17_51,2080,2088
f_21_20,2080,2084
b_14_46,2080,2088
f_19_79,2080,2084
b_12_33,2084,2092
f_22_10,2084,2088
b_13_44,2084,2092
f_21_21,2084,2088
b_15_47,2084,2092
f_23_8,2088,2092
b_17_50,2088,2096
f_22_11,2088,2092
b_14_45,2088,2096
f_21_22,2088,2092
b_12_32,2092,2100
f_23_9,2092,2096
b_13_43,2092,2100
f_22_12,2092,2096
b_15_46,2092,2100
b_16_47,2092,2100
b_17_49,2096,2104
f_23_10,2096,2100
b_14_44,2096,2104
f_22_13,2096,2100
f_24_8,2100,2104
b_13_42,2100,2108
f_23_11,2100,2104
b_15_45,2100,2108
f_22_14,2100,2104
f_20_79,2100,2104
b_17_48,2104,2112
f_24_9,2104,2108
b_14_43,2104,2112
f_23_12,2104,2108
b_16_46,2104,2112
b_12_31,2104,2112
b_13_41,2108,2116
f_24_10,2108,2112
b_15_44,2108,2116
f_23_13,2108,2112
f_25_8,2112,2116
b_14_42,2112,2120
f_24_11,2112,2116
b_16_45,2112,2120
f_23_14,2112,2116
b_17_47,2112,2120
b_13_40,2116,2124
f_25_9,2116,2120
b_15_43,2116,2124
f_24_12,2116,2120
b_12_30,2116,2124
b_14_41,2120,2128
f_25_10,2120,2124
b_16_44,2120,2128
f_24_13,2120,2124
b_19_79,2120,2128
f_26_8,2124,2128
b_15_42,2124,2132
f_25_11,2124,2128
b_12_29,2124,2132
f_24_14,2124,2128
b_14_40,2128,2136
f_26_9,2128,2132
b_16_43,2128,2136
f_25_12,2128,2132
b_17_46,2128,2136
f_21_23,2128,2132
b_15_41,2132,2140
f_26_10,2132,2136
b_12_28,2132,2140
f_25_13,2132,2136
b_13_39,2132,2140
f_21_24,2136,2140
b_16_42,2136,2144
f_26_11,2136,2140
b_17_45,2136,2144
f_25_14,2136,2140
b_15_40,2140,2148
f_21_25,2140,2144
b_12_27,2140,2148
f_26_12,2140,2144
b_13_38,2140,2148
f_22_15,2140,2144
b_16_41,2144,2152
f_21_26,2144,2148
b_17_44,2144,2152
f_26_13,2144,2148
b_14_39,2144,2152
f_22_16,2148,2152
b_12_26,2148,2156
f_21_27,2148,2152
b_13_37,2148,2156
f_26_14,2148,2152
b_16_40,2152,2160
f_22_17,2152,2156
b_17_43,2152,2160
f_21_28,2152,2156
b_14_38,2152,2160
f_23_15,2152,2156
b_12_25,2156,2164
f_22_18,2156,2160
b_13_36,2156,2164
f_21_29,2156,2160
b_15_39,2156,2164
f_23_16,2160,2164
b_17_42,2160,2168
f_22_19,2160,2164
b_14_37,2160,2168
f_21_30,2160,2164
b_12_24,2164,2172
f_23_17,2164,2168
b_13_35,2164,2172
f_22_20,2164,2168
b_15_38,2164,2172
f_21_31,2164,2168
b_17_41,2168,2176
f_23_18,2168,2172
b_14_36,2168,2176
f_22_21,2168,2172
b_16_39,2168,2176
f_21_32,2172,2176
b_13_34,2172,2180
f_23_19,2172,2176
b_15_37,2172,2180
f_22_22,2172,2176
b_17_40,2176,2184
f_21_33,2176,2180
b_14_35,2176,2184
f_23_20,2176,2180
b_16_38,2176,2184
f_22_23,2176,2180
b_13_33,2180,2188
f_21_34,2180,2184
b_15_36,2180,2188
f_23_21,2180,2184
b_12_23,2180,2188
f_22_24,2184,2188
b_14_34,2184,2192
f_21_35,2184,2188
b_16_37,2184,2192
f_23_22,2184,2188
b_13_32,2188,2196
f_22_25,2188,2192
b_15_35,2188,2196
f_21_36,2188,2192
b_12_22,2188,2196
f_23_23,2188,2192
b_14_33,2192,2200
f_22_26,2192,2196
b_16_36,2192,2200
f_21_37,2192,2196
b_17_39,2192,2200
f_23_24,2196,2200
b_15_34,2196,2204
f_22_27,2196,2200
b_12_21,2196,2204
f_21_38,2196,2200
b_14_32,2200,2208
f_23_25,2200,2204
b_16_35,2200,2208
f_22_28,2200,2204
b_17_38,2200,2208
f_21_39,2200,2204
b_15_33,2204,2212
f_23_26,2204,2208
b_12_20,2204,2212
f_22_29,2204,2208
b_13_31,2204,2212
f_21_40,2208,2212
b_16_34,2208,2216
f_23_27,2208,2212
b_17_37,2208,2216
f_22_30,2208,2212
b_15_32,2212,2220
f_21_41,2212,2216
b_12_19,2212,2220
f_23_28,2212,2216
b_13_30,2212,2220
f_22_31,2212,2216
b_16_33,2216,2224
f_21_42,2216,2220
b_17_36,2216,2224
f_23_29,2216,2220
b_14_31,2216,2224
f_22_32,2220,2224
b_12_18,2220,2228
f_21_43,2220,2224
b_13_29,2220,2228
f_23_30,2220,2224
b_16_32,2224,2232
f_22_33,2224,2228
b_17_35,2224,2232
f_21_44,2224,2228
b_14_30,2224,2232
f_23_31,2224,2228
b_12_17,2228,2236
f_22_34,2228,2232
b_13_28,2228,2236
f_21_45,2228,2232
b_15_31,2228,2236
f_23_32,2232,2236
b_17_34,2232,2240
f_22_35,2232,2236
b_14_29,2232,2240
f_21_46,2232,2236
b_12_16,2236,2244
f_23_33,2236,2240
b_13_27,2236,2244
f_22_36,2236,2240
b_15_30,2236,2244
f_21_47,2236,2240
b_17_33,2240,2248
f_23_34,2240,2244
b_14_28,2240,2248
f_22_37,2240,2244
b_16_31,2240,2248
f_21_48,2244,2248
b_13_26,2244,2252
f_23_35,2244,2248
b_15_29,2244,2252
f_22_38,2244,2248
b_17_32,2248,2256
f_21_49,2248,2252
b_14_27,2248,2256
f_23_36,2248,2252
b_16_30,2248,2256
f_22_39,2248,2252
b_13_25,2252,2260
f_21_50,2252,2256
b_15_28,2252,2260
f_23_37,2252,2256
b_12_15,2252,2260
f_22_40,2256,2260
b_14_26,2256,2264
f_21_51,2256,2260
b_16_29,2256,2264
f_23_38,2256,2260
b_13_24,2260,2268
f_22_41,2260,2264
b_15_27,2260,2268
f_21_52,2260,2264
b_12_14,2260,2268
f_23_39,2260,2264
b_14_25,2264,2272
f_22_42,2264,2268
b_16_28,2264,2272
f_21_53,2264,2268
b_17_31,2264,2272
f_23_40,2268,2272
b_15_26,2268,2276
f_22_43,2268,2272
b_12_13,2268,2276
f_21_54,2268,2272
b_14_24,2272,2280
f_23_41,2272,2276
b_16_27,2272,2280
f_22_44,2272,2276
b_17_30,2272,2280
f_21_55,2272,2276
b_15_25,2276,2284
f_23_42,2276,2280
b_12_12,2276,2284
f_22_45,2276,2280
b_13_23,2276,2284
f_21_56,2280,2284
b_16_26,2280,2288
f_23_43,2280,2284
b_17_29,2280,2288
f_22_46,2280,2284
b_15_24,2284,2292
f_21_57,2284,2288
b_12_11,2284,2292
f_23_44,2284,2288
b_13_22,2284,2292
f_22_47,2284,2288
b_16_25,2288,2296
f_21_58,2288,2292
b_17_28,2288,2296
f_23_45,2288,2292
b_14_23,2288,2296
f_22_48,2292,2296
b_12_10,2292,2300
f_21_59,2292,2296
b_13_21,2292,2300
f_23_46,2292,2296
b_16_24,2296,2304
f_22_49,2296,2300
b_17_27,2296,2304
f_21_60,2296,2300
b_14_22,2296,2304
f_23_47,2296,2300
b_12_9,2300,2308
f_22_50,2300,2304
b_13_20,2300,2308
f_21_61,2300,2304
b_15_23,2300,2308
f_23_48,2304,2308
b_17_26,2304,2312
f_22_51,2304,2308
b_14_21,2304,2312
f_21_62,2304,2308
b_12_8,2308,2316
f_23_49,2308,2312
b_13_19,2308,2316
f_22_52,2308,2312
b_15_22,2308,2316
f_21_63,2308,2312
b_17_25,2312,2320
f_23_50,2312,2316
b_14_20,2312,2320
f_22_53,2312,2316
b_16_23,2312,2320
f_21_64,2316,2320
b_13_18,2316,2324
f_23_51,2316,2320
b_15_21,2316,2324
f_22_54,2316,2320
b_17_24,2320,2328
f_21_65,2320,2324
b_14_19,2320,2328
f_23_52,2320,2324
b_16_22,2320,2328
f_22_55,2320,2324
b_13_17,2324,2332
f_21_66,2324,2328
b_15_20,2324,2332
f_23_53,2324,2328
b_12_7,2324,2332
f_22_56,2328,2332
b_14_18,2328,2336
f_21_67,2328,2332
b_16_21,2328,2336
f_23_54,2328,2332
b_13_16,2332,2340
f_22_57,2332,2336
b_15_19,2332,2340
f_21_68,2332,2336
b_12_6,2332,2340
f_23_55,2332,2336
b_14_17,2336,2344
f_22_58,2336,2340
b_16_20,2336,2344
f_21_69,2336,2340
b_17_23,2336,2344
f_23_56,2340,2344
b_15_18,2340,2348
f_22_59,2340,2344
b_12_5,2340,2348
f_21_70,2340,2344
b_14_16,2344,2352
f_23_57,2344,2348
b_16_19,2344,2352
f_22_60,2344,2348
b_17_22,2344,2352
f_21_71,2344,2348
b_15_17,2348,2356
f_23_58,2348,2352
b_12_4,2348,2356
f_22_61,2348,2352
b_13_15,2348,2356
f_21_72,2352,2356
b_16_18,2352,2360
f_23_59,2352,2356
b_17_21,2352,2360
f_22_62,2352,2356
b_15_16,2356,2364
f_21_73,2356,2360
b_12_3,2356,2364
f_23_60,2356,2360
b_13_14,2356,2364
f_22_63,2356,2360
b_16_17,2360,2368
f_21_74,2360,2364
b_17_20,2360,2368
f_23_61,2360,2364
b_14_15,2360,2368
f_22_64,2364,2368
b_12_2,2364,2372
f_21_75,2364,2368
b_13_13,2364,2372
f_23_62,2364,2368
b_16_16,2368,2376
f_22_65,2368,2372
b_17_19,2368,2376
f_21_76,2368,2372
b_14_14,2368,2376
f_23_63,2368,2372
b_12_1,2372,2380
f_22_66,2372,2376
b_13_12,2372,2380
f_21_77,2372,2376
b_15_15,2372,2380
f_23_64,2376,2380
b_17_18,2376,2384
f_22_67,2376,2380
b_14_13,2376,2384
f_21_78,2376,2380
b_12_0,2380,2388
f_23_65,2380,2384
b_13_11,2380,2388
f_22_68,2380,2384
b_15_14,2380,2388
f_21_79,2380,2384
b_17_17,2384,2392
f_23_66,2384,2388
b_14_12,2384,2392
f_22_69,2384,2388
b_16_15,2384,2392
f_31_0,2388,2392
b_13_10,2388,2396
f_23_67,2388,2392
b_15_13,2388,2396
f_22_70,2388,2392
b_17_16,2392,2400
f_31_1,2392,2396
b_14_11,2392,2400
f_23_68,2392,2396
b_16_14,2392,2400
b_20_79,2392,2400
b_13_9,2396,2404
f_31_2,2396,2400
b_15_12,2396,2404
f_23_69,2396,2400
b_14_10,2400,2408
f_31_3,2400,2404
b_16_13,2400,2408
f_23_70,2400,2404
f_22_71,2400,2404
b_13_8,2404,2412
b_15_11,2404,2412
f_31_4,2404,2408
b_18_78,2404,2412
b_17_15,2404,2412
b_14_9,2408,2416
b_16_12,2408,2416
f_31_5,2408,2412
f_22_72,2412,2416
b_15_10,2412,2420
b_18_77,2412,2420
f_30_6,2412,2416
f_23_71,2412,2416
b_14_8,2416,2424
f_22_73,2416,2420
b_16_11,2416,2424
b_17_14,2416,2424
b_13_7,2416,2424
b_15_9,2420,2428
f_22_74,2420,2424
b_18_76,2420,2428
f_23_72,2424,2428
b_16_10,2424,2432
f_22_75,2424,2428
b_17_13,2424,2432
f_31_6,2424,2428
f_24_15,2424,2428
b_15_8,2428,2436
f_23_73,2428,2432
b_18_75,2428,2436
f_22_76,2428,2432
b_13_6,2428,2436
b_14_7,2428,2436
b_16_9,2432,2440
f_23_74,2432,2436
b_17_12,2432,2440
f_22_77,2432,2436
f_24_16,2436,2440
b_18_74,2436,2444
f_23_75,2436,2440
b_13_5,2436,2444
f_22_78,2436,2440
f_25_15,2436,2440
b_16_8,2440,2448
f_24_17,2440,2444
b_17_11,2440,2448
f_23_76,2440,2444
b_14_6,2440,2448
b_15_7,2440,2448
b_18_73,2444,2452
f_24_18,2444,2448
b_13_4,2444,2452
f_23_77,2444,2448
f_25_16,2448,2452
b_17_10,2448,2456
f_24_19,2448,2452
b_14_5,2448,2456
f_23_78,2448,2452
f_22_79,2448,2452
b_18_72,2452,2460
f_25_17,2452,2456
b_13_3,2452,2460
f_24_20,2452,2456
b_15_6,2452,2460
b_16_7,2452,2460
b_17_9,2456,2464
f_25_18,2456,2460
b_14_4,2456,2464
f_24_21,2456,2460
b_13_2,2460,2468
f_25_19,2460,2464
b_15_5,2460,2468
f_24_22,2460,2464
b_18_71,2460,2468
b_17_8,2464,2472
b_14_3,2464,2472
f_25_20,2464,2468
b_16_6,2464,2472
b_13_1,2468,2476
b_15_4,2468,2476
f_25_21,2468,2472
b_21_79,2468,2476
b_14_2,2472,2480
b_16_5,2472,2480
f_25_22,2472,2476
b_13_0,2476,2484
b_15_3,2476,2484
b_18_70,2476,2484
f_23_79,2476,2480
b_14_1,2480,2488
b_16_4,2480,2488
b_17_7,2480,2488
b_15_2,2484,2492
b_18_69,2484,2492
b_19_78,2484,2492
b_14_0,2488,2496
b_16_3,2488,2496
b_22_79,2488,2496
b_15_1,2492,2500
b_18_68,2492,2500
b_19_77,2492,2500
b_17_6,2492,2500
b_16_2,2496,2504
f_24_23,2496,2500
b_15_0,2500,2508
b_18_67,2500,2508
b_19_76,2500,2508
b_17_5,2500,2508
b_20_78,2500,2508
b_23_79,2500,2508
b_16_1,2504,2512
f_24_24,2508,2512
b_18_66,2508,2516
b_19_75,2508,2516
b_17_4,2508,2516
b_20_77,2508,2516
b_21_78,2508,2516
f_25_23,2508,2512
b_16_0,2512,2520
f_24_25,2512,2516
f_26_15,2512,2516
b_18_65,2516,2524
f_24_26,2516,2520
b_17_3,2516,2524
b_20_76,2516,2524
b_21_77,2516,2524
b_22_78,2516,2524
f_27_7,2516,2520
f_25_24,2520,2524
b_19_74,2520,2528
f_28_7,2520,2524
b_18_64,2524,2532
f_25_25,2524,2528
f_24_27,2524,2528
b_21_76,2524,2532
b_22_77,2524,2532
b_23_78,2524,2532
f_29_7,2524,2528
b_19_73,2528,2536
f_25_26,2528,2532
b_20_75,2528,2536
f_30_7,2528,2532
f_26_16,2532,2536
b_17_2,2532,2540
f_24_28,2532,2536
b_23_77,2532,2540
b_18_63,2532,2540
b_19_72,2536,2544
f_26_17,2536,2540
f_25_27,2536,2540
b_22_76,2536,2544
b_17_1,2540,2548
f_26_18,2540,2544
b_21_75,2540,2548
f_24_29,2540,2544
b_18_62,2540,2548
f_31_7,2540,2544
f_27_8,2544,2548
b_20_74,2544,2552
f_25_28,2544,2548
b_19_71,2544,2552
b_17_0,2548,2556
f_27_9,2548,2552
f_26_19,2548,2552
b_23_76,2548,2556
b_18_61,2548,2556
f_24_30,2548,2552
b_20_73,2552,2560
f_27_10,2552,2556
b_22_75,2552,2560
b_19_70,2552,2560
f_24_31,2552,2556
f_24_32,2556,2560
b_21_74,2556,2564
f_26_20,2556,2560
f_25_29,2556,2560
b_20_72,2560,2568
f_24_33,2560,2564
f_27_11,2560,2564
b_18_60,2560,2568
b_19_69,2560,2568
f_25_30,2560,2564
b_21_73,2564,2572
f_24_34,2564,2568
b_23_75,2564,2572
f_25_31,2564,2568
f_25_32,2568,2572
b_22_74,2568,2576
f_27_12,2568,2572
f_26_21,2568,2572
b_20_71,2568,2576
b_21_72,2572,2580
f_25_33,2572,2576
f_24_35,2572,2576
b_19_68,2572,2580
f_27_13,2572,2576
f_26_22,2572,2576
b_22_73,2576,2584
f_25_34,2576,2580
b_18_59,2576,2584
b_20_70,2576,2584
f_26_23,2576,2580
f_26_24,2580,2584
b_23_74,2580,2588
f_24_36,2580,2584
b_21_71,2580,2588
b_22_72,2584,2592
f_26_25,2584,2588
f_25_35,2584,2588
b_20_69,2584,2592
f_27_14,2584,2588
b_23_73,2588,2596
f_26_26,2588,2592
b_19_67,2588,2596
f_25_36,2588,2592
b_21_70,2588,2596
f_27_15,2588,2592
f_27_16,2592,2596
b_18_58,2592,2600
b_20_68,2592,2600
f_24_37,2592,2596
b_22_71,2592,2600
b_23_72,2596,2604
f_27_17,2596,2600
f_26_27,2596,2600
b_21_69,2596,2604
f_24_38,2596,2600
b_18_57,2600,2608
f_27_18,2600,2604
b_20_67,2600,2608
f_26_28,2600,2604
b_22_70,2600,2608
f_24_39,2600,2604
f_24_40,2604,2608
b_19_66,2604,2612
b_21_68,2604,2612
f_25_37,2604,2608
b_23_71,2604,2612
b_18_56,2608,2616
f_24_41,2608,2612
f_27_19,2608,2612
b_22_69,2608,2616
f_25_38,2608,2612
b_19_65,2612,2620
f_24_42,2612,2616
b_21_67,2612,2620
f_27_20,2612,2616
b_23_70,2612,2620
f_25_39,2612,2616
f_25_40,2616,2620
b_20_66,2616,2624
b_22_68,2616,2624
f_26_29,2616,2620
b_18_55,2616,2624
b_19_64,2620,2628
f_25_41,2620,2624
f_24_43,2620,2624
b_23_69,2620,2628
f_26_30,2620,2624
b_20_65,2624,2632
f_25_42,2624,2628
b_22_67,2624,2632
f_24_44,2624,2628
b_18_54,2624,2632
f_26_31,2624,2628
f_26_32,2628,2632
b_21_66,2628,2636
b_23_68,2628,2636
f_24_45,2628,2632
b_19_63,2628,2636
b_20_64,2632,2640
f_26_33,2632,2636
f_25_43,2632,2636
b_18_53,2632,2640
f_24_46,2632,2636
b_21_65,2636,2644
f_26_34,2636,2640
b_23_67,2636,2644
f_25_44,2636,2640
b_19_62,2636,2644
f_24_47,2636,2640
f_24_48,2640,2644
b_22_66,2640,2648
b_18_52,2640,2648
f_25_45,2640,2644
b_20_63,2640,2648
b_21_64,2644,2652
f_24_49,2644,2648
f_26_35,2644,2648
b_19_61,2644,2652
f_25_46,2644,2648
b_22_65,2648,2656
f_24_50,2648,2652
b_18_51,2648,2656
f_26_36,2648,2652
b_20_62,2648,2656
f_25_47,2648,2652
f_25_48,2652,2656
b_23_66,2652,2660
b_19_60,2652,2660
f_26_37,2652,2656
b_21_63,2652,2660
b_22_64,2656,2664
f_25_49,2656,2660
f_24_51,2656,2660
b_20_61,2656,2664
f_26_38,2656,2660
b_23_65,2660,2668
f_25_50,2660,2664
b_19_59,2660,2668
f_24_52,2660,2664
b_21_62,2660,2668
f_26_39,2660,2664
f_26_40,2664,2668
b_18_50,2664,2672
b_20_60,2664,2672
f_24_53,2664,2668
b_22_63,2664,2672
b_23_64,2668,2676
f_26_41,2668,2672
f_25_51,2668,2672
b_21_61,2668,2676
f_24_54,2668,2672
b_18_49,2672,2680
f_26_42,2672,2676
b_20_59,2672,2680
f_25_52,2672,2676
b_22_62,2672,2680
f_24_55,2672,2676
f_24_56,2676,2680
b_19_58,2676,2684
b_21_60,2676,2684
f_25_53,2676,2680
b_23_63,2676,2684
b_18_48,2680,2688
f_24_57,2680,2684
f_26_43,2680,2684
b_22_61,2680,2688
f_25_54,2680,2684
b_19_57,2684,2692
f_24_58,2684,2688
b_21_59,2684,2692
f_26_44,2684,2688
b_23_62,2684,2692
f_25_55,2684,2688
f_25_56,2688,2692
b_20_58,2688,2696
b_22_60,2688,2696
f_26_45,2688,2692
b_18_47,2688,2696
b_19_56,2692,2700
f_25_57,2692,2696
f_24_59,2692,2696
b_23_61,2692,2700
f_26_46,2692,2696
b_20_57,2696,2704
f_25_58,2696,2700
b_22_59,2696,2704
f_24_60,2696,2700
b_18_46,2696,2704
f_26_47,2696,2700
f_26_48,2700,2704
b_21_58,2700,2708
b_23_60,2700,2708
f_24_61,2700,2704
b_19_55,2700,2708
b_20_56,2704,2712
f_26_49,2704,2708
f_25_59,2704,2708
b_18_45,2704,2712
f_24_62,2704,2708
b_21_57,2708,2716
f_26_50,2708,2712
b_23_59,2708,2716
f_25_60,2708,2712
b_19_54,2708,2716
f_24_63,2708,2712
f_24_64,2712,2716
b_22_58,2712,2720
b_18_44,2712,2720
f_25_61,2712,2716
b_20_55,2712,2720
b_21_56,2716,2724
f_24_65,2716,2720
f_26_51,2716,2720
b_19_53,2716,2724
f_25_62,2716,2720
b_22_57,2720,2728
f_24_66,2720,2724
b_18_43,2720,2728
f_26_52,2720,2724
b_20_54,2720,2728
f_25_63,2720,2724
f_25_64,2724,2728
b_23_58,2724,2732
b_19_52,2724,2732
f_26_53,2724,2728
b_21_55,2724,2732
b_22_56,2728,2736
f_25_65,2728,2732
f_24_67,2728,2732
b_20_53,2728,2736
f_26_54,2728,2732
b_23_57,2732,2740
f_25_66,2732,2736
b_19_51,2732,2740
f_24_68,2732,2736
b_21_54,2732,2740
f_26_55,2732,2736
f_26_56,2736,2740
b_18_42,2736,2744
b_20_52,2736,2744
f_24_69,2736,2740
b_22_55,2736,2744
b_23_56,2740,2748
f_26_57,2740,2744
f_25_67,2740,2744
b_21_53,2740,2748
f_24_70,2740,2744
b_18_41,2744,2752
f_26_58,2744,2748
b_20_51,2744,2752
f_25_68,2744,2748
b_22_54,2744,2752
f_24_71,2744,2748
f_24_72,2748,2752
b_19_50,2748,2756
b_21_52,2748,2756
f_25_69,2748,2752
b_23_55,2748,2756
b_18_40,2752,2760
f_24_73,2752,2756
f_26_59,2752,2756
b_22_53,2752,2760
f_25_70,2752,2756
b_19_49,2756,2764
f_24_74,2756,2760
b_21_51,2756,2764
f_26_60,2756,2760
b_23_54,2756,2764
f_25_71,2756,2760
f_25_72,2760,2764
b_20_50,2760,2768
b_22_52,2760,2768
f_26_61,2760,2764
b_18_39,2760,2768
b_19_48,2764,2772
f_25_73,2764,2768
f_24_75,2764,2768
b_23_53,2764,2772
f_26_62,2764,2768
b_20_49,2768,2776
f_25_74,2768,2772
b_22_51,2768,2776
f_24_76,2768,2772
b_18_38,2768,2776
f_26_63,2768,2772
f_26_64,2772,2776
b_21_50,2772,2780
b_23_52,2772,2780
f_24_77,2772,2776
b_19_47,2772,2780
b_20_48,2776,2784
f_26_65,2776,2780
f_25_75,2776,2780
b_18_37,2776,2784
f_24_78,2776,2780
b_21_49,2780,2788
f_26_66,2780,2784
b_23_51,2780,2788
f_25_76,2780,2784
b_19_46,2780,2788
f_24_79,2780,2784
f_28_8,2784,2788
b_22_50,2784,2792
b_18_36,2784,2792
f_25_77,2784,2788
b_20_47,2784,2792
b_21_48,2788,2796
f_28_9,2788,2792
f_26_67,2788,2792
b_19_45,2788,2796
f_25_78,2788,2792
b_22_49,2792,2800
f_28_10,2792,2796
b_18_35,2792,2800
f_26_68,2792,2796
b_20_46,2792,2800
f_25_79,2792,2796
f_29_8,2796,2800
b_23_50,2796,2804
b_19_44,2796,2804
f_26_69,2796,2800
b_21_47,2796,2804
b_22_48,2800,2808
f_29_9,2800,2804
f_28_11,2800,2804
b_20_45,2800,2808
f_26_70,2800,2804
b_23_49,2804,2812
f_29_10,2804,2808
b_19_43,2804,2812
f_28_12,2804,2808
b_21_46,2804,2812
b_24_79,2804,2812
f_30_8,2808,2812
b_18_34,2808,2816
b_20_44,2808,2816
f_27_21,2808,2812
b_23_48,2812,2820
f_30_9,2812,2816
f_29_11,2812,2816
b_21_45,2812,2820
f_27_22,2812,2816
f_26_71,2812,2816
b_18_33,2816,2824
f_30_10,2816,2820
b_20_43,2816,2824
f_29_12,2816,2820
b_24_78,2816,2824
b_22_47,2816,2824
f_26_72,2820,2824
b_19_42,2820,2828
b_21_44,2820,2828
f_28_13,2820,2824
b_18_32,2824,2832
f_26_73,2824,2828
f_30_11,2824,2828
b_24_77,2824,2832
f_28_14,2824,2828
f_27_23,2824,2828
b_19_41,2828,2836
f_26_74,2828,2832
b_21_43,2828,2836
f_30_12,2828,2832
b_22_46,2828,2836
b_23_47,2828,2836
f_27_24,2832,2836
b_20_42,2832,2840
b_24_76,2832,2840
f_29_13,2832,2836
b_19_40,2836,2844
f_27_25,2836,2840
f_26_75,2836,2840
b_22_45,2836,2844
f_29_14,2836,2840
f_28_15,2836,2840
b_20_41,2840,2848
f_27_26,2840,2844
b_24_75,2840,2848
f_26_76,2840,2844
b_23_46,2840,2848
b_18_31,2840,2848
f_28_16,2844,2848
b_21_42,2844,2852
b_22_44,2844,2852
f_26_77,2844,2848
b_20_40,2848,2856
f_28_17,2848,2852
f_27_27,2848,2852
b_23_45,2848,2856
f_26_78,2848,2852
f_29_15,2848,2852
b_21_41,2852,2860
f_28_18,2852,2856
b_22_43,2852,2860
f_27_28,2852,2856
b_18_30,2852,2860
b_19_39,2852,2860
f_29_16,2856,2860
b_24_74,2856,2864
b_23_44,2856,2864
f_27_29,2856,2860
b_21_40,2860,2868
f_29_17,2860,2864
f_28_19,2860,2864
b_18_29,2860,2868
f_27_30,2860,2864
f_26_79,2860,2864
b_24_73,2864,2872
f_29_18,2864,2868
b_23_43,2864,2872
f_28_20,2864,2868
b_19_38,2864,2872
b_20_39,2864,2872
f_31_8,2868,2872
b_22_42,2868,2876
b_18_28,2868,2876
f_28_21,2868,2872
b_24_72,2872,2880
f_31_9,2872,2876
f_29_19,2872,2876
b_19_37,2872,2880
f_28_22,2872,2876
b_21_39,2872,2880
b_22_41,2876,2884
f_31_10,2876,2880
b_18_27,2876,2884
f_29_20,2876,2880
b_20_38,2876,2884
b_23_42,2880,2888
b_19_36,2880,2888
f_29_21,2880,2884
b_24_71,2880,2888
b_22_40,2884,2892
f_31_11,2884,2888
b_20_37,2884,2892
f_29_22,2884,2888
b_23_41,2888,2896
b_18_26,2888,2896
b_19_35,2888,2896
f_31_12,2888,2892
b_21_38,2888,2896
b_25_79,2888,2896
b_20_36,2892,2900
f_30_13,2892,2896
b_23_40,2896,2904
b_18_25,2896,2904
b_19_34,2896,2904
b_21_37,2896,2904
f_30_14,2896,2900
f_27_31,2896,2900
b_20_35,2900,2908
b_24_70,2900,2908
b_22_39,2900,2908
f_27_32,2904,2908
b_19_33,2904,2912
b_21_36,2904,2912
f_31_13,2904,2908
b_18_24,2908,2916
b_20_34,2908,2916
b_24_69,2908,2916
f_31_14,2908,2912
f_28_23,2908,2912
f_27_33,2912,2916
b_21_35,2912,2920
b_22_38,2912,2920
b_23_39,2912,2920
f_28_24,2916,2920
b_20_33,2916,2924
f_27_34,2916,2920
b_24_68,2916,2924
b_19_32,2920,2928
b_21_34,2920,2928
f_27_35,2920,2924
b_22_37,2920,2928
b_23_38,2920,2928
f_29_23,2920,2924
f_28_25,2924,2928
b_24_67,2924,2932
f_27_36,2924,2928
b_18_23,2924,2932
f_29_24,2928,2932
b_21_33,2928,2936
f_28_26,2928,2932
b_22_36,2928,2936
f_27_37,2928,2932
b_25_78,2928,2936
b_20_32,2932,2940
b_24_66,2932,2940
f_28_27,2932,2936
b_23_37,2932,2940
f_30_15,2932,2936
f_29_25,2936,2940
b_22_35,2936,2944
f_28_28,2936,2940
f_27_38,2936,2940
b_19_31,2936,2944
f_30_16,2940,2944
b_24_65,2940,2948
f_29_26,2940,2944
b_23_36,2940,2948
f_28_29,2940,2944
b_18_22,2940,2948
b_21_32,2944,2952
b_22_34,2944,2952
f_29_27,2944,2948
b_25_77,2944,2952
f_27_39,2944,2948
f_30_17,2948,2952
b_23_35,2948,2956
f_29_28,2948,2952
f_28_30,2948,2952
b_20_31,2948,2956
f_27_40,2952,2956
b_22_33,2952,2960
f_30_18,2952,2956
b_25_76,2952,2960
f_29_29,2952,2956
b_19_30,2952,2960
b_24_64,2956,2964
b_23_34,2956,2964
f_30_19,2956,2960
b_18_21,2956,2964
f_28_31,2956,2960
f_27_41,2960,2964
b_25_75,2960,2968
f_30_20,2960,2964
f_29_30,2960,2964
b_21_31,2960,2968
f_28_32,2964,2968
b_23_33,2964,2972
f_27_42,2964,2968
b_18_20,2964,2972
f_30_21,2964,2968
b_20_30,2964,2972
b_22_32,2968,2976
b_25_74,2968,2976
f_27_43,2968,2972
b_19_29,2968,2976
f_29_31,2968,2972
f_28_33,2972,2976
b_18_19,2972,2980
f_27_44,2972,2976
f_30_22,2972,2976
b_24_63,2972,2980
f_29_32,2976,2980
b_25_73,2976,2984
f_28_34,2976,2980
b_19_28,2976,2984
f_27_45,2976,2980
b_21_30,2976,2984
b_23_32,2980,2988
b_18_18,2980,2988
f_28_35,2980,2984
b_20_29,2980,2988
f_30_23,2980,2984
f_29_33,2984,2988
b_19_27,2984,2992
f_28_36,2984,2988
f_27_46,2984,2988
b_22_31,2984,2992
f_30_24,2988,2992
b_18_17,2988,2996
f_29_34,2988,2992
b_20_28,2988,2996
f_28_37,2988,2992
b_24_62,2988,2996
b_25_72,2992,3000
b_19_26,2992,3000
f_29_35,2992,2996
b_21_29,2992,3000
f_27_47,2992,2996
f_30_25,2996,3000
b_20_27,2996,3004
f_29_36,2996,3000
f_28_38,2996,3000
b_23_31,2996,3004
f_27_48,3000,3004
b_19_25,3000,3008
f_30_26,3000,3004
b_21_28,3000,3008
f_29_37,3000,3004
b_22_30,3000,3008
b_18_16,3004,3012
b_20_26,3004,3012
f_30_27,3004,3008
b_24_61,3004,3012
f_28_39,3004,3008
f_27_49,3008,3012
b_21_27,3008,3016
f_30_28,3008,3012
f_29_38,3008,3012
b_25_71,3008,3016
f_28_40,3012,3016
b_20_25,3012,3020
f_27_50,3012,3016
b_24_60,3012,3020
f_30_29,3012,3016
b_23_30,3012,3020
b_19_24,3016,3024
b_21_26,3016,3024
f_27_51,3016,3020
b_22_29,3016,3024
f_29_39,3016,3020
f_28_41,3020,3024
b_24_59,3020,3028
f_27_52,3020,3024
f_30_30,3020,3024
b_18_15,3020,3028
f_29_40,3024,3028
b_21_25,3024,3032
f_28_42,3024,3028
b_22_28,3024,3032
f_27_53,3024,3028
b_25_70,3024,3032
b_20_24,3028,3036
b_24_58,3028,3036
f_28_43,3028,3032
b_23_29,3028,3036
f_30_31,3028,3032
f_29_41,3032,3036
b_22_27,3032,3040
f_28_44,3032,3036
f_27_54,3032,3036
b_19_23,3032,3040
f_30_32,3036,3040
b_24_57,3036,3044
f_29_42,3036,3040
b_23_28,3036,3044
f_28_45,3036,3040
b_18_14,3036,3044
b_21_24,3040,3048
b_22_26,3040,3048
f_29_43,3040,3044
b_25_69,3040,3048
f_27_55,3040,3044
f_30_33,3044,3048
b_23_27,3044,3052
f_29_44,3044,3048
f_28_46,3044,3048
b_20_23,3044,3052
f_27_56,3048,3052
b_22_25,3048,3056
f_30_34,3048,3052
b_25_68,3048,3056
f_29_45,3048,3052
b_19_22,3048,3056
b_24_56,3052,3060
b_23_26,3052,3060
f_30_35,3052,3056
b_18_13,3052,3060
f_28_47,3052,3056
f_27_57,3056,3060
b_25_67,3056,3064
f_30_36,3056,3060
f_29_46,3056,3060
b_21_23,3056,3064
f_28_48,3060,3064
b_23_25,3060,3068
f_27_58,3060,3064
b_18_12,3060,3068
f_30_37,3060,3064
b_20_22,3060,3068
b_22_24,3064,3072
b_25_66,3064,3072
f_27_59,3064,3068
b_19_21,3064,3072
f_29_47,3064,3068
f_28_49,3068,3072
b_18_11,3068,3076
f_27_60,3068,3072
f_30_38,3068,3072
b_24_55,3068,3076
f_29_48,3072,3076
b_25_65,3072,3080
f_28_50,3072,3076
b_19_20,3072,3080
f_27_61,3072,3076
b_21_22,3072,3080
b_23_24,3076,3084
b_18_10,3076,3084
f_28_51,3076,3080
b_20_21,3076,3084
f_30_39,3076,3080
f_29_49,3080,3084
b_19_19,3080,3088
f_28_52,3080,3084
f_27_62,3080,3084
b_22_23,3080,3088
f_30_40,3084,3088
b_18_9,3084,3092
f_29_50,3084,3088
b_20_20,3084,3092
f_28_53,3084,3088
b_24_54,3084,3092
b_25_64,3088,3096
b_19_18,3088,3096
f_29_51,3088,3092
b_21_21,3088,3096
f_27_63,3088,3092
f_30_41,3092,3096
b_20_19,3092,3100
f_29_52,3092,3096
f_28_54,3092,3096
b_23_23,3092,3100
f_27_64,3096,3100
b_19_17,3096,3104
f_30_42,3096,3100
b_21_20,3096,3104
f_29_53,3096,3100
b_22_22,3096,3104
b_18_8,3100,3108
b_20_18,3100,3108
f_30_43,3100,3104
b_24_53,3100,3108
f_28_55,3100,3104
f_27_65,3104,3108
b_21_19,3104,3112
f_30_44,3104,3108
f_29_54,3104,3108
b_25_63,3104,3112
f_28_56,3108,3112
b_20_17,3108,3116
f_27_66,3108,3112
b_24_52,3108,3116
f_30_45,3108,3112
b_23_22,3108,3116
b_19_16,3112,3120
b_21_18,3112,3120
f_27_67,3112,3116
b_22_21,3112,3120
f_29_55,3112,3116
f_28_57,3116,3120
b_24_51,3116,3124
f_27_68,3116,3120
f_30_46,3116,3120
b_18_7,3116,3124
f_29_56,3120,3124
b_21_17,3120,3128
f_28_58,3120,3124
b_22_20,3120,3128
f_27_69,3120,3124
b_25_62,3120,3128
b_20_16,3124,3132
b_24_50,3124,3132
f_28_59,3124,3128
b_23_21,3124,3132
f_30_47,3124,3128
f_29_57,3128,3132
b_22_19,3128,3136
f_28_60,3128,3132
f_27_70,3128,3132
b_19_15,3128,3136
f_30_48,3132,3136
b_24_49,3132,3140
f_29_58,3132,3136
b_23_20,3132,3140
f_28_61,3132,3136
b_18_6,3132,3140
b_21_16,3136,3144
b_22_18,3136,3144
f_29_59,3136,3140
b_25_61,3136,3144
f_27_71,3136,3140
f_30_49,3140,3144
b_23_19,3140,3148
f_29_60,3140,3144
f_28_62,3140,3144
b_20_15,3140,3148
f_27_72,3144,3148
b_22_17,3144,3152
f_30_50,3144,3148
b_25_60,3144,3152
f_29_61,3144,3148
b_19_14,3144,3152
b_24_48,3148,3156
b_23_18,3148,3156
f_30_51,3148,3152
b_18_5,3148,3156
f_28_63,3148,3152
f_27_73,3152,3156
b_25_59,3152,3160
f_30_52,3152,3156
f_29_62,3152,3156
b_21_15,3152,3160
f_28_64,3156,3160
b_23_17,3156,3164
f_27_74,3156,3160
b_18_4,3156,3164
f_30_53,3156,3160
b_20_14,3156,3164
b_22_16,3160,3168
b_25_58,3160,3168
f_27_75,3160,3164
b_19_13,3160,3168
f_29_63,3160,3164
f_28_65,3164,3168
b_18_3,3164,3172
f_27_76,3164,3168
f_30_54,3164,3168
b_24_47,3164,3172
f_29_64,3168,3172
b_25_57,3168,3176
f_28_66,3168,3172
b_19_12,3168,3176
f_27_77,3168,3172
b_21_14,3168,3176
b_23_16,3172,3180
b_18_2,3172,3180
f_28_67,3172,3176
b_20_13,3172,3180
f_30_55,3172,3176
f_29_65,3176,3180
b_19_11,3176,3184
f_28_68,3176,3180
f_27_78,3176,3180
b_22_15,3176,3184
f_30_56,3180,3184
b_18_1,3180,3188
f_29_66,3180,3184
b_20_12,3180,3188
f_28_69,3180,3184
b_24_46,3180,3188
b_25_56,3184,3192
b_19_10,3184,3192
f_29_67,3184,3188
b_21_13,3184,3192
f_27_79,3184,3188
f_30_57,3188,3192
b_20_11,3188,3196
f_29_68,3188,3192
f_28_70,3188,3192
b_23_15,3188,3196
b_18_0,3192,3200
b_19_9,3192,3200
f_30_58,3192,3196
b_21_12,3192,3200
f_29_69,3192,3196
b_22_14,3192,3200
b_20_10,3196,3204
f_30_59,3196,3200
b_24_45,3196,3204
b_25_55,3196,3204
b_19_8,3200,3208
b_21_11,3200,3208
f_30_60,3200,3204
f_29_70,3200,3204
b_20_9,3204,3212
b_24_44,3204,3212
f_30_61,3204,3208
b_23_14,3204,3212
b_26_79,3204,3212
b_21_10,3208,3216
b_22_13,3208,3216
b_20_8,3212,3220
b_24_43,3212,3220
f_30_62,3212,3216
f_28_71,3212,3216
b_21_9,3216,3224
b_22_12,3216,3224
b_23_13,3216,3224
b_25_54,3216,3224
b_19_7,3216,3224
f_28_72,3220,3224
b_24_42,3220,3228
b_21_8,3224,3232
f_28_73,3224,3228
b_22_11,3224,3232
b_23_12,3224,3232
b_25_53,3224,3232
b_19_6,3224,3232
f_29_71,3224,3228
b_24_41,3228,3236
f_28_74,3228,3232
b_20_7,3228,3236
f_29_72,3232,3236
b_22_10,3232,3240
f_28_75,3232,3236
b_25_52,3232,3240
b_19_5,3232,3240
b_26_78,3232,3240
b_24_40,3236,3244
f_29_73,3236,3240
b_23_11,3236,3244
f_30_63,3236,3240
b_22_9,3240,3248
f_29_74,3240,3244
f_28_76,3240,3244
b_26_77,3240,3248
b_20_6,3240,3248
b_21_7,3240,3248
f_30_64,3244,3248
b_23_10,3244,3252
f_29_75,3244,3248
b_19_4,3244,3252
b_22_8,3248,3256
f_30_65,3248,3252
b_25_51,3248,3256
f_28_77,3248,3252
b_21_6,3248,3256
f_31_15,3248,3252
b_23_9,3252,3260
f_30_66,3252,3256
f_29_76,3252,3256
b_20_5,3252,3260
b_24_39,3252,3260
f_31_16,3256,3260
b_25_50,3256,3264
f_30_67,3256,3260
b_26_76,3256,3264
f_28_78,3256,3260
b_23_8,3260,3268
f_31_17,3260,3264
b_19_3,3260,3268
f_29_77,3260,3264
b_24_38,3260,3268
f_28_79,3260,3264
b_25_49,3264,3272
f_31_18,3264,3268
f_30_68,3264,3268
b_21_5,3264,3272
b_22_7,3264,3272
b_19_2,3268,3276
f_31_19,3268,3272
b_20_4,3268,3276
f_29_78,3268,3272
b_25_48,3272,3280
b_26_75,3272,3280
f_30_69,3272,3276
b_22_6,3272,3280
f_29_79,3272,3276
b_19_1,3276,3284
f_31_20,3276,3280
b_24_37,3276,3284
b_23_7,3276,3284
b_26_74,3280,3288
b_20_3,3280,3288
b_21_4,3280,3288
f_30_70,3280,3284
b_19_0,3284,3292
f_31_21,3284,3288
b_23_6,3284,3292
b_25_47,3284,3292
b_26_73,3288,3296
b_20_2,3288,3296
b_21_3,3288,3296
b_24_36,3288,3296
b_22_5,3288,3296
f_31_22,3292,3296
b_27_79,3292,3300
b_26_72,3296,3304
b_20_1,3296,3304
b_21_2,3296,3304
b_24_35,3296,3304
b_22_4,3296,3304
b_23_5,3296,3304
b_25_46,3296,3304
f_30_71,3300,3304
f_30_72,3304,3308
b_21_1,3304,3312
b_24_34,3304,3312
b_22_3,3304,3312
b_23_4,3304,3312
b_25_45,3304,3312
b_27_78,3304,3312
b_26_71,3304,3312
b_20_0,3308,3316
f_30_73,3312,3316
b_22_2,3312,3320
b_23_3,3312,3320
b_25_44,3312,3320
b_27_77,3312,3320
b_26_70,3312,3320
f_31_23,3312,3316
f_31_24,3316,3320
b_24_33,3316,3324
b_28_79,3316,3324
b_21_0,3320,3328
f_30_74,3320,3324
b_25_43,3320,3328
b_27_76,3320,3328
b_26_69,3320,3328
f_31_25,3324,3328
b_23_2,3324,3332
b_28_78,3324,3332
b_29_79,3324,3332
b_24_32,3328,3336
b_22_1,3328,3336
f_30_75,3328,3332
b_26_68,3328,3336
f_31_26,3332,3336
b_27_75,3332,3340
b_28_77,3332,3340
b_29_78,3332,3340
b_22_0,3336,3344
b_23_1,3336,3344
b_25_42,3336,3344
f_30_76,3336,3340
b_24_31,3336,3344
f_31_27,3340,3344
b_28_76,3340,3348
f_30_77,3340,3344
b_23_0,3344,3352
b_25_41,3344,3352
b_27_74,3344,3352
b_26_67,3344,3352
b_29_77,3344,3352
f_30_78,3344,3348
f_31_28,3348,3352
b_24_30,3348,3356
f_30_79,3348,3352
b_25_40,3352,3360
b_27_73,3352,3360
b_26_66,3352,3360
b_28_75,3352,3360
b_29_76,3352,3360
f_31_29,3352,3356
b_30_79,3352,3360
b_24_29,3356,3364
f_31_30,3356,3360
b_27_72,3360,3368
b_26_65,3360,3368
b_28_74,3360,3368
b_29_75,3360,3368
b_30_78,3360,3368
f_31_31,3360,3364
b_24_28,3364,3372
b_25_39,3364,3372
f_31_32,3368,3372
b_28_73,3368,3376
b_29_74,3368,3376
b_30_77,3368,3376
b_26_64,3372,3380
b_24_27,3372,3380
b_25_38,3372,3380
b_27_71,3372,3380
f_31_33,3376,3380
b_30_76,3376,3384
b_28_72,3380,3388
b_29_73,3380,3388
f_31_34,3380,3384
b_25_37,3380,3388
b_27_70,3380,3388
b_26_63,3380,3388
b_24_26,3384,3392
f_31_35,3384,3388
b_29_72,3388,3396
b_30_75,3388,3396
f_31_36,3388,3392
b_27_69,3388,3396
b_26_62,3388,3396
b_28_71,3388,3396
b_24_25,3392,3400
b_25_36,3392,3400
b_30_74,3396,3404
f_31_37,3396,3400
b_28_70,3396,3404
b_29_71,3396,3404
b_24_24,3400,3408
b_25_35,3400,3408
b_27_68,3400,3408
b_26_61,3400,3408
b_30_73,3404,3412
f_31_38,3404,3408
b_25_34,3408,3416
b_27_67,3408,3416
b_26_60,3408,3416
b_28_69,3408,3416
b_29_70,3408,3416
f_31_39,3408,3412
f_31_40,3412,3416
b_24_23,3412,3420
b_30_72,3416,3424
f_31_41,3416,3420
b_27_66,3416,3424
b_26_59,3416,3424
b_28_68,3416,3424
b_29_69,3416,3424
b_25_33,3420,3428
b_24_22,3420,3428
f_31_42,3424,3428
b_28_67,3424,3432
b_29_68,3424,3432
b_30_71,3424,3432
b_25_32,3428,3436
b_27_65,3428,3436
b_26_58,3428,3436
b_24_21,3428,3436
f_31_43,3432,3436
b_30_70,3432,3440
b_27_64,3436,3444
b_26_57,3436,3444
b_28_66,3436,3444
b_29_67,3436,3444
f_31_44,3436,3440
b_25_31,3436,3444
b_24_20,3440,3448
f_31_45,3440,3444
b_26_56,3444,3452
b_28_65,3444,3452
b_29_66,3444,3452
b_30_69,3444,3452
f_31_46,3444,3448
b_27_63,3444,3452
b_24_19,3448,3456
b_25_30,3448,3456
b_28_64,3452,3460
b_29_65,3452,3460
b_30_68,3452,3460
f_31_47,3452,3456
b_24_18,3456,3464
b_25_29,3456,3464
b_27_62,3456,3464
b_26_55,3456,3464
f_31_48,3460,3464
b_30_67,3460,3468
b_29_64,3464,3472
f_31_49,3464,3468
b_25_28,3464,3472
b_27_61,3464,3472
b_26_54,3464,3472
b_28_63,3464,3472
b_24_17,3468,3476
f_31_50,3468,3472
b_30_66,3472,3480
f_31_51,3472,3476
b_27_60,3472,3480
b_26_53,3472,3480
b_28_62,3472,3480
b_29_63,3472,3480
b_24_16,3476,3484
b_25_27,3476,3484
b_30_65,3480,3488
f_31_52,3480,3484
b_28_61,3480,3488
b_29_62,3480,3488
b_25_26,3484,3492
b_27_59,3484,3492
b_26_52,3484,3492
b_24_15,3484,3492
b_30_64,3488,3496
f_31_53,3488,3492
b_25_25,3492,3500
b_27_58,3492,3500
b_26_51,3492,3500
b_28_60,3492,3500
b_29_61,3492,3500
f_31_54,3492,3496
b_24_14,3496,3504
f_31_55,3496,3500
f_31_56,3500,3504
b_27_57,3500,3508
b_26_50,3500,3508
b_28_59,3500,3508
b_29_60,3500,3508
b_30_63,3500,3508
b_25_24,3504,3512
b_24_13,3504,3512
f_31_57,3508,3512
b_28_58,3508,3516
b_29_59,3508,3516
b_30_62,3508,3516
b_27_56,3512,3520
b_26_49,3512,3520
b_24_12,3512,3520
b_25_23,3512,3520
f_31_58,3516,3520
b_30_61,3516,3524
b_26_48,3520,3528
b_28_57,3520,3528
b_29_58,3520,3528
f_31_59,3520,3524
b_25_22,3520,3528
b_27_55,3520,3528
b_24_11,3524,3532
f_31_60,3524,3528
b_28_56,3528,3536
b_29_57,3528,3536
b_30_60,3528,3536
f_31_61,3528,3532
b_27_54,3528,3536
b_26_47,3528,3536
b_24_10,3532,3540
b_25_21,3532,3540
b_29_56,3536,3544
b_30_59,3536,3544
f_31_62,3536,3540
b_28_55,3536,3544
b_24_9,3540,3548
b_25_20,3540,3548
b_27_53,3540,3548
b_26_46,3540,3548
b_30_58,3544,3552
f_31_63,3544,3548
f_31_64,3548,3552
b_25_19,3548,3556
b_27_52,3548,3556
b_26_45,3548,3556
b_28_54,3548,3556
b_29_55,3548,3556
b_24_8,3552,3560
f_31_65,3552,3556
b_30_57,3556,3564
f_31_66,3556,3560
b_27_51,3556,3564
b_26_44,3556,3564
b_28_53,3556,3564
b_29_54,3556,3564
b_25_18,3560,3568
b_24_7,3560,3568
b_30_56,3564,3572
f_31_67,3564,3568
b_28_52,3564,3572
b_29_53,3564,3572
b_25_17,3568,3576
b_27_50,3568,3576
b_26_43,3568,3576
b_24_6,3568,3576
f_31_68,3572,3576
b_30_55,3572,3580
b_25_16,3576,3584
b_27_49,3576,3584
b_26_42,3576,3584
b_28_51,3576,3584
b_29_52,3576,3584
f_31_69,3576,3580
b_24_5,3580,3588
f_31_70,3580,3584
b_27_48,3584,3592
b_26_41,3584,3592
b_28_50,3584,3592
b_29_51,3584,3592
b_30_54,3584,3592
f_31_71,3584,3588
b_24_4,3588,3596
b_25_15,3588,3596
f_31_72,3592,3596
b_28_49,3592,3600
b_29_50,3592,3600
b_30_53,3592,3600
b_26_40,3596,3604
b_24_3,3596,3604
b_25_14,3596,3604
b_27_47,3596,3604
f_31_73,3600,3604
b_30_52,3600,3608
b_28_48,3604,3612
b_29_49,3604,3612
f_31_74,3604,3608
b_25_13,3604,3612
b_27_46,3604,3612
b_26_39,3604,3612
b_24_2,3608,3616
f_31_75,3608,3612
b_29_48,3612,3620
b_30_51,3612,3620
f_31_76,3612,3616
b_27_45,3612,3620
b_26_38,3612,3620
b_28_47,3612,3620
b_24_1,3616,3624
b_25_12,3616,3624
b_30_50,3620,3628
f_31_77,3620,3624
b_28_46,3620,3628
b_29_47,3620,3628
b_24_0,3624,3632
b_25_11,3624,3632
b_27_44,3624,3632
b_26_37,3624,3632
b_30_49,3628,3636
f_31_78,3628,3632
b_25_10,3632,3640
b_27_43,3632,3640
b_26_36,3632,3640
b_28_45,3632,3640
b_29_46,3632,3640
f_31_79,3632,3636
b_30_48,3636,3644
b_31_79,3636,3644
b_25_9,3640,3648
b_27_42,3640,3648
b_26_35,3640,3648
b_28_44,3640,3648
b_29_45,3640,3648
b_31_78,3644,3652
b_30_47,3644,3652
b_25_8,3648,3656
b_27_41,3648,3656
b_26_34,3648,3656
b_28_43,3648,3656
b_29_44,3648,3656
b_31_77,3652,3660
b_30_46,3652,3660
b_27_40,3656,3664
b_26_33,3656,3664
b_28_42,3656,3664
b_29_43,3656,3664
b_25_7,3656,3664
b_31_76,3660,3668
b_30_45,3660,3668
b_26_32,3664,3672
b_28_41,3664,3672
b_29_42,3664,3672
b_25_6,3664,3672
b_27_39,3664,3672
b_31_75,3668,3676
b_30_44,3668,3676
b_28_40,3672,3680
b_29_41,3672,3680
b_25_5,3672,3680
b_27_38,3672,3680
b_26_31,3672,3680
b_31_74,3676,3684
b_30_43,3676,3684
b_29_40,3680,3688
b_25_4,3680,3688
b_27_37,3680,3688
b_26_30,3680,3688
b_28_39,3680,3688
b_31_73,3684,3692
b_30_42,3684,3692
b_25_3,3688,3696
b_27_36,3688,3696
b_26_29,3688,3696
b_28_38,3688,3696
b_29_39,3688,3696
b_31_72,3692,3700
b_30_41,3692,3700
b_25_2,3696,3704
b_27_35,3696,3704
b_26_28,3696,3704
b_28_37,3696,3704
b_29_38,3696,3704
b_30_40,3700,3708
b_31_71,3700,3708
b_25_1,3704,3712
b_27_34,3704,3712
b_26_27,3704,3712
b_28_36,3704,3712
b_29_37,3704,3712
b_31_70,3708,3716
b_30_39,3708,3716
b_25_0,3712,3720
b_27_33,3712,3720
b_26_26,3712,3720
b_28_35,3712,3720
b_29_36,3712,3720
b_31_69,3716,3724
b_30_38,3716,3724
b_27_32,3720,3728
b_26_25,3720,3728
b_28_34,3720,3728
b_29_35,3720,3728
b_31_68,3724,3732
b_30_37,3724,3732
b_26_24,3728,3736
b_28_33,3728,3736
b_29_34,3728,3736
b_27_31,3728,3736
b_31_67,3732,3740
b_30_36,3732,3740
b_28_32,3736,3744
b_29_33,3736,3744
b_27_30,3736,3744
b_26_23,3736,3744
b_31_66,3740,3748
b_30_35,3740,3748
b_29_32,3744,3752
b_27_29,3744,3752
b_26_22,3744,3752
b_28_31,3744,3752
b_31_65,3748,3756
b_30_34,3748,3756
b_27_28,3752,3760
b_26_21,3752,3760
b_28_30,3752,3760
b_29_31,3752,3760
b_31_64,3756,3764
b_30_33,3756,3764
b_27_27,3760,3768
b_26_20,3760,3768
b_28_29,3760,3768
b_29_30,3760,3768
b_30_32,3764,3772
b_31_63,3764,3772
b_27_26,3768,3776
b_26_19,3768,3776
b_28_28,3768,3776
b_29_29,3768,3776
b_31_62,3772,3780
b_30_31,3772,3780
b_27_25,3776,3784
b_26_18,3776,3784
b_28_27,3776,3784
b_29_28,3776,3784
b_31_61,3780,3788
b_30_30,3780,3788
b_27_24,3784,3792
b_26_17,3784,3792
b_28_26,3784,3792
b_29_27,3784,3792
b_31_60,3788,3796
b_30_29,3788,3796
b_26_16,3792,3800
b_28_25,3792,3800
b_29_26,3792,3800
b_27_23,3792,3800
b_31_59,3796,3804
b_30_28,3796,3804
b_28_24,3800,3808
b_29_25,3800,3808
b_27_22,3800,3808
b_26_15,3800,3808
b_31_58,3804,3812
b_30_27,3804,3812
b_29_24,3808,3816
b_27_21,3808,3816
b_26_14,3808,3816
b_28_23,3808,3816
b_31_57,3812,3820
b_30_26,3812,3820
b_27_20,3816,3824
b_26_13,3816,3824
b_28_22,3816,3824
b_29_23,3816,3824
b_31_56,3820,3828
b_30_25,3820,3828
b_27_19,3824,3832
b_26_12,3824,3832
b_28_21,3824,3832
b_29_22,3824,3832
b_30_24,3828,3836
b_31_55,3828,3836
b_27_18,3832,3840
b_26_11,3832,3840
b_28_20,3832,3840
b_29_21,3832,3840
b_31_54,3836,3844
b_30_23,3836,3844
b_27_17,3840,3848
b_26_10,3840,3848
b_28_19,3840,3848
b_29_20,3840,3848
b_31_53,3844,3852
b_30_22,3844,3852
b_27_16,3848,3856
b_26_9,3848,3856
b_28_18,3848,3856
b_29_19,3848,3856
b_31_52,3852,3860
b_30_21,3852,3860
b_26_8,3856,3864
b_28_17,3856,3864
b_29_18,3856,3864
b_27_15,3856,3864
b_31_51,3860,3868
b_30_20,3860,3868
b_28_16,3864,3872
b_29_17,3864,3872
b_27_14,3864,3872
b_26_7,3864,3872
b_31_50,3868,3876
b_30_19,3868,3876
b_29_16,3872,3880
b_27_13,3872,3880
b_26_6,3872,3880
b_28_15,3872,3880
b_31_49,3876,3884
b_30_18,3876,3884
b_27_12,3880,3888
b_26_5,3880,3888
b_28_14,3880,3888
b_29_15,3880,3888
b_31_48,3884,3892
b_30_17,3884,3892
b_27_11,3888,3896
b_26_4,3888,3896
b_28_13,3888,3896
b_29_14,3888,3896
b_30_16,3892,3900
b_31_47,3892,3900
b_27_10,3896,3904
b_26_3,3896,3904
b_28_12,3896,3904
b_29_13,3896,3904
b_31_46,3900,3908
b_30_15,3900,3908
b_27_9,3904,3912
b_26_2,3904,3912
b_28_11,3904,3912
b_29_12,3904,3912
b_31_45,3908,3916
b_30_14,3908,3916
b_27_8,3912,3920
b_26_1,3912,3920
b_28_10,3912,3920
b_29_11,3912,3920
b_31_44,3916,3924
b_30_13,3916,3924
b_26_0,3920,3928
b_28_9,3920,3928
b_29_10,3920,3928
b_27_7,3920,3928
b_31_43,3924,3932
b_30_12,3924,3932
b_28_8,3928,3936
b_29_9,3928,3936
b_27_6,3928,3936
b_31_42,3932,3940
b_30_11,3932,3940
b_29_8,3936,3944
b_27_5,3936,3944
b_28_7,3936,3944
b_31_41,3940,3948
b_30_10,3940,3948
b_27_4,3944,3952
b_28_6,3944,3952
b_29_7,3944,3952
b_31_40,3948,3956
b_30_9,3948,3956
b_27_3,3952,3960
b_28_5,3952,3960
b_29_6,3952,3960
b_30_8,3956,3964
b_31_39,3956,3964
b_27_2,3960,3968
b_28_4,3960,3968
b_29_5,3960,3968
b_31_38,3964,3972
b_30_7,3964,3972
b_27_1,3968,3976
b_28_3,3968,3976
b_29_4,3968,3976
b_31_37,3972,3980
b_30_6,3972,3980
b_27_0,3976,3984
b_28_2,3976,3984
b_29_3,3976,3984
b_31_36,3980,3988
b_30_5,3980,3988
b_28_1,3984,3992
b_29_2,3984,3992
b_31_35,3988,3996
b_30_4,3988,3996
b_28_0,3992,4000
b_29_1,3992,4000
b_31_34,3996,4004
b_30_3,3996,4004
b_29_0,4000,4008
b_31_33,4004,4012
b_30_2,4004,4012
b_31_32,4012,4020
b_30_1,4012,4020
b_30_0,4020,4028
b_31_31,4020,4028
b_31_30,4028,4036
b_31_29,4036,4044
b_31_28,4044,4052
b_31_27,4052,4060
b_31_26,4060,4068
b_31_25,4068,4076
b_31_24,4076,4084
b_31_23,4084,4092
b_31_22,4092,4100
b_31_21,4100,4108
b_31_20,4108,4116
b_31_19,4116,4124
b_31_18,4124,4132
b_31_17,4132,4140
b_31_16,4140,4148
b_31_15,4148,4156
b_31_14,4156,4164
b_31_13,4164,4172
b_31_12,4172,4180
b_31_11,4180,4188
b_31_10,4188,4196
b_31_9,4196,4204
b_31_8,4204,4212
b_31_7,4212,4220
b_31_6,4220,4228
b_31_5,4228,4236
b_31_4,4236,4244
b_31_3,4244,4252
b_31_2,4252,4260
b_31_1,4260,4268
b_31_0,4268,4276
"""
    unified_scheduler = order_result_mutichunk(input_str,stage_alignment)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)
    return stage_alignment, unified_scheduler, comm_graph
if __name__ == '__main__':
    stage_alignment = [[0,4],[1,5],[2,6],[3,7]]
    input_str = '''f_0_0,0,12
f_1_0,12,24
f_0_1,12,24
f_2_0,24,36
f_1_1,24,36
f_0_2,24,36
f_3_0,36,48
f_2_1,36,48
f_1_2,36,48
f_0_3,36,48
f_0_4,48,60
f_3_1,48,60
f_2_2,48,60
f_1_3,48,60
f_1_4,60,72
f_0_5,60,72
f_3_2,60,72
f_2_3,60,72
f_2_4,72,84
f_1_5,72,84
f_0_6,72,84
f_3_3,72,84
f_3_4,84,96
f_2_5,84,96
f_1_6,84,96
f_0_7,84,96
f_4_0,96,108
f_3_5,96,108
b_0_7,96,120
f_5_0,108,120
f_2_6,120,132
f_1_7,120,132
b_0_6,132,156
b_1_7,132,156
f_4_1,156,168
f_3_6,156,168
f_2_7,156,168
b_0_5,168,192
b_1_6,168,192
b_2_7,168,192
f_6_0,192,204
f_5_1,192,204
f_4_2,192,204
f_3_7,192,204
b_0_4,204,228
b_1_5,204,228
b_2_6,204,228
b_3_7,204,228
f_7_0,228,240
f_6_1,228,240
f_5_2,228,240
f_4_3,228,240
b_1_4,240,264
b_2_5,240,264
b_3_6,240,264
b_0_3,240,264
f_4_4,264,276
f_7_1,264,276
f_6_2,264,276
f_5_3,264,276
b_2_4,276,300
b_3_5,276,300
b_0_2,276,300
b_1_3,276,300
f_5_4,300,312
f_4_5,300,312
f_7_2,300,312
f_6_3,300,312
b_3_4,312,336
b_0_1,312,336
b_1_2,312,336
b_2_3,312,336
f_6_4,336,348
f_5_5,336,348
f_4_6,336,348
f_7_3,336,348
b_0_0,348,372
b_1_1,348,372
b_2_2,348,372
b_3_3,348,372
f_7_4,372,384
f_6_5,372,384
f_5_6,372,384
f_4_7,372,384
b_1_0,384,408
b_2_1,384,408
b_3_2,384,408
b_4_7,384,408
f_8_0,408,420
f_7_5,408,420
f_6_6,408,420
f_5_7,408,420
b_2_0,420,444
b_3_1,420,444
b_4_6,420,444
b_5_7,420,444
f_9_0,444,456
f_8_1,444,456
f_7_6,444,456
f_6_7,444,456
b_3_0,456,480
b_4_5,456,480
b_5_6,456,480
b_6_7,456,480
f_10_0,480,492
f_9_1,480,492
f_8_2,480,492
f_7_7,480,492
b_4_4,492,516
b_5_5,492,516
b_6_6,492,516
b_7_7,492,516
f_11_0,516,528
f_10_1,516,528
f_9_2,516,528
f_8_3,516,528
b_5_4,528,552
b_6_5,528,552
b_7_6,528,552
b_4_3,528,552
f_8_4,552,564
f_11_1,552,564
f_10_2,552,564
f_9_3,552,564
b_6_4,564,588
b_7_5,564,588
b_4_2,564,588
b_5_3,564,588
f_9_4,588,600
f_8_5,588,600
f_11_2,588,600
f_10_3,588,600
b_7_4,600,624
b_4_1,600,624
b_5_2,600,624
b_6_3,600,624
f_10_4,624,636
f_9_5,624,636
f_8_6,624,636
f_11_3,624,636
b_4_0,636,660
b_5_1,636,660
b_6_2,636,660
b_7_3,636,660
f_11_4,660,672
f_10_5,660,672
f_9_6,660,672
f_8_7,660,672
b_5_0,672,696
b_6_1,672,696
b_7_2,672,696
b_8_7,672,696
f_12_0,696,708
f_11_5,696,708
f_10_6,696,708
f_9_7,696,708
b_6_0,708,732
b_7_1,708,732
b_8_6,708,732
b_9_7,708,732
f_13_0,732,744
f_12_1,732,744
f_11_6,732,744
f_10_7,732,744
b_7_0,744,768
b_8_5,744,768
b_9_6,744,768
b_10_7,744,768
f_14_0,768,780
f_13_1,768,780
f_12_2,768,780
f_11_7,768,780
b_8_4,780,804
b_9_5,780,804
b_10_6,780,804
b_11_7,780,804
f_15_0,804,816
f_14_1,804,816
f_13_2,804,816
f_12_3,804,816
b_9_4,816,840
b_10_5,816,840
b_11_6,816,840
b_8_3,816,840
f_12_4,840,852
f_15_1,840,852
f_14_2,840,852
f_13_3,840,852
b_10_4,852,876
b_11_5,852,876
b_8_2,852,876
b_9_3,852,876
f_13_4,876,888
f_12_5,876,888
f_15_2,876,888
f_14_3,876,888
b_11_4,888,912
b_8_1,888,912
b_9_2,888,912
b_10_3,888,912
f_14_4,912,924
f_13_5,912,924
f_12_6,912,924
f_15_3,912,924
b_8_0,924,948
b_9_1,924,948
b_10_2,924,948
b_11_3,924,948
f_15_4,948,960
f_14_5,948,960
f_13_6,948,960
f_12_7,948,960
b_9_0,960,984
b_10_1,960,984
b_11_2,960,984
b_12_7,960,984
b_10_0,984,1008
f_15_5,984,996
f_14_6,984,996
f_13_7,984,996
b_11_1,996,1020
b_12_6,996,1020
b_13_7,996,1020
b_11_0,1020,1044
b_12_5,1020,1044
f_15_6,1020,1032
f_14_7,1020,1032
b_13_6,1032,1056
b_14_7,1032,1056
b_12_4,1044,1068
b_13_5,1056,1080
b_14_6,1056,1080
f_15_7,1056,1068
b_15_7,1068,1092
b_13_4,1080,1104
b_14_5,1080,1104
b_15_6,1092,1116
b_12_3,1092,1116
b_14_4,1104,1128
b_15_5,1116,1140
b_12_2,1116,1140
b_13_3,1116,1140
b_15_4,1140,1164
b_12_1,1140,1164
b_13_2,1140,1164
b_14_3,1140,1164
b_12_0,1164,1188
b_13_1,1164,1188
b_14_2,1164,1188
b_15_3,1164,1188
b_13_0,1188,1212
b_14_1,1188,1212
b_15_2,1188,1212
b_14_0,1212,1236
b_15_1,1212,1236
b_15_0,1236,1260'''
    unified_scheduler = order_result_mutichunk(input_str,stage_alignment)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)

#     input_str='''0_f_0_0,0.0,14.0
# 0_b_0_0,288.0,300.0
# 0_w_0_0,300.0,312.0
# 0_f_1_0,14.0,28.0
# 0_b_1_0,392.0,404.0
# 0_w_1_0,404.0,416.0
# 0_f_2_0,28.0,42.0
# 0_b_2_0,496.0,508.0
# 0_w_2_0,508.0,520.0
# 0_f_3_0,42.0,56.0
# 0_b_3_0,600.0,612.0
# 0_w_3_0,612.0,624.0
# 0_f_4_0,90.0,104.0
# 0_b_4_0,704.0,716.0
# 0_w_4_0,734.0,746.0
# 0_f_5_0,104.0,118.0
# 0_b_5_0,808.0,820.0
# 0_w_5_0,820.0,832.0
# 0_f_6_0,180.0,194.0
# 0_b_6_0,878.0,890.0
# 0_w_6_0,890.0,902.0
# 0_f_7_0,194.0,208.0
# 0_b_7_0,920.0,932.0
# 0_w_7_0,932.0,944.0
# 0_f_0_1,14.0,26.0
# 0_b_0_1,276.0,288.0
# 0_w_0_1,344.0,356.0
# 0_f_1_1,38.0,50.0
# 0_b_1_1,380.0,392.0
# 0_w_1_1,448.0,460.0
# 0_f_2_1,62.0,74.0
# 0_b_2_1,484.0,496.0
# 0_w_2_1,520.0,532.0
# 0_f_3_1,86.0,98.0
# 0_b_3_1,508.0,520.0
# 0_w_3_1,544.0,556.0
# 0_f_4_1,110.0,122.0
# 0_b_4_1,594.0,606.0
# 0_w_4_1,630.0,642.0
# 0_f_5_1,190.0,202.0
# 0_b_5_1,698.0,710.0
# 0_w_5_1,722.0,734.0
# 0_f_6_1,214.0,226.0
# 0_b_6_1,768.0,780.0
# 0_w_6_1,780.0,792.0
# 0_f_7_1,320.0,332.0
# 0_b_7_1,890.0,902.0
# 0_w_7_1,902.0,914.0
# 0_f_0_2,26.0,38.0
# 0_b_0_2,238.0,250.0
# 0_w_0_2,332.0,344.0
# 0_f_1_2,50.0,62.0
# 0_b_1_2,368.0,380.0
# 0_w_1_2,404.0,416.0
# 0_f_2_2,74.0,86.0
# 0_b_2_2,416.0,428.0
# 0_w_2_2,428.0,440.0
# 0_f_3_2,98.0,110.0
# 0_b_3_2,496.0,508.0
# 0_w_3_2,520.0,532.0
# 0_f_4_2,178.0,190.0
# 0_b_4_2,582.0,594.0
# 0_w_4_2,594.0,606.0
# 0_f_5_2,214.0,226.0
# 0_b_5_2,686.0,698.0
# 0_w_5_2,710.0,722.0
# 0_f_6_2,264.0,276.0
# 0_b_6_2,756.0,768.0
# 0_w_6_2,768.0,780.0
# 0_f_7_2,356.0,368.0
# 0_b_7_2,878.0,890.0
# 0_w_7_2,902.0,914.0
# 0_f_0_3,56.0,90.0
# 0_b_0_3,118.0,146.0
# 0_w_0_3,208.0,226.0
# 0_f_1_3,146.0,180.0
# 0_b_1_3,226.0,254.0
# 0_w_1_3,312.0,330.0
# 0_f_2_3,254.0,288.0
# 0_b_2_3,330.0,358.0
# 0_w_2_3,416.0,434.0
# 0_f_3_3,358.0,392.0
# 0_b_3_3,434.0,462.0
# 0_w_3_3,582.0,600.0
# 0_f_4_3,462.0,496.0
# 0_b_4_3,554.0,582.0
# 0_w_4_3,686.0,704.0
# 0_f_5_3,520.0,554.0
# 0_b_5_3,658.0,686.0
# 0_w_5_3,756.0,774.0
# 0_f_6_3,624.0,658.0
# 0_b_6_3,728.0,756.0
# 0_w_6_3,832.0,850.0
# 0_f_7_3,774.0,808.0
# 0_b_7_3,850.0,878.0
# 0_w_7_3,902.0,920.0
# 1_f_0_0,0.0,14.0
# 1_b_0_0,288.0,300.0
# 1_w_0_0,300.0,312.0
# 1_f_1_0,14.0,28.0
# 1_b_1_0,392.0,404.0
# 1_w_1_0,404.0,416.0
# 1_f_2_0,28.0,42.0
# 1_b_2_0,496.0,508.0
# 1_w_2_0,508.0,520.0
# 1_f_3_0,42.0,56.0
# 1_b_3_0,600.0,612.0
# 1_w_3_0,612.0,624.0
# 1_f_4_0,90.0,104.0
# 1_b_4_0,704.0,716.0
# 1_w_4_0,716.0,728.0
# 1_f_5_0,104.0,118.0
# 1_b_5_0,808.0,820.0
# 1_w_5_0,820.0,832.0
# 1_f_6_0,180.0,194.0
# 1_b_6_0,878.0,890.0
# 1_w_6_0,890.0,902.0
# 1_f_7_0,194.0,208.0
# 1_b_7_0,920.0,932.0
# 1_w_7_0,932.0,944.0
# 1_f_0_1,14.0,26.0
# 1_b_0_1,276.0,288.0
# 1_w_0_1,344.0,356.0
# 1_f_1_1,38.0,50.0
# 1_b_1_1,380.0,392.0
# 1_w_1_1,392.0,404.0
# 1_f_2_1,62.0,74.0
# 1_b_2_1,472.0,484.0
# 1_w_2_1,484.0,496.0
# 1_f_3_1,86.0,98.0
# 1_b_3_1,508.0,520.0
# 1_w_3_1,532.0,544.0
# 1_f_4_1,166.0,178.0
# 1_b_4_1,606.0,618.0
# 1_w_4_1,618.0,630.0
# 1_f_5_1,190.0,202.0
# 1_b_5_1,698.0,710.0
# 1_w_5_1,722.0,734.0
# 1_f_6_1,226.0,238.0
# 1_b_6_1,804.0,816.0
# 1_w_6_1,816.0,828.0
# 1_f_7_1,320.0,332.0
# 1_b_7_1,890.0,902.0
# 1_w_7_1,914.0,926.0
# 1_f_0_2,26.0,38.0
# 1_b_0_2,240.0,252.0
# 1_w_0_2,264.0,276.0
# 1_f_1_2,50.0,62.0
# 1_b_1_2,356.0,368.0
# 1_w_1_2,368.0,380.0
# 1_f_2_2,74.0,86.0
# 1_b_2_2,460.0,472.0
# 1_w_2_2,472.0,484.0
# 1_f_3_2,98.0,110.0
# 1_b_3_2,496.0,508.0
# 1_w_3_2,532.0,544.0
# 1_f_4_2,178.0,190.0
# 1_b_4_2,566.0,578.0
# 1_w_4_2,642.0,654.0
# 1_f_5_2,202.0,214.0
# 1_b_5_2,654.0,666.0
# 1_w_5_2,710.0,722.0
# 1_f_6_2,252.0,264.0
# 1_b_6_2,792.0,804.0
# 1_w_6_2,804.0,816.0
# 1_f_7_2,332.0,344.0
# 1_b_7_2,860.0,872.0
# 1_w_7_2,914.0,926.0
# 1_f_0_3,56.0,90.0
# 1_b_0_3,118.0,146.0
# 1_w_0_3,208.0,226.0
# 1_f_1_3,146.0,180.0
# 1_b_1_3,226.0,254.0
# 1_w_1_3,312.0,330.0
# 1_f_2_3,254.0,288.0
# 1_b_2_3,330.0,358.0
# 1_w_2_3,416.0,434.0
# 1_f_3_3,358.0,392.0
# 1_b_3_3,434.0,462.0
# 1_w_3_3,520.0,538.0
# 1_f_4_3,462.0,496.0
# 1_b_4_3,538.0,566.0
# 1_w_4_3,652.0,670.0
# 1_f_5_3,566.0,600.0
# 1_b_5_3,624.0,652.0
# 1_w_5_3,716.0,734.0
# 1_f_6_3,670.0,704.0
# 1_b_6_3,746.0,774.0
# 1_w_6_3,860.0,878.0
# 1_f_7_3,774.0,808.0
# 1_b_7_3,832.0,860.0
# 1_w_7_3,902.0,920.0
# '''
#     stage_alignment = [[0,3],[1,2],[2,1],[3,0]]#[[0,7],[1,6],[2,5],[3,4]]#[[0,4],[1,5],[2,6],[3,7]]##[[0],[1],[2],[3],[4],[5],[6],[7]]
#     result = order_result_mutistream(input_str,stage_alignment)
#     comm_graph_mutistream(result,stage_alignment)

