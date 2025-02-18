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
        print(step)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    print('[')
    for d in range(len(stage_alignment)):
        device_steps[d].sort(key=lambda x: x[-2])
        print(f'{device_steps[d]},')
    print(']')
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
        print(needrecv)
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
    communication_graph = detect_cycle_deadlock_mutichunk(communication_graph,stage_alignment)
    communication_graph = detect_cross_deadlock_mutichunk(communication_graph,stage_alignment)
    print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(communication_graph):
        print(f'{comm_stage},')
    print(']')
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

def generate_():
    stage_alignment = [[0,4],[1,5],[2,6],[3,7]]
    input_str = '''f_0_0,0.0,12.0
f_1_0,12.0,24.0
f_0_1,12.0,24.0
f_2_0,24.0,36.0
f_1_1,24.0,36.0
f_0_2,24.0,36.0
f_3_0,36.0,48.0
f_2_1,36.0,48.0
f_1_2,36.0,48.0
f_0_3,36.0,48.0
f_0_4,48.0,60.0
f_3_1,48.0,60.0
f_2_2,48.0,60.0
f_1_3,48.0,60.0
f_1_4,60.0,72.0
f_0_5,60.0,72.0
f_3_2,60.0,72.0
f_2_3,60.0,72.0
f_2_4,72.0,84.0
f_1_5,72.0,84.0
f_0_6,72.0,84.0
f_3_3,72.0,84.0
f_3_4,84.0,96.0
f_2_5,84.0,96.0
f_1_6,84.0,96.0
f_0_7,84.0,96.0
f_4_0,96.0,108.0
f_3_5,96.0,108.0
f_2_6,96.0,108.0
b_0_7,96.0,132.0
f_5_0,108.0,120.0
f_4_1,108.0,120.0
f_6_0,120.0,132.0
b_0_6,132.0,168.0
f_1_7,132.0,144.0
b_1_7,144.0,180.0
b_0_5,168.0,204.0
f_3_6,168.0,180.0
b_1_6,180.0,216.0
f_2_7,180.0,192.0
b_2_7,192.0,228.0
b_0_4,204.0,240.0
f_5_1,204.0,216.0
b_1_5,216.0,252.0
f_4_2,216.0,228.0
b_2_6,228.0,264.0
f_3_7,228.0,240.0
f_7_0,240.0,252.0
b_0_3,240.0,276.0
b_1_4,252.0,288.0
f_6_1,252.0,264.0
b_2_5,264.0,300.0
f_5_2,264.0,276.0
b_0_2,276.0,312.0
f_4_3,276.0,288.0
f_4_4,288.0,300.0
b_1_3,288.0,324.0
b_2_4,300.0,336.0
f_4_5,300.0,312.0
b_0_1,312.0,348.0
f_4_6,312.0,324.0
b_1_2,324.0,360.0
f_4_7,324.0,336.0
f_8_0,336.0,348.0
b_2_3,336.0,372.0
b_0_0,348.0,384.0
f_7_1,348.0,360.0
b_1_1,360.0,396.0
f_6_2,360.0,372.0
b_2_2,372.0,408.0
f_5_3,372.0,384.0
f_5_4,384.0,396.0
b_3_7,384.0,420.0
b_1_0,396.0,432.0
f_5_5,396.0,408.0
b_2_1,408.0,444.0
f_5_6,408.0,420.0
b_3_6,420.0,456.0
f_5_7,420.0,432.0
f_9_0,432.0,444.0
b_4_7,432.0,468.0
b_2_0,444.0,480.0
f_8_1,444.0,456.0
b_3_5,456.0,492.0
f_7_2,456.0,468.0
b_4_6,468.0,504.0
f_6_3,468.0,480.0
f_6_4,480.0,492.0
b_5_7,480.0,516.0
b_3_4,492.0,528.0
f_6_5,492.0,504.0
b_4_5,504.0,540.0
f_6_6,504.0,516.0
b_5_6,516.0,552.0
f_6_7,516.0,528.0
f_10_0,528.0,540.0
b_3_3,528.0,564.0
b_4_4,540.0,576.0
f_9_1,540.0,552.0
b_5_5,552.0,588.0
f_8_2,552.0,564.0
b_3_2,564.0,600.0
f_7_3,564.0,576.0
f_7_4,576.0,588.0
b_4_3,576.0,612.0
b_5_4,588.0,624.0
f_7_5,588.0,600.0
b_3_1,600.0,636.0
f_7_6,600.0,612.0
b_4_2,612.0,648.0
f_7_7,612.0,624.0
f_11_0,624.0,636.0
b_5_3,624.0,660.0
b_3_0,636.0,672.0
f_10_1,636.0,648.0
b_4_1,648.0,684.0
f_9_2,648.0,660.0
b_5_2,660.0,696.0
f_8_3,660.0,672.0
f_8_4,672.0,684.0
b_6_7,672.0,708.0
b_4_0,684.0,720.0
f_8_5,684.0,696.0
b_5_1,696.0,732.0
f_8_6,696.0,708.0
b_6_6,708.0,744.0
f_8_7,708.0,720.0
f_12_0,720.0,732.0
b_7_7,720.0,756.0
b_5_0,732.0,768.0
f_11_1,732.0,744.0
b_6_5,744.0,780.0
f_10_2,744.0,756.0
b_7_6,756.0,792.0
f_9_3,756.0,768.0
f_9_4,768.0,780.0
b_8_7,768.0,804.0
b_6_4,780.0,816.0
f_9_5,780.0,792.0
b_7_5,792.0,828.0
f_9_6,792.0,804.0
b_8_6,804.0,840.0
f_9_7,804.0,816.0
f_13_0,816.0,828.0
b_6_3,816.0,852.0
b_7_4,828.0,864.0
f_12_1,828.0,840.0
b_8_5,840.0,876.0
f_11_2,840.0,852.0
b_6_2,852.0,888.0
f_10_3,852.0,864.0
f_10_4,864.0,876.0
b_7_3,864.0,900.0
b_8_4,876.0,912.0
f_10_5,876.0,888.0
b_6_1,888.0,924.0
f_10_6,888.0,900.0
b_7_2,900.0,936.0
f_10_7,900.0,912.0
f_14_0,912.0,924.0
b_8_3,912.0,948.0
b_6_0,924.0,960.0
f_13_1,924.0,936.0
b_7_1,936.0,972.0
f_12_2,936.0,948.0
b_8_2,948.0,984.0
f_11_3,948.0,960.0
f_11_4,960.0,972.0
b_9_7,960.0,996.0
b_7_0,972.0,1008.0
f_11_5,972.0,984.0
b_8_1,984.0,1020.0
f_11_6,984.0,996.0
b_9_6,996.0,1032.0
f_11_7,996.0,1008.0
f_15_0,1008.0,1020.0
b_10_7,1008.0,1044.0
b_8_0,1020.0,1056.0
f_14_1,1020.0,1032.0
b_9_5,1032.0,1068.0
f_13_2,1032.0,1044.0
b_10_6,1044.0,1080.0
f_12_3,1044.0,1056.0
f_12_4,1056.0,1068.0
b_11_7,1056.0,1092.0
b_9_4,1068.0,1104.0
f_12_5,1068.0,1080.0
b_10_5,1080.0,1116.0
f_12_6,1080.0,1092.0
b_11_6,1092.0,1128.0
f_12_7,1092.0,1104.0
b_9_3,1104.0,1140.0
b_10_4,1116.0,1152.0
f_15_1,1116.0,1128.0
b_11_5,1128.0,1164.0
f_14_2,1128.0,1140.0
b_9_2,1140.0,1176.0
f_13_3,1140.0,1152.0
f_13_4,1152.0,1164.0
b_10_3,1152.0,1188.0
b_11_4,1164.0,1200.0
f_13_5,1164.0,1176.0
b_9_1,1176.0,1212.0
f_13_6,1176.0,1188.0
b_10_2,1188.0,1224.0
f_13_7,1188.0,1200.0
b_11_3,1200.0,1236.0
b_9_0,1212.0,1248.0
b_10_1,1224.0,1260.0
f_15_2,1224.0,1236.0
b_11_2,1236.0,1272.0
f_14_3,1236.0,1248.0
f_14_4,1248.0,1260.0
b_12_7,1248.0,1284.0
b_10_0,1260.0,1296.0
f_14_5,1260.0,1272.0
b_11_1,1272.0,1308.0
f_14_6,1272.0,1284.0
b_12_6,1284.0,1320.0
f_14_7,1284.0,1296.0
b_13_7,1296.0,1332.0
b_11_0,1308.0,1344.0
b_12_5,1320.0,1356.0
b_13_6,1332.0,1368.0
f_15_3,1332.0,1344.0
f_15_4,1344.0,1356.0
b_14_7,1344.0,1380.0
b_12_4,1356.0,1392.0
f_15_5,1356.0,1368.0
b_13_5,1368.0,1404.0
f_15_6,1368.0,1380.0
b_14_6,1380.0,1416.0
f_15_7,1380.0,1392.0
b_12_3,1392.0,1428.0
b_13_4,1404.0,1440.0
b_14_5,1416.0,1452.0
b_12_2,1428.0,1464.0
b_15_7,1428.0,1464.0
b_14_4,1452.0,1488.0
b_12_1,1464.0,1500.0
b_15_6,1464.0,1500.0
b_13_3,1464.0,1500.0
b_12_0,1500.0,1536.0
b_15_5,1500.0,1536.0
b_13_2,1500.0,1536.0
b_14_3,1500.0,1536.0
b_15_4,1536.0,1572.0
b_13_1,1536.0,1572.0
b_14_2,1536.0,1572.0
b_13_0,1572.0,1608.0
b_14_1,1572.0,1608.0
b_15_3,1572.0,1608.0
b_14_0,1608.0,1644.0
b_15_2,1608.0,1644.0
b_15_1,1644.0,1680.0
b_15_0,1680.0,1716.0'''
    unified_scheduler = order_result_mutichunk(input_str,stage_alignment)
    comm_graph = comm_graph_muti_chunk(unified_scheduler,stage_alignment)
    return stage_alignment,unified_scheduler,comm_graph


