import math
import itertools
import copy
from collections import deque


def generate_binary_tuples(n):
    for i in range(2 ** n):
        # 将整数i转换为二进制字符串，补齐前导零，再拆分为元组
        binary_str = bin(i)[2:].zfill(n)
        yield tuple(int(bit) for bit in binary_str)
        
def generate_unique_placement_mapping(n):
    if n < 0:
        return None  # 根据实际情况处理非法输入
    if n == 0:
        return None  # 题目要求空数组返回None
    
    seen = set()
    result = []
    
    # 生成所有可能的二进制数组
    for arr in itertools.product([0, 1], repeat=n):
        arr_tuple = arr
        if arr_tuple not in seen:
            # 添加到结果列表
            result.append(list(arr_tuple))
            # 计算补集并标记为已处理
            complement = tuple(1 - x for x in arr_tuple)
            seen.add(arr_tuple)
            seen.add(complement)
    
    return result

class PipelinePlacement:
    def __init__(self, 
                layer_num:int, 
                layer_computation_cost:list[float], 
                layer_para: list[float],
                dev_num:int, 
                dev_max_memory:list[float], 
                dev_compute_power:list[float]):
        self.layer_num = layer_num
        self.layer_computation_cost = layer_computation_cost
        self.layer_para = layer_para
        self.dev_num = dev_num
        self.dev_max_memory = dev_max_memory
        self.dev_compute_power = dev_compute_power

    def add_layer_to_device(self, placement, lid, step, order):

        for i in range(lid, min(lid + step, self.layer_num)):
            if order:
                placement[i - lid].append(i)
            else:
                placement[self.dev_num - 1 - i + lid].append(i)

    def get_reduced_placements(self):
        max_chunk_num = (self.layer_num + self.dev_num - 1) // self.dev_num
        gen = generate_binary_tuples(max_chunk_num - 1) # - 1 to reduce reverse but equal placements (example, (0,1) = (1.0))
        placements = []
        for g in gen:
            placement = [[] for _ in range(self.dev_num)]
            lid = 0

            self.add_layer_to_device(placement, lid, self.dev_num, 1)
            lid += self.dev_num
            for o in g:
                self.add_layer_to_device(placement, lid, self.dev_num, o)
                lid += self.dev_num
            
            placements.append(copy.deepcopy(placement))
        return placements

    def get_reduced_possilble_placements(self):
        base, rem = divmod(self.layer_num, self.dev_num)
        s = [base + 1] * rem + [base] * (self.dev_num - rem)
        
        # 初始化状态：当前分配，剩余可用层号
        from collections import deque
        stack = deque([(tuple([] for _ in range(self.dev_num)), set(range(self.layer_num)))])
        seen = set()

        while stack:
            current, available = stack.pop()
            
            # 终止条件：所有层分配完成
            if not available:
                if all(len(dev) == s[i] for i, dev in enumerate(current)):
                    yield [list(dev) for dev in current]
                continue
            
            # 修复后的列索引计算
            active_columns = []
            for i, dev in enumerate(current):
                if len(dev) < s[i]:
                    active_columns.append(len(dev))
            if not active_columns:
                continue
            col = min(active_columns)
            
            # 获取需要分配当前列的设备
            active_devices = [i for i, dev in enumerate(current) if len(dev) == col and len(dev) < s[i]]
            k = len(active_devices)
            
            # 生成候选连续数字块
            sorted_avail = sorted(available)
            candidates = []
            for start in range(len(sorted_avail) - k + 1):
                block = sorted_avail[start:start+k]
                if all(block[i+1] - block[i] == 1 for i in range(k-1)):
                    candidates.append(block)
                    candidates.append(block[::-1])  # 添加反向序列
            
            # 验证并生成新状态
            for block in candidates:
                # 检查设备内部递增性
                valid = True
                new_assignment = [list(dev) for dev in current]
                new_available = set(available)
                for i, dev_idx in enumerate(active_devices):
                    if new_assignment[dev_idx] and block[i] <= new_assignment[dev_idx][-1]:
                        valid = False
                        break
                    new_assignment[dev_idx].append(block[i])
                    new_available.remove(block[i])
                if not valid:
                    continue
                
                # 转换为不可变类型用于哈希
                frozen = tuple(tuple(dev) for dev in new_assignment)
                if frozen not in seen:
                    seen.add(frozen)
                    stack.append( (frozen, new_available) )
    
    def get_possible_placements(self):
        """生成器：每次调用返回一个新的合法placement"""
        base = self.layer_num // self.dev_num
        remainder = self.layer_num % self.dev_num
        s = [base + 1 if i < remainder else base for i in range(self.dev_num)]
        
        def is_monotonic(sequence, direction):
            """验证序列是否符合严格单调性"""
            if direction == 'asc':
                return all(x < y for x, y in zip(sequence, sequence[1:]))
            return all(x > y for x, y in zip(sequence, sequence[1:]))

        def validate(placement):
            """验证所有设备的对应位置满足严格单调性"""
            for pos in range(max(len(dev) for dev in placement)):
                layers = []
                for dev in placement:
                    if pos < len(dev):
                        layers.append(dev[pos])
                if len(layers) < 2:
                    continue
                direction = 'asc' if layers[0] < layers[1] else 'desc'
                if not all(layers[i] < layers[i+1] for i in range(len(layers)-1)) if direction == 'asc' \
                   else not all(layers[i] > layers[i+1] for i in range(len(layers)-1)):
                    return False
            return True

        def backtrack(position, remaining, current_placement):
            if not remaining:
                if all(len(current_placement[d]) == s[d] for d in range(self.dev_num)) and validate(current_placement):
                    yield [tuple(dev) for dev in current_placement]  # 返回不可变对象确保唯一性
                return

            active_devices = [d for d in range(self.dev_num) if s[d] > position]
            if not active_devices:
                return

            # 生成所有可能的组合分配
            for combo in itertools.permutations(remaining, len(active_devices)):
                # 尝试两种排序方向
                for direction in ['asc', 'desc']:
                    sorted_combo = sorted(combo) if direction == 'asc' else sorted(combo, reverse=True)
                    new_placement = [list(dev) for dev in current_placement]
                    for dev, layer in zip(active_devices, sorted_combo):
                        new_placement[dev].append(layer)
                    
                    # 立即验证当前层单调性
                    current_layers = [new_placement[dev][position] for dev in active_devices]
                    if len(current_layers) >= 2:
                        if direction == 'asc' and not is_monotonic(current_layers, 'asc'):
                            continue
                        if direction == 'desc' and not is_monotonic(current_layers, 'desc'):
                            continue
                    
                    # 剪枝：剩余层必须足够后续分配
                    remaining_layers = [x for x in remaining if x not in combo]
                    required_layers = sum(s[d] - (position + 1) for d in range(self.dev_num))
                    if len(remaining_layers) < required_layers:
                        continue
                    
                    yield from backtrack(position + 1, remaining_layers, new_placement)

        # 使用集合去重
        seen = set()
        for placement in backtrack(0, list(range(self.layer_num)), [[] for _ in range(self.dev_num)]):
            key = tuple(tuple(dev) for dev in placement)
            if key not in seen:
                seen.add(key)
                yield [list(dev) for dev in placement]
    
    def get_placements(self):
        # Sort layers by computation cost in descending order, keeping their original indices
        sorted_layers = sorted(enumerate(self.layer_computation_cost), key=lambda x: -x[1])
        
        # Initialize devices: memory_used, total_time, and layers list
        devices = [
            {
                'memory_used': 0.0,
                'total_time': 0.0,
                'layers': []
            }
            for _ in range(self.dev_num)
        ]
        
        for layer_idx, comp_cost in sorted_layers:
            layer_mem = self.layer_para[layer_idx]
            best_dev = None
            min_max_time = float('inf')
            min_new_time = float('inf')
            
            # Find the best device to place the current layer
            for did in range(self.dev_num):
                # Check if the device has enough memory
                if devices[did]['memory_used'] + layer_mem > self.dev_max_memory[did]:
                    continue
                
                # Calculate new total time for this device if the layer is placed here
                new_time = devices[did]['total_time'] + (comp_cost / self.dev_compute_power[did])
                
                # Calculate the candidate max time across all devices after placement
                current_max_time = 0.0
                for other_dev_idx in range(self.dev_num):
                    if other_dev_idx == did:
                        current_time = new_time
                    else:
                        current_time = devices[other_dev_idx]['total_time']
                    if current_time > current_max_time:
                        current_max_time = current_time
                
                # Update the best device based on the minimal max_time and new_time
                if (current_max_time < min_max_time) or \
                   (current_max_time == min_max_time and new_time < min_new_time):
                    best_dev = did
                    min_max_time = current_max_time
                    min_new_time = new_time
            
            if best_dev is None:
                raise RuntimeError(f"Cannot place layer {layer_idx} due to insufficient memory on all devices.")
            
            # Update the best device's status
            devices[best_dev]['memory_used'] += layer_mem
            devices[best_dev]['total_time'] = min_new_time
            devices[best_dev]['layers'].append(layer_idx)
        
        # Prepare the result, sorting the layer indices for each device
        dsa = []
        for dev in devices:
            dev['layers'].sort()
            dev['layer_num'] = len(dev['layers'])
            dsa.append(dev['layers'])
        for device in devices:
            print(device)
        for dsa_in_pp in dsa:
            print(dsa_in_pp, ",")
        return dsa
    
    def get_placements_dp(self):
        # 按计算成本降序排序，保留原始索引
        sorted_layers = sorted(enumerate(self.layer_computation_cost), 
                             key=lambda x: (-x[1], x[0]))
        
        # 初始化动态规划表
        dp = [[] for _ in range(self.layer_num + 1)]
        initial_memory = tuple([0.0] * self.dev_num)
        initial_time = tuple([0.0] * self.dev_num)
        initial_layers = tuple([tuple() for _ in range(self.dev_num)])
        dp[0].append((initial_memory, initial_time, initial_layers))
        
        # 处理每个层
        for step in range(self.layer_num):
            layer_idx, comp_cost = sorted_layers[step]
            layer_mem = self.layer_para[layer_idx]
            
            next_states = []
            for state in dp[step]:
                current_memory, current_time, current_layers = state
                
                # 尝试将当前层分配到每个设备
                for dev in range(self.dev_num):
                    # 检查显存限制
                    if current_memory[dev] + layer_mem > self.dev_max_memory[dev]:
                        continue
                    
                    # 更新显存和计算时间
                    new_memory = list(current_memory)
                    new_memory[dev] += layer_mem
                    new_memory = tuple(new_memory)
                    
                    new_time = list(current_time)
                    new_time[dev] += comp_cost / self.dev_compute_power[dev]
                    new_time = tuple(new_time)
                    
                    # 更新层分配
                    new_layers = list(current_layers)
                    new_layers[dev] = tuple(list(new_layers[dev]) + [layer_idx])
                    new_layers = tuple(new_layers)
                    
                    next_states.append((new_memory, new_time, new_layers))
            
            # 剪枝：去除被支配的状态
            pruned_states = []
            for state in next_states:
                dominated = False
                for other in next_states:
                    if state == other:
                        continue
                    if self.is_dominated(state, other):
                        dominated = True
                        break
                if not dominated:
                    pruned_states.append(state)
            dp[step+1] = pruned_states
        
        # 寻找最优解
        if not dp[self.layer_num]:
            raise RuntimeError("No valid placement found")
        
        min_variance = float('inf')
        best_layers = None
        for state in dp[self.layer_num]:
            times = state[1]
            mean = sum(times) / self.dev_num
            variance = sum((t - mean)**2 for t in times) / self.dev_num
            if variance < min_variance:
                min_variance = variance
                best_layers = state[2]
        
        # 转换为要求的格式并排序
        dsa = []
        for layers in best_layers:
            sorted_layers = sorted(layers)
            dsa.append(sorted_layers)
        for dsa_in_pp in dsa:
            print(dsa_in_pp)
        return dsa
    
    def is_dominated(self, state_a, state_b):
        """检查state_a是否被state_b支配"""
        mem_a, time_a, _ = state_a
        mem_b, time_b, _ = state_b
        
        all_less_equal = True
        has_strict_less = False
        
        for m_a, m_b in zip(mem_a, mem_b):
            if m_b > m_a:
                all_less_equal = False
                break
        
        for t_a, t_b in zip(time_a, time_b):
            if t_b > t_a:
                all_less_equal = False
                break
        
        if not all_less_equal:
            return False
        
        for m_a, m_b in zip(mem_a, mem_b):
            if m_b < m_a:
                has_strict_less = True
                break
        
        for t_a, t_b in zip(time_a, time_b):
            if t_b < t_a:
                has_strict_less = True
                break
        
        return has_strict_less

# 示例测试
if __name__ == "__main__":
    # 输入参数
    # L = 64
    # D = 8
    # Tl = [1 for _ in range(L)]
    # Cd = [1 for _ in range(D)]  # 机器0算力2, 机器1算力1
    
    # # 运行分配算法
    # allocation, chunk_max, total_max = allocate_layers(L, D, Tl, Cd)
    
    # # 打印结果
    # print("层到机器的分配结果:")
    # for layer in sorted(allocation):
    #     print(f"层 {layer} → 机器 {allocation[layer]}")
    
    # print("\n每个chunk的最大时间:", chunk_max)
    # print("流水线整体最大时间:", total_max)
    pp_size = 8
    layer_num = 80
    layer_computation_cost = [1 for _ in range(layer_num)]
    layer_computation_cost[-1] = 2
    pp_compute_power = [1 for _ in range(pp_size)]
    pp_compute_power[0] = 2
    pp_compute_power[1] = 2
    pp_compute_power[2] = 2
    pp_compute_power[3] = 2
    
    test_placement = PipelinePlacement(layer_num=layer_num, 
                                    layer_computation_cost=layer_computation_cost,
                                    layer_para=[1 for _ in range(layer_num)],
                                    dev_num=pp_size,
                                    dev_max_memory=[100000 for _ in range(layer_num)],
                                    dev_compute_power=pp_compute_power,
                                    )
    
    # pg = test_placement.get_reduced_possilble_placements()
    # for p in pg:
    #     print(p)
    # ps = test_placement.get_reduced_placements()
    ps = test_placement.get_placements()
    # test_placement.get_placements()
    # pp4 layer8 也不行
    # test_placement.get_placements_dp()
