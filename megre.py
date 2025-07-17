import os
import json
from preprocess import find_mismatch
def merge_json_folder(folder_path, output_file):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith('iter')]  # 获取文件夹中的所有 JSON 文件路径
    pp_size=len(json_files)  # 计算 JSON 文件的数量
    # merged_data = []  # 创建一个空列表，用于存储合并后的 JSON 数据
    comm_matrix = [[[] for __ in range(pp_size)] for _ in range(pp_size)]  # 初始化通信矩阵
    for file in json_files:
        file_path = os.path.join(folder_path, file)  # 构建完整的文件路径
        with open(file_path, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                line = line.strip()
                if line:  # 跳过空行
                    obj = json.loads(line)
                    operation = obj['operation']
                    local_rank = obj['local_rank']
                    step_id = obj['step_id']
                    match_rank = obj.get('match_rank', None)
                    if operation in ('SA','RA','SG','RG'):
                        comm_matrix[local_rank][match_rank].append((operation,step_id))  # 将通信信息存储到通信矩阵中
    
    # # 将列表中的 JSON 数据写入目标文件
    # with open(output_file, 'w') as output:
    #     output.write('\n'.join(merged_data))
    print(find_mismatch(comm_matrix))
    # print("JSON 文件合并完成！")

if __name__ == "__main__":
    folder_path = "/cpfs01/user/matenghui/InternEvo/jsonResult/het/Het_pp4_layers32_mb16"  # 存放 JSON 文件的文件夹路径
    output_file = folder_path + "/merged.json"  
    merge_json_folder(folder_path, output_file)