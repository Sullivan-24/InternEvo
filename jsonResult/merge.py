import os
import json
 
def merge_json_folder(folder_path, output_file):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith('iter')]  # 获取文件夹中的所有 JSON 文件路径
 
    merged_data = []  # 创建一个空列表，用于存储合并后的 JSON 数据
 
    for file in json_files:
        file_path = os.path.join(folder_path, file)  # 构建完整的文件路径
        with open(file_path, 'r') as json_file:
            data = json_file.read()  # 读取 JSON 文件内容
            merged_data.append(data)  # 将 JSON 数据添加到列表中
 
    # 将列表中的 JSON 数据写入目标文件
    with open(output_file, 'w') as output:
        output.write('\n'.join(merged_data))
 
    print("JSON 文件合并完成！")
 
# 合并后的 JSON 文件路径

if __name__ == "__main__":

    folder_path = "/mnt/petrelfs/matenghui/InternEvo/jsonResult/async/Interleaved_pp4_chunk20_mb16"  # 存放 JSON 文件的文件夹路径
    output_file = folder_path + "/merged.json"  
    merge_json_folder(folder_path, output_file)