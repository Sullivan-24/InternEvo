# import numpy as np
# file_name = "/mnt/shared-storage-user/lusitian/data/data_jsonl/github/tokenized_llama2/train_folder/data/output.bin.meta"

# with open(file_name, "rb") as f:
#     meta = np.load(f)
#     lengths = meta[:,1]
#     lengths_total = sum(lengths)
#     print("Total tokens:", lengths_total)
    
# class test:
    
#     def __init__(self):
#         self.a = 1
        
#     def print_a(self):
#         return self.a
        
        
# t = test()
# t.a += 1
# print(t.a, t.print_a())
import pandas as pd

file_path = './attn_record/github/16*1024/Mround_robin_B1024_mb32/dp1_tp2_pp4_profile_True_20260120_080645attn_stats.csv'

# 1. 读取 CSV 文件
df = pd.read_csv(file_path)

# 2. 将第一列 'attn_flops' 的所有值除以 8
df['attn_flops'] = df['attn_flops'] / 8

# 3. 将修改后的数据保存回原文件 (如果不希望覆盖原文件，可以修改下面的文件名为新名字)
df.to_csv(file_path, index=False)

print(f"成功处理文件: {file_path}")
print(df.head()) # 打印前几行确认