import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义文件路径和对应的标签
# 注意：这里使用了你提供的绝对路径
files_info = [
    # {
    #     "path": "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/attn_record/github/4*1024/Mround_robin_B1024_mb32/dp1_tp2_pp4_profile_True_20260120_123320attn_stats.csv",
    #     "label": "4*1024 Sequence Length (tp2_pp4)"
    # },
    # {
    #     "path": "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/attn_record/github/8*1024/Mround_robin_B1024_mb32/dp1_tp2_pp4_profile_True_20260120_122748attn_stats.csv",
    #     "label": "8*1024 Sequence Length (tp2_pp4)"
    # },
    # {
    #     "path": "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/attn_record/github/16*1024/Mround_robin_B1024_mb32/dp1_tp2_pp4_profile_True_20260120_122606attn_stats.csv",
    #     "label": "16*1024 Sequence Length (tp2_pp4)"
    # },
    # {
    #     "path": "./attn_record/github/4*1024/Mround_robin_B1024_mb32/dp1_tp4_pp2_profile_True_20260120_124129attn_stats.csv",
    #     "label": "4*1024 Sequence Length (tp4_pp2)"
    # },
    # {
    #     "path": "./attn_record/github/8*1024/Mround_robin_B1024_mb32/dp1_tp4_pp2_profile_True_20260120_124626attn_stats.csv",
    #     "label": "8*1024 Sequence Length (tp4_pp2)"
    # },
    # {
    #     "path": "./attn_record/github/16*1024/Mround_robin_B1024_mb32/dp1_tp4_pp2_profile_True_20260120_124154attn_stats.csv",
    #     "label": "16*1024 Sequence Length (tp4_pp2)"
    # },
    # {
    #     "path": "./attn_record/github/4*1024/Mround_robin_B1024_mb8/dp1_tp4_pp2_profile_True_20260120_130912attn_stats.csv",
    #     "label": "4*1024 Sequence Length (tp4_pp2 mb8)"
    # },
    # {
    #     "path": "./attn_record/github/8*1024/Mround_robin_B1024_mb8/dp1_tp4_pp2_profile_True_20260120_130403attn_stats.csv",
    #     "label": "8*1024 Sequence Length (tp4_pp2 mb8)"  
    # },
    # {
    #     "path": "./attn_record/github/16*1024/Mround_robin_B1024_mb8/dp1_tp4_pp2_profile_True_20260120_130440attn_stats.csv",
    #     "label": "16*1024 Sequence Length (tp4_pp2 mb8)"
    # },
    # {
    #     "path": "./attn_record/github/4*1024/Mround_robin_B1024_mb16/dp1_tp4_pp2_profile_True_20260120_134525attn_stats.csv",
    #     "label": "4*1024 Sequence Length (tp4_pp2 mb16)"
    # },
    {
        "path": "./attn_record/github/30B_llama2/8*1024/Mround_robin_B512_mb8/dp2_tp2_pp8_profile_True_20260122_151101attn_stats.csv",
        "label": "8*1024 Sequence Length (30B_llama mb8)"
    },
    # {
    #     "path": "./attn_record/github/16*1024/Mround_robin_B1024_mb16/dp1_tp4_pp2_profile_True_20260120_140253attn_stats.csv",
    #     "label": "16*1024 Sequence Length (tp4_pp2 mb16)"
    # },
]

# 创建图表
plt.figure(figsize=(12, 8))

# 遍历文件并绘图
colors = [ 'red']
#colors = ['blue', 'green', 'red', 'yellow','pink', 'purple', 'grey', 'orange', 'black', 'brown', 'cyan', 'magenta']  # 为每组数据指定颜色, , 'yellow','pink', 'purple', 'grey', 'orange', 'black'
for idx, info in enumerate(files_info):
    file_path = info["path"]
    label = info["label"]
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # 绘制散点
            # color = np.random.rand(3,)
            plt.scatter(df['attn_flops'], df['attn_time'], label=label, color=colors[idx], alpha=0.6, edgecolors='w', s=60)
            print(f"Loaded {len(df)} points from {label}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# 设置图表属性
plt.title('Attention FLOPS vs Execution Time Comparison', fontsize=16)
plt.xlabel('Attention FLOPS', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.legend(title="Dataset", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 保存图片
output_filename = 'attn_flops_8k_30B_mb8.png'
plt.savefig(output_filename, dpi=300)
print(f"\n图表已保存为: {output_filename}")