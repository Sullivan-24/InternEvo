import re
import pandas as pd

# 日志文件路径
log_file_path = '/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/attn_record'
# 根据log文件名去掉后缀来确定输出文件名
log_file_path = './attn_record/github/30B_llama2/4*1024/Mround_robin_B512_mb8/dp2_tp2_pp8_profile_True_20260122_160327.log'
output_file_path = log_file_path.rsplit('.', 1)[0] + 'attn_stats.csv'

data = []

try:
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 使用正则表达式查找 attn_flops 和 attn_time
            # 假设格式如：... attn_flops=16.868 attn_time=0.244 ...
            match_flops = re.search(r'attn_flops=([\d\.]+)', line)
            match_time = re.search(r'attn_time=([\d\.]+)', line)
            
            if match_flops and match_time:
                data.append({
                    'attn_flops': float(match_flops.group(1)),
                    'attn_time': float(match_time.group(1))
                })

        if len(data) > 0:
            data = data[1:]

    # 使用 pandas 创建 DataFrame 并打印（这样看起来就是一张表）
    if data:
        df = pd.DataFrame(data)
        print(df)
        
        # 如果你想保存为 Excel 或 CSV，取消下面的注释:
        df.to_csv(output_file_path, index=False)
        print("已保存到 attn_stats.csv")
    else:
        print("未在日志中找到 attn_flops 和 attn_time 数据。")

except FileNotFoundError:
    print(f"无法找到文件: {log_file_path}")