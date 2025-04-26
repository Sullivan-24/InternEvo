ZBV = ["OOM"]
ZBV_RECOMP = [51.357444763183594, 51.4356689453125, 28.747970581054688, 28.810470581054688]
UPP_DYNA_RECOMP = [51.357444763183594, 51.4356689453125, 50.74797058105469, 61.81047058105469]
UPP_DYNA_RECOMP_LAYERWISE = [74.8760986328125, 73.4980697631836, 73.4980697631836, 73.06047058105469]
import matplotlib.pyplot as plt

# 数据
ZBV = ["OOM"]  # 这里没有实际数据，仅作为占位符
ZBV_RECOMP = [51.357444763183594, 51.4356689453125, 28.747970581054688, 28.810470581054688]
UPP_DYNA_RECOMP = [51.357444763183594, 51.4356689453125, 50.74797058105469, 61.81047058105469]
UPP_DYNA_RECOMP_LAYERWISE = [74.8760986328125, 73.4980697631836, 73.4980697631836, 73.06047058105469]

# X轴标签（假设是层数或步骤）
x_labels = [f"Device {i+1}" for i in range(len(ZBV_RECOMP))]
colors = {
    "UPP":'#D62728',
    "1F1B":'#1f77b4',
    "I1F1B":'#ff7f0e',
    "ZBV":'#2ca02c',
    "ZBH":'#e377c2',
}
# 创建图形
plt.figure(figsize=(5, 3))
# 绘制 GPU_MAX_MEM 红线
# GPU_MAX_MEM = 80
# plt.axhline(y=GPU_MAX_MEM, color='r', linestyle='--', label="GPU Max Mem.")
# 绘制折线图
plt.plot(x_labels, UPP_DYNA_RECOMP_LAYERWISE, marker='^', c=colors['UPP'],  label="UPP+Dyna. Recomp.+Layerwise")
plt.plot(x_labels, UPP_DYNA_RECOMP, marker='s', linestyle='--', c=colors['I1F1B'], label="UPP+Dyna. Recomp.")
plt.plot(x_labels, ZBV_RECOMP, marker='o', c=colors['ZBV'],  label="ZBV+Full Recomp.")



# 添加标题和标签
import numpy as np
plt.title("Memory Usage in Different Parameter Situations")
plt.ylabel("Memory Usage (GB)")
plt.yticks(np.arange(0, 81, 10))  # 从 0 到 100，每隔 10
# 添加图例
plt.legend(fontsize=9)

# 显示图形
plt.grid(True)
plt.show()