pp4mb16_1f1b = [29.810272216796875, 23.747772216796875, 17.685272216796875, 14.446990966796875]
pp4mb16_Interleaved1f1b = [38.904022216796875, 32.841522216796875, 26.779022216796875, 23.540740966796875]
pp4mb16_zbv = [33.00959014892578, 30.18537139892578, 30.18537139892578, 32.811065673828125]
pp4mb16_upp = [38.65802764892578, 42.31037139892578, 45.34162139892578, 45.34162139892578]

import matplotlib.pyplot as plt
import numpy as np

# 数据
pp4mb16_1f1b = [29.810272216796875, 23.747772216796875, 17.685272216796875, 14.446990966796875]
pp4mb16_Interleaved1f1b = [38.904022216796875, 32.841522216796875, 26.779022216796875, 23.540740966796875]
pp4mb16_zbv = [33.00959014892578, 30.18537139892578, 30.18537139892578, 32.811065673828125]
pp4mb16_upp = [38.65802764892578, 42.31037139892578, 45.34162139892578, 45.34162139892578]
pp4mb16_upp_layerwise = [79.49858093261719, 64.62396240234375, 49.123069763183594, 50.998565673828125]  # 新增数据

# 设备数量
x_labels = [f"Device {i+1}" for i in range(len(pp4mb16_Interleaved1f1b))]

colors = {
    "UPP":'#D62728',
    "1F1B":'#1f77b4',
    "I1F1B":'#ff7f0e',
    "ZBV":'#2ca02c',
    "ZBH":'#e377c2',
}
# 创建画布
plt.figure(figsize=(6, 3))

# 绘制折线图
plt.plot(x_labels, pp4mb16_1f1b, marker='o', c=colors['1F1B'], label='1F1B')
plt.plot(x_labels, pp4mb16_Interleaved1f1b, marker='s', c=colors['I1F1B'], label='Interleaved1F1B')
plt.plot(x_labels, pp4mb16_zbv, marker='^', c=colors['ZBV'], label='ZBV')
plt.plot(x_labels, pp4mb16_upp, marker='d', linestyle='--',c=colors['UPP'], label='UPP')
plt.plot(x_labels, pp4mb16_upp_layerwise, marker='*', c=colors['UPP'], label='UPP Layerwise')  # 新增数据绘图

# 添加数据标签
# for i, (a, b, c, d, e) in enumerate(zip(pp4mb16_1f1b, pp4mb16_Interleaved1f1b, pp4mb16_zbv, pp4mb16_upp, pp4mb16_upp_layerwise)):
#     plt.text(i + 1, a, f'{a:.2f}', ha='center', va='bottom')
#     plt.text(i + 1, b, f'{b:.2f}', ha='center', va='bottom')
#     plt.text(i + 1, c, f'{c:.2f}', ha='center', va='bottom')
#     plt.text(i + 1, d, f'{d:.2f}', ha='center', va='bottom')
#     plt.text(i + 1, e, f'{e:.2f}', ha='center', va='bottom')  # 新增数据标签

# 设置标题和标签
plt.title('Memory Usage Comparison of Different Schedule Algorithms')
plt.ylabel("Memory Usage (GB)")

# 添加图例
plt.legend(fontsize=7.5)
plt.yticks(np.arange(0, 81, 10))  # 从 0 到 100，每隔 10
# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()