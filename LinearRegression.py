import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 准备数据
file_path = './attn_record/github/16*1024/Mround_robin_B512_mb8/dp2_tp2_pp2_profile_True_20260121_080103attn_stats.csv'

def plot_single_regression(file_path):
    df = pd.read_csv(file_path)
    
    # scikit-learn 需要 X 是一个二维数组 (DataFrame 或 list of lists)
    X = df[['attn_flops']] 
    y = df['attn_time']

    # 2. 创建并训练模型
    model = LinearRegression()
    model.fit(X, y)

    # 3. 进行预测和评估
    y_pred = model.predict(X)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    # 4. 打印结果
    print("-" * 30)
    print("Scikit-Learn 线性回归结果")
    print("-" * 30)
    print(f"回归方程: y = {slope:.6f}x + {intercept:.6f}")
    print(f"斜率 (Slope/Coefficient): {slope:.6f}")
    print(f"截距 (Intercept): {intercept:.6f}")
    print(f"R² (决定系数): {r2:.6f}")
    print("-" * 30)

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='#1f77b4', alpha=0.6, label='Points') # 经典蓝色
    plt.plot(X, y_pred, color='#d62728', linewidth=2, label=f'Function: y={slope:.4f}x+{intercept:.4f}') # 经典红色
    
    plt.title(f'Linear Regression Analysis \n$R^2 = {r2:.4f}$', fontsize=14)
    plt.xlabel('Attention FLOPS', fontsize=12)
    plt.ylabel('Attention Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 保存图片
    output_img = 'sklearn_regression_result.png'
    plt.savefig(output_img, dpi=300)
    print(f"图表已保存为: {output_img}")

def plot_triple_regression(files_list):
    plt.figure(figsize=(12, 8))
    output_filename = 'linear_regression_comparison_13B_4K_8K_MB8.png'
    
    # 颜色列表
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    x_max = 0
    
    for idx, info in enumerate(files_list):
        file_path = info["path"]
        label = info["label"]
        color = colors[idx % len(colors)]
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        
        df = pd.read_csv(file_path)
        X = df[['attn_flops']]
        x_max = max(X.max().iloc[0], x_max) 
        y = df['attn_time']
        
        # 训练模型
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # 获取参数
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        
        # 打印信息
        print(f"--- {label} ---")
        print(f"Equation: y = {slope:.6f}x + {intercept:.6f}")
        print(f"R²: {r2:.6f}")
        
        # 绘图 - 散点
        plt.scatter(X, y, color=color, alpha=0.4, s=30)
        
        # 绘图 - 拟合线
        # 为了画线好看，我们生成从 min(X) 到 max(X) 的点
        print(f"{x_max=}")
        X_range = pd.DataFrame({'attn_flops': [X.min().iloc[0], x_max]})
        y_range = model.predict(X_range)
        
        plt.plot(X_range, y_range, color=color, linewidth=2, 
                    label=f'{label}: y={slope:.4f}x+{intercept:.4f} ($R^2={r2:.3f}$)')
        

    plt.title('Linear Regression Analysis Comparison (LLama-7B)', fontsize=16)
    plt.xlabel('Attention FLOPS', fontsize=14)
    plt.ylabel('Attention Time (seconds)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(output_filename, dpi=300)
    print(f"\nGraph saved to: {output_filename}")

if __name__ == "__main__":
    
    files_list = [ 
    {
        "path": "./attn_record/github/13B_llama2/8*1024/Mround_robin_B512_mb8/dp2_tp2_pp4_profile_True_20260122_093211attn_stats.csv",
        "label": "8K Sequence Length"            
    },
    {
        "path": "./attn_record/github/13B_llama2/4*1024/Mround_robin_B512_mb8/dp2_tp2_pp4_profile_True_20260122_093527attn_stats.csv",
        "label": "4K Sequence Length"
    },
    ]
    
    # plot_single_regression(file_path)
    plot_triple_regression(files_list)
    
