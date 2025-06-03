import os
import glob
import joblib
from optuna.importance import get_param_importances
import csv
import random
import numpy as np
import sys

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 目录路径
# dir_path = "./Result/optuna/"
dir_path = "./total_optuna/"

model = sys.argv[1]

# 检索文件，要求文件名包含'MLP'且以.pkl结尾
pattern = os.path.join(dir_path, "*"+model+"*.pkl")
pkl_files = glob.glob(pattern)

# 用于存储所有结果
results = []

# 遍历找到的每个.pkl文件
for file_path in pkl_files:
    try:
        study = joblib.load(file_path)
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        continue
    random.seed(42)
    np.random.seed(42)
    # 计算参数重要性
    param_importance = get_param_importances(study)

    # 输出每个参数的重要性，并将结果保存到列表中
    for param_name, importance_value in param_importance.items():
        print(f"文件 {os.path.basename(file_path)}: 参数 {param_name} 的重要性为 {importance_value}")
        results.append({
            "file_name": os.path.basename(file_path),
            "parameter": param_name,
            "importance": importance_value
        })

# 将结果写入 CSV 文件
output_csv = "14-" + model + "_HP_importance.csv"
try:
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["file_name", "parameter", "importance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已成功保存到 {output_csv}")
except Exception as e:
    print(f"保存 CSV 文件时出错: {e}")
