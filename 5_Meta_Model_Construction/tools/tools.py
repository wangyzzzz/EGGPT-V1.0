# 导入 NumPy 库，用于数学运算
import numpy as np

# 定义计算 MSE 的函数
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 定义计算 R² 的函数
def r2_score(y_true, y_pred):
    sstot = np.sum((y_true - np.mean(y_true)) ** 2)
    ssres = np.sum((y_true - y_pred) ** 2)
    return 1 - (ssres / sstot)
