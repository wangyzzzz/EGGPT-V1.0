#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
import tools.tools as tools1
import argparse
from pathlib import Path
import json
import time
import os
from torch.utils.data import DataLoader
from tools.EarlyStopping import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import ttest_rel

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--root_path', type=str, default = 'None')
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--aim_index', type=int, default = 0)
parser.add_argument('--k_fold', type=str, default = '10')
parser.add_argument('--merge_component_arr', nargs='+', type=str, default = ['1D_MLP'])

args = parser.parse_args()
trait = args.trait_name
lr = float(args.lr)
root_path = args.root_path
aim_index = args.aim_index
k_fold = int(args.k_fold)
merge_component_arr = args.merge_component_arr
merge_component_arr = ['1D_MLP', 'PCA_MLP', 'GRM_MLP', '2D_MLP', '3D_MLP',
                '1D_SVR', 'PCA_SVR', 'GRM_SVR', '2D_SVR', '3D_SVR',
                '1D_RF', 'PCA_RF', 'GRM_RF', '2D_RF', '3D_RF',
                '1D_LGB', 'PCA_LGB', 'GRM_LGB', '2D_LGB', '3D_LGB',
                '1D_CNN', 'PCA_CNN', 'GRM_CNN', '2D_CNN', '3D_CNN',
                '1D_TE', '2D_TE', '3D_TE',
                '1D_LSTM', '2D_LSTM', '3D_LSTM'
             ]
os.makedirs(root_path + 'Result/MR_result/', exist_ok=True)

def set_seed(seed):
    """固定所有的随机种子以确保实验可重复性"""
    # Python 内置随机库的种子
    random.seed(seed)

    # Numpy 的种子
    np.random.seed(seed)

    # PyTorch 的种子
    torch.manual_seed(seed)

    # 如果你在使用 CUDA，则需要额外设置以下内容
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

        # 下面两个设置可以使得计算结果更加确定性，但是可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 示例：设置种子
set_seed(42)

# 定义Dataset类加载数据
class CustomDataset(Dataset):
    def __init__(self, label_file, temp):
        # 从excel文件中读取标签到dataframe
        self.data = pd.read_excel(label_file)
        self.gene_id = self.data['gene_id'].values
        self.temp = temp
        self.total_result = {}
        if self.temp == 'train' :
            temp1 = 'standard_train'
            # ML_temp1 = 'standard_Svr_train'
            # ML_temp2 = 'standard_RF_train'
        if self.temp == 'original' :
            temp1 = 'standard'
            # ML_temp1 = 'standard_svr'
            # ML_temp2 = 'standard_rf'
        self.labels = self.data[temp1 + '_real_1D_MLP'].values
        
        for idx in range(len(self.gene_id)) :
            temp_result = []
            for temp2 in merge_component_arr:
                # print (temp1 + '_predict_' + temp2)
                temp_result.append(self.data[temp1 + '_predict_' + temp2].values[idx])
            temp_result = np.array(temp_result)
            self.total_result[self.gene_id[idx]] = temp_result
            del temp_result
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.total_result[self.gene_id[idx]]).to(dtype=torch.float32), torch.tensor(self.labels[idx]).to(dtype=torch.float32), self.gene_id[idx]

def get_net(in_features, ratio=0.05):
    count = int(in_features*ratio)
    net = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(in_features, 1),
                        )
    return net

def get_loss(loss_name):
    if loss_name == "MSELoss":
        loss = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    else:
        raise NotImplementedError
    return optimizer

def get_schduler(scheduler_name, optimizer, T0):
    # print (scheduler_name)
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T0)
    else:
        raise NotImplementedError
    return scheduler

# 定义L1 Loss
def l1_penalty(weights):
    return torch.abs(weights).sum()

def inverse_standardize(data, mean, std):
    return data * std + mean

save_index = 'test'
data_format = 'merge'

device = 'cpu'

fold = 0
mean_std_file = args.root_path + '1_Data_Collection/P/'+trait+ '/10-fold/'+ trait+'_Mean_Std.xlsx'

data_df = pd.read_excel(mean_std_file)

train_label_ind = args.root_path + 'Result/' +str(trait)+ '/merge_result/Final_merged_train_data'
test_label_ind = args.root_path + 'Result/' +str(trait)+ '/merge_result/Final_merged_test_data'

mean = data_df['Mean'].values
std = data_df['Std'].values
data_mean = mean[int(fold)]
data_std = std[int(fold)]
train_label_file = train_label_ind + str(fold) + '.xlsx' 
test_label_file = test_label_ind + str(fold) + '.xlsx' 
train_dataset_1 = CustomDataset(train_label_file, 'train')
result_total, label, id = train_dataset_1.__getitem__(0)

gene_count = len(result_total)

best_R2 = -10000
final_R2 = -10000
best_rmse = 10000
best_rmse_arr = [1000, 1000, 1000, 1000, 1000 ,1000, 1000, 1000, 1000, 1000]
final_rmse = 10000

now = time.time()

bs1=64
bs2=1000

train_dataloader_array = []
test_dataloader_array = []

for i in range(k_fold) :
    data_mean = mean[i]
    data_std = std[i]
    train_label_file = train_label_ind + str(i) + '.xlsx' 
    test_label_file = test_label_ind + str(i) + '.xlsx' 
    train_dataset = CustomDataset(train_label_file, 'train')
    test_dataset = CustomDataset(test_label_file, 'original')
    train_dataloader = DataLoader(train_dataset, batch_size=bs1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=bs2)
    train_dataloader_array.append(train_dataloader)
    test_dataloader_array.append(test_dataloader)

# 设置 CSV 文件路径
csv_file_path = root_path + 'Result/MR_result/'+str(trait)+'.csv'
csv_new_flag = 0
# 检查文件是否存在
csv_file = Path(csv_file_path)

if csv_file.is_file():
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 获取 "model_index" 列的最后一个值
    last_model_index_value = df["model_index"].iloc[-1]
    print (last_model_index_value)
    last_r2_value = df["R2"].iloc[-1]
    last_r_value = df["R"].iloc[-1]
    last_mse_value = df["MSE"].iloc[-1]
    last_mape_value = df["MAPE"].iloc[-1]
    print (last_mape_value)
    last_rmse_value = df["RMSE"].iloc[-1]
    last_single_model = df["single_model"].iloc[-1]
    last_p_values = df["p_values"].iloc[-1]
    last_r2_array = df["R2_folds"].iloc[-1]
    last_r_array = df["R_folds"].iloc[-1]
    last_mse_array = df["MSE_folds"].iloc[-1]
    last_mape_array = df["MAPE_folds"].iloc[-1]
    last_rmse_array = df["RMSE_folds"].iloc[-1]
    try:
        # 删除可能存在的单引号
        last_model_index_value = last_model_index_value.replace("'", '"')
        # 安全地将字符串转换为数组
        model_index_array = json.loads(last_model_index_value)

        # 删除可能存在的单引号
        last_p_values = last_p_values.replace("'", '"')
        # 安全地将字符串转换为数组
        last_p_values_array = json.loads(last_p_values)

        last_r2_array = last_r2_array.replace("'", '"')
        last_r2_array = json.loads(last_r2_array)

        last_r_array = last_r_array.replace("'", '"')
        last_r_array = json.loads(last_r_array)

        last_mse_array = last_mse_array.replace("'", '"')
        last_mse_array = json.loads(last_mse_array)

        last_mape_array = last_mape_array.replace("'", '"')
        last_mape_array = last_mape_array.replace('inf', '0')
        last_mape_array = json.loads(last_mape_array)

        last_rmse_array = last_rmse_array.replace("'", '"')
        last_rmse_array = json.loads(last_rmse_array)
    except json.JSONDecodeError as e:
        print(f"Error converting the 'model_index' value to an array: {e}")
        last_mse_value = None
        last_mape_value = None
        last_rmse_value = None
        last_r2_value = None
        last_r_value = None
        # 在这里处理错误，例如设置 model_index_array 为 None 或者退出
        model_index_array = None
        last_p_values_array = None
        last_r2_array = None
        last_r_array = None
        last_mse_array = None
        last_mape_array = None
        last_rmse_array = None
    
else:
    # 文件不存在，设置 model_index_array 为 [0]
    model_index_array = [0]
    last_mse_value = 0
    last_mape_value = 0
    last_rmse_value = 0
    last_r2_value = 0
    last_r_value = 0
    last_single_model = 0
    last_p_values_array = []
    last_r2_array = []
    last_r_array = []
    last_mse_array = []
    last_mape_array = []
    last_rmse_array = []
    for i in range(10):
        for j, data in enumerate(test_dataloader_array[i], 0):
            gene, hd, gene_id = data
            gene = gene[:, 0]
            # hd = hd
            original_init_real = inverse_standardize(hd, mean[i], std[i])
            original_init_predict = inverse_standardize(gene, mean[i], std[i])
            last_mse_value += mean_squared_error(original_init_real, original_init_predict)
            last_mape_value += tools1.mean_absolute_percentage_error(original_init_real, original_init_predict)
            last_rmse_value += np.sqrt(mean_squared_error(original_init_real, original_init_predict))
            last_r2_value += r2_score(original_init_real, original_init_predict)
            last_r_value += pearsonr(original_init_real, original_init_predict)[0]
            last_r2_array = last_r2_array + [r2_score(original_init_real, original_init_predict)]
            last_r_array = last_r_array + [pearsonr(original_init_real, original_init_predict)[0]]
            last_mse_array = last_mse_array + [mean_squared_error(original_init_real, original_init_predict)]
            last_mape_array = last_mape_array + [tools1.mean_absolute_percentage_error(original_init_real, original_init_predict)]
            last_rmse_array = last_rmse_array + [np.sqrt(mean_squared_error(original_init_real, original_init_predict))]
    last_r2_value /= 10
    last_r_value /= 10
    last_mse_value /= 10
    last_rmse_value /= 10
    last_mape_value /= 10
    csv_new_flag = 1

# 现在 model_index_array 包含了数组
merge_arr = model_index_array + [int(aim_index)]

best_features = merge_arr.copy()
final_features = merge_arr.copy()
last_features = merge_arr.copy()

aim_last_mse_value = 0
aim_last_mape_value = 0
aim_last_rmse_value = 0
aim_last_r2_value = 0
aim_last_r_value = 0
aim_last_p_values_array = []
aim_last_r2_array = []
aim_last_r_array = []
aim_last_mse_array = []
aim_last_mape_array = []
aim_last_rmse_array = []

for i in range(10):
    for j, data in enumerate(test_dataloader_array[i], 0):
        gene, hd, gene_id = data
        gene = gene[:, aim_index]
        # hd = hd
        aim_original_init_real = inverse_standardize(hd, mean[i], std[i])
        aim_original_init_predict = inverse_standardize(gene, mean[i], std[i])
        aim_last_mse_value += mean_squared_error(aim_original_init_real, aim_original_init_predict)
        aim_last_mape_value += tools1.mean_absolute_percentage_error(aim_original_init_real, aim_original_init_predict)
        aim_last_rmse_value += np.sqrt(mean_squared_error(aim_original_init_real, aim_original_init_predict))
        aim_last_r2_value += r2_score(aim_original_init_real, aim_original_init_predict)
        aim_last_r_value += pearsonr(aim_original_init_real, aim_original_init_predict)[0]
        aim_last_r2_array = aim_last_r2_array + [r2_score(aim_original_init_real, aim_original_init_predict)]
        aim_last_r_array = aim_last_r_array + [pearsonr(aim_original_init_real, aim_original_init_predict)[0]]
        aim_last_mse_array = aim_last_mse_array + [mean_squared_error(aim_original_init_real, aim_original_init_predict)]
        aim_last_mape_array = aim_last_mape_array + [tools1.mean_absolute_percentage_error(aim_original_init_real, aim_original_init_predict)]
        aim_last_rmse_array = aim_last_rmse_array + [np.sqrt(mean_squared_error(aim_original_init_real, aim_original_init_predict))]

aim_last_r2_value /= 10
aim_last_r_value /= 10
aim_last_mse_value /= 10
aim_last_rmse_value /= 10
aim_last_mape_value /= 10

is_first = True
total_fold_best_R2 = 0
total_fold_best_RMSE = 0
total_fold_best_MAPE = 0
total_fold_best_R = 0
total_fold_best_MSE = 0

total_fold_best_RMSE_array = []
total_fold_best_R2_array = []
total_fold_best_MAPE_array = []
total_fold_best_R_array = []
total_fold_best_MSE_array = []

for fold in range(k_fold) :
    set_seed(42)
    current_features = best_features.copy()
    model = get_net(len(current_features))
    model = model.to(device)
    loss_curve = np.array([])
    test_loss_curve = np.array([])
    early_stopping = EarlyStopping(patience=200, min_delta=0.001)    
    min_loss, min_rmse, max_r2, max_r = 5000, 10000, -1000, -1000
    min_mape = 1000
    min_mse = 100000000
    max_r_p = 1
    criterion = get_loss("MSELoss")
    optimizer = get_optimizer("AdamW", model, lr)
    scheduler = get_schduler("CosineAnnealingWarmRestarts", optimizer,50)
    for epoch in range(5000):
        model.train()
        tot_loss = 0
        tot_acc = 0
        
        for i, data in enumerate(train_dataloader_array[fold], 0):
            gene, hd, gene_id = data
            gene = gene[:, current_features]
            hd = hd.to(device)
            gene = gene.to(device)
            outputs = model(gene)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, hd)
            l1_loss = sum(l1_penalty(w) for w in model.parameters())
            loss = loss + 0. * l1_loss  # alpha * L1损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss
        scheduler.step()
        tot_loss = tot_loss.detach().to("cpu").numpy()
        if (tot_loss / len(train_dataloader_array[fold]) > 2):
            tot_loss = np.nan
        else :
            tot_loss /=  len(train_dataloader_array[fold])
        predict = []
        real = []
        model.eval()
        test_loss = 0
        test_total_acc = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader_array[fold], 0):
                gene, hd, gene_id = data
                gene = gene[:, current_features]
                hd = hd
                real.append(hd.detach().cpu().numpy())
                hd = hd.to(device)
                gene = gene.to(device)
                start = time.time()
                outputs = model(gene)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, hd)
                test_loss += loss
                end = time.time()
                predict.append(outputs.detach().to("cpu").numpy())
                cost_time = end - start
        real = np.array(real).flatten()
        predict = np.array(predict).flatten()
        # print(real.shape, predict.shape)
        original_real = inverse_standardize(real, mean[int(fold)], std[int(fold)])
        original_predict = inverse_standardize(predict, mean[int(fold)], std[int(fold)])
        RMSE = np.sqrt(mean_squared_error(original_real, original_predict))
        MSE = mean_squared_error(original_real, original_predict)
        R, p_value = pearsonr(original_real, original_predict)
        R2 = r2_score(original_real, original_predict)
        MAPE = tools1.mean_absolute_percentage_error(original_real, original_predict)

        if (RMSE < min_rmse):
            min_rmse = RMSE
            min_mse = MSE
            max_r2 = R2
            max_r = R
            min_mape = MAPE
            max_r_p = p_value
        early_stopping(-R2)

        if early_stopping.early_stop:
            total_fold_best_RMSE += min_rmse
            total_fold_best_RMSE_array = total_fold_best_RMSE_array + [min_rmse]
            total_fold_best_R2 += max_r2
            total_fold_best_R2_array = total_fold_best_R2_array + [max_r2]
            total_fold_best_MAPE += min_mape
            total_fold_best_MAPE_array = total_fold_best_MAPE_array + [min_mape]
            total_fold_best_R += max_r
            total_fold_best_R_array = total_fold_best_R_array + [max_r]
            total_fold_best_MSE += min_mse
            total_fold_best_MSE_array = total_fold_best_MSE_array + [min_mse]
            print (f"early stop, epoch:{epoch}, train_los: {tot_loss:.2f}, test_loss: {test_loss:.2f}, R2: {max_r2:.2f}, r: {max_r:.2f}, RMSE: {min_rmse:.2f}, {model[1].weight.data}")
            break

        if epoch == 4999:
            total_fold_best_RMSE += min_rmse
            total_fold_best_RMSE_array = total_fold_best_RMSE_array + [min_rmse]
            total_fold_best_R2 += max_r2
            total_fold_best_R2_array = total_fold_best_R2_array + [max_r2]
            total_fold_best_MAPE += min_mape
            total_fold_best_MAPE_array = total_fold_best_MAPE_array + [min_mape]
            total_fold_best_R += max_r
            total_fold_best_R_array = total_fold_best_R_array + [max_r]
            total_fold_best_MSE += min_mse
            total_fold_best_MSE_array = total_fold_best_MSE_array + [min_mse]

            print(f"epoch end, fold : , {str(fold)},  max R2: , {max_r2:.2f}, min RMSE: , {min_rmse:.2f}, min MAPE: , {max_r:.2f}")

total_fold_best_MAPE /= 10
total_fold_best_R2 /= 10
total_fold_best_MSE /= 10
total_fold_best_R /= 10
total_fold_best_RMSE /= 10

print(f"total RMSE: , {total_fold_best_RMSE:.2f}, last RMSE: , {last_rmse_value:.2f}")
print (f"total R2: , {total_fold_best_R2:.2f}, last R2: , {last_r2_value:.2f}")

final_fold_best_R2 = 0
final_fold_best_R = 0
final_fold_best_RMSE = 0
final_fold_best_MAPE = 0
final_fold_best_MSE = 0
final_fold_p_values = []
final_fold_r2_array = []
final_fold_r_array = []
final_fold_mse_array = []
final_fold_mape_array = []
final_fold_rmse_array = []
single_model = -1

if (total_fold_best_R2 < last_r2_value) :
    if (last_r2_value <= aim_last_r2_value):
        final_fold_best_R2 = aim_last_r2_value
        final_fold_best_R = aim_last_r_value
        final_fold_best_RMSE = aim_last_rmse_value
        final_fold_best_MAPE = aim_last_mape_value
        final_fold_best_MSE = aim_last_mse_value
        final_fold_p_values = aim_last_p_values_array
        final_fold_r2_array = aim_last_r2_array
        final_fold_r_array = aim_last_r_array
        final_fold_mse_array = aim_last_mse_array
        final_fold_mape_array = aim_last_mape_array
        final_fold_rmse_array = aim_last_rmse_array
        final_features = [aim_index]
        single_model = aim_index
    else :
        final_fold_best_R2 = last_r2_value
        final_fold_best_R = last_r_value
        final_fold_best_RMSE = last_rmse_value
        final_fold_best_MAPE = last_mape_value
        final_fold_best_MSE = last_mse_value
        final_fold_p_values = last_p_values_array
        final_fold_r2_array = last_r2_array
        final_fold_r_array = last_r_array
        final_fold_mse_array = last_mse_array
        final_fold_mape_array = last_mape_array
        final_fold_rmse_array = last_rmse_array
        final_features = model_index_array
        single_model = last_single_model

else:

    record_features = -1
    record_rmse = 10000
    record_r2 = -10000
    record_rmse_arr = [1000, 1000, 1000, 1000, 1000 ,1000, 1000, 1000, 1000, 1000]
    record_r2_arr = [-1000, -1000, -1000, -1000, -1000 , -1000, -1000, -1000, -1000, -1000]
    record_mse = 100000000
    record_mape = 1000
    record_r = -1000
    record_mse_arr = [100000000, 100000000, 100000000, 100000000, 100000000 ,100000000, 100000000, 100000000, 100000000, 100000000]
    record_mape_arr = [1000, 1000, 1000, 1000, 1000 ,1000, 1000, 1000, 1000, 1000]
    record_r_arr = [-1000, -1000, -1000, -1000, -1000 , -1000, -1000, -1000, -1000, -1000]
    flag = 1
    while flag :
        for feature_index in last_features :
            if len(last_features) == 2:
                break
            fold_best_RMSE = 0
            fold_best_RMSE_array = []
            fold_best_R2 = 0
            fold_best_R2_array = []
            fold_best_MSE = 0
            fold_best_MAPE = 0
            fold_best_R = 0
            fold_best_MSE_array = []
            fold_best_MAPE_array = []
            fold_best_R_array = []
            for fold in range(k_fold) :
                set_seed(42)
                current_features = best_features.copy()
                current_features.remove(feature_index)
                if (current_features == model_index_array) and csv_new_flag == 0:
                    print (last_rmse_value, last_r2_value, last_mape_value, last_r_value, last_mse_value)
                    fold_best_RMSE = last_rmse_value * 10
                    fold_best_RMSE_array = last_rmse_array
                    fold_best_R2 = last_r2_value * 10
                    fold_best_R2_array = last_r2_array
                    fold_best_MAPE = last_mape_value * 10
                    fold_best_MAPE_array = last_mape_array
                    fold_best_R = last_r_value * 10
                    fold_best_R_array = last_r_array
                    fold_best_MSE = last_mse_value * 10
                    fold_best_MSE_array = last_mse_array
                    break
                model = get_net(len(current_features))
                model = model.to(device)
                loss_curve = np.array([])
                test_loss_curve = np.array([])
                early_stopping = EarlyStopping(patience=200, min_delta=0.001)    
                min_loss, min_rmse, max_r2, max_r = 5000, 10000, -1000, -1000
                min_mape = 1000
                min_mse = 100000000
                max_r_p = 1
                criterion = get_loss("MSELoss")
                optimizer = get_optimizer("AdamW", model, lr)
                scheduler = get_schduler("CosineAnnealingWarmRestarts", optimizer,50)
                for epoch in range(5000):
                    model.train()
                    tot_loss = 0
                    tot_acc = 0

                    for i, data in enumerate(train_dataloader_array[fold], 0):
                        gene, hd, gene_id = data
                        gene = gene[:, current_features]
                        hd = hd.to(device)
                        gene = gene.to(device)
                        outputs = model(gene)
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, hd)
                        l1_loss = sum(l1_penalty(w) for w in model.parameters())
                        loss = loss + 0. * l1_loss  # alpha * L1损失
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        tot_loss += loss
                    scheduler.step()
                    tot_loss = tot_loss.detach().to("cpu").numpy()
                    if (tot_loss / len(train_dataloader_array[fold]) > 2):
                        tot_loss = np.nan
                    else :
                        tot_loss /= len(train_dataloader_array[fold])
                    predict = []
                    real = []
                    model.eval()
                    test_loss = 0
                    test_total_acc = 0
                    with torch.no_grad():
                        for i, data in enumerate(test_dataloader_array[fold], 0):
                            gene, hd, gene_id = data
                            gene = gene[:, current_features]
                            hd = hd
                            real.append(hd.detach().cpu().numpy())
                            hd = hd.to(device)
                            gene = gene.to(device)
                            start = time.time()
                            outputs = model(gene)
                            outputs = outputs.squeeze(-1)
                            loss = criterion(outputs, hd)
                            test_loss += loss
                            end = time.time()
                            predict.append(outputs.detach().to("cpu").numpy())
                            cost_time = end - start
                    real = np.array(real).flatten()
                    predict = np.array(predict).flatten()
                    # print(real.shape, predict.shape)
                    original_real = inverse_standardize(real, mean[int(fold)], std[int(fold)])
                    original_predict = inverse_standardize(predict, mean[int(fold)], std[int(fold)])
                    RMSE = np.sqrt(mean_squared_error(original_real, original_predict))
                    MSE = mean_squared_error(original_real, original_predict)
                    R, p_value = pearsonr(original_real, original_predict)
                    R2 = r2_score(original_real, original_predict)
                    MAPE = tools1.mean_absolute_percentage_error(original_real, original_predict)
                    if (RMSE < min_rmse):
                        min_rmse = RMSE
                        min_mse = MSE
                        max_r2 = R2
                        max_r = R
                        min_mape = MAPE
                    early_stopping(-R2)
                    if early_stopping.early_stop:
                        fold_best_RMSE += min_rmse
                        fold_best_RMSE_array = fold_best_RMSE_array + [min_rmse]
                        fold_best_R2 += max_r2
                        fold_best_R2_array = fold_best_R2_array + [max_r2]
                        fold_best_MAPE += min_mape
                        fold_best_MAPE_array = fold_best_MAPE_array + [min_mape]
                        fold_best_R += max_r
                        fold_best_R_array = fold_best_R_array + [max_r]
                        fold_best_MSE += min_mse
                        fold_best_MSE_array = fold_best_MSE_array + [min_mse]
                        print (f"early stop, epoch:{epoch}, fold:, {str(fold)}, train_los: {tot_loss:.2f}, test_loss: {test_loss:.2f}, R2: {max_r2:.2f}, r: {max_r:.2f}, RMSE: {min_rmse:.2f}")

                        # print(f"Early stopping epoch", epoch, " fold : ", str(fold), " max R2: ", max_r2, "min RMSE: ", min_rmse, "min MAPE: ", min_mape, "min MSE: ", min_mse, "max R: ", max_r, "avg RMSE: ", fold_best_RMSE / (fold + 1))
                        break
                    if epoch == 4999:
                        fold_best_RMSE += min_rmse
                        fold_best_RMSE_array = fold_best_RMSE_array + [min_rmse]
                        fold_best_R2 += max_r2
                        fold_best_R2_array = fold_best_R2_array + [max_r2]
                        fold_best_MAPE += min_mape
                        fold_best_MAPE_array = fold_best_MAPE_array + [min_mape]
                        fold_best_R += max_r
                        fold_best_R_array = fold_best_R_array + [max_r]
                        fold_best_MSE += min_mse
                        fold_best_MSE_array = fold_best_MSE_array + [min_mse]
                        print (f"epoch end, epoch:{epoch}, fold:, {str(fold)}, train_los: {tot_loss:.2f}, test_loss: {test_loss:.2f}, R2: {max_r2:.2f}, r: {max_r:.2f}, RMSE: {min_rmse:.2f}")

                        # print("epoch end, fold : ", str(fold), " max R2: ", max_r2, "min RMSE: ", min_rmse, "min MAPE: ", min_mape, "min MSE: ", min_mse, "max R: ", max_r, "avg RMSE: ", fold_best_RMSE / (fold + 1))
            fold_best_MAPE /= 10
            fold_best_MSE /= 10
            fold_best_RMSE /= 10
            fold_best_R /= 10
            fold_best_R2 /= 10
            print(f"now_feature,  {best_features}, feature index: , {feature_index}, R2 : {fold_best_R2:.2f}")

            # if ((fold_best_RMSE / 10 < best_rmse) and (fold_best_RMSE / 10 - best_rmse < record_rmse) and (fold_best_RMSE/10 < total_fold_best_RMSE/10) ):
            if (((fold_best_R2) > (record_r2)) and ((fold_best_R2) >= (total_fold_best_R2)) and (fold_best_R2 > aim_last_r2_value*1.01) and (fold_best_R2 > last_r2_value*1.01)):
                print ("存在删除某一元素后，比total好的情况, 且大于上一次merge的元素, 删除元素： ", feature_index)
                print ("当前 R2 数组：", fold_best_R2_array, "record R2 数组：", record_r2_arr, "total R2 数组：", total_fold_best_R2_array)
                print (f"当前 R2：, {fold_best_R2:.2f}, record R2：, {record_r2:.2f}, total R2：, {total_fold_best_R2:.2f}")
                record_rmse = fold_best_RMSE
                record_r2 = fold_best_R2
                record_rmse_arr = fold_best_RMSE_array
                record_r2_arr = fold_best_R2_array
                record_mape = fold_best_MAPE
                record_mse = fold_best_MSE
                record_r = fold_best_R
                record_mape_arr = fold_best_MAPE_array
                record_mse_arr = fold_best_MSE_array
                record_r_arr = fold_best_R_array
                record_features = feature_index
                print(f"temp best r2: , {record_r2:.2f},  temp best append features: , {feature_index}, now_feature,  {best_features}")

        if record_features == -1:
            print("不存在 删除某一个之后，比total好的存在，直接比较，last、total、aim_add")
            print (f"last R2: , {last_r2_value:.2f},  total R2: , {total_fold_best_R2:.2f},  aim_add R2: , {aim_last_r2_value:.2f}")

            if last_r2_value*1.01 < (total_fold_best_R2) or last_r2_value*1.01 < (aim_last_r2_value):
                if aim_last_r2_value >= total_fold_best_R2 :
                    final_fold_best_R2 = aim_last_r2_value
                    final_fold_best_R = aim_last_r_value
                    final_fold_best_RMSE = aim_last_rmse_value
                    final_fold_best_MAPE = aim_last_mape_value
                    final_fold_best_MSE = aim_last_mse_value
                    final_fold_r2_array = aim_last_r2_array
                    final_fold_r_array = aim_last_r_array
                    final_fold_mse_array = aim_last_mse_array
                    final_fold_mape_array = aim_last_mape_array
                    final_fold_rmse_array = aim_last_rmse_array
                    final_features = [aim_index]
                    single_model = aim_index
                else :
                    final_fold_best_R2 = total_fold_best_R2
                    final_fold_best_R = total_fold_best_R
                    final_fold_best_RMSE = total_fold_best_RMSE
                    final_fold_best_MAPE = total_fold_best_MAPE
                    final_fold_best_MSE = total_fold_best_MSE
                    final_fold_r2_array = total_fold_best_R2_array
                    final_fold_r_array = total_fold_best_R_array
                    final_fold_mse_array = total_fold_best_MSE_array
                    final_fold_mape_array = total_fold_best_MAPE_array
                    final_fold_rmse_array = total_fold_best_RMSE_array
            else :
                final_fold_best_R2 = last_r2_value
                final_fold_best_R = last_r_value
                final_fold_best_RMSE = last_rmse_value
                final_fold_best_MAPE = last_mape_value
                final_fold_best_MSE = last_mse_value
                final_fold_r2_array = last_r2_array
                final_fold_r_array = last_r_array
                final_fold_mse_array = last_mse_array
                final_fold_mape_array = last_mape_array
                final_fold_rmse_array = last_rmse_array
                final_features = model_index_array
                single_model = last_single_model
            break
        else:
        # 当前只改到这里
            if (len(last_features) > 2) and record_features in best_features:
                best_features.remove(record_features)
                last_features.remove(record_features)
                # final_features.remove(record_features)
                continue
            else :
                if record_features in best_features:
                    best_features.remove(record_features)
                final_features = best_features.copy()
                # print (record_r2_arr, record_r_arr, record_mse_arr, record_mape_arr, record_rmse_arr)
                # print (record_r2, record_r, record_mse, record_mape, record_rmse)
                final_fold_best_R2 = record_r2
                final_fold_best_R = record_r
                final_fold_best_RMSE = record_rmse
                final_fold_best_MAPE = record_mape
                final_fold_best_MSE = record_mse
                final_fold_r2_array = record_r2_arr
                final_fold_r_array = record_r_arr
                final_fold_mse_array = record_mse_arr
                final_fold_mape_array = record_mape_arr
                final_fold_rmse_array = record_rmse_arr
                flag = 0



df_to_append = pd.DataFrame([[final_fold_best_R2, final_fold_best_R, final_fold_best_MSE, final_fold_best_RMSE,final_fold_best_MAPE, str(final_features), single_model, str(final_fold_p_values), str(final_fold_r2_array), str(final_fold_r_array), str(final_fold_mse_array), str(final_fold_mape_array), str(final_fold_rmse_array)]],
                            columns=['R2', 'R', 'MSE', 'RMSE', 'MAPE', 'model_index', 'single_model', 'p_values', 'R2_folds', 'R_folds', 'MSE_folds', 'MAPE_folds', 'RMSE_folds'])
# # df_to_append = df_to_append.round(3)
# 检查文件是否存在
csv_file = Path(csv_file_path)
if csv_file.is_file():
    # 文件存在，追加模式（注意header参数，避免重复写入列名）
    df_to_append.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    # 文件不存在，写入模式（创建文件）
    df_to_append.to_csv(csv_file_path, mode='w', index=False)

print("Values appended to CSV file successfully.")
