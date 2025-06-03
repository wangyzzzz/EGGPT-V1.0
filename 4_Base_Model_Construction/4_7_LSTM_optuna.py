#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import random
import numpy as np
import os
import argparse
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import time
from tools.EarlyStopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import optuna
# import torch.optim as optim
import joblib
def set_seed(seed):
    """固定所有的随机种子以确保实验可重复性"""
    # Python 内置随机库的种子
    random.seed(seed)

    # Numpy 的种子
    np.random.seed(seed)

    # PyTorch 的种子
    torch.manual_seed(seed)

    # 如果你在使用 CUDA，则需要额外设置以下内容
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 下面两个设置可以使得计算结果更加确定性，但是可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 示例：设置种子
set_seed(42)

# 定义Dataset类加载数据
class CustomDataset(Dataset):
    def __init__(self, gene_dir, label_file):
        self.gene_dir = gene_dir
        # 从excel文件中读取标签到dataframe
        df = pd.read_excel(label_file)

        self.labels = df[str(trait)].values
        self.gene_id = df['gene_id'].values
        self.total_gene = {}
        for i in self.gene_id:
            gene = np.load(self.gene_dir + "/" + str(i)+ ".npz")
            total_gene_list = [gene[key].reshape(1, -1) for key in gene if key == 'arr_0' or key.startswith('arr_')]
        # 将列表中的数组堆叠起来形成一个NumPy数组
            total_gene = np.vstack(total_gene_list)
            self.total_gene[i] = total_gene

        self.data = df
        print("Total number of samples: ", len(self.data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gene_id = self.gene_id[idx]
        total_gene = self.total_gene[gene_id]
        gene = torch.tensor(total_gene).to(dtype=torch.float32)
        label = torch.tensor(self.labels[idx]).to(dtype=torch.float32)
        return gene, label, gene_id

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--data_format', type=str, default = '1800')
parser.add_argument('--gene_model', type=str, default = '1D')
parser.add_argument('--root_path', type=str, default = 'None')
parser.add_argument('--k_fold', type=str, default = 'None')
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--bs', type=int, default = 64)
parser.add_argument('--dropout', type=float, default = 0.5)
parser.add_argument('--device', type=str, default = 'cpu')
parser.add_argument('--epoch', type=int, default = -1)
parser.add_argument('--flag', type=int, default = 0)
parser.add_argument('--result_name', type=str, default = 'None')
parser.add_argument('--optuna_count', type=str, default = 'None')

args = parser.parse_args()
data_format = args.data_format
gene_model = args.gene_model
k_fold = args.k_fold
trait = args.trait_name
lr = float(args.lr)
bs = int(args.bs)
dropout = float(args.dropout)
aim_epoch = int(args.epoch)
root_path = args.root_path



if (int(k_fold) == -1) or args.flag == 1:
    epoch_df = pd.read_csv(root_path + 'Result/' + str(trait) + '/Base_model_result/LSTM/group_means.csv')
    # epoch = epoch_df['epoch'].values
    # 找到第一列=gene_model的行
    gene_model_list = epoch_df[epoch_df['gene_model'] == gene_model]
    aim_epoch = gene_model_list['epoch'].values + 1



embid_ratio = 1
if gene_model == '3D' :
    embid_ratio = 1.5
if gene_model == '1D':
    embid_ratio = 0.5
gene_dir = root_path + 'Result/' + trait + '/SNPs/encode/' + str(gene_model) + '/'

mean_std_file = root_path + '1_Data_Collection/P/'+trait+ '/10-fold/'+ trait+'_Mean_Std.xlsx'

data_df = pd.read_excel(mean_std_file)

mean = data_df['Mean'].values
std = data_df['Std'].values

data_mean = mean[int(k_fold)]
data_std = std[int(k_fold)]



train_dataset_array = []
test_dataset_array = []

for i in range(int(k_fold)) :
    if gene_model == 'PCA' or gene_model == 'GE_PCA':
        train_gene_dir = gene_dir + 'train_' + str(i) + '.csv'
        test_gene_dir = gene_dir + 'test_' + str(i) + '.csv'
    else:
        train_gene_dir = gene_dir + 'TE/'+str(i)
        test_gene_dir = gene_dir + 'TE/'+str(i)

    train_label_file = root_path + '1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_train.xlsx'
    test_label_file = root_path + '1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_test.xlsx'

    train_dataset = CustomDataset(train_gene_dir, train_label_file)
    test_dataset = CustomDataset(test_gene_dir, test_label_file)
    train_dataset_array.append(train_dataset)
    test_dataset_array.append(test_dataset)


gene, label, id = train_dataset.__getitem__(0)
gene_count = len(gene)

threshold = 3
# 假设dataset是一个torch.utils.data.Dataset对象


class LSTMWrapper(nn.Module):
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # 选择最后一个时间步的输出



def get_loss(loss_name):
    if loss_name == "MSELoss":
        loss = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError
    return optimizer
def get_scheduler(scheduler_name, optimizer, T0):
    # print(scheduler_name)
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T0)
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)
    # elif scheduler_name == "ReduceLROnPlateau":
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=10)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T0)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} is not implemented.")
    return scheduler
os.makedirs(root_path + 'Result/' + str(trait) + '/Base_model_result/LSTM/temp', exist_ok=True)
save_url = root_path + 'Result/' + str(trait) + '/Base_model_result/LSTM/temp'
total_save_url = root_path + 'Result/' + str(trait) + '/Base_model_result/LSTM/'

train_scale = 0.9



record = {
    'trail': [],
    'num_layer': [],
    # 'depth': [],
    # 'mlp_ratio': [],
    'dropout_rate': [],
    'optimizer': [],
    'lr': [],
    # 'activation': [],
    'batch_size': [],
    'scheduler': [],
    'hidden_units': [],
    'fold': [],
    'epoch': [],
    'loss': [],
    'R2': [],
    'R': [],
    'RMSE': []
}


now_r2 = -10000

# 定义目标函数
def objective(trial):
    global now_r2
    set_seed(42)
    # global now_r2
    print ("==================== R2:", now_r2)
    # 超参数搜索空间
    hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128, 256, 512])

    # Adam
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
    lr = trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingWarmRestarts', 'StepLR'])

    losses = []
    # 定义超参数搜索范围
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    num_layer = trial.suggest_categorical("num_layer", [2, 4, 6, 8]) # 注意力头数


    losses = []
    R2s = []
    Rs = []
    RMSEs = []


    total_train_df_arr = []
    total_test_df_arr = []
    for fold in range(int(k_fold)):
        if aim_epoch == -1:
            early_stopping = EarlyStopping(patience=150, min_delta=0.001)
        else:
            early_stopping = EarlyStopping(patience=aim_epoch)
        record['trail'].append(trial.number)
        record['num_layer'].append(num_layer)
        record['dropout_rate'].append(dropout)
        record['optimizer'].append(optimizer_name)
        record['lr'].append(lr)
        # record['activation'].append(activation_name)
        record['batch_size'].append(batch_size)
        record['scheduler'].append(scheduler_name)
        record['hidden_units'].append(hidden_units)
        record['fold'].append(fold)
        early_loss  = 10000


        train_dataset = train_dataset_array[fold]
        test_dataset = test_dataset_array[fold]
            # 自己的逻辑
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


        if data_format == '1800':
            if gene_model == '3D':
                lstm_model = nn.LSTM(input_size=180, hidden_size=hidden_units, num_layers=num_layer, dropout=dropout, bidirectional=True, batch_first=True)
            elif gene_model == '1D':
                lstm_model = nn.LSTM(input_size=60, hidden_size=hidden_units, num_layers=num_layer, dropout=dropout, bidirectional=True, batch_first=True)
            else:
                lstm_model = nn.LSTM(input_size=120, hidden_size=hidden_units, num_layers=num_layer, dropout=dropout, bidirectional=True, batch_first=True)


        def get_net(in_features, ratio=0.05):
            if gene_model == 'PCA':
                net = nn.Sequential(
                        LSTMWrapper(lstm_model),
                        nn.Dropout(0.),
                        nn.Linear(128, 1)
                        )
            else:
                net = nn.Sequential(
                        LSTMWrapper(lstm_model),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_units*2, 1)
                        )
            return net

        model = get_net(300)
        device = args.device
        model = model.to(device)

        now = int(time.time())
        min_loss, min_rmse, max_r2, max_r = 5000, 1000, -1000, -1000

        # 存储每一epoch的准确率

        record_epoch = 0
        criterion = get_loss("MSELoss")
        optimizer = get_optimizer(optimizer_name, model, lr)
        scheduler = get_scheduler(scheduler_name, optimizer, 10)

        aim_epoch1 = 5000

        # 训练模型
        for epoch in range(aim_epoch1):  # 设定一个较大的epoch数，以便早停
            model.train()
            train_predict = np.array([])
            train_real = np.array([])
            train_id = np.array([])
            tot_loss = 0
            tot_acc = 0
            for i, data in enumerate(train_dataloader, 0):
                gene, hd, gene_id = data
                train_id = np.append(train_id, gene_id)
                # print (gene.shape)
                hd = hd
                train_real = np.append(train_real, hd)
                hd = hd.to(device)
                gene = gene.to(device)
                start = time.time()
                outputs = model(gene)
                outputs = outputs.squeeze(-1)
                # print (outputs[:5])
                loss = criterion(outputs, hd)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss
                end = time.time()
                train_predict = np.append(train_predict, outputs.detach().to("cpu").numpy())
                cost_time = end - start
                # print ("epoch:{},batch:{},train_loss:{},cost_time:{}".format(epoch, i, loss, cost_time))
                # break
            scheduler.step()
            # print("epoch:{},train loss:{}".format(epoch, tot_loss / len(train_dataloader)))

            predict = np.array([])
            real = np.array([])
            ids = np.array([])
            model.eval()
            test_loss = 0
            test_total_acc = 0
            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    gene, hd, gene_id = data
                    ids = np.append(ids, gene_id)
                    hd = hd

                    real = np.append(real, hd)
                    hd = hd.to(device)
                    gene = gene.to(device)
                    start = time.time()
                    outputs = model(gene)
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, hd)
                    test_loss += loss
                    end = time.time()
                    predict = np.append(predict, outputs.detach().to("cpu").numpy())
                    cost_time = end - start
                    # print ("epoch:{},batch:{},test_loss:{},cost_time:{}".format(epoch, i, loss, cost_time))

            # print("test loss:", test_loss / len(test_dataloader))
            test_loss = test_loss.detach().to("cpu").numpy()
            # test_loss_curve = np.append(test_loss_curve, test_loss / len(test_dataloader))

            def inverse_standardize(data, mean, std):
                return data * std + mean
            data_mean = mean[int(fold)]
            data_std = std[int(fold)]
            original_real = inverse_standardize(real, data_mean, data_std)
            original_predict = inverse_standardize(predict, data_mean, data_std)

            RMSE = np.sqrt(mean_squared_error(original_real, original_predict))
            # R, _ = pearsonr(original_real, original_predict)
            R = 0
            R2 = r2_score(original_real, original_predict)

            if (RMSE < min_rmse):
                min_rmse = RMSE
                max_r2 = R2
                max_r = R
                record_epoch = epoch
                test_df = pd.DataFrame({
                    'original_real': original_real,
                    'original_predict': original_predict,
                    'standard_real': real, 
                    'standard_predict': predict,
                    'gene_id': ids,
                })
                train_df = pd.DataFrame({
                    'train_real': inverse_standardize(train_real, data_mean, data_std),
                    'train_predict': inverse_standardize(train_predict, data_mean, data_std),
                    'standard_train_real': train_real,
                    'standard_train_predict': train_predict,
                    'gene_id': train_id,
                })
                # print ("epoch:{},batch:{},R:{},R2:{}".format(epoch, i, R, R2))

            if (test_loss < min_loss):
                min_loss = test_loss

            early_stopping(RMSE)

            if early_stopping.early_stop or epoch == aim_epoch1 - 1:
                print("fold: ", fold, " epoch:", epoch, " min_loss:", min_loss, "R2:", max_r2, "RMSE:", min_rmse)
                print("Early stopping")
                total_train_df_arr.append(train_df)
                total_test_df_arr.append(test_df)
                break
        record['loss'].append(min_loss)
        record['R2'].append(max_r2)
        record['R'].append(max_r)
        record['RMSE'].append(min_rmse)
        record['epoch'].append(record_epoch)
        losses.append(min_loss)
        R2s.append(max_r2)
        RMSEs.append(min_rmse)
        Rs.append(max_r)
    record['trail'].append(trial.number)
    # record['depth'].append(depth)
    # record['mlp_ratio'].append(mlp_ratio)
    record['num_layer'].append(num_layer)
    record['dropout_rate'].append(dropout)
    record['optimizer'].append(optimizer_name)
    record['lr'].append(lr)
    # record['activation'].append(activation_name)
    record['batch_size'].append(batch_size)
    record['scheduler'].append(scheduler_name)
    record['hidden_units'].append(hidden_units)
    record['loss'].append(np.mean(losses))
    record['R2'].append(np.mean(R2s))
    record['R'].append(np.mean(Rs))
    record['RMSE'].append(np.mean(RMSEs))
    record['fold'].append(-1)
    record['epoch'].append(-1)
        

    if (np.mean(R2s) > now_r2):
        now_r2 = np.mean(R2s)
        for i in range(int(k_fold)):
            total_save_url = args.root_path + 'Result/' + str(trait) + '/Base_model_result/LSTM/'
            excel_path = total_save_url + "/" + data_format  + "_" + gene_model + "_" + str(i) + "_test_predict" + '.xlsx'
            total_test_df_arr[i].to_excel(excel_path, index=True)
            excel_path = total_save_url + "/" + data_format + "_" + gene_model + "_" + str(i) + "_train_predict" + '.xlsx'
            total_train_df_arr[i].to_excel(excel_path, index=True)
    return 1-np.mean(R2s)

def callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# 使用固定的随机种子
set_seed(42)
# 配置Optuna日志记录
optuna.logging.set_verbosity(optuna.logging.INFO)

# 创建一个Optuna study
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)


# study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=int(args.optuna_count))

# 转换为 DataFrame
df_info = pd.DataFrame.from_dict(record, orient='index')


import pickle
save_result_name = args.result_name

# 保存为 CSV 文件
# df_info.to_csv('info_record.csv', index=False)
df_info.to_csv(root_path + 'Result/optuna/'+trait+'_'+save_result_name+ '_LSTM_'+str(data_format)+'_' + str(gene_model)+'_record.csv', index_label='key')

# 保存Study对象
joblib.dump(study, root_path + 'Result/optuna/'+trait+'_'+save_result_name + '_LSTM_'+str(data_format)+'_' + str(gene_model)+'.pkl')
