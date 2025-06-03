#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import argparse
from tools.EarlyStopping import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import optuna
import torch.optim as optim
import joblib
import random
import warnings
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--gene_model', type=str, default = '1D')
parser.add_argument('--data_format', type=str, default = '1800')
parser.add_argument('--root_path', type=str, default = '1800')
parser.add_argument('--lr', type=str, default = '1800')
parser.add_argument('--bs', type=str, default = '1800')
parser.add_argument('--dropout', type=str, default = '1800')

parser.add_argument('--k_fold', type=str, default = 'None')
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--device', type=str, default = 'None')
parser.add_argument('--result_name', type=str, default = 'None')
parser.add_argument('--optuna_count', type=str, default = 'None')


args = parser.parse_args()
gene_model = args.gene_model
k_fold = args.k_fold
trait = args.trait_name
data_format = args.data_format
root_path = args.root_path

# 定义模型
class SimplePCAModel(nn.Module):
    def __init__(self, in_features, hidden_units, dropout_rate, activation_name):
        super(SimplePCAModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout_rate),
            getattr(nn, activation_name)(),
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout_rate),
            getattr(nn, activation_name)(),
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout_rate),
            getattr(nn, activation_name)(),
            nn.Linear(hidden_units, 1)
        )
    def forward(self, x):
        return self.net(x)
    

class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden_units, dropout_rate, activation_name):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.Dropout(dropout_rate),
            getattr(nn, activation_name)(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.net(x)

# 定义Dataset类加载数据
class CustomDataset(Dataset):
    def __init__(self, gene_dir, label_file):
        self.gene_dir = gene_dir
        # 从excel文件中读取标签到dataframe
        df = pd.read_excel(label_file)
        self.labels = df[str(trait)].values
        self.gene_id = df['gene_id'].values
        if gene_model == 'PCA' or gene_model == 'GE_PCA':
            self.total_gene = pd.read_csv(gene_dir)
        elif gene_model == 'GRM':
            self.total_gene = pd.read_csv(gene_dir, index_col=0)
        else:
            self.total_gene = {}
            for i in self.gene_id:
                gene = np.load(self.gene_dir + "/" + str(i)+".npz")
                total_gene_list = [gene[key] for key in gene if key == 'arr_0' or key.startswith('arr_')]
            # 将列表中的数组堆叠起来形成一个NumPy数组
                total_gene = np.vstack(total_gene_list)
                self.total_gene[i] = total_gene
        self.data = df
        print("Total number of samples: ", len(self.data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gene_id = self.gene_id[idx]
        if gene_model == 'PCA' or gene_model == 'GE_PCA':
            total_gene = self.total_gene.iloc[idx, :].values
        elif gene_model == 'GRM':
            # 直接取第一行
            total_gene = np.atleast_2d(self.total_gene.loc[gene_id, :].values)[0]
        else:
            total_gene = self.total_gene[gene_id]
        gene = torch.tensor(total_gene.flatten()).to(dtype=torch.float32)
        label = torch.tensor(self.labels[idx]).to(dtype=torch.float32)
        return gene, label, gene_id
        # return gene, label

gene_dir = root_path + 'Result/'+trait+'/SNPs/encode/' + str(gene_model) + '/'

mean_std_file = root_path + '1_Data_Collection/P/'+trait+'/10-fold/'+trait+'_Mean_Std.xlsx'

data_df = pd.read_excel(mean_std_file)

mean = data_df['Mean'].values
std = data_df['Std'].values

train_dataset_array = []
test_dataset_array = []
gene_count_array = []
for i in range(int(k_fold)) :
    if gene_model == 'PCA' or gene_model == 'GE_PCA':
        train_gene_dir = gene_dir + 'train_' + str(i) + '.csv'
        test_gene_dir = gene_dir + 'test_' + str(i) + '.csv'
    elif gene_model == 'GRM':
        train_gene_dir = gene_dir + 'train_' + str(i) + '.csv'
        test_gene_dir = gene_dir + 'test_' + str(i) + '.csv'
    else:
        train_gene_dir = gene_dir + 'One_hot/'+str(i)
        test_gene_dir = gene_dir + 'One_hot/'+str(i)

    train_label_file = root_path + '1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_train.xlsx'
    test_label_file = root_path + '1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_test.xlsx' 

    train_dataset = CustomDataset(train_gene_dir, train_label_file)
    test_dataset = CustomDataset(test_gene_dir, test_label_file)
    gene, label, gene_id = train_dataset.__getitem__(0)
    gene_count = len(gene)
    print (gene_count)

    train_dataset_array.append(train_dataset)
    test_dataset_array.append(test_dataset)
    gene_count_array.append(gene_count)

device = args.device
def get_scheduler(scheduler_name, optimizer, T0):
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T0)
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=T0, gamma=0.9)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T0)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} is not implemented.")
    return scheduler

record = {
    'trail': [],
    'hidden_units': [],
    'dropout_rate': [],
    'optimizer': [],
    'lr': [],
    'activation': [],
    'batch_size': [],
    'scheduler': [],
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
    print ("==================== R2:", now_r2)
    # 超参数搜索空间
    hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256, 512, 1024])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
    lr = trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    activation_name = trial.suggest_categorical('activation', ['LeakyReLU', 'Tanh'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingWarmRestarts', 'StepLR'])

    losses = []
    R2s = []
    Rs = []
    RMSEs = []

    total_train_df_arr = []
    total_test_df_arr = []
    for fold in range(int(k_fold)):
        record['trail'].append(trial.number)
        record['hidden_units'].append(hidden_units)
        record['dropout_rate'].append(dropout_rate)
        record['optimizer'].append(optimizer_name)
        record['lr'].append(lr)
        record['activation'].append(activation_name)
        record['batch_size'].append(batch_size)
        record['scheduler'].append(scheduler_name)
        record['fold'].append(fold)

        train_dataset = train_dataset_array[fold]
        test_dataset = test_dataset_array[fold]
            # 自己的逻辑
        if gene_model == 'GRM':
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
        if gene_model == 'PCA' or gene_model == 'GE_PCA':
            model = SimplePCAModel(in_features=gene_count_array[fold], hidden_units=hidden_units, dropout_rate=dropout_rate, activation_name=activation_name)
        else:
            model = SimpleModel(in_features=gene_count_array[fold], hidden_units=hidden_units, dropout_rate=dropout_rate, activation_name=activation_name)
        model = model.to(device)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = get_scheduler(scheduler_name, optimizer, 10)
        criterion = nn.MSELoss()

        early_stopping = EarlyStopping(patience=50, min_delta=0.001)
        min_loss, min_rmse, max_r2, max_r = 5000, 1000, -1000, -1000

        record_epoch = 0
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
                # print (data)
                gene, hd, gene_id = data
                train_id = np.append(train_id, gene_id)
                train_real = np.append(train_real, hd)
                hd = hd.to(device)
                gene = gene.to(device)
                start = time.time()  
                outputs = model(gene)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, hd)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss
                end = time.time()
                train_predict = np.append(train_predict, outputs.detach().to("cpu").numpy())
                cost_time = end - start
            scheduler.step()

            predict = np.array([])
            real = np.array([])
            ids = np.array([])
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    gene, hd, gene_id = data
                    ids = np.append(ids, gene_id)
                    hd = hd

                    real = np.append(real, hd)
                    hd = hd.to(device)
                    gene = gene.to(device)

                    start = time.time()
                    outputs = model(gene.float())
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, hd)
                    test_loss += loss
                    end = time.time()
                    predict = np.append(predict, outputs.detach().to("cpu").numpy())
                    cost_time = end - start
                    # print ("epoch:{},batch:{},test_loss:{},cost_time:{}".format(epoch, i, loss, cost_time))

            # print("test loss:", test_loss / len(test_dataloader))
            test_loss = test_loss.detach().to("cpu").numpy()

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
            # 匹配早停
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
    record['hidden_units'].append(hidden_units)
    record['dropout_rate'].append(dropout_rate)
    record['optimizer'].append(optimizer_name)
    record['lr'].append(lr)
    record['activation'].append(activation_name)
    record['batch_size'].append(batch_size)
    record['scheduler'].append(scheduler_name)
    record['loss'].append(np.mean(losses))
    record['R2'].append(np.mean(R2s))
    record['R'].append(np.mean(Rs))
    record['RMSE'].append(np.mean(RMSEs))
    record['fold'].append(-1)
    record['epoch'].append(-1)

    if (np.mean(R2s) > now_r2):
        now_r2 = np.mean(R2s)
        for i in range(int(k_fold)):
            total_save_url = root_path + 'Result/' + str(trait) + '/Base_model_result/MLP/'
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

import os
os.makedirs(root_path + 'Result/' + str(trait) + '/Base_model_result/MLP/temp', exist_ok=True)

# 创建一个Optuna study
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)

# study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=int(args.optuna_count))

# 转换为 DataFrame
df_info = pd.DataFrame.from_dict(record, orient='index')

save_result_name = args.result_name

# 保存为 CSV 文件
# df_info.to_csv('info_record.csv', index=False)
df_info.to_csv(root_path+'Result/optuna/'+trait+'_'+save_result_name+ '_MLP_'+str(data_format)+'_' + str(gene_model)+'_record.csv', index_label='key')

# 保存Study对象
joblib.dump(study, root_path+'Result/optuna/'+trait+'_'+save_result_name + '_MLP_'+str(data_format)+'_' + str(gene_model)+'.pkl')
