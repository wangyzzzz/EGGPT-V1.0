#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import joblib
import optuna
import lightgbm as lgb

def set_seed(seed):
    """固定所有的随机种子以确保实验可重复性"""
    # Python 内置随机库的种子
    random.seed(seed)

    # Numpy 的种子
    np.random.seed(seed)

import argparse

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--data_format', type=str, default = '1800')
parser.add_argument('--gene_model', type=str, default = '1D')
parser.add_argument('--root_path', type=str, default = 'None')
parser.add_argument('--k_fold', type=str, default = 'None')
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--result_name', type=str, default = 'None')
parser.add_argument('--optuna_count', type=str, default = 'None')

args = parser.parse_args()

data_format = args.data_format
gene_model = args.gene_model
k_fold = args.k_fold
trait = args.trait_name
root_path = args.root_path
set_seed(42)
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
            total_gene = np.atleast_2d(self.total_gene.loc[gene_id, :].values)[0]
        else:
            total_gene = self.total_gene[gene_id]
        gene = torch.tensor(total_gene.flatten()).to(dtype=torch.float32)
        label = torch.tensor(self.labels[idx]).to(dtype=torch.float32)
        return gene, label, gene_id

ML_NAME = data_format + "_" + gene_model + "_" + k_fold

gene_dir = root_path + 'Result/' + trait + '/SNPs/encode/' +  str(gene_model) + '/'

mean_std_file = root_path + '1_Data_Collection/P/'+trait+ '/10-fold/' +trait+'_Mean_Std.xlsx'

data_df = pd.read_excel(mean_std_file)

mean = data_df['Mean'].values
std = data_df['Std'].values

data_mean = mean[int(k_fold)]
data_std = std[int(k_fold)]

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
    gene, label, id = train_dataset.__getitem__(0)
    gene_count = len(gene)
    print (gene_count)

    train_dataset_array.append(train_dataset)
    test_dataset_array.append(test_dataset)
    gene_count_array.append(gene_count)


count = 0
record = {
    'trail': [],
    'n_estimators': [],
    # 'min_samples_split': [],
    'min_data_in_leaf': [],
    'max_depth': [],
    # 'max_features': [],
    'lr': [],
    'C': [],
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
    
    n_estimators = trial.suggest_categorical('n_estimators', np.arange(50, 1001, 50))
    min_data_in_leaf = trial.suggest_categorical('min_data_in_leaf', [1, 2, 4, 6])  # Python 列表也可以
    max_depth = trial.suggest_categorical('max_depth', [10, 20, 30, 40, 50, None])
    lr = trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    C = trial.suggest_categorical('C', [1e-3, 1e-2, 1e-1, 1, 10, 100])


    losses = []
    R2s = []
    Rs = []
    RMSEs = []

    total_train_df_arr = []
    total_test_df_arr = []
    for fold in range(int(k_fold)):
        record['trail'].append(trial.number)
        record['n_estimators'].append(n_estimators)
        record['min_data_in_leaf'].append(min_data_in_leaf)
        record['max_depth'].append(max_depth)
        record['lr'].append(lr)
        record['C'].append(C)
        record['fold'].append(fold)

        # LightGBM 模型
        lgb_model = lgb.LGBMRegressor(
            boosting_type="gbdt",
            objective='regression', # 或其他适合你问题的目标函数
            n_estimators=n_estimators,  # 树的数量，需要根据你的数据进行调整
            learning_rate=lr,  # 学习率，需要根据你的数据进行调整
            reg_alpha=C, #  L1正则化，对应SVR的C参数
            min_data_in_leaf=min_data_in_leaf,
            max_depth=max_depth,
            verbose=-1,
            # ...其他LightGBM参数...
        )


        # print ("haha1")
        train_dataset = train_dataset_array[fold]
        test_dataset = test_dataset_array[fold]
            # 自己的逻辑
        train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
        for i, data in enumerate(train_dataloader, 0):
            train_inputs, train_labels, train_ids = data
            lgb_model.fit(train_inputs, train_labels)

            train_lgb_pred = lgb_model.predict(train_inputs)

            r, p_value = pearsonr(train_labels, train_lgb_pred)
            lgb_r2 = r2_score(train_labels, train_lgb_pred)
            lgb_rmse = np.sqrt(mean_squared_error(train_lgb_pred, train_labels))

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels, ids = data
            lgb_pred = lgb_model.predict(inputs)

            r, p_value = pearsonr(labels, lgb_pred)
            lgb_r2 = r2_score(labels, lgb_pred)
            lgb_rmse = np.sqrt(mean_squared_error(lgb_pred, labels))

            print ("lgb r2: ", lgb_r2, " r: ", r, " rmse: ", lgb_rmse)
        record['loss'].append(1-lgb_r2)
        record['R2'].append(lgb_r2)
        record['R'].append(0)
        record['RMSE'].append(lgb_rmse)
        # record['epoch'].append(record_epoch)

        losses.append(1-lgb_r2)
        R2s.append(lgb_r2)
        RMSEs.append(lgb_rmse)
        data_mean = mean[fold]
        data_std = std[fold]
        def inverse_standardize(data, mean, std):
            return data * std + mean
        original_real = inverse_standardize(labels, data_mean, data_std)
        original_lgb_predict = inverse_standardize(lgb_pred, data_mean, data_std)
        # now = int(time.time())
    
        test_df = pd.DataFrame({
            'original_real': original_real,
            'original_predict': original_lgb_predict,
            'standard_real': labels, 
            'original_predict': lgb_pred,
            'gene_id': ids,
        })

        standard_train_lgb = lgb_model.predict(train_inputs)
        train_df = pd.DataFrame({
            'train_real': inverse_standardize(train_labels, data_mean, data_std),
            'train_predict': inverse_standardize(standard_train_lgb, data_mean, data_std),
            'standard_train_real': train_labels,
            'standard_train_predict': standard_train_lgb,
            'gene_id': train_ids,
        })

        total_train_df_arr.append(train_df)
        total_test_df_arr.append(test_df)
    
    record['trail'].append(trial.number)
    record['n_estimators'].append(n_estimators)
    record['min_data_in_leaf'].append(min_data_in_leaf)
    record['max_depth'].append(max_depth)
    record['lr'].append(lr)
    record['C'].append(C)
    record['fold'].append(-1)
    record['loss'].append(np.mean(losses))
    record['R2'].append(np.mean(R2s))
    record['RMSE'].append(np.mean(RMSEs))
    record['R'].append(0)
        
    if (np.mean(R2s) > now_r2):
        now_r2 = np.mean(R2s)
        for i in range(int(k_fold)):
            total_save_url = root_path + 'Result/' + str(trait) + '/Base_model_result/LGB/'
            os.makedirs(root_path + 'Result/' + str(trait) + '/Base_model_result/LGB/', exist_ok=True)
            excel_path = total_save_url + "/" + data_format  + "_" + gene_model + "_" + str(i) + "_test_predict" + '.xlsx'
            total_test_df_arr[i].to_excel(excel_path, index=True)
            excel_path = total_save_url + "/" + data_format + "_" + gene_model + "_" + str(i) + "_train_predict" + '.xlsx'
            total_train_df_arr[i].to_excel(excel_path, index=True)
    

    return np.mean(1-np.mean(R2s))

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

study.optimize(objective, n_trials=int(args.optuna_count))

# 转换为 DataFrame
df_info = pd.DataFrame.from_dict(record, orient='index')

import pickle
save_result_name = args.result_name

# 保存为 CSV 文件
# df_info.to_csv('info_record.csv', index=False)
df_info.to_csv(root_path + 'Result/optuna/'+trait+'_'+save_result_name+ '_LGB_'+str(data_format)+'_' + str(gene_model)+'_record.csv', index_label='key')

# 保存Study对象
joblib.dump(study, root_path + 'Result/optuna/'+trait+'_'+save_result_name + '_LGB_'+str(data_format)+'_' + str(gene_model)+'.pkl')
