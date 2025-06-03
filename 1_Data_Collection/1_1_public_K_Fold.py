#!/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import KFold
import os
import argparse
# from EggPT_v1 import config


parser = argparse.ArgumentParser(description='K_fold')
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--root_path', type=str, default = 'None')
parser.add_argument('--k_fold', type=int, default = 10)
parser.add_argument('--threshold', type=int, default = 100)

args = parser.parse_args()

k_fold = args.k_fold
trait_name = args.trait_name
root_path = args.root_path
threshold = args.threshold

label_file = root_path + '1_Data_Collection/P/' + trait_name+'.xlsx'

def normalize_labels(dataset, indices):
    labels = np.array([dataset[idx][0] for idx in indices])
    normalized_labels = scaler.transform(labels.reshape(-1, 1)).flatten()
    return normalized_labels

def filter_by_label(dataset, indices, threshold):
    # 获取归一化的标签
    normalized_labels = normalize_labels(dataset, indices)
    # 筛选出标签值小于等于阈值的索引和大于阈值的索引
    kept_indices = [idx for idx, label in zip(indices, normalized_labels) if label <= threshold]
    removed_indices = [idx for idx, label in zip(indices, normalized_labels) if label > threshold]
    return kept_indices, removed_indices

class PhenoDataset(Dataset):
    def __init__(self, label_file):
        df = pd.read_excel(label_file, header=0)
        self.labels = df[trait_name].astype('float32').values
        self.data = df
        print("Total number of samples: ", len(self.data))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample_id = self.data['ID'].values[idx]
        label = torch.tensor(self.labels[idx]).to(dtype=torch.float32)
        return label, sample_id

dataset = PhenoDataset(label_file)
label, id = dataset.__getitem__(0)
total_save_url = root_path + '1_Data_Collection/P/' + trait_name+'/10-fold'
os.makedirs(total_save_url, exist_ok=True)

data_mean = []
data_std = []

# 下面是预测最终结果需要的数据，fold设置为-1

all_labels = np.array([dataset[idx][0] for idx in range(len(dataset))])
all_ids = np.array([dataset[idx][1] for idx in range(len(dataset))])

all_scaler = StandardScaler()
all_scaler.fit(all_labels.reshape(-1, 1))



# 归一化all_labels
all_normalized_labels = all_scaler.transform(all_labels.reshape(-1, 1)).flatten()

# 保存所有样本的 gene_id 和标签到 CSV 文件
df = pd.DataFrame({
    'gene_id': all_ids,
    trait_name: all_labels,
})
os.makedirs(total_save_url + '/-1', exist_ok=True)

excel_path = total_save_url + "/" + "-1" + "/" + trait_name +"_train.xlsx"
df.to_excel(excel_path, index=True)
# 到此结束


kfold = KFold(n_splits=k_fold, shuffle=True, random_state=123)

for fold, (train_dataset, test_dataset) in enumerate(kfold.split(dataset)):
    os.makedirs(total_save_url + '/' + str(fold), exist_ok=True)
    dataset = PhenoDataset(label_file)

    print(f'FOLD {fold}')
    print('--------------------------------')
    start = time.time()  
    scaler = StandardScaler()
    train_labels = np.array([dataset[idx][0] for idx in train_dataset])
    test_id = np.array([dataset[idx][1] for idx in test_dataset])

    # 将 numpy 数组转换为 pandas DataFrame
    df1 = pd.DataFrame(test_id, columns=['Test_ID'])
    scaler.fit(train_labels.reshape(-1, 1))
    end1 = time.time()
    data_mean.append(scaler.mean_[0])
    data_std.append(scaler.scale_[0])

    # print("Mean:", data_mean)
    # print("Standard deviation:", data_std)

    end2 = time.time()

    # 过滤训练集和测试集
    filtered_train_indices, removed_train_indices = filter_by_label(dataset, train_dataset, threshold)
    filtered_test_indices, removed_test_indices = filter_by_label(dataset, test_dataset, threshold)

    # 更新训练集和测试集的标签
    train_dataset = Subset(dataset, train_dataset)
    train_dataset.dataset.labels[train_dataset.indices] = normalize_labels(dataset, train_dataset.indices)
    test_dataset = Subset(dataset, test_dataset)
    test_dataset.dataset.labels[test_dataset.indices] = normalize_labels(dataset, test_dataset.indices)
    end3 = time.time()

    # 根据过滤后的索引创建新的训练集和测试集
    filtered_train_dataset = Subset(dataset, filtered_train_indices)
    filtered_test_dataset = Subset(dataset, filtered_test_indices)

    # 提取被删除的 gene_id
    removed_train_gene_ids = [dataset[idx][2] for idx in removed_train_indices]
    removed_test_gene_ids = [dataset[idx][2] for idx in removed_test_indices]

    # 保存被删除的 gene_id 到 CSV 文件
    removed_gene_ids = {'Removed_Train_Gene_IDs': removed_train_gene_ids, 'Removed_Test_Gene_IDs': removed_test_gene_ids}
    df_removed = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in removed_gene_ids.items()]))

    # Now you can create DataLoaders for your train and test sets
    train_dataloader = DataLoader(filtered_train_dataset, batch_size=5000, shuffle=False)
    test_dataloader = DataLoader(filtered_test_dataset, batch_size=5000)
    print ("Check info : Mean - ", scaler.mean_[0], " Std - ", scaler.scale_[0])

    for i, data in enumerate(train_dataloader, 0):
        label, sample_id = data
        print ("Train Check info :  gene_id - ", sample_id[0], " label - ", label[0], " del standard - ", label[0] * scaler.scale_[0] + scaler.mean_[0])
        print ("Train Check info :  gene_id - ", sample_id[1], " label - ", label[1], " del standard - ", label[1] * scaler.scale_[0] + scaler.mean_[0])
        print ("Train Check info :  gene_id - ", sample_id[2], " label - ", label[2], " del standard - ", label[2] * scaler.scale_[0] + scaler.mean_[0])
        df = pd.DataFrame({
            'gene_id': sample_id,
            trait_name: label,
        })
        excel_path = total_save_url + "/" + str(fold) + "/" + trait_name +"_train.xlsx"
        df.to_excel(excel_path, index=True)

    for i, data in enumerate(test_dataloader, 0):
        label, sample_id = data
        print ("Test Check info :  gene_id - ", sample_id[0], " label - ", label[0], " del standard - ", label[0] * scaler.scale_[0] + scaler.mean_[0])
        print ("Test Check info :  gene_id - ", sample_id[1], " label - ", label[1], " del standard - ", label[1] * scaler.scale_[0] + scaler.mean_[0])
        print ("Test Check info :  gene_id - ", sample_id[2], " label - ", label[2], " del standard - ", label[2] * scaler.scale_[0] + scaler.mean_[0])
        df = pd.DataFrame({
            'gene_id': sample_id,
            trait_name: label,
        })
        excel_path = total_save_url + "/" + str(fold) + "/" + trait_name +"_test.xlsx"
        df.to_excel(excel_path, index=True)
    end4 = time.time()
   # 训练完scaler后，你可以打印均值和标准差

# all 数据存到最后一行
data_mean.append(all_scaler.mean_[0])
data_std.append(all_scaler.scale_[0])

df_1 = pd.DataFrame({
            'Mean': data_mean,
            'Std': data_std,
        })
excel_path = total_save_url + "/" + trait_name +"_Mean_Std.xlsx"
df_1.to_excel(excel_path, index=True)

