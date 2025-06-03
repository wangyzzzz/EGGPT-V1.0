#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from functools import reduce
import os
import argparse

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')

parser.add_argument('--root_path', type=str, default = 'None')
parser.add_argument('--gene_count', type=int , default = 1800)
parser.add_argument('--trait_name', type=str, default = 'None')
parser.add_argument('--temp_gene_count', type=int, default = 1800)
parser.add_argument('--k_fold', type=str, default = '10')
parser.add_argument('--base_model', nargs='+', type=str, default = ['MLP'])
parser.add_argument('--data_encoding', nargs='+', type=str, default = ['1D'])
parser.add_argument('--data_processing', nargs='+', type=str, default = ['GWAS'])



args = parser.parse_args()

indexs = ['train', 'test']
k_fold = int(args.k_fold)
trait = args.trait_name
gene_count = str(args.gene_count)
temp_gene_count = str(args.temp_gene_count)
root_path = args.root_path
os.makedirs(root_path + 'Result/'+ trait + '/model_result/', exist_ok=True)
os.makedirs(root_path + 'Result/'+ trait + '/merge_result/', exist_ok=True)
models = args.base_model
print (models)
encoding = args.data_encoding
processing = args.data_processing

for index in indexs:
    for model in models:
        for fold in range(k_fold):
            dataframes = []
            suffixes = []
            encoding_num = len(encoding)
            for pci in processing:
                if pci == 'GWAS':
                    for encode in encoding:
                        file_encode = pd.read_excel(root_path + 'Result/'+ trait + '/Base_model_result/' + model + '/' + gene_count + '_' + encode + '_' + str(fold) + '_' + index +'_predict.xlsx')
                        # gene_id重复的只保留一行
                        file_encode = file_encode.drop_duplicates(subset=['gene_id'])
                        dataframes.append(file_encode)
                        suffixes.append('_'+encode)
                if (pci == 'PCA' or pci == 'GRM') and model in ['MLP', 'RF', 'LGB', 'SVR', 'CNN']:
                    # if pci == 'GRM' and model in ['CNN', 'RF', 'SVR']:
                    #     continue
                    file_PCA = pd.read_excel(root_path + 'Result/'+ trait + '/Base_model_result/' + model + '/All_' + pci + '_' + str(fold) + '_' + index +'_predict.xlsx')
                    file_PCA = file_PCA.drop_duplicates(subset=['gene_id'])
                    dataframes.append(file_PCA)
                    suffixes.append('_' + pci)

            # 重命名每个DataFrame中的列，添加相应的后缀，但跳过'gene_id'列
            renamed_dataframes = []
            for df, suffix in zip(dataframes, suffixes):
                df_renamed = df.rename(columns=lambda x: x + suffix if x != 'gene_id' else x)
                renamed_dataframes.append(df_renamed)
            
            # 使用reduce函数进行连续合并
            merged_train = reduce(lambda left, right: pd.merge(left, right, on='gene_id'), renamed_dataframes)

            # 将合并后的DataFrame保存为新的Excel文件
            train_output_filename = root_path + 'Result/'+ trait + '/model_result/' + model + '_merged_' + index + '_data' + str(fold) + '.xlsx'
            merged_train.to_excel(train_output_filename, index=False)


for index in indexs:
    for fold in range(k_fold):
        dataframes = []
        suffixes = []
        for model in models:
            merge_train = pd.read_excel(root_path + 'Result/'+ trait + '/model_result/' + model + '_merged_' + index + '_data' + str(fold) + '.xlsx')
            dataframes.append(merge_train)
            suffixes.append('_'+model)

        renamed_dataframes = []
        for df, suffix in zip(dataframes, suffixes):
            df_renamed = df.rename(columns=lambda x: x + suffix if x != 'gene_id' else x)
            renamed_dataframes.append(df_renamed)
        # 使用reduce函数进行连续合并
        final_merged_train = reduce(lambda left, right: pd.merge(left, right, on='gene_id'), renamed_dataframes)
        train_output_filename = root_path + 'Result/'+ trait + '/merge_result/'+'/Final_merged_' + index + '_data' + str(fold) +'.xlsx'
        final_merged_train.to_excel(train_output_filename, index=False)


