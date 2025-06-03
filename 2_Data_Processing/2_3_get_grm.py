import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--k_fold', type=int, default = 10)
parser.add_argument('--trait_name', type=str, default = '50000')
parser.add_argument('--root_path', type=str, default = '0')
parser.add_argument('--grm_file_name', type=str, default = '0')

args = parser.parse_args()


trait = args.trait_name
total_grm_dir = args.root_path + '1_Data_Collection/GRM/' + args.grm_file_name
aim_dir = args.root_path + 'Result/'+trait+'/SNPs/encode/GRM/'

os.makedirs(aim_dir, exist_ok=True)
grm_df = pd.read_excel(total_grm_dir, index_col=0)

for i in range(args.k_fold):
    i = i
    train_gene_dir = aim_dir + 'train_' + str(i) + '.csv'
    test_gene_dir = aim_dir + 'test_' + str(i) + '.csv'
    train_label_file = '/home/user/code/git/EGGPT_optuna/1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_train.xlsx'
    test_label_file = '/home/user/code/git/EGGPT_optuna/1_Data_Collection/P/'+trait+'/10-fold/' + str(i) + '/'+trait+'_test.xlsx' 
    train_df = pd.read_excel(train_label_file)
    # train_labels = train_df[str(trait)].values
    train_gene_id = train_df['gene_id'].astype(str).values
    # print (train_gene_id)
    # print (grm_df)
    # 同时也转换 grm_df 的行和列索引为字符串
    grm_df.index = grm_df.index.astype(str)
    grm_df.columns = grm_df.columns.astype(str)
    train_total_gene = grm_df.loc[train_gene_id, train_gene_id]
    train_total_gene.to_csv(train_gene_dir)
    test_df = pd.read_excel(test_label_file)
    # test_labels = test_df[str(trait)].values
    test_gene_id = test_df['gene_id'].astype(str).values
    test_total_gene = grm_df.loc[test_gene_id, train_gene_id]
    test_total_gene.to_csv(test_gene_dir)
