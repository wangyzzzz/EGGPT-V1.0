import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--k_fold', type=int, default = 10)
parser.add_argument('--trait_name', type=str, default = '50000')
parser.add_argument('--root_path', type=str, default = '0')
parser.add_argument('--snp_file_name', type=str, default = '0')
parser.add_argument('--aim_num', type=int, default = 1800)

args = parser.parse_args()

aim_dir = args.root_path + 'Result/' + args.trait_name + '/SNPs/encode/' 
GWAS_result_path = args.root_path + 'Result/' + args.trait_name + '/GWAS/'
trait = args.trait_name
gene_csv_path = args.root_path + '1_Data_Collection/SNP/' + args.snp_file_name
aim_num = args.aim_num

def custom_encode_all(X):
    # 对 X 中的每个值进行自定义编码
    # 这里假设 X 是一个 numpy 数组
    encoded = []
    for x in X:
        if x == -1:
            encoded.append([0, 0, 0])  # -1 编码为 [1, 0]
        elif x == 1:
            encoded.append([0, 1, 0])  # 1 编码为 [0, 1]
        elif x == 2:
            encoded.append([0, 0, 1])  # 2 编码为 [0.5, 0.5]
        elif x == 0:
            encoded.append([1, 0, 0])
        else:
            raise ValueError("Unknown category: {}".format(x))
    return np.array(encoded)

def custom_encode_additive(X):
    # 对 X 中的每个值进行自定义编码
    # 这里假设 X 是一个 numpy 数组
    encoded = []
    for x in X:
        if x == 0:
            encoded.append([1, 0])  # -1 编码为 [1, 0]
        elif x == 2:
            encoded.append([0, 1])  # 1 编码为 [0, 1]
        elif x == 1:
            encoded.append([0.5, 0.5])  # 2 编码为 [0.5, 0.5]
        elif x == -1:
            encoded.append([0, 0])
        else:
            raise ValueError("Unknown category: {}".format(x))
    return np.array(encoded)

def custom_encode_1D(X):
    encoded = []
    for x in X:
        if x == 0:
            encoded.append([0])  # -1 编码为 [1, 0]
        elif x == 2:
            encoded.append([2])  # 1 编码为 [0, 1]
        elif x == 1:
            encoded.append([1])  # 2 编码为 [0.5, 0.5]
        elif x == -1:
            encoded.append([-1])
        else:
            raise ValueError("Unknown category: {}".format(x))
    return np.array(encoded)

# 读取CSV文件（假设第一行是样本数据的列索引）
csv_data = pd.read_csv(gene_csv_path, usecols=range(0, 10), header=None)

# 生成SNP IDs
snp_id = ['SNP' + str(i) for i in range(csv_data.shape[0] - 1)]

snps_count_arr = [aim_num]

for fold in range(args.k_fold + 1):
    fold = fold - 1
    csv_file_path = GWAS_result_path + str(trait) + '_' + str(fold) + '.csv'
    df = pd.read_csv(csv_file_path)

    for snps_count in snps_count_arr:
        print (snps_count)
        cutoff = snps_count
        # 根据B列排序并获取top%的SNP名字
        top_snps = df.sort_values(by='P', ascending=False).tail(cutoff)['SNP'].tolist()

        # 找出我们感兴趣的SNP在VCF文件中的索引位置
        # 第一行是样本名，所以我们从第二行开始
        indices_of_interest = [i + 1 for i, variant_id in enumerate(snp_id) if variant_id in top_snps]
        aim_indices = [0] + indices_of_interest
        print (len(aim_indices))
        # 计算要跳过的行号
        skip_rows = set(range(len(snp_id) + 1)) - set(aim_indices)

        # 读取 CSV 文件，跳过不感兴趣的行
        csv_data = pd.read_csv(gene_csv_path, skiprows=lambda x: x in skip_rows, header=None)
        
        # csv_data = pd.read_csv(gene_csv_path, usecols=aim_indices, header=None)
        id = csv_data.iloc[0, :].values  # 列索引从1开始，因为第0列是SNP_ID
        # print (id)
        snp_values = csv_data.iloc[1:, :].values
        print(snp_values.shape)

        # 选择的切割长度
        cut_length = len(snp_values)
        for sample_index in range(snp_values.shape[1]):
            cuts_ignore = []
            cuts_all = []
            cuts_additive = []
            cuts_dominant = []
            snps = snp_values[:, sample_index]
            cut_count = len(snps) // cut_length
            for i in range(cut_count):
                cut = snps[i*cut_length:(i+1)*cut_length]
                cut = np.array([int(float(item)) for item in cut])
                one_hot_cut_all = custom_encode_all(cut)
                one_hot_cut_additive = custom_encode_additive(cut)
                cuts_all.append(one_hot_cut_all)
                cuts_additive.append(one_hot_cut_additive)

            # 如果SNP数量不能被切割长度整除，生成一个额外的片段包含剩余的SNP，用0填充不足的部分
            if len(snps) % cut_length != 0:
                start = cut_count * cut_length
                end = len(snps)
                extra_cut = np.full(cut_length, 0)
                extra_cut[:end-start] = snps[start:end]
                one_hot_cut_all_extra = custom_encode_all(extra_cut)
                one_hot_cut_additive_extra = custom_encode_additive(extra_cut)
                cuts_all.append(one_hot_cut_all_extra)
                cuts_additive.append(one_hot_cut_additive_extra)

            id_temp = id[sample_index]
            os.makedirs(aim_dir + '2D/One_hot/'+ str(fold), exist_ok=True)
            os.makedirs(aim_dir + '3D/One_hot/'+ str(fold), exist_ok=True)

            np.savez(aim_dir + '2D/One_hot/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_additive)
            np.savez(aim_dir + '3D/One_hot/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_all)

        print (snps_count, 'One_hot, ', 'Fold:', fold, ' over')

        if snps_count == 1800:
            cut_length = 60
        elif snps_count == 720:
            cut_length = 36

        for sample_index in range(snp_values.shape[1]):
            cuts_ignore = []
            cuts_all = []
            cuts_additive = []
            cuts_dominant = []
            cuts_1D = []

            snps = snp_values[:, sample_index]
            cut_count = len(snps) // cut_length
            for i in range(cut_count):
                cut = snps[i*cut_length:(i+1)*cut_length]
                cut = np.array([int(float(item)) for item in cut])
                one_hot_cut_all = custom_encode_all(cut)
                one_hot_cut_additive = custom_encode_additive(cut)
                cuts_all.append(one_hot_cut_all)
                cuts_additive.append(one_hot_cut_additive)
                encode_1D = custom_encode_1D(cut)
                cuts_1D.append(encode_1D)
            # 如果SNP数量不能被切割长度整除，生成一个额外的片段包含剩余的SNP，用0填充不足的部分
            if len(snps) % cut_length != 0:
                start = cut_count * cut_length
                end = len(snps)
                extra_cut = np.full(cut_length, 0)
                extra_cut[:end-start] = snps[start:end]
                one_hot_cut_all_extra = custom_encode_all(extra_cut)
                one_hot_cut_additive_extra = custom_encode_additive(extra_cut)
                cuts_all.append(one_hot_cut_all_extra)
                cuts_additive.append(one_hot_cut_additive_extra)
                encode_1D_extra = custom_encode_1D(extra_cut)
                cuts_1D.append(encode_1D_extra)
            id_temp = id[sample_index]

            os.makedirs(aim_dir + '2D/TE/'+ str(fold), exist_ok=True)
            os.makedirs(aim_dir + '3D/TE/'+ str(fold), exist_ok=True)
            os.makedirs(aim_dir + '1D/TE/'+ str(fold), exist_ok=True)

            np.savez(aim_dir + '2D/TE/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_additive)
            np.savez(aim_dir + '3D/TE/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_all)
            np.savez(aim_dir + '1D/TE/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_1D)

        print (snps_count, 'TE, ', 'Fold:', fold, ' over')
        cut_length = len(snp_values)
        for sample_index in range(snp_values.shape[1]):
            cuts_1D = []
            snps = snp_values[:, sample_index]
            cut_count = len(snps) // cut_length
            for i in range(cut_count):
                cut = snps[i*cut_length:(i+1)*cut_length]
                cut = np.array([int(float(item)) for item in cut])

                encode_1D = custom_encode_1D(cut)
                cuts_1D.append(encode_1D)

            # 如果SNP数量不能被切割长度整除，生成一个额外的片段包含剩余的SNP，用0填充不足的部分
            if len(snps) % cut_length != 0:
                start = cut_count * cut_length
                end = len(snps)
                extra_cut = np.full(cut_length, 0)
                extra_cut[:end-start] = snps[start:end]

                encode_1D_extra = custom_encode_1D(extra_cut)
                cuts_1D.append(encode_1D_extra)
            id_temp = id[sample_index]

            os.makedirs(aim_dir + '1D/One_hot/'+ str(fold), exist_ok=True)
            np.savez(aim_dir + '1D/One_hot/' + str(fold) + '/' +  str(id_temp) +'.npz', *cuts_1D)
        print (snps_count, '1D, ', 'Fold:', fold, ' over')