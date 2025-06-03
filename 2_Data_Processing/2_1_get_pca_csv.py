import argparse
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.decomposition import PCA as PCA_CPU
from sklearn.preprocessing import StandardScaler


def process_PCA_CPU_MAX(df, test_df):
    # 选择数据类型为float32的列
    float_cols = df.select_dtypes(include=['float32']).columns
    # 对这些列进行标准化
    scaler = StandardScaler()
    float_data_scaled = scaler.fit_transform(df[float_cols].values)

    # 初始化PCA模型
    pca = PCA_CPU(n_components=None, random_state=42)
    # 训练PCA模型并转换数据
    scores = pca.fit_transform(float_data_scaled)

    test_float_cols = test_df.select_dtypes(include=['float32']).columns

    # 使用相同的缩放器来转换新数据
    new_data_scaled = scaler.transform(test_df[test_float_cols].values)

    # 使用训练好的PCA模型来转换新数据
    new_scores = pca.transform(new_data_scaled)

    # 获取解释率和累计解释率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # 找到累计解释率超过95%的最少组件数
    n_components_95 = (cumulative_explained_variance >= 0.95).argmax() + 1

    # 将得分转换为DataFrame，并为列命名，仅保留到n_components_95
    scores_df = pd.DataFrame(scores[:, :n_components_95], columns=['PC' + str(x) for x in range(1, n_components_95+1)])
    test_scores_df = pd.DataFrame(new_scores[:, :n_components_95], columns=['PC' + str(x) for x in range(1, n_components_95+1)])
    # 打印累计解释率
    print("累计解释率: ", cumulative_explained_variance)
    print(f"选择的组件数（解释率超过95%）: {n_components_95}")
    return scores_df, test_scores_df


warnings.filterwarnings('ignore', 'Expected')
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--csv_path', type=str, default = '/home/user/code/multi_omics_predict/data/all_2229_m0.05g0.05.vcf.vcf')
parser.add_argument('--phenotype_path', type=str, default = '')
parser.add_argument('--fold', type=int, default = 0)
parser.add_argument('--trait_name', type=str, default = '50000')
parser.add_argument('--GPU', type=str, default = '0')
parser.add_argument('--root_path', type=str, default = '0')

args = parser.parse_args()


fold = str(args.fold)
phenotype_path = str(args.phenotype_path) + '/' + fold + '/' + str(args.trait_name) + '_train.xlsx'
test_phenotype_path = str(args.phenotype_path) + '/' + fold + '/' + str(args.trait_name) + '_test.xlsx'

phenotype_df = pd.read_excel(phenotype_path)
test_phenotype_df = pd.read_excel(test_phenotype_path)

# 读取CSV文件
csv_data = pd.read_csv(args.csv_path, header=None)

# 生成SNP IDs
snp_ids = ['SNP' + str(i) for i in range(csv_data.shape[0] - 1)]
samples = csv_data.iloc[0, :].values

# 根据表型数据筛选样本列索引
sample_column = 'gene_id'
sample_indices = [list(samples).index(s) for s in phenotype_df[sample_column] if s in samples]

test_sample_indices = [list(samples).index(s) for s in test_phenotype_df[sample_column] if s in samples]

# 筛选基因型数据
filtered_genotype_data = csv_data.iloc[:, 1].values  # +1 是因为我们添加了SNP_ID列
test_filtered_genotype_data = csv_data.iloc[:, 1]

# 遍历所有变异，添加基因型数据到DataFrame中
count = 0

# 在循环外部初始化一个列表，用于存储基因型数值
genotype_data = []

for i in range(len(sample_indices)):
    # 将本次循环得到的基因型数值添加到列表中
    genotype_data.append(csv_data.iloc[1:, sample_indices[i]].values)
    count += 1
print (count)

# 遍历所有变异，添加基因型数据到DataFrame中
count = 0

# 在循环外部初始化一个列表，用于存储基因型数值
test_genotype_data = []

for i in range(len(test_sample_indices)):
    test_genotype_data.append(csv_data.iloc[1:, test_sample_indices[i]].values)
    count += 1

# 将列表转换为 NumPy 数组并检查类型
genotype_data = np.array(genotype_data, dtype=np.float32)
test_genotype_data = np.array(test_genotype_data, dtype=np.float32)

# 将列表转换为 Pandas DataFrame，并指定列名
genotype_data_df = pd.DataFrame(genotype_data, dtype='float32', columns=snp_ids)
test_genotype_data_df = pd.DataFrame(test_genotype_data, dtype='float32', columns=snp_ids)


pca_df, test_pca_df = process_PCA_CPU_MAX(genotype_data_df, test_genotype_data_df) #  直接使用 Pandas DataFrame


os.makedirs(args.root_path + 'Result/' + args.trait_name + '/SNPs/encode/PCA/', exist_ok=True)

pca_df.to_csv(args.root_path + 'Result/' + args.trait_name + '/SNPs/encode/PCA'+'/train_' + fold + ".csv", index = False)
test_pca_df.to_csv(args.root_path + 'Result/' + args.trait_name + '/SNPs/encode/PCA'+'/test_' + fold + ".csv", index = False)


