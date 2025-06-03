
# root_config
root_path = '/Users/wangyuze/Desktop/github/EggPT_V1.0/'
k_fold = "5"
device = 'cuda:0'
cnn_device = '0'

# 1_1_public_K_Fold config
trait_name = 'DTT'
# label_file = 'label_file'
threshold = "100"

# 2_1_get_pca_csv config
snp_file_name = 'final_5K.csv'

# 2_2_2_workflow_public config
snp_count = 180000
step = 40000

# 3_SNPs_process_public config
aim_num = 1800

# 4_1_MLP config
mlp_lr = 0.001
mlp_bs = 64
mlp_dropout = 0.5

# 4_2_ML config


# 4_3_TE config
te_lr = 0.001
te_bs = 64
te_dropout = 0.5

# 4_4_CNN config
cnn_lr = 0.001
cnn_bs = 64
cnn_dropout = 0.5


# 4_5_LSTM config
lstm_lr = 0.001
lstm_bs = 64
lstm_dropout = 0.5

Stacking_model = [
    'MLP_p_value_1D',
    'MLP_p_value_2D',
    'MLP_p_value_3D',
    'MLP_pca_1D',
    'TE_p_value_1D',
    'TE_p_value_2D',
    'TE_p_value_3D',
    'CNN_p_value_1D',
    'CNN_p_value_2D',
    'CNN_p_value_3D',
    'CNN_pca_1D',
    'LSTM_p_value_1D',
    'LSTM_p_value_2D',
    'LSTM_p_value_3D',
    'ML_p_value_1D',
    'ML_p_value_2D',
    'ML_p_value_3D',
    'ML_pca_1D',
]

# stacking config
gene_count = 1800
temp_gene_count = 1800
aim_index = 0
stacking_lr = 0.001
