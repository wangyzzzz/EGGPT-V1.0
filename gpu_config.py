
# root_config
root_path = '/home/user/code/git/EGGPT_optuna/'
k_fold = "10"
device = 'cuda:1'
cnn_device = '1'

# 1_1_public_K_Fold config
trait_name = 'Cotton_FibMic_17_18'
# label_file = 'label_file'
threshold = "100"

# 2_1_get_pca_csv config
snp_file_name = '1245Cotton_180K.csv'

# 2_2_2_workflow_public config
snp_count = 180000
step = 40000

# 3_SNPs_process_public config
aim_num = 1800

# 4_1_MLP config
mlp_lr = 0.0001
mlp_bs = 64
mlp_dropout = 0.3
mlp_device = 'cuda:1'

# 4_2_ML config


# 4_3_TE config
te_lr = 0.0001
te_bs = 64
te_dropout = 0.3

# 4_4_CNN config
cnn_lr = 0.0001
cnn_bs = 64
cnn_dropout = 0.3

# 4_5_LSTM config
lstm_lr = 0.0001
lstm_bs = 64
lstm_dropout = 0.3

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
stacking_lr = 0.0001
merge_device = 'cuda:0'
