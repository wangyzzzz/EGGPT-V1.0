import os
# import config
import gpu_config as config_old
import toml

config = toml.load("test_config.toml")
# 定义Shell脚本内容

print (config)

snp_pca_csv_path = config["train_parameter"]["root_path"] + "1_Data_Collection/SNP/" + config["train_parameter"]["snp_file_name"]
phenotype_path = config["train_parameter"]["root_path"] + "1_Data_Collection/P/" + config["train_parameter"]["trait"] + "/10-fold"
base_model = config["base_model"]["components"]
data_encoding = config["data_encoding"]["components"]
data_processing = config["data_processing"]["components"]
# Define paths from config
root_path = config["train_parameter"]["root_path"]
trait_name = config["train_parameter"]["trait"]
k_fold = int(config["train_parameter"]["k_fold"])
snp_file_name = config["train_parameter"]["snp_file_name"]
aim_num = str(config["train_parameter"]["aim_num"])
threshold = int(config["train_parameter"]["threshold"])
optuna_num = int(config["train_parameter"]["optuna"])
dataset = str(config["train_parameter"]["dataset"])


mlp_device = str(config["device"]["MLP"])
cnn_device = str(config["device"]["CNN"])
te_device = str(config["device"]["TE"])
lstm_device = str(config["device"]["LSTM"])

#-----------------------------------#
# conda
conda_init = 'eval "$(conda shell.bash hook)"'
rapids_conda_activate = '# conda activate rapids-24.02'
torch1_conda_activate = 'conda activate torch1'
tf_conda_activate = 'conda activate tf_env1'
#-----------------------------------#

#-----------------------------------#
# K-Fold Component
k_fold_component = f"# python {root_path}1_Data_Collection/1_1_public_K_Fold.py --trait_name  {trait_name} --root_path {root_path} --k_fold {k_fold} --threshold {threshold}"
#-----------------------------------#

#-----------------------------------#
# SNP PCA Component
snp_pca_component = ""
snp_pca_cmd = f"# python {root_path}2_Data_Processing/2_1_get_pca_csv.py --trait_name {trait_name} --root_path {root_path} --csv_path {snp_pca_csv_path} --phenotype_path {phenotype_path} --fold "
for i in range(int(config["train_parameter"]["k_fold"])):
    snp_pca_component += f"{snp_pca_cmd}{i}\n"
#-----------------------------------#

#-----------------------------------#
# SNP GWAS Component

gwas_component = ""
for i in range(int(config["train_parameter"]["k_fold"])):
    # gwas_component += f"bash {root_path}no_double_sample_name.sh {root_path} {trait_name} {i} {dataset}\n"
    gwas_component += f"\n"


#-----------------------------------#


#-----------------------------------#
# SNP encoding Component
snp_encoding_component = f"# python {root_path}3_Data_Encoding/3_SNPs_process.py --trait_name {trait_name} --root_path {root_path} --k_fold {k_fold} --snp_file_name {snp_file_name} --aim_num {aim_num}"
#-----------------------------------#

# Define base model component commands
def generate_base_model_commands(model_type, gene_model, root_path, trait_name, k_fold, aim_num, mlp_device=None, device=None, cnn_device=None, default_hp=None):
    commands = []
    # for gene_model in gene_models:
    if model_type == "MLP":
        cmd = f"python {root_path}4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name {trait_name} --optuna {optuna_num} --device {mlp_device} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "RF":
        cmd = f"python {root_path}4_Base_Model_Construction/4_2_RF_optuna.py --trait_name {trait_name} --optuna {optuna_num} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "SVR":
        cmd = f"python {root_path}4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name {trait_name} --optuna {optuna_num} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "LGB":
        cmd = f"python {root_path}4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name {trait_name} --optuna {optuna_num} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "TE":
        cmd = f"python {root_path}4_Base_Model_Construction/4_5_TE_optuna.py --trait_name {trait_name} --optuna {optuna_num} --device {device} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "CNN":
        cmd = f"python {root_path}4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name {trait_name} --optuna {optuna_num} --device {cnn_device} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    elif model_type == "LSTM":
        cmd = f"python {root_path}4_Base_Model_Construction/4_7_LSTM_optuna.py --trait_name {trait_name} --optuna {optuna_num} --device {device} --root_path {root_path} --gene_model {gene_model} --data_format {aim_num if gene_model not in ['PCA', 'GRM'] else 'All'} --k_fold "
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    # commands.extend([f"{cmd}{i}" for i in range(k_fold)])
    commands.extend([f"{cmd}{k_fold} --result_name 6_2"])



    return "\n".join(commands)


mlp_gwas_1D_component = generate_base_model_commands("MLP", "1D", root_path, trait_name, k_fold, aim_num, mlp_device=mlp_device)
mlp_gwas_2D_component = generate_base_model_commands("MLP", "2D", root_path, trait_name, k_fold, aim_num, mlp_device=mlp_device)
mlp_gwas_3D_component = generate_base_model_commands("MLP", "3D", root_path, trait_name, k_fold, aim_num, mlp_device=mlp_device)
mlp_pca_component = generate_base_model_commands("MLP", "PCA", root_path, trait_name, k_fold, aim_num, mlp_device=mlp_device)
mlp_grm_component = generate_base_model_commands("MLP", "GRM", root_path, trait_name, k_fold, aim_num, mlp_device=mlp_device)

rf_gwas_1D_component = generate_base_model_commands("RF", '1D', root_path, trait_name, k_fold, aim_num)
rf_gwas_2D_component = generate_base_model_commands("RF", '2D', root_path, trait_name, k_fold, aim_num)
rf_gwas_3D_component = generate_base_model_commands("RF", '3D', root_path, trait_name, k_fold, aim_num)
rf_pca_component = generate_base_model_commands("RF", 'PCA', root_path, trait_name, k_fold, aim_num)
rf_grm_component = generate_base_model_commands("RF", 'GRM', root_path, trait_name, k_fold, aim_num)

svr_gwas_1D_component = generate_base_model_commands("SVR", '1D', root_path, trait_name, k_fold, aim_num)
svr_gwas_2D_component = generate_base_model_commands("SVR", '2D', root_path, trait_name, k_fold, aim_num)
svr_gwas_3D_component = generate_base_model_commands("SVR", '3D', root_path, trait_name, k_fold, aim_num)
svr_pca_component = generate_base_model_commands("SVR", 'PCA', root_path, trait_name, k_fold, aim_num)
svr_grm_component = generate_base_model_commands("SVR", 'GRM', root_path, trait_name, k_fold, aim_num)

lgb_gwas_1D_component = generate_base_model_commands("LGB", '1D', root_path, trait_name, k_fold, aim_num)
lgb_gwas_2D_component = generate_base_model_commands("LGB", '2D', root_path, trait_name, k_fold, aim_num)
lgb_gwas_3D_component = generate_base_model_commands("LGB", '3D', root_path, trait_name, k_fold, aim_num)
lgb_pca_component = generate_base_model_commands("LGB", 'PCA', root_path, trait_name, k_fold, aim_num)
lgb_grm_component = generate_base_model_commands("LGB", 'GRM', root_path, trait_name, k_fold, aim_num)

te_gwas_1D_component = generate_base_model_commands("TE", '1D', root_path, trait_name, k_fold, aim_num, device=te_device)
te_gwas_2D_component = generate_base_model_commands("TE", '2D', root_path, trait_name, k_fold, aim_num, device=te_device)
te_gwas_3D_component = generate_base_model_commands("TE", '3D', root_path, trait_name, k_fold, aim_num, device=te_device)
# te_pca_component = generate_base_model_commands("TE", 'PCA', root_path, trait_name, k_fold, aim_num, device=config_old.device)
# te_grm_component = generate_base_model_commands("TE", 'GRM', root_path, trait_name, k_fold, aim_num, device=config_old.device)

cnn_gwas_1D_component = generate_base_model_commands("CNN", '1D', root_path, trait_name, k_fold, aim_num, cnn_device=cnn_device)
cnn_gwas_2D_component = generate_base_model_commands("CNN", '2D', root_path, trait_name, k_fold, aim_num, cnn_device=cnn_device)
cnn_gwas_3D_component = generate_base_model_commands("CNN", '3D', root_path, trait_name, k_fold, aim_num, cnn_device=cnn_device)
cnn_pca_component = generate_base_model_commands("CNN", 'PCA', root_path, trait_name, k_fold, aim_num, cnn_device=cnn_device)
cnn_grm_component = generate_base_model_commands("CNN", 'GRM', root_path, trait_name, k_fold, aim_num, cnn_device=cnn_device)

lstm_gwas_1D_component = generate_base_model_commands("LSTM", '1D', root_path, trait_name, k_fold, aim_num, device=lstm_device)
lstm_gwas_2D_component = generate_base_model_commands("LSTM", '2D', root_path, trait_name, k_fold, aim_num, device=lstm_device)
lstm_gwas_3D_component = generate_base_model_commands("LSTM", '3D', root_path, trait_name, k_fold, aim_num, device=lstm_device)
# lstm_pca_component = generate_base_model_commands("LSTM", 'PCA', root_path, trait_name, k_fold, aim_num, device=config_old.device)
# lstm_grm_component = generate_base_model_commands("LSTM", 'GRM', root_path, trait_name, k_fold, aim_num, device=config_old.device)


#-----------------------------------#
# Stacking Component
result_base_model_list = ""
for model in base_model:
    result_base_model_list = result_base_model_list + model + " "

result_data_encoding_list = ""
for encode in data_encoding:
    result_data_encoding_list = result_data_encoding_list + encode + " "

result_merge_component = f"python {root_path}5_Meta_Model_Construction/result_merge.py --root_path {root_path} --trait_name  {trait_name} --k_fold {k_fold}  --base_model {base_model} --data_encoding {data_encoding} --data_processing {data_processing}"

model_pruning_component = ""
model_pruning_cmd = f"python {root_path}5_Meta_Model_Construction/merge_MLP_reduce.py --root_path {root_path} --trait_name {trait_name} --k_fold {k_fold} --aim_index "

for i in range(int(21)):
    model_pruning_component += f"{model_pruning_cmd}{i+1}\n"
#-----------------------------------#
rename_component = ""
# 定义文件名和时间戳
file_o="/home/user/code/git/EggPT_V1.0/Result/MR_result/" + config["train_parameter"]["trait"] + ".csv"
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 格式化时间戳
timestamp = now.strftime("%Y%m%d_%H%M%S")

# 打印时间戳
print(timestamp)

# 提取文件的扩展名和基本名
extension=config["train_parameter"]["trait"]
basename="csv"

# 使用 mv 命令重命名文件
rename_component = "mv " + file_o + " /home/user/code/git/EggPT_V1.0/Result/MR_result/" + extension +"_" + timestamp+"." +basename

rename_component_1 = "mv " + "/home/user/code/git/EggPT_V1.0/Result/"+config["train_parameter"]["trait"]+"/Base_model_result" + " /home/user/code/git/EggPT_V1.0/Result/"+config["train_parameter"]["trait"]+"/Base_model_result"+"_" + timestamp

config_cmd_map = {
    'SNP_PCA': snp_pca_component,
    'SNP_GWAS': gwas_component,
    'SNP_GWAS_Encoding': snp_encoding_component,

    'SNP_PCA_MLP': mlp_pca_component,
    'SNP_GRM_MLP': mlp_grm_component,
    'SNP_GWAS_1D_MLP': mlp_gwas_1D_component,
    'SNP_GWAS_2D_MLP': mlp_gwas_2D_component,
    'SNP_GWAS_3D_MLP': mlp_gwas_3D_component,

    'SNP_PCA_RF': rf_pca_component,
    'SNP_GRM_RF': rf_grm_component,
    'SNP_GWAS_1D_RF': rf_gwas_1D_component,
    'SNP_GWAS_2D_RF': rf_gwas_2D_component,
    'SNP_GWAS_3D_RF': rf_gwas_3D_component,

    'SNP_PCA_SVR': svr_pca_component,
    'SNP_GRM_SVR': svr_grm_component,
    'SNP_GWAS_1D_SVR': svr_gwas_1D_component,
    'SNP_GWAS_2D_SVR': svr_gwas_2D_component,
    'SNP_GWAS_3D_SVR': svr_gwas_3D_component,

    'SNP_PCA_LGB': lgb_pca_component,
    'SNP_GRM_LGB': lgb_grm_component,
    'SNP_GWAS_1D_LGB': lgb_gwas_1D_component,
    'SNP_GWAS_2D_LGB': lgb_gwas_2D_component,
    'SNP_GWAS_3D_LGB': lgb_gwas_3D_component,

    'SNP_GWAS_1D_TE': te_gwas_1D_component,
    'SNP_GWAS_2D_TE': te_gwas_2D_component,
    'SNP_GWAS_3D_TE': te_gwas_3D_component,

    'SNP_PCA_CNN': cnn_pca_component,
    'SNP_GRM_CNN': cnn_grm_component,
    'SNP_GWAS_1D_CNN': cnn_gwas_1D_component,
    'SNP_GWAS_2D_CNN': cnn_gwas_2D_component,
    'SNP_GWAS_3D_CNN': cnn_gwas_3D_component,

    'SNP_GWAS_1D_LSTM': lstm_gwas_1D_component,
    'SNP_GWAS_2D_LSTM': lstm_gwas_2D_component,
    'SNP_GWAS_3D_LSTM': lstm_gwas_3D_component,
}

final_cmd = [
    conda_init,
    rapids_conda_activate,
    k_fold_component
]

# 特征工程部分
feature_en = []
for layer1_com in config["data_collection"]["components"]: 
    for layer2_com in config["data_processing"]["components"]:
        final_cmd += [layer1_com + "_" + layer2_com]
        if layer2_com == 'GWAS':
            final_cmd += [torch1_conda_activate]
            final_cmd += [layer1_com + "_" + layer2_com + "_Encoding"]
            for layer3_com in config["data_encoding"]["components"]:
                feature_en += [layer1_com + "_" + layer2_com + "_" + layer3_com]
        else :
            feature_en += [layer1_com + "_" + layer2_com]
                
# + 模型结构
final_cmd += [torch1_conda_activate]
for feature_com in feature_en:
    for layer4_com in config["base_model"]["components"]:
        final_cmd += [feature_com + "_" + layer4_com]

# + 5 layer

# final_cmd += [result_merge_component, model_pruning_component]


# 定义Shell脚本文件名
shell_script_filename = "pipe_line.sh"

# 创建并写入Shell脚本文件
with open(shell_script_filename, 'w') as file:
    for cmd in final_cmd:
        if cmd in config_cmd_map.keys():
            file.write(config_cmd_map[cmd])
        else :
            file.write(cmd)
        file.write("\n")

# 设置Shell脚本文件为可执行
os.chmod(shell_script_filename, 0o755)

print(f"Shell script '{shell_script_filename}' has been created and made executable.")
