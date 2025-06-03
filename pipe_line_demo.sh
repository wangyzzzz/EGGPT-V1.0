eval "$(conda shell.bash hook)"
# conda activate rapids-24.02
# python /home/user/code/git/EGGPT-V1.0/1_Data_Collection/1_1_public_K_Fold.py --trait_name  Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --k_fold 10 --threshold 100
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 0
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 1
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 2
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 3
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 4
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 5
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 6
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 7
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 8
# python /home/user/code/git/EGGPT-V1.0/2_Data_Processing/2_1_get_pca_csv.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --csv_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/SNP/1404maize_180K.csv --phenotype_path /home/user/code/git/EGGPT-V1.0/1_Data_Collection/P/Maize_PH/10-fold --fold 9












conda activate torch1
# python /home/user/code/git/EGGPT-V1.0/3_Data_Encoding/3_SNPs_process.py --trait_name Maize_PH --root_path /home/user/code/git/EGGPT-V1.0/ --k_fold 10 --snp_file_name 1404maize_180K.csv --aim_num 1800
SNP_GRM
conda activate torch1
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name Maize_PH --optuna 40 --device cpu --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model PCA --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_2_RF_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model PCA --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model PCA --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model PCA --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model PCA --data_format All --k_fold 10 --result_name 6_2
SNP_PCA_LSTM
SNP_PCA_TE
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name Maize_PH --optuna 40 --device cpu --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_2_RF_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_7_LSTM_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_5_TE_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 1D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name Maize_PH --optuna 40 --device cpu --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_2_RF_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_7_LSTM_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_5_TE_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 2D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name Maize_PH --optuna 40 --device cpu --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_2_RF_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_7_LSTM_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_5_TE_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model 3D --data_format 1800 --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_1_MLP_optuna.py --trait_name Maize_PH --optuna 40 --device cpu --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model GRM --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_2_RF_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model GRM --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_3_SVR_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model GRM --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_4_LGB_optuna.py --trait_name Maize_PH --optuna 40 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model GRM --data_format All --k_fold 10 --result_name 6_2
python /home/user/code/git/EGGPT-V1.0/4_Base_Model_Construction/4_6_1DCNN_optuna.py --trait_name Maize_PH --optuna 40 --device cuda:0 --root_path /home/user/code/git/EGGPT-V1.0/ --gene_model GRM --data_format All --k_fold 10 --result_name 6_2
SNP_GRM_LSTM
SNP_GRM_TE
