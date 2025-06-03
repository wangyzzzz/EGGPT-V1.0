import joblib

# with open("/home/user/code/git/EGGPT_optuna/Result/optuna/pigeon_w_15_TE_1800_3D.pkl", "rb") as f:
#     study = pickle.load(f)

study = joblib.load('/home/user/code/git/EGGPT_optuna/Result/optuna/Rice_YD_HZ_15_LSTM_1800_3D.pkl')

for trial in study.trials:
    print(f"Trial {trial.number}: Started at {trial.datetime_start}, Completed at {trial.datetime_complete}")
