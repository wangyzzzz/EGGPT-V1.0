import optuna
import joblib
import matplotlib.pyplot as plt

# study = joblib.load('11_7_2_Rice_GL.pkl')
study = joblib.load('/home/user/code/git/EGGPT_optuna/Result/optuna/Rice_YD_HZ_15_LSTM_1800_3D.pkl')
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# 可视化优化历史
fig1 = optuna.visualization.plot_optimization_history(study)

# 保存图像
fig1.write_image("optimization_history.png")
# fig1.show()

# 可视化参数重要性
fig2 = optuna.visualization.plot_param_importances(study)
fig2.write_image("param_importances.png")
fig2.show()

# 可视化平行坐标图
fig3 = optuna.visualization.plot_parallel_coordinate(study)
fig3.write_image("parallel_coordinate.png")
fig3.show()
