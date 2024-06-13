import pandas as pd

# 读取txt文件
df = pd.read_csv('/data/users/lijing/PycharmProjects/PLpro_yuan/TSE/from_zhang/V2/results/all_T4SE_train_ESM_features.txt', header=None)

# 保存为csv文件
df.to_csv('/data/users/lijing/PycharmProjects/PLpro_yuan/TSE/T4SE/features/all_T4SE_train_ESM_features.csv', index=False)
# 读取txt文件
df = pd.read_csv('/data/users/lijing/PycharmProjects/PLpro_yuan/TSE/from_zhang/V2/results/all_T4SE_test_ESM_features.txt', header=None)

# 保存为csv文件
df.to_csv('/data/users/lijing/PycharmProjects/PLpro_yuan/TSE/T4SE/features/all_T4SE_test_ESM_features.csv', index=False)