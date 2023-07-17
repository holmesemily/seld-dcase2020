#%% 
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

min_i = 28.056959861353036
max_i = -73.14660305467869

dump_dir = '../dataset/myfeatures/dump'
og_dir = '../dataset/myfeatures/foa_dev_label'
og_file = os.path.join(og_dir, "fold1_room1_mix001_ov1.csv")
dump_file = os.path.join(dump_dir, "model_cnn_nq_norm_td_predict_1_b256_tanh.csv")
dump_file2 = os.path.join(dump_dir, "model_cnn_nq_norm_td_predict_1_b256_tanh_nobm.csv")
df_dump = pd.read_csv(dump_file, header=None)
df_dump2 = pd.read_csv(dump_file2, header=None)
df_og = pd.read_csv(og_file, header=None)/40
plt.plot(df_dump.iloc[:,1], color='green')
plt.plot(df_dump2.iloc[:,1], color='red')
plt.plot(df_og.iloc[:,1], color='blue')
plt.show(block=True)
# %%
