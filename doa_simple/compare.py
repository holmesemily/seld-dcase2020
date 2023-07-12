#%% 
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


dump_dir = '../dataset/myfeatures/dump'
og_dir = '../dataset/myfeatures/foa_dev_label'
og_file = os.path.join(og_dir, "fold1_room1_mix001_ov1.csv")
dump_file = os.path.join(dump_dir, "model_cnn_nq_predict_0.csv")
# dump_file2 = os.path.join(dump_dir, "fold1_room1_mix001_ov1_pred_model12_comp.csv")
df_dump = pd.read_csv(dump_file, header=None)
# df_dump2 = pd.read_csv(dump_file2, header=None)
df_og = pd.read_csv(og_file, header=None)
plt.plot(df_dump.iloc[:,1], color='green')
# plt.plot(df_dump2.iloc[:,1], color='red')
plt.plot(df_og.iloc[:,1], color='blue')
plt.show(block=True)
# %%
