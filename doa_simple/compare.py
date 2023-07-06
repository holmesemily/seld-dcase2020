#%% 
import pandas as pd
import os
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


dump_dir = '../dataset/myfeatures/dump'
og_dir = '../dataset/myfeatures/foa_dev_label'
og_file = os.path.join(og_dir, "fold1_room1_mix001_ov1.csv")
dump_file = os.path.join(dump_dir, "fold1_room1_mix001_ov1_pred.csv")
df_dump = pd.read_csv(dump_file)
df_og = pd.read_csv(og_file)
plt.plot(df_dump)
plt.plot(df_og)
plt.show(block=True)
# %%
