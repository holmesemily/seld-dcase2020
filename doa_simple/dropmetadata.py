import pandas as pd 
import os

directory = '../dataset/metadata_dev'
col = ['frames', 'event', 'x', 'y', 'z']

for filename in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory, filename), names=col, encoding = "ISO-8859-1")
    df.drop('event', axis=1, inplace=True)
    df.to_csv(os.path.join(directory, filename), encoding = 'utf-8', index=False, header=False)