import os
import pandas as pd
import numpy as np



# Data Loading
file = "/Users/Dennis Pkemoi/OneDrive/Desktop/College Education/2020 NESBE Research internship/Methods and work/Code and dependencies/model1.h5"
filepath = os.path.dirname(file)
dataset = []
dataset_h = []
with os.scandir('healthy Data') as health:
    for entry in health:
        data = pd.read_csv('healthy Data/' + entry.name, sep='\t', header=None)
        data.columns = ['s1', 's2', 's3', 's4', 's5']
        data.drop(data.columns[[4]], axis=1, inplace=True)
        f_name = int(entry.name[5:6])
        idd = np.full(len(data), f_name)
        data['id'] = idd
        data['anomaly'] = np.full(len(data), 0, dtype=int)
        dataset.append(data)
        dataset_h.append(data)
with os.scandir('Data') as entries:
    for entry in entries:
        data = pd.read_csv('Data/' + entry.name, sep='\t', header=None)
        data.columns = ['s1', 's2', 's3', 's4', 's5']
        data.drop(data.columns[[4]], axis=1, inplace=True)
        f_name = int(entry.name[5:6])
        idd = np.full(len(data), f_name)
        data['id'] = idd
        data['anomaly'] = np.full(len(data), 1, dtype=int)
        dataset.append(data)
dataset = pd.concat(dataset, ignore_index=True)
dataset = dataset.iloc[np.random.permutation(dataset.index)].reset_index(drop=True)
dataset_h = pd.concat(dataset_h, ignore_index=True)
# dataset_h = dataset_h.iloc[np.random.permutation(dataset_h.index)].reset_index(drop=True)
