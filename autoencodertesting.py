import keras
import pandas as pd
import numpy as np
from dataprocessor import gen_sequence
from sklearn import preprocessing
import matplotlib.pyplot as plt
model = keras.models.load_model('FaultModel.h5')
tester = pd.read_csv('Data/b30hz0.txt', sep="\t", header=None)
tester.columns = ['s1', 's2', 's3', 's4', 's5']
tester.drop(tester.columns[[4]], axis=1, inplace=True)
tester['cycle'] = np.linspace(1, len(tester), len(tester))
idd = np.full(len(tester), 1, dtype=float)
tester['id'] = idd
tester = tester.sort_values(['id', 'cycle'])
tester['cycle_norm'] = tester['cycle']
cols_normalize = tester.columns.difference(['id', 'cycle'])
scaler = preprocessing.MinMaxScaler()
norm_tester = pd.DataFrame(scaler.fit_transform(tester[cols_normalize]),
                           columns=cols_normalize,
                           index=tester.index)
tester_join = tester[tester.columns.difference(cols_normalize)].join(norm_tester)
tester = tester_join.reindex(columns=tester.columns)
tester = tester.reset_index(drop=True)
print(tester.head())
sequence_length = 50
sensor_cols = ['s' + str(i) for i in range(1, 4)]
seq_gen = (list(gen_sequence(tester[tester['id'] == id], sequence_length, sensor_cols))
           for id in tester['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)
threshold = 0.09763295
scores_test = model.predict(seq_array, verbose=1)
test_mse_loss = np.mean(np.abs(scores_test - seq_array), axis=1)
test_mse_loss = test_mse_loss.reshape((-1))

plt.hist(test_mse_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()
anomalies = test_mse_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))