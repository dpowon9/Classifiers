import pandas as pd
import numpy as np
from dataprocessor import gen_sequence
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

tester = pd.read_csv('healthy Data/h30hz0.txt', sep="\t", header=None)
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
seq_length = 50
sensor_cols = ['s' + str(i) for i in range(1, 4)]
seq_gen = (list(gen_sequence(tester[tester['id'] == id], seq_length, sensor_cols))
           for id in tester['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)

dim = seq_array.shape[2]
timesteps = 50
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(LSTM(50, input_shape=(timesteps, dim), return_sequences=True))
model.add(Dense(3))
model.compile(loss='mae', optimizer='adam')
print(model.summary())

history = model.fit(seq_array, seq_array, epochs=20, batch_size=200, validation_data=(seq_array, seq_array), verbose=1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
# Getting the training metrics
scores = model.predict(seq_array)
train_mse_loss = np.mean(np.abs(scores - seq_array), axis=1)
plt.hist(train_mse_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()
# Get reconstruction loss threshold.
threshold = np.max(train_mse_loss)
print("Reconstruction error threshold: ", threshold)
with open('Results/Reconstruction error threshold.txt', 'a') as rec:
    rec.write('\nAt 0 loading the reconstruction error threshold is ' + str(threshold))
    rec.close()
