import pandas as pd
import numpy as np
from model_builder import rul_LSTM
from PM_Data import PM_data_format
from sklearn.metrics import confusion_matrix, recall_score, precision_score
np.random.seed(7)
sequence_length = 50
[seq_array, label_array, seq_array_test_last, label_array_test_last] = PM_data_format()
# build the network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]
model = rul_LSTM(sequence_length, nb_features, nb_out)
# fit the network
model.fit(seq_array, label_array, epochs=10, batch_size=128, validation_split=0.05, verbose=1)
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accuracy: {}'.format(scores[1]))

y_pred = model.predict_classes(seq_array, verbose=1, batch_size=200)
y_true = label_array
cm = confusion_matrix(y_true, y_pred)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print('precision = ', precision, '\n', 'recall = ', recall)

scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
print('Accurracy: {}'.format(scores_test[1]))

y_pred_test = model.predict_classes(seq_array_test_last)
y_true_test = label_array_test_last
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true_test, y_pred_test)

precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print('Precision: ', precision_test, '\n', 'Recall: ', recall_test, '\n', 'F1-score:', f1_test)

results_df = pd.DataFrame([[scores_test[1], precision_test, recall_test, f1_test],
                           [0.94, 0.952381, 0.8, 0.869565]],
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                          index=['LSTM',
                                 'Template Best Model'])

print(results_df)
