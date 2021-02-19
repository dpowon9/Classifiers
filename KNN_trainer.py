import os
import pandas as pd
import numpy as np
from model_builder import KNN_builder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from two_dim_data import data_format

np.random.seed(7)
# Data Loading
[X, Y, X_test, Y_test, dist] = data_format()
# fit the network
model = KNN_builder(dist)
history = model.fit(X, Y)
Y_pred = model.predict(X_test)
res = pd.DataFrame({'predict': Y_pred,
                    'True': Y_test})
print(res)
print('Accuracy of KNN regression classifier on test set: {:.2f}'.format(model.score(X_test, Y_test)))
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(classification_report(Y_test, Y_pred))
