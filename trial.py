from model_builder import Kfold_DenseModel
from two_dim_data import data_format
[X, Y, X_test, Y_test, dist] = data_format()
print(Kfold_DenseModel(4, 5, 64, 2, X, Y))

