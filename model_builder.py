def rul_LSTM(seq_length, features, out_length):
    """
    :param seq_length: How far back you want your model to look
    :param features: Number of features
    :param out_length: Output dimensions
    :return: Model
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    model = Sequential()
    model.add(LSTM(
        input_shape=(seq_length, features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=out_length, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def LSTM_autoencoder(timesteps, dim):
    """
    :param timesteps: How many steps you want your LSTM to look back
    :param dim: input features
    :return: Model
    """
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
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
    return model


def KNN_builder(n_neighbours):
    """
    :param n_neighbours: This is the K value, the nearest neighbours to include
    :return: model
    """
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=int(n_neighbours))
    return model


def Dense_model(input_dim):
    """
    :param input_dim: Number of features
    :return: model
    """
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add((Dense(200, input_dim=input_dim, activation='relu')))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Kfold_DenseModel(input_dim, epochs, batch, split, X, Y):
    """
    :param input_dim: Number of features
    :param epochs: number of epochs to train
    :param batch: batch size
    :param split: k_fold split, how many times your model will train
    :param X: Train data
    :param Y: Train labels
    :return: Returns results
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    def build_model():
        model = Sequential()
        model.add((Dense(200, input_dim=input_dim, activation='relu')))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch, verbose=1)
    kfold = KFold(n_splits=split, shuffle=True)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    res = "Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100)
    return res


def Log_reg():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, solver='saga', multi_class='auto',
                               warm_start=True)
    return model


def many_models():
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    num = input("LinearSVC = 1\n"
                "GaussianNB = 2\n"
                "RandomForest = 3\n"
                "ExtraTrees = 4\n"
                "Please select a model: ")
    try:
        num = int(num)
    except ValueError:
        raise ValueError("Only integers expected")
    if num == 1:
        model = LinearSVC()
        return model
    elif num == 2:
        model = GaussianNB()
        return model
    elif num == 3:
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0)
        return model
    elif num == 4:
        model = ExtraTreesClassifier(class_weight='balanced', random_state=0)
        return model
    else:
        print("Number exceeds the options listed. Please try again")
