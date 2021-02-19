def data_format():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    dataset = []
    datah = pd.read_csv('healthy Data/h30hz0.txt', sep='\t', header=None)
    datah.columns = ['s1', 's2', 's3', 's4', 's5']
    datah.drop(datah.columns[[4]], axis=1, inplace=True)
    f_name = 0
    idd = np.full(len(datah), f_name)
    datah['id'] = idd
    datah['label'] = np.full(len(datah), 0, dtype=int)
    dataset.append(datah)
    data = pd.read_csv('Data/b30hz0.txt', sep='\t', header=None)
    data.columns = ['s1', 's2', 's3', 's4', 's5']
    data.drop(data.columns[[4]], axis=1, inplace=True)
    idd = np.full(len(data), f_name, dtype=int)
    data['id'] = idd
    data['label'] = np.full(len(data), 1, dtype=int)
    dataset.append(data)
    dataset = pd.concat(dataset, ignore_index=True)
    dataset = dataset.iloc[np.random.permutation(dataset.index)].reset_index(drop=True)
    train, test = train_test_split(dataset, test_size=0.3, random_state=0)
    #  Data preprocessing
    # Normalize train
    col_normalize = train.columns.difference(['id', 'label'])
    scaler = preprocessing.MinMaxScaler()
    train_norm = pd.DataFrame(preprocessing.scale(train[col_normalize]),
                              columns=col_normalize,
                              index=train.index)
    join_df = train[train.columns.difference(col_normalize)].join(train_norm)
    train = join_df.reindex(columns=train.columns)
    # Normalize test
    test_norm = pd.DataFrame(preprocessing.scale(test[col_normalize]),
                             columns=col_normalize,
                             index=test.index)
    test_join_df = test[test.columns.difference(col_normalize)].join(test_norm)
    test = test_join_df.reindex(columns=test.columns)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)
    dist1 = np.linalg.norm(train['s2'] - train['s1'])
    dist2 = np.linalg.norm(train['s3'] - train['s2'])
    dist3 = np.linalg.norm(train['s4'] - train['s3'])
    dist11 = np.linalg.norm(train['s3'] - train['s1'])
    dist12 = np.linalg.norm(train['s4'] - train['s2'])
    dist13 = np.linalg.norm(train['s4'] - train['s1'])
    dist14 = np.linalg.norm(train['s3'] - train['s1'])
    distt = [dist1, dist2, dist3, dist11, dist12, dist13, dist14]
    dist = np.min(distt)
    X = train[col_normalize]
    Y = train['label']
    X_test = test[col_normalize]
    Y_test = test['label']
    return X, Y, X_test, Y_test, dist
