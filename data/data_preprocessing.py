from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def train_test_instance_scaled_data(data_file, timesteps, test_train_split_point):
    # load data from file
    data = pd.read_csv(data_file, usecols=['open', 'high', 'low', 'close', 'volume'], dtype='float32')
    log_returns_data = np.log(data) - np.log(data.shift(1))
    log_returns_data.loc[0] = 0
    log_returns_data = log_returns_data.to_numpy()

    n = log_returns_data.shape[0]
    train_end = int(test_train_split_point * n)

    scaler = MinMaxScaler(feature_range=(0.1, 0.9))

    x, y = [], []
    for i in range(n - timesteps):
        t = scaler.fit_transform(log_returns_data[i: i + timesteps])
        x.append(log_returns_data[i: i + timesteps])
        y.append(log_returns_data[i + timesteps][4])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test = np.split(x, [train_end])
    y_train, y_test = np.split(y, [train_end])

    return x_train, y_train, x_test, y_test


