from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def train_test_instance_scaled_data(data_file, timesteps, test_train_split_point):
    # load data from file
    data = pd.read_csv(data_file, usecols=['close'], dtype='float32')
    log_returns_data = np.log(data) - np.log(data.shift(1))
    log_returns_data.loc[0] = 0
    log_returns_data = log_returns_data.to_numpy()

    n = log_returns_data.shape[0]
    train_end = int(test_train_split_point * n)

    scaler = MinMaxScaler(feature_range=(-0.9, 0.9))

    x, y = [], []
    for i in range(n - timesteps):
        scaler.fit(log_returns_data[i: i + timesteps + 1])
        t = scaler.transform(log_returns_data[i: i + timesteps + 1])
        x.append(t[0: timesteps])
        y.append(t[timesteps])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test = np.split(x, [train_end])
    y_train, y_test = np.split(y, [train_end])

    return x_train, y_train, x_test, y_test

'''

x_train, y_train, x_test, y_test = train_test_instance_scaled_data('data/binance_btcusd_1h_v0.csv', 256, 1)

print(x_train.shape)
plt.plot(x_train[0, :, 3])
plt.plot(x_train[0, :, 2])
plt.show()

'''
