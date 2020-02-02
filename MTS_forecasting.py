import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Add, LayerNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from TCN.tcn import TCN
import data.data_preprocessing as preprocessing
from pathlib import Path

##
# It's a very naive (toy) example to show how to do time series forecasting.
# - There are no training-testing sets here. Everything is training set for simplicity.
# - There is no input/output normalization.
# - The model is simple.
##

'''
close = pd.read_csv(data_file, usecols=['close'], dtype='float64')
volume = pd.read_csv(data_file, usecols=['volume'], dtype='float64')
open = pd.read_csv(data_file, usecols=['open'], dtype='float64')
high = pd.read_csv(data_file, usecols=['high'], dtype='float64')
low = pd.read_csv(data_file, usecols=['low'], dtype='float64')

close_train = close.truncate(after=10000)
close_test = close.truncate(before=10001, after=10100)
volume_train = volume.truncate(after=10000)
volume_test = volume.truncate(before=10001, after=10100)
open_train = open.truncate(after=10000)
open_test = open.truncate(before=10001, after=10100)
low_train = low.truncate(after=10000)
low_test = low.truncate(before=10001, after=10100)
high_train = high.truncate(after=10000)
high_test = high.truncate(before=10001, after=10100)

lookback_window = 24  # months.


close_train = close_train.values
close_test = close_test.values
volume_train = volume_train.values
volume_test = volume_test.values
open_train = open_train.values
open_test = open_test.values
high_train = high_train.values
high_test = high_test.values
low_train = low_train.values
low_test = low_test.values

c_train, y_train, c_test, y_test = [], [], [], []
v_train, v_test = [], []
o_train, o_test = [], []
h_train, h_test = [], []
l_train, l_test = [], []

for i in range(lookback_window, len(close_train)):
    c_train.append(close_train[i - lookback_window:i])
    y_train.append(close_train[i])
    v_train.append(volume_train[i - lookback_window:i])
    o_train.append(open_train[i - lookback_window:i])
    h_train.append(high_train[i - lookback_window:i])
    l_train.append(low_train[i - lookback_window:i])
for i in range(lookback_window, len(close_test)):
    c_test.append(close_test[i - lookback_window:i])
    y_test.append(close_test[i])
    v_test.append(volume_test[i - lookback_window:i])
    o_test.append(open_test[i - lookback_window:i])
    h_test.append(high_test[i - lookback_window:i])
    l_test.append(low_test[i - lookback_window:i])

c_test = np.array(c_test)
y_test = np.array(y_test)
c_train = np.array(c_train)
y_train = np.array(y_train)

v_train = np.array(v_train)
v_test = np.array(v_test)
o_train = np.array(o_train)
o_test = np.array(o_test)
h_train = np.array(h_train)
h_test = np.array(h_test)
l_train = np.array(l_train)
l_test = np.array(l_test)
'''

matplotlib.use('TkAgg')

data_file = 'data/binance_btcusd_1h_v0.csv'
lookback_window = 32  # hours.

x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data(data_file, timesteps=lookback_window, test_train_split_point=0.8)
o_train = x_train[:, :, [0]]
h_train = x_train[:, :, [1]]
l_train = x_train[:, :, [2]]
c_train = x_train[:, :, [3]]
v_train = x_train[:, :, [4]]

o_test = x_test[:, :, [0]]
h_test = x_test[:, :, [1]]
l_test = x_test[:, :, [2]]
c_test = x_test[:, :, [3]]
v_test = x_test[:, :, [4]]



i_o = Input(shape=(lookback_window, 1))
skip_o = Conv1D(filters=1, kernel_size=1, padding='causal')(i_o)
l_o = Conv1D(filters=64, kernel_size=2, padding='causal', kernel_initializer='he_normal')(i_o)
l_o = LayerNormalization()(l_o)
l_o = Activation('relu')(l_o)

i_h = Input(shape=(lookback_window, 1))
skip_h = Conv1D(filters=1, kernel_size=1, padding='causal')(i_h)
l_h = Conv1D(filters=64, kernel_size=2, padding='causal', kernel_initializer='he_normal')(i_h)
l_h = LayerNormalization()(l_h)
l_h = Activation('relu')(l_h)

i_l = Input(shape=(lookback_window, 1))
skip_l = Conv1D(filters=1, kernel_size=1, padding='causal')(i_l)
l_l = Conv1D(filters=64, kernel_size=2, padding='causal', kernel_initializer='he_normal')(i_l)
l_l = LayerNormalization()(l_l)
l_l = Activation('relu')(l_l)

i_c = Input(shape=(lookback_window, 1))
skip_c = Conv1D(filters=1, kernel_size=1, padding='causal')(i_c)
l_c = Conv1D(filters=64, kernel_size=2, padding='causal', kernel_initializer='he_normal')(i_c)
l_c = LayerNormalization()(l_c)
l_c = Activation('relu')(l_c)

i_v = Input(shape=(lookback_window, 1))
skip_v = Conv1D(filters=1, kernel_size=1, padding='causal')(i_v)
l_v = Conv1D(filters=64, kernel_size=2, padding='causal', kernel_initializer='he_normal')(i_v)
l_v = LayerNormalization()(l_v)
l_v = Activation('relu')(l_v)


m = Add()([skip_o, l_o, skip_h, l_h, skip_l, l_l, skip_c, l_c, skip_v, l_v])
m = TCN(activation='relu', dropout_rate=0.4, use_layer_norm=True, nb_stacks=3)(m)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i_o, i_h, i_l, i_c, i_v], outputs=[m])

model.summary()

model_file = "best_model.hdf5"
model.load_weights(model_file)

model.compile(optimizer=Adam(learning_rate=.001), loss='mae')


checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)


print('Train...')
batch_size = 128
model.fit([o_train, h_train, l_train, c_train, v_train], y_train, epochs=50, batch_size=batch_size, verbose=2, callbacks=[checkpoint])

score = model.evaluate([o_test, h_test, l_test, c_test, v_test], y_test, verbose=0)

print('Test loss: ', score)


p = model.predict([o_train, h_train, l_train, c_train, v_train])

plt.plot(p)
plt.plot(y_train)
plt.title('btc_use_1h_closing')
plt.legend(['predicted', 'actual'])
plt.show()
