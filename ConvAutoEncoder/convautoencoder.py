import datetime

from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import kullback_leibler_divergence
import data.data_preprocessing as preprocessing
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def conv1Dautoencoder(l_input, filters):
    print('input shape: ', l_input.shape)
    x = Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu')(l_input)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(filters=int(filters * 2), kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(filters=int(filters * 4), kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(filters=int(filters * 8), kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(filters=int(filters * 16), kernel_size=2, padding='same', activation='relu')(x)
    encoded = MaxPooling1D(2, padding='same')(x)
    print('encoder output shape: ', encoded.shape)

    x = Conv1D(filters=int(filters * 16), kernel_size=2, padding='same', activation='relu')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=int(filters * 8), kernel_size=2, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=int(filters * 4), kernel_size=2, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=int(filters * 2), kernel_size=2, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(filters=1, kernel_size=2, padding='same', activation='sigmoid')(x)
    print('decoder output shape: ', decoded.shape)

    encoder = Model(l_input, encoded)
    autoencoder = Model(l_input, decoded)

    return encoder, autoencoder


input_size = 1024
test_split = .8

x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data(
    '~/PycharmProjects/TCN/data/binance_btcusd_1h_v0.csv', input_size, test_split)

l_input = Input(shape=(input_size, 1))
encoder, autoencoder = conv1Dautoencoder(l_input, 16)
adam = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None,
            decay=0.0, amsgrad=True)

autoencoder.compile(loss='mape', optimizer=adam, metrics=['mse'])
autoencoder.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                   write_grads=True, profile_batch=0)

early_stopping = EarlyStopping(patience=20000, min_delta=0.0005, verbose=1)
mcp_save = ModelCheckpoint('SeriesNet_Closing.hdf5', save_best_only=True, monitor='loss', mode='min', verbose=1)
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', min_delta=0.00001, factor=0.1, patience=50, verbose=1, mode='min')

#autoencoder.load_weights('SeriesNet_Closing.hdf5')

autoencoder.fit(x_train, x_train, batch_size=128, epochs=100, verbose=1, validation_split=.2, shuffle=True,
                callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensorboard_callback])
