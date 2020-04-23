from WaveNet import wavenet
import data.data_preprocessing as preprocessing
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

input_size = 512
num_filters = 32
kernel_size = 2
num_residual_blocks = 9

test_split = .8
validation_split = .2

#x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data('~/PycharmProjects/TCN/data/binance_btcusd_1h_v0.csv',
                                                                                 input_size, test_split)

model = wavenet.build_wavenet_model(input_size, num_filters, kernel_size, num_residual_blocks)
adam = Adam(lr=100, beta_1=0.9, beta_2=0.999, epsilon=None,
                       decay=0.0, amsgrad=False)

model.compile(loss='mse', optimizer=adam, metrics=['mape'])
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, write_grads=True, profile_batch=0)


early_stopping = EarlyStopping(patience=20000, min_delta=0.0005, verbose=1)
mcp_save = ModelCheckpoint('SeriesNet_Closing.hdf5', save_best_only=True, monitor='loss', mode='min', verbose=1)
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', min_delta=0.00001, factor=0.1, patience=50, verbose=1, mode='min')

#model.load_weights('SeriesNet_Closing.hdf5')

model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_split=validation_split, shuffle=True, callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensorboard_callback])
