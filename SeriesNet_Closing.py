from SeriesNet.SeriesNet import DC_CNN_Model
import data.data_preprocessing as preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import datetime



matplotlib.use('TkAgg')

x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data('data/binance_btcusd_1h_v0.csv', 2048, .8)
x_train = x_train[:500]
y_train = y_train[:500]
model = DC_CNN_Model(2048)
model.load_weights('SeriesNet_Closing.hdf5')

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


early_stopping = EarlyStopping(patience=20000, min_delta=0.0005, verbose=1)
mcp_save = ModelCheckpoint('SeriesNet_Closing.hdf5', save_best_only=True, monitor='loss', mode='min', verbose=1)
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', min_delta=0.00001, factor=0.1, patience=50, verbose=1, mode='min')

model.fit(x_train, y_train, batch_size=10, epochs=3000, verbose=1, validation_split=0.2, shuffle=True, callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensorboard_callback])

'''
evaluation = model.evaluate(x_test, y_test, batch_size=1, verbose=1)

prediction = model.predict(x_test)

plt.plot(y_test)
plt.plot(prediction)
plt.legend(['actual', 'prediction'])
plt.show()
'''
