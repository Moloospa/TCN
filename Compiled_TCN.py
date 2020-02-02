import TCN.tcn as tcn
import data.data_preprocessing as preprocessing

data_file = 'binance_btcusd_1h_v0.csv'
lookback_window = 32
x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data(data_file, timesteps=lookback_window,
                                                                                 test_train_split_point=0.8)

model = tcn.compiled_tcn(num_feat=5, nb_filters=64, kernel_size=2, dilations=(1, 2, 4, 8, 16, 32), nb_stacks=3,
                         max_len=32, output_len=1, padding='causal', skip_connections=True, return_sequences=True,
                         regression=True, dropout_rate=0.05, name='tcn', kernel_initializer='he_normal', activation='linear',
                         opt='adam', lr=0.002, use_batch_norm=False,use_layer_norm=True)


