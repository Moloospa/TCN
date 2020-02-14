import TCN.tcn as tcn
import data.data_preprocessing as preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Input, Model
import GPy
import GPyOpt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class TCN():
    def __init__(self, data_file,
                 num_feat=5,
                 num_classes=1,
                 nb_filters=32,
                 kernel_size=2,
                 dilations=(1, 2, 4, 8, 16, 32),
                 nb_stacks=3,
                 max_len=32,
                 output_len=1,
                 padding='causal',
                 use_skip_connections=True,
                 return_sequences=False,
                 regression=True,
                 dropout_rate=0.05,
                 name='tcn',
                 kernel_initializer='he_normal',
                 activation='linear',
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=True,
                 batch_size=128,
                 epochs=50,
                 test_split=0.2,
                 validation_split=0.1):
        self.data_file = data_file
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.max_len = max_len
        self.output_len = output_len
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.regression = regression
        self.dropout_rate = dropout_rate
        self.name = name
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.opt = opt
        self.lr = lr
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_split = test_split
        self.validation_split = validation_split
        self.__x_train, self.__y_train, self.__x_test, self.__y_test = self.TCN_data()
        self.__model = self.TCN_model()

    def TCN_data(self):
        x_train, y_train, x_test, y_test = preprocessing.train_test_instance_scaled_data(data_file=self.data_file,
                                                                                         timesteps=self.max_len,
                                                                                         test_train_split_point=(
                                                                                                 1 - self.test_split))
        return x_train, y_train, x_test, y_test

    def TCN_model(self):
        model = tcn.compiled_tcn(num_feat=self.num_feat,
                                 num_classes=self.num_classes,
                                 nb_filters=self.nb_filters,
                                 kernel_size=self.kernel_size,
                                 dilations=self.dilations,
                                 nb_stacks=self.nb_stacks,
                                 max_len=self.max_len,
                                 output_len=self.output_len,
                                 padding=self.padding,
                                 use_skip_connections=self.use_skip_connections,
                                 return_sequences=self.return_sequences,
                                 regression=self.regression,
                                 dropout_rate=self.dropout_rate,
                                 name=self.name,
                                 kernel_initializer=self.kernel_initializer,
                                 activation=self.activation,
                                 opt=self.opt,
                                 lr=self.lr,
                                 use_batch_norm=self.use_batch_norm,
                                 use_layer_norm=self.use_layer_norm)
        model.load_weights('.mdl_wts.hdf5')
        model.summary()
        return model

    def TCN_fit_evaluate(self):
        # fit
        early_stopping = EarlyStopping(patience=20000, min_delta=0.0005, verbose=1)
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, factor=0.1, patience=100, verbose=1, mode='min')

        self.__model.fit(self.__x_train,
                         self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=1,
                         validation_split=self.validation_split,
                         shuffle=True,
                         callbacks=[early_stopping, mcp_save, reduce_lr_loss])

        # evaluate
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=1)
        return evaluation

    def TCN_predict(self):
        prediction = self.__model.predict(self.__x_test)
        return self.__y_test, prediction


def run_TCN(data_file='data/binance_btcusd_1h_v0.csv',
            num_feat=4,
            num_classes=1,
            nb_filters=52,
            kernel_size=3,
            dilations=(1, 2, 4, 8, 16, 32, 64,128),
            nb_stacks=2,
            max_len=128,
            output_len=1,
            padding='causal',
            use_skip_connections=True,
            return_sequences=False,
            regression=True,
            dropout_rate=0.05,
            kernel_initializer='he_normal',
            activation='relu',
            opt='adam',
            lr=0.02,
            use_batch_norm=False,
            use_layer_norm=True,
            batch_size=128,
            epochs=50,
            test_split=0.2,
            validation_split=0.1):
    _tcn = TCN(data_file=data_file,
               num_feat=num_feat,
               num_classes=num_classes,
               nb_filters=nb_filters,
               kernel_size=kernel_size,
               dilations=dilations,
               nb_stacks=nb_stacks,
               max_len=max_len,
               output_len=output_len,
               padding=padding,
               use_skip_connections=use_skip_connections,
               return_sequences=return_sequences,
               regression=regression,
               dropout_rate=dropout_rate,
               kernel_initializer=kernel_initializer,
               activation=activation,
               opt=opt,
               lr=lr,
               use_batch_norm=use_batch_norm,
               use_layer_norm=use_layer_norm,
               batch_size=batch_size,
               epochs=epochs,
               test_split=test_split,
               validation_split=validation_split)

    evaluation = _tcn.TCN_fit_evaluate()
    y_test, prediction = _tcn.TCN_predict()

    return evaluation, y_test, prediction


bounds = [{'name': 'validation_split', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'droupout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
          {'name': 'lr', 'type': 'continuous', 'domain': (0.000001, 0.001)}]


#          {'name': 'nb_filters', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
#         {'name': 'kernel_size', 'type': 'discrete', 'domain': (2, 3, 4)},
#        {'name': 'dilation', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
#       {'name': 'nb_stacks', 'type': 'discrete', 'domain': (2, 3, 4, 5, 6)}]

def f(x):
    print(x)
    evaluation = run_TCN(
        validation_split=float(x[:, 0]),
        dropout_rate=float(x[:, 1]),
        lr=float(x[:, 2]))
    print("LOSS:\t".format(evaluation))
    print(evaluation)
    return evaluation


#evaluation, y_test, prediction = run_TCN(validation_split=0.2, dropout_rate=0.1, lr=0.001, epochs=1)

print('Evaluation: {}'.format(evaluation))
plt.plot(prediction)
plt.plot(y_test)
plt.title('btc_use_1h_closing')
plt.legend(['predicted', 'actual'])
plt.show()

'''
opt_tcn = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_tcn.run_optimization(max_iter=10)

print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
""".format(bounds[0]["name"],opt_tcn.x_opt[0],
           bounds[1]["name"],opt_tcn.x_opt[1],
           bounds[2]["name"],opt_tcn.x_opt[2]))
print("optimized loss: {0}".format(opt_tcn.fx_opt))

opt_tcn.x_opt

'''

