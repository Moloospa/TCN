import numpy as np
import pandas as pd
import ta
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import itertools
from matplotlib import pyplot as plt


class Model:
    def __init__(self, datafile):
        self.datafile = datafile
        self.candlesticks = self.get_raw_ohlc()
        self.model = self.get_model()
        self.direction = self.get_direction()

    def get_raw_ohlc(self):
        candlesticks = pd.read_csv(filepath_or_buffer=self.datafile, dtype='float32',
                                   usecols=['open', 'high', 'low', 'close'])
        return np.array(candlesticks)

    def normalize_ohlc(self):
        # normalize by dividing all columns by
        return np.divide(self.candlesticks, self.candlesticks[:, 0][:, None])

    def get_ohlc_relative_position(self):

        ohlc_midpoint = (self.candlesticks[:, 1] - self.candlesticks[:, 2]) / 2 + self.candlesticks[:, 2]
        return (ohlc_midpoint - np.roll(ohlc_midpoint, 1)) / np.roll(ohlc_midpoint, 1)

    def compute_ma(self, periods=3):
        weights = np.ones(periods) / periods
        ma = np.convolve(self.candlesticks[:, 3], weights, mode='same')
        return np.roll(ma, 1)

    def get_trend(self):
        ma = self.compute_ma()
        condlist = [np.logical_and(ma > np.roll(ma, 1), np.roll(ma, 1) > np.roll(ma, 2)),
                    np.logical_and(ma < np.roll(ma, 1), np.roll(ma, 1) < np.roll(ma, 2))]
        choicelist = [1, -1]
        return np.select(condlist, choicelist)

    def get_direction(self):
        a = self.candlesticks[:, 3]
        b = np.roll(a, -1)
        condlist = [b > a, b < a]
        choicelist = [1, -1]
        return np.select(condlist, choicelist)

    def get_model(self):
        cs_t = self.normalize_ohlc()[:, 1:]
        cs_t_1 = np.roll(cs_t, 1)
        cs_t_2 = np.roll(cs_t, 2)
        delta_t = self.get_ohlc_relative_position()
        delta_t_1 = np.roll(delta_t, 1)
        trend = self.get_trend()
        t_t = np.roll(trend, 2)
        model = np.concatenate((cs_t_2, cs_t_1, cs_t_2, delta_t[:, None], delta_t_1[:, None], t_t[:, None]), axis=1)
        return model



class Pattern:
    def __init__(self, feature, instance_count, direction_count):
         self.feature = feature
         self.instance_count = instance_count
         self.direction_count = direction_count
         self.reliability = np.amax(self.direction_count) / self.instance_count * 100

    def to_string(self):
          print('feature: ', self.feature, '\ninstance_count: ', self.instance_count, '\ndirection_count: ',
                self.direction_count, '\nreliability: ', self.reliability, ' %')


datafile = 'test.csv'
m = Model(datafile)
model = m.model
direction = m.direction
print(model.shape)
n = 10
print('just before clustering')
clusters = AgglomerativeClustering(linkage='average').fit(model[n:, :])

patterns = []

for x in range(model.shape[0]):
    if direction[x] == 1 : direction_count = np.array([1, 0, 0])
    elif direction[x] == -1 : direction_count = np.array([0, 1, 0])
    else : direction_count = np.array([0, 0, 1])

    patterns.append(Pattern(model[x], 1, direction_count))

for x in clusters.children_:
    feature = np.average((patterns[x[0]].feature, patterns[x[1]].feature), axis=0)
    instance_count = patterns[x[0]].instance_count + patterns[x[1]].instance_count
    direction_count = patterns[x[0]].direction_count + patterns[x[1]].direction_count

    patterns.append(Pattern(feature, instance_count, direction_count))

i = 0
for p in patterns :
    print('Node ', i)
    i += 1
    p.to_string()



'''
print(clusters.labels_)
print(clusters.children_)

ii = itertools.count(model.shape[0])
tree = [{'node_id': next(ii), 'left': x[0], 'right': x[1]} for x in clusters.children_]

print(tree)'''
