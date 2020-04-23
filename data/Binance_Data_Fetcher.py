import ccxt
import time
import pandas as pd


def binance_klines_fetcher():
    binance = ccxt.binance()
    outputfile = 'binance_btcusd_4h_v0.csv'
    step = 3600000 * 1001 * 4
    symbol = 'BTCUSDT'
    interval = '4h'
    limit = 1000
    startTime = 1502942400000

    i = limit
    with open(outputfile, 'a') as file:
        while (i >= limit):
            params = {'symbol': symbol, 'interval': interval, 'startTime': startTime, 'limit': limit}
            print(params['startTime'])
            klines = binance.publicGetKlines(params)
            for item in klines:
                file.write("%s\n" % item)
            startTime += step
            i = len(klines)


binance_klines_fetcher()