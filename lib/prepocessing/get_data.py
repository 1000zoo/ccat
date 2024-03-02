import pyupbit as pu
import pandas as pd
import csv

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

from lib.etc.private_constants import DATA_PATH
from lib.etc.util import *

"""
1, 3, 5, 10, 15, 30, 60, 240
"""

def save_ts_data(interval, count=10000000000):
    tt = os.path.join(DATA_PATH, "ex")
    check_dir(tt)
    interval = 'minute' + interval
    c = os.path.join(tt, interval + ".csv")
    with open(c, 'w') as f:
        d = pu.get_ohlcv("KRW-BTC", count=count, interval=interval)
        w = csv.writer(f, delimiter=',')
        print(d.shape)

        for t, i in zip(d.index, d.values):
            data = [str(t)]
            for val in i[:-1]:
                data.append(str(val))

            w.writerow(data)

def load_data(interval, count=10000000000, ticker="KRW-BTC"):
    df = pu.get_ohlcv(ticker, count=count, interval=interval)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    return df

if __name__ == '__main__':
    ll = ['1']
    ll.reverse()
    import time
    for l in ll:
        start = time.time()
        print(l)
        save_ts_data(l, 128)
        print(time.time() - start)




