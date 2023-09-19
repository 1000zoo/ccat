import pyupbit as pu
import os
from lib.etc.private_constants import *
from lib.etc.util import *
import csv

"""
1, 3, 5, 10, 15, 30, 60, 240
"""

def save_ts_data(interval, count=10000000000):
    tt = os.path.join(TRAIN_DATA_PATH, "ex")
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



if __name__ == '__main__':
    ll = ['1', '3', '5', '10', '15', '30', '60', '240']
    ll.reverse()
    import time
    for l in ll:
        start = time.time()
        print(l)
        save_ts_data(l)
        print(time.time() - start)




