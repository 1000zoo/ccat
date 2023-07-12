import pyupbit as pu
import os
from lib.etc.private_constants import *
from lib.etc.util import *

"""
1, 3, 5, 10, 15, 30, 60, 240
"""



if __name__ == '__main__':
    tt = os.path.join(TRAIN_DATA_PATH, "ex")
    check_dir(tt)

    csv = os.path.join(tt, "test.csv")

    with open(csv, 'w') as f:
        d = pu.get_ohlcv("KRW-BTC", count=1, interval="day")





