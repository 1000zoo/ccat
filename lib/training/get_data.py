import pyupbit as pu
"""
1, 3, 5, 10, 15, 30, 60, 240
"""

if __name__ == '__main__':
    c = pu.get_ohlcv("KRW-BTC", count=1, interval='day')
    print(c)


