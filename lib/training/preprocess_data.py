import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lib.etc.private_constants import TRAIN_DATA_PATH


def load_data(interval):
    interval = interval.split(".")[0] + ".csv"
    path = os.path.join(TRAIN_DATA_PATH, 'ex', interval)
    d = pd.read_csv(path)
    d.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    d['date'] = pd.to_datetime(d['date'])
    return d


def split_data(d: pd.DataFrame):
    nr = d.shape[0]
    split_idx = int(nr * 0.8)

    _train_data = d.iloc[:split_idx]
    _test_data = d.iloc[split_idx:]

    return _train_data, _test_data


def preprocessing_only_close(train_data, test_data):
    train_data = train_data[['close']]
    test_data = test_data[['close']]

    # Fit the scaler on the training data
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    # Scale the training data
    train_data = scaler.transform(train_data)

    # Scale the test data
    test_data = scaler.transform(test_data)

    # Convert the data to numpy arrays
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    return train_data, test_data


def moving_avg(df: pd.DataFrame):
    df['open'] = df['open'].pct_change()  # Create arithmetic returns column
    df['high'] = df['high'].pct_change()  # Create arithmetic returns column
    df['low'] = df['low'].pct_change()  # Create arithmetic returns column
    df['close'] = df['close'].pct_change()  # Create arithmetic returns column
    df['volume'] = df['volume'].pct_change()

    df.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values

    times = sorted(df.index.values)
    # last_10pct = sorted(df.index.values)[-int(0.1 * len(times))]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2 * len(times))]  # Last 20% of series

    min_return = min(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].min(axis=0))
    max_return = max(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].max(axis=0))

    df['open'] = (df['open'] - min_return) / (max_return - min_return)
    df['high'] = (df['high'] - min_return) / (max_return - min_return)
    df['low'] = (df['low'] - min_return) / (max_return - min_return)
    df['close'] = (df['close'] - min_return) / (max_return - min_return)

    min_volume = df[(df.index < last_20pct)]['volume'].min(axis=0)
    max_volume = df[(df.index < last_20pct)]['volume'].max(axis=0)

    df['volume'] = (df['volume'] - min_volume) / (max_volume - min_volume)

    df_train = df[(df.index < last_20pct)]
    df_test = df[(df.index >= last_20pct)]

    df_train = drop_and_to_np(df_train)
    df_test = drop_and_to_np(df_test)

    return df_train, df_test




def drop_and_to_np(df: pd.DataFrame):
    temp = df.copy()
    temp.drop(columns=['date'], inplace=True)
    return temp.values



def to_sequences(df, config):
    X, y = [], []
    sl = config.sequence_length
    for index in range(len(df) - sl):
        X.append(df[index: index + sl])
        y.append(df[:, 3][index + sl])

    return np.array(X), np.array(y)
