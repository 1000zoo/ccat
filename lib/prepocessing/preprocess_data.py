import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

import pandas as pd
import numpy as np

from keras import utils
from sklearn.preprocessing import MinMaxScaler
from lib.etc.private_constants import DATA_PATH
from lib.etc.train_parameters import CustomConfig

def load_categorical_data(data_path, rate=0.1):
    data = np.load(data_path)

    X_result = data["X_result"]
    y_result = data["y_result"]

    X_result = X_result[int(len(X_result) * (1 - rate)) : ]
    y_result = y_result[int(len(y_result) * (1 - rate)) : ]

    X_result_shape = X_result.shape
    X_result_2D = X_result.reshape(-1, X_result_shape[2])

    X_result_2D_scaled = MinMaxScaler().fit_transform(X_result_2D)
    X_result_scaled = X_result_2D_scaled.reshape((X_result_shape))
    y_result_encoded = utils.to_categorical(y_result, num_classes=3)
        
    split_val = int(len(X_result) * 0.2)
    split_test = int(len(X_result) * 0.1)

    train = X_result_scaled[split_val : ], y_result_encoded[split_val : ]
    val = X_result_scaled[split_test : split_val], y_result_encoded[split_test : split_val]
    test = X_result_scaled[: split_test], y_result_encoded[: split_test]

    return train, val, test


def load_preprocessed_data(sequence_length, reduce=0):
    data = load_data(reduce)
    data = scaling(data)
    print(data.shape)
    X, y = slicing(data, sequence_length)
    val = int(0.7 * len(data))
    test = int(0.85 * len(data))

    return (X[:val], y[:val]), (X[val:test], y[val:test]), (X[test:], y[test:])

def load_dataset(sequence_length, batch_size, reduce=0):
    data = load_data(reduce)
    print(data.shape)

    data = scaling(data)
    return slicing_to_dataset(data, sequence_length, batch_size)

def slicing_to_dataset(data, sequence_length, batch_size):
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window : window.batch(sequence_length + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:][0][3]))

    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    test_size = int(0.15 * len(data))

    # dataset = dataset.shuffle(len(data))

    train_dataset = dataset.take(train_size)
    test_and_val = dataset.skip(train_size)
    
    val_dataset = test_and_val.take(val_size)
    test_dataset = test_and_val.skip(val_size)

    return train_dataset.batch(batch_size), val_dataset.batch(batch_size), test_dataset.batch(batch_size)

def slicing(data, sequence_length):
    X = data[:]
    y = data[:, 3]

    X_window, y_window = [], []
    for i in range(len(X) - sequence_length):
        X_window.append(X[i : i+sequence_length, :])
        y_window.append(y[i + sequence_length])

    return np.array(X_window), np.array(y_window)

def scaling(data):
    scaler = MinMaxScaler()
    ohlc = data[["open", "high", "low", "close"]]
    v = data[["volume"]]

    ohlc_scaled = scaler.fit_transform(ohlc)
    v_scaled = scaler.fit_transform(v)

    return np.concatenate((ohlc_scaled, v_scaled), axis = 1)


def load_data(reduce=0):
    reduce = min(0.999, max(0, reduce))
    d = pd.read_csv(DATA_PATH)
    d = d.drop(["Unnamed: 0", "value"], axis=1)
    return d[int((1 - reduce) * len(d)) : ]

