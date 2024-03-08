import tensorflow as tf

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

from matplotlib import pyplot as plt

from lib.etc.train_parameters import CustomConfig
from lib.etc.private_constants import PathConfig

from train_model import get_model, get_categorical_model
from lib.prepocessing.preprocess_data import *
from sklearn.utils.class_weight import compute_class_weight

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, prefix : str):
        super(CustomModelCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save(self.prefix.format(epoch=epoch + 1))

def fit(train_dataset, val_dataset, config : CustomConfig, path_config : PathConfig):
    model = get_model(config)

    if (type(train_dataset) == tuple and type(val_dataset) == tuple):
        return model.fit(
            train_dataset[0], train_dataset[1], epochs = config.epochs,
            batch_size=config.batch_size,
            validation_data = val_dataset,
            validation_batch_size=config.batch_size,
            callbacks = [
                CustomModelCheckpoint(save_freq=5, prefix=path_config.checkpoint_path)
            ]
        )

    return model.fit(
        train_dataset, epochs = config.epochs,
        validation_data = val_dataset,
        callbacks = [
            CustomModelCheckpoint(save_freq=5, prefix=path_config.checkpoint_path)
        ]
    )

def fit_categorical(X_train, y_train, X_val, y_val, config : CustomConfig, path_config : PathConfig):
    model = get_categorical_model(config)

    #모델 설정
    y_train_int_labels = np.argmax(y_train, axis=1)

    # 고유한 클래스 레이블을 찾습니다.
    classes = np.unique(y_train_int_labels)

    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int_labels)
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train, 
        epochs=config.epochs, batch_size=config.batch_size, 
        class_weight=class_weight_dict,
        validation_data=(X_val, y_val),
        validation_batch_size=config.batch_size,
        callbacks=[CustomModelCheckpoint(save_freq=5, prefix=path_config.checkpoint_path)]
    )
    model.save(path_config.model_path)
    return history

def main():
    config = CustomConfig()
    path_config = PathConfig(config)
    print("loaded configs")

    # train, val, test = load_dataset(config.sequence_length, config.batch_size)
    train, val, test = load_categorical_data(
        "/Users/cjswl/python_data/cryptoAI-data/data/anomaly-detection data.npz",
        rate=0.001
    )
    print(train[0].shape, train[1].shape)
    print(val[0].shape, val[1].shape)
    print(test[0].shape, test[1].shape)
    print("dataset load success")

    history = fit_categorical(train[0], train[1], val[0], val[1], config, path_config)



if __name__ == '__main__':
    main()