import tensorflow as tf

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_path)

from matplotlib import pyplot as plt

from lib.etc.train_parameters import CustomConfig
from lib.etc.private_constants import PathConfig

from train_model import get_model
from lib.prepocessing.preprocess_data import *

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


def main():
    config = CustomConfig()
    path_config = PathConfig(config)
    print("loaded configs")

    # train, val, test = load_dataset(config.sequence_length, config.batch_size)
    train, val, test = load_preprocessed_data(config.sequence_length, reduce=0.2)
    print("dataset load success")

    history = fit(train, val, config, path_config)



if __name__ == '__main__':
    main()