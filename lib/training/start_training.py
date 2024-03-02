import keras.callbacks

from lib.etc.train_parameters import CustomConfig
from lib.etc.private_constants import MODEL_PATH, FIGURE_PATH

from train_model import *
from lib.prepocessing.preprocess_data import *
from matplotlib import pyplot as plt

def main():
    config = CustomConfig()
    data = load_data(config.interval)
    train, test = moving_avg(data)
    train_X, train_y = to_sequences(train, config)
    test_X, test_y = to_sequences(test, config)
    model = get_model(config)
    save_path = os.path.join(MODEL_PATH, f'Transformer+TimeEmbedding-{config.interval}.hdf5')
    figure_path = os.path.join(FIGURE_PATH, f'Transformer+TimeEmbedding-results-{config.interval}.png')

    callback = keras.callbacks.ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(train_X, train_y,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        callbacks=[callback],
                        validation_split=0.1)

    model.save(save_path)

    lc = plt.figure()
    st = lc.suptitle(f"Learning Curve {config.interval}")

    ax1 = lc.add_subplot(311)
    ax1.plot(history.history['loss'], label='Training loss (MSE)')
    ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12)

    ax2 = lc.add_subplot(312)
    ax2.plot(history.history['mae'], label='Training loss (MAE)')
    ax2.plot(history.history['val_mae'], label='Validation loss (MAE)')
    ax2.set_title("Model metric - MAE", fontsize=18)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MAE)')
    ax2.legend(loc="best", fontsize=12)


    ax3 = lc.add_subplot(313)
    ax3.plot(history.history['mape'], label='Training loss (MAPE)')
    ax3.plot(history.history['val_mape'], label='Validation loss (MAPE)')
    ax3.set_title("Model metric - MAPE", fontsize=18)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MAPE)')
    ax3.legend(loc="best", fontsize=12)

    lc.savefig(os.path.join(FIGURE_PATH, f'Transformer+TimeEmbedding-learning-curve-{config.interval}.png'))

    model = tf.keras.models.load_model(save_path,
                                       custom_objects={'Time2Vector': Time2Vector,
                                                       'SingleAttention': SingleAttention,
                                                       'MultiAttention': MultiAttention,
                                                       'TransformerEncoder': TransformerEncoder})


    # Calculate predication for training, validation and test data
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    # Print evaluation metrics for all datasets
    train_eval = model.evaluate(train_X, train_y, verbose=0)
    test_eval = model.evaluate(test_X, test_y, verbose=0)
    print(' ')
    print('Evaluation metrics')
    print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
    print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

    ###############################################################################
    '''Display results'''

    fig = plt.figure(figsize=(15, 20))
    st = fig.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
    st.set_y(0.92)

    # Plot training data results
    ax11 = fig.add_subplot(211)
    ax11.plot(train[:, 3], label='IBM Closing Returns')
    ax11.plot(np.arange(config.sequence_length, train_pred.shape[0] + config.sequence_length), train_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('IBM Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    # Plot test data results
    ax31 = fig.add_subplot(212)
    ax31.plot(test[:, 3], label='IBM Closing Returns')
    ax31.plot(np.arange(config.sequence_length, test_pred.shape[0] + config.sequence_length), test_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('IBM Closing Returns')
    ax31.legend(loc="best", fontsize=12)

    fig.savefig(figure_path)



if __name__ == '__main__':
    main()