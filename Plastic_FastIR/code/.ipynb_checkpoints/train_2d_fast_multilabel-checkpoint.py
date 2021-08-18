import argparse
import pickle
import platform
from timeit import default_timer as timer

import numpy as np
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description='2d fast with pc multilabel')
parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
parser.add_argument('--bs', default=128, type=int, metavar='N', help='batch size')
parser.add_argument('--mcp', default='loss', type=str, metavar='N', help='model check point')
parser.add_argument('--nf', default=64, type=int, metavar='N', help='number of filters')
parser.add_argument('--im', default=25, type=int, metavar='N', help='2D ißmage size')

osname = platform.system()
if osname == 'Darwin':
    data_dir = r'/Users/sjiang87/plastic/data/'
    weight_dir = r'/Users/sjiang87/plastic/weights/'
    result_dir = r'/Users/sjiang87/machinelearning2/plastic/result/'
elif osname == 'Windows':
    data_dir = r'D:/plastic/data/'
    weight_dir = r'D:/plastic/weights/'
    result_dir = r'D:/machinelearning2/plastic/result/'
elif osname == 'Linux':
    data_dir = r'./'
    weight_dir = r'./'
    result_dir = r'./'


def cnn2d(shape, num_filters):
    """
    2D CNN model.
    :param shape: tuple, shape of the input, e.g. (25, 25, 2).
    :param num_filters: int, number of convolutional filters in each layer.
    :return: Keras model.
    """
    np.random.seed(0)
    tf.random.set_seed(0)
    inputs = layers.Input(shape)
    x = layers.Conv2D(num_filters, 3, activation='relu')(inputs)
    x = layers.Conv2D(num_filters, 3, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(num_filters, 3, activation='relu')(x)
    x = layers.Conv2D(num_filters, 3, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    for _ in range(3):
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(12, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='2dcnn')
    return model


def cnn1d(shape, num_filters):
    """
    1D CNN model.
    :param shape: tuple, shape of the input, e.g. (1600, 1).
    :param num_filters: int, number of convolutional filters in each layer.
    :return: Keras model.
    """
    np.random.seed(0)
    tf.random.set_seed(0)
    inputs = layers.Input(shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(num_filters, 3, activation='relu')(x)
    x = layers.Conv1D(num_filters, 3, activation='relu')(x)
    x = layers.MaxPool1D()(x)
    x = layers.Conv1D(num_filters, 3, activation='relu')(x)
    x = layers.Conv1D(num_filters, 3, activation='relu')(x)
    x = layers.MaxPool1D()(x)
    x = layers.Flatten()(x)
    for _ in range(3):
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(12, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='1dcnn')
    return model


def main(k, image_size, learning_rate, batch_size, model_checkpoint, num_filters, train):
    """
    The function to train the CNN for plastic IR classification.
    :param k: int, index for 5-fold cross-validation.
    :param image_size: int, image size, when it is 0, it is 1D CNN.
    :param learning_rate: float, learning rate.
    :param batch_size: float, batch size.
    :param model_checkpoint: str, Keras model check point.
    :param num_filters: float, number of convolutional filters in each layer.
    :param train: bool, True if you want to train, False if you want to test
    :return:
    """
    # if image size is not equal to 0, we load 2D CNN data.
    if image_size != 0:
        with open(data_dir + f'fast_with_pc_{image_size}.pickle', 'rb') as handle:
            x = pickle.load(handle)
            y_single_label = pickle.load(handle)
    # if image size is equal to 0, we load 1D CNN data.
    else:
        with open(data_dir + 'fast_with_pc.pickle', 'rb') as handle:
            x = pickle.load(handle)
            y_single_label = pickle.load(handle)
        # scale the 1D CNN data to be between 0 and 1.
        x = skp.minmax_scale(x, axis=1)
        # add another dimension for 1D CNN channel.
        x = x[..., None]
    # load multilabel labels.
    with open(data_dir + 'fast_with_pc_multilabel.pickle', 'rb') as handle:
        y = pickle.load(handle)
    # shuffle x and y with a fixed random seed.
    x = np.random.RandomState(0).permutation(x)
    y_single_label = np.random.RandomState(0).permutation(y_single_label)
    y = np.random.RandomState(0).permutation(y)
    # define stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # if i = k, then we stop
    i = 0
    for train_index, test_index in skf.split(x, y_single_label):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # train test split to select the validation data from the training data.
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=0, test_size=0.3)
        i += 1
        if i == k:
            print(f'This is fold {i}.')
            break
    # define the name for saving results
    name = f'multilabel_fast_wtih_pc_{k}_{image_size}_{learning_rate}_{batch_size}_{model_checkpoint}_{num_filters}'
    # if image size is not equal to 0, load 2D CNN model.
    if image_size != 0:
        model = cnn2d(shape=(x_train.shape[1:]), num_filters=num_filters)
    # if image size is equal to 0, load 1D CNN model.
    else:
        model = cnn1d(shape=(x_train.shape[1:]), num_filters=num_filters)
    # define the CNN training optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # compile the CNN model.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    # define Keras model checkpoint.
    if model_checkpoint == 'loss':
        checkpoint_cb = keras.callbacks.ModelCheckpoint(weight_dir + f'{name}.h5',
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        verbose=2)
    elif model_checkpoint == 'acc':
        checkpoint_cb = keras.callbacks.ModelCheckpoint(weight_dir + f'{name}.h5',
                                                        save_best_only=True,
                                                        monitor='val_acc',
                                                        mode='max',
                                                        verbose=2)
    # define early stopping.
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min')
    # train the model.
    if train:
        hist = model.fit(x_train, y_train,
                         validation_data=(x_valid, y_valid),
                         epochs=50,
                         shuffle=True,
                         verbose=2,
                         batch_size=batch_size,
                         callbacks=[checkpoint_cb, early_stopping_cb])
    # load the best trained weights.
    model.load_weights(weight_dir + f'{name}.h5')
    # calculate the prediction time. Use this on the same device for benchmarking.
    t1 = timer()
    # calculate predicted labels.
    y_pred = np.argmax(model.predict(x_test), axis=1)
    t2 = timer()
    # print prediction time and accuracy.
    print(f'{t2 - t1:0.3f} sec')
    print(f'acc: {skm.accuracy_score(y_test, y_pred):0.3f}')
    # during training, save testing data label, predicted label and history to the pickle file.
    if train:
        with open(result_dir + f'{name}.pickle', 'wb') as f:
            pickle.dump(y_test, f)
            pickle.dump(y_pred, f)
            pickle.dump(hist.history, f)
    # during testing, output the testing data, testing label, predicted label and CNN model.
    else:
        return x_test, y_test, y_pred, model


if __name__ == '__main__':
    args = parser.parse_args()
    image_size = args.im
    learning_rate = args.lr
    batch_size = args.bs
    model_checkpoint = args.mcp
    num_filters = args.nf
    train = True
    # run 5-fold cross-validation.
    for k in range(1, 6):
        main(k, image_size, learning_rate, batch_size, model_checkpoint, num_filters, train)
