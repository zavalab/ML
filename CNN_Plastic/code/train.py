import pickle
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from pyts.image import GramianAngularField
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(k, dim, image_size=None, train=False):
    with open('../data/plastic.pickle', 'rb') as handle:
        X_train = pickle.load(handle)

    X_train = (X_train - X_train.min()) / (X_train.max()-X_train.min())

    y_train = np.concatenate((np.ones(70) * 0, np.ones(70),
                              np.ones(70) * 2, np.ones(70) * 3,
                              np.ones(70) * 4, np.ones(70) * 5,
                              np.ones(70) * 6, np.ones(70) * 7,
                              np.ones(70) * 8, np.ones(70) * 9,))

    X_train = np.random.RandomState(0).permutation(X_train)
    y_train = np.random.RandomState(0).permutation(y_train)

    if dim == 2:
        gasf = GramianAngularField(image_size=image_size, method='summation')
        gadf = GramianAngularField(image_size=image_size, method='difference')
        X_train1 = gasf.fit_transform(X_train)[..., np.newaxis]
        X_train2 = gadf.fit_transform(X_train)[..., np.newaxis]

        X_train = np.concatenate([X_train1, X_train2], axis=-1)
    else:
        X_train = X_train[..., np.newaxis]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test = X_train[train_index], X_train[test_index]
        y_train_fold, y_test = y_train[train_index], y_train[test_index]
        X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
            X_train_fold, y_train_fold, random_state=0, test_size=0.3)

        i += 1
        if i == k:
            print(f"This is fold {i}.")
            break

    def cnn2d(shape, seed):
        np.random.seed(seed)
        if tf.__version__ == '1.14.0':
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
        inputs = layers.Input(shape)
        x = layers.Conv2D(64, 3, activation='relu')(inputs)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPool2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name="fcnn")
        return model

    def cnn1d(shape, seed):
        np.random.seed(seed)
        if tf.__version__ == '1.14.0':
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
        inputs = layers.Input(shape)
        x = layers.Conv1D(64, 3, activation='relu')(inputs)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPool1D()(x)

        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPool1D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name="fcnn")
        return model

    name = f"mix-{dim}-{i}-{image_size}"
    print(name)

    if dim == 2:
        model = cnn2d(
            shape=(
                X_train_fold.shape[1],
                X_train_fold.shape[2],
                X_train_fold.shape[3]),
            seed=0)
    else:
        model = cnn1d(
            shape=(
                X_train_fold.shape[1],
                X_train_fold.shape[2]),
            seed=0)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f'../weights/{name}.h5',
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=0)
    if dim == 2:
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, mode="min")
    else:
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, mode="min")

    if train:
        hist = model.fit(
            X_train_fold,
            keras.utils.to_categorical(
                y_train_fold,
                num_classes=10),
            validation_data=(
                X_valid_fold,
                keras.utils.to_categorical(
                    y_valid_fold,
                    num_classes=10)),
            epochs=1000,
            shuffle=True,
            verbose=0,
            batch_size=64,
            callbacks=[
                checkpoint_cb,
                early_stopping_cb])
    model.load_weights(f'../weights/{name}.h5')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f'acc: {skm.accuracy_score(y_test, y_pred):0.3f}')
    if train:
        with open(f'../result/pickle/{name}.pickle', 'wb') as f:
            pickle.dump(y_test, f)
            pickle.dump(y_pred, f)
            pickle.dump(hist.history, f)
    else:
        return X_test, y_test, y_pred, model, gasf, gadf


if __name__ == '__main__':
    for dim in [2]:  # [2, 1]:
        for k in [5]:  # [1, 2, 3, 4, 5]:
            if dim == 2:
                for image_size in [250]:  # [50, 100, 150, 200]:
                    main(k, dim, image_size, train=True)
            else:
                main(k, dim, train=True)
