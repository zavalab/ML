import pickle
import itertools
import numpy as np
from timeit import default_timer as timer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, r2_score


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def hp_dict():
    hp_all = {'filters': [8, 16, 24, 32],
              'units': [16, 32, 48, 64],
              'learning_rate': [5e-3, 1e-3, 5e-4],
              'batch_size': [8, 16, 32]}

    keys = sorted(hp_all)
    combinations = list(itertools.product(*(hp_all[key] for key in keys)))

    hp = []
    for i in range(len(combinations)):
        hp_ = {}

        for j, key in enumerate(keys):
            hp_[key] = combinations[i][j]

        hp_ = dotdict(hp_)
        hp.append(hp_)
    return hp


def cnn(hp, shape, job='cls'):
    x_in = Input(shape=shape)

    x = Conv3D(hp.filters, 3, activation='relu')(x_in)
    x = Conv3D(hp.filters, 3, activation='relu')(x)
    x = MaxPool3D((3, 2, 2))(x)

    for _ in range(2):
        x = Conv3D(hp.filters, 3, activation='relu')(x)
        x = Conv3D(hp.filters, 3, activation='relu')(x)
        x = MaxPool3D((3, 2, 2))(x)

    x = Flatten()(x)
    x = Dense(hp.units, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(hp.units, activation='relu')(x)
    x = Dropout(0.2)(x)

    if job == 'cls':
        x = Dense(4, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        metrics = ['acc']
    elif job == 'reg':
        x = Dense(1, activation='linear')(x)
        loss = 'mse'
        metrics = ['mae']

    model = Model(inputs=x_in, outputs=x)

    model.compile(optimizer=Adam(learning_rate=hp.learning_rate),
                  loss=loss, metrics=metrics)
    return model


def load_data(data='main', chem='o3'):
    with open('../data/o3cl2/video.pickle', 'rb') as handle:
        x = pickle.load(handle)
        y = pickle.load(handle)

    x = x - x.min()
    x = x / x.max()

    yo = y[:, 0]
    yc = y[:, 1]

    if data == 'main':
        idx = np.where((yo != 0) & (yc != 3.5))[0]
    elif data == 'unseen':
        idx = np.where(yc == 3.5)[0]

    x = x[idx]
    yo = yo[idx]
    yc = yc[idx]

    x = np.random.RandomState(0).permutation(x)[:32]
    yo = np.random.RandomState(0).permutation(yo)[:32]
    yc = np.random.RandomState(0).permutation(yc)[:32]

    if chem == 'o3':
        return x, yo
    elif chem == 'cl2':
        return x, yc


def rotation(x, y):
    x1 = np.rot90(x, axes=(2, 3))
    x2 = np.rot90(x1, axes=(2, 3))
    x3 = np.rot90(x2, axes=(2, 3))
    x_rot = np.concatenate((x, x1, x2, x3))
    y_rot = np.concatenate([y, y, y, y])
    x_rot = np.random.RandomState(0).permutation(x_rot)
    y_rot = np.random.RandomState(0).permutation(y_rot)
    return x_rot, y_rot


def param2name(hp):
    name = ''
    for key in hp.keys():
        name += key + '_' + str(hp[key]) + '_'
    return name


def train(hp, x_train, y_train, x_valid, y_valid):
    name = param2name(hp)

    print(f'{name} started...')
    shape = x_train.shape[1:]

    model = cnn(hp, shape, job='reg')

    cp = ModelCheckpoint(f'../weight/o3cl2/{name}.h5', monitor='val_loss',
                         verbose=2, save_best_only=True,
                         mode='min')
    els = EarlyStopping(monitor='val_loss', mode='min', patience=20)

    t0 = timer()
    history = model.fit(x_train, y_train, batch_size=hp.batch_size,
                        epochs=1, callbacks=[cp, els],
                        validation_data=(x_valid, y_valid), verbose=1).history
    t1 = timer()

    model.load_weights(f'./weight/o3cl2/{name}.h5')

    y_pred = model.predict(x_valid)

    if hp.job == 'cls':
        acc = accuracy_score(y_valid, y_pred)
    elif hp.job == 'reg':
        acc = r2_score(y_valid, y_pred)

    with open(f'../history/o3cl2/{name}.pickle', 'wb') as handle:
        pickle.dump(history, handle)

    return model, acc, name, t1 - t0


def main():
    for job in ['reg', 'cls']:
        for chem in ['o3', 'cl2']:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

            x, y = load_data(data='main', chem=chem)

            le = LabelEncoder()
            yc = le.fit_transform(y)

            xu, yu = load_data(data='unseen', chem=chem)

            count = 1
            for train_id, test_id in cv.split(x, yc):
                x_train, x_test = x[train_id], x[test_id]
                yc_train, yc_test = yc[train_id], yc[test_id]

                if job == 'cls':
                    y = yc

                y_train, y_test = y[train_id], y[test_id]
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                                      stratify=yc_train,
                                                                      test_size=0.25,
                                                                      random_state=0)
                x_train, y_train = rotation(x_train, y_train)

                model = []
                acc = []
                model_name = []
                time = []

                hps = hp_dict()
                for hp in hps:
                    hp['cv'] = count
                    hp['job'] = job
                    hp['chem'] = chem
                    model_, acc_, model_name_, time_ = train(
                        hp, x_train, y_train, x_valid, y_valid)
                    model.append(model_)
                    acc.append(acc_)
                    model_name.append(model_name_)
                    time.append(time_)

                idx = np.argmax(acc)

                acc_best = acc[idx]
                model_best = model[idx]
                name_best = model_name[idx]
                time_best = time[idx]

                y_pred = model_best.predict(x_test)
                name = str(count) + '_' + job + '_' + chem
                with open(f'../result/cnn/o3cl2/test_{name}.pickle', 'wb') as handle:
                    pickle.dump(y_test, handle)
                    pickle.dump(y_pred, handle)
                    pickle.dump(name_best, handle)
                    pickle.dump(time_best, handle)

                if job == 'reg':
                    y_pred_u = model_best.predict(xu)
                    with open(f'../result/cnn/o3cl2/unseen_{name}.pickle', 'wb') as handle:
                        pickle.dump(yu, handle)
                        pickle.dump(y_pred_u, handle)

                count += 1


if __name__ == '__main__':
    main()
