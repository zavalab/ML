import os
import ast
import pickle
import platform

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from gnn_uq.load_data import load_data
from gnn_uq.gnn_model import RegressionUQSpace, nll


def post_train(arch, name, dataset, seed):
    """
    A function to train the best k models with certain architecture choices.
    Args:
        dataset:
        arch: str, string of a list of choice
        name: str, saving name
        x_train: np.array, training data
        y_train: np.array, training label
        x_valid: np.array, validation data
        y_valid: np.array, validation label

    Returns:
        saving a pickle file
    """


    # load data
    # clear the tensorflow session
    tf.keras.backend.clear_session()

    (x_train, y_train), (x_valid, y_valid), (x_test,
                                             y_test), (mean, std) = load_data(dataset, test=1, split_type='random', seed=seed)

    # turn str of architecture choice to a list
    arch = ast.literal_eval(arch)

    # define model input and output shapes
    input_shape = [item.shape[1:] for item in x_train]
    output_shape = y_train.shape[1:]
    shapes = dict(input_shape=input_shape, output_shape=output_shape)

    # define regression uncertainty quantification space
    space = RegressionUQSpace(**shapes).build()

    # load the model with a certain architecture
    model = space.sample(choice=arch)
    print(model.summary())

    if dataset == 'qm7':
        batch_size = 100
        epochs = 200
        patience = 100
        lr = 1e-3
    else:
        batch_size = 50
        epochs = 200
        patience = 100
        lr = 1e-3

    # compile and train the model
    model.compile(loss=nll, optimizer=Adam(learning_rate=lr))
    if not os.path.exists(f'./NEW_POST_MODEL/post_model_{dataset}_random_{seed}/'):
        os.mkdir(f'./NEW_POST_MODEL/post_model_{dataset}_random_{seed}/')
    if not os.path.exists(f'./NEW_POST_RESULT/post_result_{dataset}_random_{seed}/'):
        os.mkdir(f'./NEW_POST_RESULT/post_result_{dataset}_random_{seed}/')

    cp = ModelCheckpoint(f'./NEW_POST_MODEL/post_model_{dataset}_random_{seed}/best_{name}.h5', monitor='val_loss',
                         verbose=2, save_best_only=True,
                         save_weights_only=True, mode='min')

    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, callbacks=[cp, es],
                        validation_data=(x_valid, y_valid), verbose=2).history
    # make prediction
    model.load_weights(f'./NEW_POST_MODEL/post_model_{dataset}_random_{seed}/best_{name}.h5')
    y_test = y_test.squeeze()
    y_pred = []
    for i in range(len(x_test[0])):
        x_test_ = [x_test[j][i][None, ...] for j in range(len(x_test))]
        y_dist_ = model(x_test_)
        y_pred.append([y_dist_.loc, y_dist_.scale])

    y_pred = np.array(y_pred).squeeze()

    unexplained_error = np.mean((y_test-y_pred[:, 0])**2)
    total_error = np.mean((y_test-np.mean(y_test))**2)
    R_squared = 1 - unexplained_error / total_error

    print(f'RMSE: {unexplained_error ** 0.5: 0.4f}')
    print(f'R2: {R_squared: 0.4f}')

    with open(f'./result/NEW_POST_RESULT/post_result_{dataset}_random_{seed}/{name}.pickle', 'wb') as handle:
        pickle.dump(y_test, handle)
        pickle.dump(y_pred, handle)
        pickle.dump(history, handle)

    y_valid = y_valid.squeeze()
    
    y_pred_valid = []
    for i in range(len(x_valid[0])):
        x_valid_ = [x_valid[j][i][None, ...] for j in range(len(x_valid))]
        y_dist_ = model(x_valid_)
        y_pred_valid.append([y_dist_.loc, y_dist_.scale])

    y_pred_valid = np.array(y_pred_valid).squeeze()
    
    with open(f'./result/NEW_POST_RESULT/post_result_{dataset}_random_{seed}/val_{name}.pickle', 'wb') as handle:
        pickle.dump(y_valid, handle)
        pickle.dump(y_pred_valid, handle)


def main(dataset='qm7', topk=20, seed=0):
    """
    A function to perform topk model post training.
    Args:
        dataset: str, data set name
        topk: int, the number of best models to pick
    """
    MODEL_DIR = f'./result/RE_{dataset}_random_{seed}/save/model/'

    arch_path = MODEL_DIR.split('save')[0] + 'results.csv'
    df = pd.read_csv(arch_path)

    loss_min = []
    arch_min = []
    id_min = []
    for i in range(len(df)):
        loss_min_ = np.argsort(df['objective'])[::-1].values[i]
        arch_min_ = df['arch_seq'][loss_min_]
        id_min_ = df['id'][loss_min_]

        if not any(np.array_equal(arch_min_, x) for x in arch_min):

            # if id_min_ > 50:
            loss_min.append(loss_min_)
            arch_min.append(arch_min_)
            id_min.append(id_min_)

    for i in range(topk):
        print(f"Model {i + 1} started... previous loss {df['objective'][loss_min_]}")
        post_train(arch_min[i], id_min[i], dataset, seed)


if __name__ == '__main__':
    for dataset in ['delaney', 'freesolv', 'lipo', 'qm7']: 
        for seed in range(8):
            main(dataset, topk=10, seed=seed)