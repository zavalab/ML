import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from focal_loss import BinaryFocalLoss
from timeit import default_timer as timer
from resnet3d import Resnet3DBuilder
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR   = "../data/"
WEIGHT_DIR = "../weight/"
PICKLE_DIR = "../result/cnn/"

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=2):
        """
        Initialize the DataGenerator object.

        Args:
            file_paths (list): List of file paths to load data from.
            labels (list): List of corresponding labels.
            batch_size (int, optional): Size of each batch. Defaults to 2.

        Returns:
            None
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        """
        Generate one batch of data.

        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: A tuple containing the batch data and corresponding labels.
                - data_batch (np.ndarray): Batch of data.
                - batch_labels (list): Corresponding batch of labels.
        """
        batch_files = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        data_batch = []

        for _, file in enumerate(batch_files):
            with open(file, 'rb') as handle:
                x = pickle.load(handle)
            data_batch.append(x)

        data_batch = np.array(data_batch)

        return data_batch, batch_labels
    
    
def get_data(file_paths, labels):
    """
    Load data from given file paths and return as numpy array along with labels.

    Args:
        file_paths (list): List of file paths to load data from.
        labels (list): List of corresponding labels.

    Returns:
        tuple: A tuple containing the loaded data and corresponding labels.
            - data (np.ndarray): Loaded data as numpy array.
            - labels (list): Corresponding labels.
    """
    data = []

    for _, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as handle:
            volume = pickle.load(handle)   
        data.append(volume)

    data = np.array(data)
    
    return data, labels

def main(fold=0, zoom_ratio=0.5, gamma=2, lr=0.001):
    """
    Execute main training loop.

    Args:
        fold (int, optional): The fold number. Defaults to 0.
        zoom_ratio (float, optional): The zoom ratio. Defaults to 0.5.
        gamma (int, optional): The gamma value. Defaults to 2.
        lr (float, optional): The learning rate. Defaults to 0.001.

    Returns:
        None
    """
    K.clear_session()
    
    with open(os.path.join(DATA_DIR, 'five_fold.pickle'), 'rb') as handle:
        fold_file  = pickle.load(handle)
        fold_group = pickle.load(handle)

    batch_size_mapping = {0.25: 8, 
                          0.125: 16, 
                          0.5: 2, 
                          1.0: 1}
    
    batch_size = batch_size_mapping.get(zoom_ratio, 1)

    file_train, file_valid, file_test = fold_file[fold]
    label_train, label_valid, label_test = fold_group[fold]
    
    file_train = [os.path.join(DATA_DIR, f"./zoom_{zoom_ratio}_data/{file_train[i].split('/')[-1]}") for i in range(len(file_train))]
    file_valid = [os.path.join(DATA_DIR, f"./zoom_{zoom_ratio}_data/{file_train[i].split('/')[-1]}") for i in range(len(file_valid))]
    file_test  = [os.path.join(DATA_DIR, f"./zoom_{zoom_ratio}_data/{file_train[i].split('/')[-1]}") for i in range(len(file_test))]
    
    le = LabelEncoder()
    label_train = le.fit_transform(label_train)
    label_valid = le.transform(label_valid)
    label_test  = le.transform(label_test)
    
    if zoom_ratio >= 0.5:
        train_generator = DataGenerator(file_train, label_train, batch_size)
        valid_generator = DataGenerator(file_valid, label_valid, batch_size)
        test_generator  = DataGenerator(file_test, label_test, 1)
        shape           = train_generator[0][0].shape[1:]
    else:
        x_train, y_train = get_data(file_train, label_train)
        x_valid, y_valid = get_data(file_valid, label_valid)
        x_test, y_test   = get_data(file_test, label_test)
        shape            = x_train.shape[1:]

    tf.random.set_seed(42)
    model = Resnet3DBuilder.build_resnet_50(shape, 1, reg_factor=0)

    weight_path = os.path.join(WEIGHT_DIR, f'z_{zoom_ratio}_f_{fold}_g_{gamma}_l_{lr}.h5')
    pickle_path  = os.path.join(PICKLE_DIR, f'z_{zoom_ratio}_g_{gamma}_l_{lr}')
    
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
        
    result_path = os.path.join(pickle_path, f'z_{zoom_ratio}_f_{fold}_g_{gamma}_l_{lr}.pickle')

    metrics = [
        keras.metrics.BinaryAccuracy(name='acc'),
        keras.metrics.Precision(name='pre'),
        keras.metrics.Recall(name='rec'),
    ]

    mcp = ModelCheckpoint(filepath=weight_path,
                          monitor='val_loss',
                          verbose=2, 
                          save_best_only=True, 
                          mode='min',
                          save_weights_only=True)
    
    es = EarlyStopping(monitor='val_loss', 
                       mode='min', 
                       patience=200)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=BinaryFocalLoss(gamma=gamma),
                  metrics=metrics)
    
    print(f'Started training: {result_path}')
    
    t0 = timer()
 
    if zoom_ratio >= 0.5:
        hist = model.fit(train_generator, validation_data=valid_generator,
                         epochs=200, callbacks=[mcp, es], verbose=0).history
    else:
        hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                         epochs=200, callbacks=[mcp, es], batch_size=8, shuffle=True, verbose=0).history

    t1 = timer()
    time_train = t1 - t0
    model.load_weights(weight_path)

    if zoom_ratio >= 0.5:
        y_pred_train = model.predict(train_generator, verbose=0)
        y_pred_valid = model.predict(valid_generator, verbose=0)
        y_pred = model.predict(test_generator, verbose=0)

    else:
        y_pred_train = model.predict(x_train, verbose=0)
        y_pred_valid = model.predict(x_valid, verbose=0)
        y_pred = model.predict(x_test, verbose=0)

    with open(result_path, 'wb') as handle:
        pickle.dump(label_test, handle)
        pickle.dump(y_pred, handle)
        pickle.dump(label_train, handle)
        pickle.dump(y_pred_train, handle)
        pickle.dump(label_valid, handle)
        pickle.dump(y_pred_valid, handle)
        pickle.dump(hist, handle)
        pickle.dump(time_train, handle)
        
    print(f'Finish training in {time_train:0.2f} sec: {result_path}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    
    parser.add_argument('--fold', type=int, required=True)
    
    parser.add_argument('--zoom_ratio', type=float, required=True)
    
    parser.add_argument('--gamma', type=int, required=True)
    
    parser.add_argument('--lr', type=float, required=True)

    args = parser.parse_args()
    
    main(args.fold, args.zoom_ratio, args.gamma, args.lr)