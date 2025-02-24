import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split

DATA_DIR = "Pastillation/data/"


def preprocess_data():
    """
    Preprocesses data from pickle files for model training.

    Loads data from pickle files, concatenates them, and applies min-max scaling.
    Each pickle file contains input features (x) and two sets of target variables (y, y2).
    """
    pickle_pattern = os.path.join(DATA_DIR, "*.pickle")
    pickle_files = sorted(glob.glob(pickle_pattern))

    x = []
    y = []
    y2 = []

    for file in pickle_files:
        with open(file, "rb") as handle:
            x_temp = pickle.load(handle)
            y_temp = pickle.load(handle)
            _ = pickle.load(handle)
            y_temp2 = pickle.load(handle)

        if "varied" in file:
            x_temp = x_temp[1:]
            y_temp = y_temp[1:]
            y_temp2 = y_temp2[1:]

        x.append(x_temp)
        y.append(y_temp)
        y2.append(y_temp2)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    y2 = np.concatenate(y2, axis=0)

    y = np.concatenate((y, y2), axis=1)

    x = x.squeeze()

    xmin = np.min(x)
    xmax = np.max(x)

    x = (x - xmin) / (xmax - xmin)
    x = x[..., None]

    return x, y, xmin, xmax


def get_data_splits(x, y, fold_number, n_splits=5):
    """
    Split data into training, validation, and test sets using k-fold cross-validation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        if fold == fold_number:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42
            )

            return x_train, x_val, x_test, y_train, y_val, y_test


def gen_cnn_data(fold_number: int) -> tuple:
    """
    Generate CNN data splits with preprocessing.

    Args:
        fold_number (int): The fold number for cross-validation split

    Returns:
        tuple: Contains the following elements:
            - x_train: Training input data
            - x_val: Validation input data
            - x_test: Test input data
            - y_train: Training target data
            - y_val: Validation target data
            - y_test: Test target data
            - xmin: Minimum value used in scaling
            - xmax: Maximum value used in scaling
    """
    # Preprocess raw data
    x, y, xmin, xmax = preprocess_data()

    # Split into train/val/test sets
    x_train, x_val, x_test, y_train, y_val, y_test = get_data_splits(x, y, fold_number)

    # Log shapes for debugging
    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, x_val, x_test, y_train, y_val, y_test, xmin, xmax)


if __name__ == "__main__":
    cnn_data = gen_cnn_data(fold_number=0)
