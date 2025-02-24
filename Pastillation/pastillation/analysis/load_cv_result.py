import os
import glob
import pickle
import numpy as np
import sklearn.metrics as skm
from tensorflow import keras

def load_cv_result(RESULT_PICKLE_DIR, fold, verbose=True, mode="1d"):
    """Load the best model results from a given cross-validation fold.

    This function loads model results from pickle files for a specific fold and selects the best
    model based on validation RMSE scores.

    Args:
        RESULT_PICKLE_DIR (str): Directory path containing the pickle result files
        fold (int): The cross-validation fold number to load results from
        verbose (bool, optional): Whether to print information about best model. Defaults to True
        mode (str, optional): Mode of CNN model ('1d' or '2d'). Defaults to "1d"

    Returns:
        tuple: A 7-element tuple containing:
            - history: Training history of the model
            - y_val: Validation target values
            - y_pred_val: Predicted values for validation set
            - y_test: Test target values  
            - y_pred: Predicted values for test set
            - time: Training time information
            - str: Filename of the best model

    Example:
        >>> history, y_val, y_pred_val, y_test, y_pred, time, filename = load_cv_result(
        ...     "results/", fold=0, verbose=True, mode="1d"
        ... )
    """
    pattern = os.path.join(RESULT_PICKLE_DIR, f"cnn_{mode}_{fold}*.pickle")
    files = sorted(glob.glob(pattern))

    rmses = []

    for file in files:

        with open(file, "rb") as handle:
            history = pickle.load(handle)
            y_val = pickle.load(handle)
            y_pred_val = pickle.load(handle)
            y_test = pickle.load(handle)
            y_pred = pickle.load(handle)
            time = pickle.load(handle)

        rmse = skm.mean_squared_error(y_val.ravel(), y_pred_val.ravel()) ** 0.5

        rmses.append(rmse)

    rmses = np.array(rmses)
    idx = np.argmin(rmses)

    with open(files[idx], "rb") as handle:
        history = pickle.load(handle)
        y_val = pickle.load(handle)
        y_pred_val = pickle.load(handle)
        y_test = pickle.load(handle)
        y_pred = pickle.load(handle)
        time = pickle.load(handle)

    if verbose:
        print(f"Best model: {os.path.basename(files[idx])}")
        print(rf"RMSE: {rmses[idx]:0.4f}")

    return (
        history,
        y_val,
        y_pred_val,
        y_test,
        y_pred,
        time,
        os.path.basename(files[idx]),
    )


def find_best_model_names(RESULT_PICKLE_DIR, folds=5, mode="*d"):
    """Find the best model names for each fold based on RMSE.

    Args:
        RESULT_PICKLE_DIR (str): Directory containing pickle files with model results
        folds (int, optional): Number of folds. Defaults to 5.
        mode (str, optional): Model mode pattern. Defaults to "*d".

    Returns:
        dict: Dictionary mapping fold numbers to best model filenames based on lowest RMSE
    """
    best_models = {}
    for fold in range(folds):
        pattern = os.path.join(RESULT_PICKLE_DIR, f"cnn_{mode}_{fold}*.pickle")
        files = sorted(glob.glob(pattern))

        rmses = []
        for file in files:
            with open(file, "rb") as handle:
                pickle.load(handle)
                y_val = pickle.load(handle)
                y_pred_val = pickle.load(handle)

            rmse = skm.mean_squared_error(y_val.ravel(), y_pred_val.ravel()) ** 0.5
            rmses.append(rmse)

        rmses = np.array(rmses)
        idx = np.argmin(rmses)

        best_models[f"{fold}"] = os.path.basename(files[idx])

    return best_models


def load_best_keras_models(WEIGHT_DIR, RESULT_PICKLE_DIR, folds=5, mode="*d"):
    """
    Load the best Keras models based on provided parameters.

    Args:
        WEIGHT_DIR (str): Directory containing model weights
        RESULT_PICKLE_DIR (str): Directory containing pickle results
        folds (int): Number of folds for cross validation, defaults to 5
        mode (str): Model architecture mode ('1d' or '2d'), defaults to '*d'

    Returns:
        list: List of loaded Keras models
    """

    best_models = find_best_model_names(RESULT_PICKLE_DIR, folds, mode=mode)

    models = []

    for fold in range(folds):
        best_model_file = best_models[f"{fold}"]
        print(best_model_file)

        mode = best_model_file.split("_")[1]
        hdim = int(best_model_file.split("_")[3])
        batch = int(best_model_file.split("_")[4])
        loss = best_model_file.split("_")[5]
        lr = float(best_model_file.split("_")[6].split(".pickle")[0])

        name = f"cnn_{mode}_{fold}_{hdim}_{batch}_{loss}_{lr}"
        weight_path = os.path.join(WEIGHT_DIR, f"{name}.h5")

        if mode == "1d":
            input_shape = (637, 65)
        else:
            input_shape = (637, 65, 1)
        inputs = keras.layers.Input(shape=input_shape)
        if mode == "1d":
            layer = keras.layers.Conv1D(hdim, 3, activation="relu", name="c1")(inputs)
            layer = keras.layers.AveragePooling1D(2, name="p1")(layer)
            layer = keras.layers.Conv1D(hdim, 3, activation="relu", name="c2")(layer)
            layer = keras.layers.AveragePooling1D(2, name="p2")(layer)
            layer = keras.layers.Conv1D(hdim, 3, activation="relu", name="c3")(layer)
            layer = keras.layers.AveragePooling1D(2, name="p3")(layer)
        elif mode == "2d":
            layer = keras.layers.Conv2D(hdim, 3, activation="relu", name="c1")(inputs)
            layer = keras.layers.AveragePooling2D(2, name="p1")(layer)
            layer = keras.layers.Conv2D(hdim, 3, activation="relu", name="c2")(layer)
            layer = keras.layers.AveragePooling2D(2, name="p2")(layer)
            layer = keras.layers.Conv2D(hdim, 3, activation="relu", name="c3")(layer)
            layer = keras.layers.AveragePooling2D(2, name="p3")(layer)
        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(128, activation="relu", name="d1")(layer)
        layer = keras.layers.Dense(128, activation="relu", name="d2")(layer)
        outputs = keras.layers.Dense(
            2,
            activation="relu",
            name="d3",
        )(layer)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=["mae", "mse"],
        )

        model.load_weights(weight_path)

        models.append(model)

    return models
