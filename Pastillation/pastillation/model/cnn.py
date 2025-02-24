import os
import pickle
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from timeit import default_timer as timer
from pastillation.data.preprocess_data import gen_cnn_data


def cnn(WEIGHT_DIR, RESULT_DIR, fold_number, params, train=False, seed=0):
    """Train and evaluate a Convolutional Neural Network model for regression.

    This function builds, trains and evaluates either a 1D or 2D CNN model based on provided parameters.
    The model architecture consists of convolutional layers followed by pooling layers and dense layers.
    Training results and predictions are saved to disk.

    Args:
        WEIGHT_DIR (str): Directory path to save model weights
        RESULT_DIR (str): Directory path to save results 
        fold_number (int): Cross validation fold number
        params (dict): Dictionary containing model hyperparameters:
            - hdim (int): Number of filters in conv layers
            - batch (int): Batch size for training
            - loss (str): Loss function name
            - lr (float): Learning rate
            - mode (str): CNN mode - either '1d' or '2d'
        train (bool, optional): Whether to train the model. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        None. Results are saved to disk including:
            - Model weights (.h5 file)
            - Training history
            - Validation and test predictions
            - Training time measurements
    """
    hdim = params["hdim"]
    batch = params["batch"]
    loss = params["loss"]
    lr = params["lr"]
    mode = params["mode"]
    name = f"cnn_{mode}_{fold_number}_{hdim}_{batch}_{loss}_{lr}"

    model_path = os.path.join(WEIGHT_DIR, f"{name}.h5")
    results_path = os.path.join(RESULT_DIR, f"{name}.pickle")

    if not os.path.exists(results_path):
        K.clear_session()

        # load data
        x_train, x_val, x_test, y_train, y_val, y_test, xmin, xmax = gen_cnn_data(
            fold_number
        )

        if mode == "1d":
            x_train = x_train.squeeze()
            x_val = x_val.squeeze()
            x_test = x_test.squeeze()

        # Set random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Build the model architecture
        inputs = keras.layers.Input(shape=(x_train.shape[1:]))
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
        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=["mae", "mse"],
        )

        # Define early stopping and model checkpoint callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=200
        )

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=model_path,
            verbose=2,
            save_weights_only=True,
            save_best_only=True,
        )
        print(name + " started ...")

        if not os.path.exists(results_path):
            t0 = timer()
            history = model.fit(
                x_train,
                y_train,
                epochs=500,
                batch_size=batch,
                validation_data=[x_val, y_val],
                callbacks=[early_stopping, model_checkpoint],
                verbose=2,
            ).history
            t1 = timer()
            model.load_weights(model_path)
            y_pred = model.predict(x_test).squeeze()
            t2 = timer()
            y_pred_val = model.predict(x_val).squeeze()

            print(f"# MAE: {skm.mean_absolute_error(y_test[:, 0], y_pred[:, 0]):0.4f}")
            print(f"# R2:  {skm.r2_score(y_test[:, 0], y_pred[:, 0]):0.4f}")

            print(f"# MAE: {skm.mean_absolute_error(y_test[:, 1], y_pred[:, 1]):0.4f}")
            print(f"# R2:  {skm.r2_score(y_test[:, 1], y_pred[:, 1]):0.4f}")

            with open(results_path, "wb") as f:
                pickle.dump(history, f)
                pickle.dump(y_val, f)
                pickle.dump(y_pred_val, f)
                pickle.dump(y_test, f)
                pickle.dump(y_pred, f)
                pickle.dump([t0, t1, t2], f)

        else:
            # Load results
            with open(results_path, "rb") as f:
                history = pickle.load(f)
                y_val = pickle.load(f)
                y_pred_val = pickle.load(f)
                y_test = pickle.load(f)
                y_pred = pickle.load(f)
                time = pickle.load(f)
