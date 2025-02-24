import itertools
import os
import random
from pastillation.model.cnn import cnn


def train_cnn(fold_number, params):
    """Train a CNN model for a given fold number and hyperparameters"""
    WEIGHT_DIR = "/Pastillation/weight/"
    RESULT_DIR = "/Pastillation/result_pickle/"
    cnn(WEIGHT_DIR, RESULT_DIR, fold_number, params, train=True, seed=0)


def job_array(idx, max_idx):
    """Generate a list of hyperparameters for a given job index with randomized order"""
    fold = [0, 1, 2, 3, 4]
    hdim = [64, 128, 256]
    batch = [32, 64, 128]
    loss = ["mse"]
    lr = [0.001, 0.0005, 0.005]
    mode = ["1d", "2d"]

    combs = list(itertools.product(fold, hdim, batch, loss, lr, mode))

    combs_new = []
    for comb in combs:
        fold, hdim, batch, loss, lr, mode = comb
        name = f"cnn_{mode}_{fold}_{hdim}_{batch}_{loss}_{lr}"
        RESULT_DIR = "/Pastillation/result_pickle/"
        results_path = os.path.join(RESULT_DIR, f"{name}.pickle")
        WEIGHT_DIR = "/Pastillation/weight/"
        model_path = os.path.join(WEIGHT_DIR, f"{name}.h5")

        if not os.path.exists(results_path) and not os.path.exists(model_path):
            combs_new.append(comb)

    combs = combs_new
    random.shuffle(combs)

    if max_idx > len(combs):
        size = 1
    else:
        size = len(combs) // max_idx
    start = idx * size
    end = start + size

    return combs[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
    combs = job_array(idx, max_idx)

    for comb in combs:
        fold, hdim, batch, loss, lr, mode = comb
        print(comb)
        params = dict()
        params["hdim"] = hdim
        params["batch"] = batch
        params["loss"] = loss
        params["lr"] = lr
        params["mode"] = mode

        train_cnn(fold, params)
