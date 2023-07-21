import glob
import pickle
import warnings

import numpy as np
import scipy.optimize as optimize

# from gekko import GEKKO
from scipy.special import erfinv
from gnn_uq.load_data import load_data
from gnn_uq.data_utils import split_data, get_data
import uncertainty_toolbox as uct


import scipy.optimize as optimize
import warnings

def load_post_training(dataset, data_path, seed):
    # load data
    _, _, _, (mean, std) = load_data(
        dataset=dataset, split_type='random', test=1, seed=seed)
    
    data = get_data(path=f'../data/{dataset}.csv')
    
    mol_train, _, mol_valid, _, mol_test, _ = split_data(
        data, split_type='random', show_mol=True, seed=seed)

    # load post training result
    file = sorted(glob.glob(data_path + f'*.pickle'))
    file_val = sorted(glob.glob(data_path + f'val*.pickle'))
    file = list(set(file) - set(file_val))

    # load test
    y_loc = []  # prediction
    y_scale = []  # variance
    y_loss = []
    for file_ in file:
        with open(file_, 'rb') as f:
            y_test = pickle.load(f)
            y_pred = pickle.load(f)
            hist = pickle.load(f)

        # scale it back before standardization
        y_test = y_test * std + mean
        y_loc_ = y_pred[:, 0] * std + mean
        y_scale_ = y_pred[:, 1] * std

 
        y_loc.append(y_loc_)
        y_scale.append(y_scale_)
        y_loss.append(min(hist['val_loss']))

    y_loc = np.array(y_loc)
    y_scale = np.array(y_scale)
    y_loss = np.array(y_loss)
    
    # load valid
    y_loc_val = []  # prediction
    y_scale_val = []  # variance
    for file_val_ in file_val:
        with open(file_val_, 'rb') as f:
            y_val = pickle.load(f)
            y_pred_val = pickle.load(f)

        # scale it back before standardization
        y_val = y_val * std + mean
        y_loc_val_ = y_pred_val[:, 0] * std + mean
        y_scale_val_ = y_pred_val[:, 1] * std

 
        y_loc_val.append(y_loc_val_)
        y_scale_val.append(y_scale_val_)

    y_loc_val = np.array(y_loc_val)
    y_scale_val = np.array(y_scale_val)

    return y_test, y_loc, y_scale, y_loss, mol_train, mol_valid, mol_test, y_val, y_loc_val, y_scale_val


def NLL(unc, y_pred, y_test):
    nll = np.mean(np.log(2 * np.pi) + np.log(unc) + np.divide((y_pred - y_test) ** 2, unc)  ) / 2

    return nll

    
def cNLL(unc, y_pred, y_test, unc_val, y_pred_val, y_val):
    def f(params):
        a, b = params
        return np.mean(np.log(2 * np.pi) + np.log(a * unc_val + b) + np.divide((y_pred_val - y_val) ** 2, (a * unc_val + b))  ) / 2 
    
    def constraint(params):
        a, b = params
        return a * unc_val + b - 1
    
    con = {'type': 'ineq', 'fun': constraint}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        initial_guess = [1.0, 0.0]
        result = optimize.minimize(f, initial_guess, constraints=con, method='SLSQP')

        a, b = result.x
        
    return np.mean(np.log(2 * np.pi) + np.log(a * unc + b) + np.divide((y_pred - y_test) ** 2, (a * unc + b))  ) / 2


def miscal_area(unc, y_pred, y_test):
    err = np.abs(y_test - y_pred) 

    fractions = np.zeros(101)
    fractions[100] = 1

    bin_scaling = [0]
    for i in range(1, 100):
        bin_scaling.append(erfinv(i / 100) * np.sqrt(2))

    for i in range(1, 100):
        bin_unc = np.sqrt(unc) * bin_scaling[i]
        bin_fraction = np.mean(bin_unc >= err)
        fractions[i] = bin_fraction

    # trapezoid rule
    auce = np.sum(0.01 * np.abs(fractions - np.arange(101) / 100))

    return auce, fractions