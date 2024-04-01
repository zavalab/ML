import os
import glob
import pickle
import warnings
import pandas as pd
import numpy as np
import scipy.optimize as optimize

from scipy.special import erfinv
from gnn_uq.load_data import load_data
from scipy.stats import spearmanr, norm
from sklearn.metrics import silhouette_score
from rdkit.Chem import Descriptors
from sklearn.cluster import KMeans


def NLL(var, y_pred, y_test):
    """
    Calculate the negative log likelihood (NLL) loss.

    Parameters:
        var (float): Variance.
        y_pred (numpy.ndarray): Predicted values.
        y_test (numpy.ndarray): True values.

    Returns:
        float: Negative log likelihood loss.
    """
    return np.mean(np.log(2 * np.pi) + np.log(var) + np.divide((y_pred - y_test) ** 2, var)  ) / 2

    
def cNLL(var, y_pred, y_test, var_val, y_pred_val, y_val):
    """
    Calculate the calibrated negative log likelihood (cNLL) loss.

    Parameters:
        var (float): Variance.
        y_pred (numpy.ndarray): Predicted values.
        y_test (numpy.ndarray): True values.
        var_val (float): Validation variance.
        y_pred_val (numpy.ndarray): Predicted values for validation set.
        y_val (numpy.ndarray): True values for validation set.

    Returns:
        float: Calibrated negative log likelihood loss.
    """
    def f(params):
        a, b = params
        return np.mean(np.log(2 * np.pi) + np.log(a * var_val + b) + np.divide((y_pred_val - y_val) ** 2, (a * var_val + b))  ) / 2 
    
    def constraint(params):
        a, b = params
        return a * var + b - 0.001
    
    con = {'type': 'ineq', 'fun': constraint}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        initial_guess = [1.0, 0.0]
        result = optimize.minimize(f, initial_guess, constraints=con, method='SLSQP')

        a, b = result.x
        
    return np.mean(np.log(2 * np.pi) + np.log(a * var + b) + np.divide((y_pred - y_test) ** 2, (a * var + b))  ) / 2


def miscal_area(var, y_pred, y_test):
    """
    Calculate the miscalibration area.

    Parameters:
        var (float): Variance.
        y_pred (numpy.ndarray): Predicted values.
        y_test (numpy.ndarray): True values.

    Returns:
        tuple: A tuple containing the miscalibration area and the fractions.
    """
    err = np.abs(y_test - y_pred) 

    fractions = np.zeros(101)
    fractions[100] = 1

    bin_scaling = [0]
    for i in range(1, 100):
        bin_scaling.append(erfinv(i / 100) * np.sqrt(2))

    for i in range(1, 100):
        bin_var = np.sqrt(var) * bin_scaling[i]
        bin_fraction = np.mean(bin_var >= err)
        fractions[i] = bin_fraction

    # trapezoid rule
    auce = np.sum(0.01 * np.abs(fractions - np.arange(101) / 100))

    return auce, fractions


def calibrate(var, y_pred, y_test, var_val, y_pred_val, y_val):
    """
    Calibrate the variance.

    Parameters:
        var (float): Variance.
        y_pred (numpy.ndarray): Predicted values.
        y_test (numpy.ndarray): True values.
        var_val (float): Validation variance.
        y_pred_val (numpy.ndarray): Predicted values for validation data.
        y_val (numpy.ndarray): True values for validation data.

    Returns:
        tuple: A tuple containing the calibrated parameters (a, b).
    """
    
    def f(params):
        a, b = params
        return np.mean(np.log(2 * np.pi) + np.log(a * var_val + b) + np.divide((y_pred_val - y_val) ** 2, (a * var_val + b))  ) / 2 
    
    def constraint(params):
        a, b = params
        return a * var + b - 0.001
    con = {'type': 'ineq', 'fun': constraint}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        initial_guess = [1.0, 0.0]
        result = optimize.minimize(f, initial_guess, constraints=con, method='SLSQP')

        a, b = result.x
        
    return a, b


def sort_best_RE(ROOT_DIR='./', dataset='delaney', seed=0, SPLIT_TYPE="523"):
    """
    Sort the best models based on their objectives.

    Parameters:
        ROOT_DIR (str): Root directory path. Defaults to './'.
        dataset (str): Dataset name. Defaults to 'delaney'.
        seed (int): Random seed. Defaults to 0.
        SPLIT_TYPE (str): Split type. Defaults to "523".

    Returns:
        tuple: A tuple containing arrays of the minimum losses, corresponding architectures, and job IDs.
    """
    
    MODEL_DIR = os.path.join(ROOT_DIR, f'NEW_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}/save/model/')

    arch_path = MODEL_DIR.split('save')[0] + 'results.csv'
    df = pd.read_csv(arch_path)

    loss_min = []
    arch_min = []
    id_min   = []
    for i in range(len(df)):
        idx_min_  = np.argsort(df['objective'])[::-1].values[i]
        loss_min_ = df['objective'][idx_min_]
        arch_min_ = df['p:arch_seq'][idx_min_]
        id_min_   = df['job_id'][idx_min_]

        if not any(np.array_equal(arch_min_, x) for x in arch_min):

            loss_min.append(loss_min_)
            arch_min.append(arch_min_)
            id_min.append(id_min_)
            
    return np.array(loss_min), np.array(arch_min), np.array(id_min)


def get_result(ROOT_DIR, datasets, split_types, range_seeds):
    """
    Retrieve results from top k results.

    Parameters:
        ROOT_DIR (str): Root directory path.
        datasets (list): List of dataset names.
        split_types (list): List of split types.
        range_seeds (list): List of random seeds.

    Returns:
        dict: A dictionary containing the results for each combination of dataset, split type, and seed.
    """
    out_result = {}
    for dataset in datasets:
        for split_type in split_types:
            sizes = (0.5, 0.2, 0.3) if split_type == "523" else (0.8, 0.1, 0.1)
            for seed in range_seeds:
                _, _, _, (mean, std) = load_data(dataset=dataset, verbose=0, test=1, norm=1, seed=seed, sizes=sizes)
                loss_min, arch_min, id_min = sort_best_RE(dataset=dataset, seed=seed, SPLIT_TYPE=split_type)

                y_mu = []
                y_std = []

                y_val_mu = []
                y_val_std = []

                for k in range(10):
                    in_file = os.path.join(ROOT_DIR, f"NEW_POST_RESULT/post_result_{dataset}_random_{seed}_split_{split_type}/test_{id_min[k]}.pickle")
                    with open(in_file, "rb") as handle:
                        y_test = pickle.load(handle)
                        y_mu_temp = pickle.load(handle)
                        y_std_temp = pickle.load(handle)
                        hist_temp = pickle.load(handle)
                    
                        y_mu_temp = y_mu_temp * std + mean
                        y_std_temp = y_std_temp * std
                    
                    y_mu.append(y_mu_temp)
                    y_std.append(y_std_temp)
                    
                    in_file2 = os.path.join(ROOT_DIR, f"NEW_POST_RESULT/post_result_{dataset}_random_{seed}_split_{split_type}/val_{id_min[k]}.pickle")
                    with open(in_file2, "rb") as handle:
                        y_val = pickle.load(handle)
                        y_vmu_temp = pickle.load(handle)
                        y_vstd_temp = pickle.load(handle)

                        y_vmu_temp = y_vmu_temp * std + mean
                        y_vstd_temp = y_vstd_temp * std
                    
                    y_val_mu.append(y_vmu_temp)
                    y_val_std.append(y_vstd_temp)
                    
                y_test_temp = y_test * std + mean
                y_test_pred = np.array(y_mu).mean(axis=0)
                y_test_epis = np.array(y_mu).var(axis=0)
                y_test_alea = (np.array(y_std)**2).mean(axis=0)
                    
                y_val_temp = y_val * std + mean
                y_val_pred = np.array(y_val_mu).mean(axis=0)
                y_val_epis = np.array(y_val_mu).var(axis=0)
                y_val_alea = (np.array(y_val_std)**2).mean(axis=0)

                out_result[(dataset, split_type, seed)] = (y_test_temp, y_test_pred, y_test_epis, y_test_alea, y_val_temp, y_val_pred, y_val_epis, y_val_alea)

    return out_result


def get_result_random(ROOT_DIR, datasets, split_types, range_seeds):
    """
    Retrieve results from random ensemble.

    Parameters:
        ROOT_DIR (str): Root directory path.
        datasets (list): List of dataset names.
        split_types (list): List of split types.
        range_seeds (list): List of random seeds.

    Returns:
        dict: A dictionary containing the results for each combination of dataset, split type, and seed.
    """
    out_result = {}
    for dataset in datasets:
        for split_type in split_types:
            sizes = (0.5, 0.2, 0.3) if split_type == "523" else (0.8, 0.1, 0.1)
            for seed in range_seeds:
                _, _, _, (mean, std) = load_data(dataset=dataset, verbose=0, test=1, norm=1, seed=seed, sizes=sizes)

                y_mu = []
                y_std = []

                y_val_mu = []
                y_val_std = []

                file_list = os.path.join(ROOT_DIR, f"NEW_POST_RESULT_RANDOM/post_result_{dataset}_random_{seed}_split_{split_type}/test_*.pickle")
                file_list = sorted(glob.glob(file_list))
        
        
                for k in range(10):
                    in_file = file_list[k]
                    with open(in_file, "rb") as handle:
                        y_test = pickle.load(handle)
                        y_mu_temp = pickle.load(handle)
                        y_std_temp = pickle.load(handle)
                        hist_temp = pickle.load(handle)
                    
                        y_mu_temp = y_mu_temp * std + mean
                        y_std_temp = y_std_temp * std
                    
                    y_mu.append(y_mu_temp)
                    y_std.append(y_std_temp)

                    with open(in_file.replace("test", "val"), "rb") as handle:
                        y_val = pickle.load(handle)
                        y_vmu_temp = pickle.load(handle)
                        y_vstd_temp = pickle.load(handle)

                        y_vmu_temp = y_vmu_temp * std + mean
                        y_vstd_temp = y_vstd_temp * std
                    
                    y_val_mu.append(y_vmu_temp)
                    y_val_std.append(y_vstd_temp)
                    
                    
                y_test_temp = y_test * std + mean
                y_test_pred = np.array(y_mu).mean(axis=0)
                y_test_epis = np.array(y_mu).var(axis=0)
                y_test_alea = (np.array(y_std)**2).mean(axis=0)
                    
                y_val_temp = y_val * std + mean
                y_val_pred = np.array(y_val_mu).mean(axis=0)
                y_val_epis = np.array(y_val_mu).var(axis=0)
                y_val_alea = (np.array(y_val_std)**2).mean(axis=0)

                out_result[(dataset, split_type, seed)] = (y_test_temp, y_test_pred, y_test_epis, y_test_alea, y_val_temp, y_val_pred, y_val_epis, y_val_alea)

    return out_result


def combine_result(result, dataset="delaney", SPLIT_TYPE="523"):
    """
    Combine results for a given dataset and split type.

    Parameters:
        result (dict): Dictionary containing the results for each combination of dataset, split type, and seed.
        dataset (str): Name of the dataset.
        SPLIT_TYPE (str): Type of split.

    Returns:
        tuple: A tuple containing NLLs, cNLLs, Miscalibration Areas, and Spearman's Correlations.
    """
    nlls = np.zeros(8)
    cnlls = np.zeros(8)
    mas = np.zeros(8)
    sps = np.zeros(8)

    for seed in range(8):
        y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = result[(dataset, SPLIT_TYPE, seed)]

        nlls[seed]  = NLL(y_epis+y_alea, y_pred, y_true)
        cnlls[seed] = cNLL(y_epis+y_alea, y_pred, y_true, v_epis+v_alea, v_pred, v_true)
        mas[seed]   = miscal_area(y_epis+y_alea, y_pred, y_true)[0]
        sps[seed]   = spearmanr(y_epis+y_alea, np.abs(y_true - y_pred)).correlation
        
    return nlls, cnlls, mas, sps


def conf_level(y_true, y_mu, y_std):
    """
    Calculate the confidence-based calibration curve.

    Parameters:
        y_true (array_like): True labels.
        y_mu (array_like): Predicted means.
        y_std (array_like): Predicted standard deviations.

    Returns:
        tuple: A tuple containing confidence levels and calibration curve.
    """
    confidence_levels = np.linspace(0, 1, 100)
    calibration_curve = []
    
    z_values = norm.ppf(1 - (1 - confidence_levels) / 2)
    
    for z, cl in zip(z_values, confidence_levels):
        lower_bound = y_mu - z * y_std
        upper_bound = y_mu + z * y_std
        
        within_interval = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        calibration_curve.append( within_interval )
        
    return confidence_levels, np.array(calibration_curve)


def extract_functional_groups(mol):
    """
    Extract functional groups from a molecule.

    Parameters:
        mol (RDKit Mol object): RDKit Mol object representing the molecule.

    Returns:
        list: List of functional group names.
    """
    fg_counts = Descriptors.MolWt(mol, True)
    fg_names = ["{}{}".format(fg[0], fg[1]) for fg in fg_counts.items()]
    return fg_names


def calculate_silhouette_score(data, k):
    """
    Calculate the silhouette score for a given dataset and number of clusters using KMeans clustering.

    Parameters:
        data (array-like): Input data.
        k (int): Number of clusters.

    Returns:
        float: Silhouette score.
    """
    model = KMeans(n_clusters=k, random_state=0)
    cluster_labels = model.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg