import os
import glob
import pickle

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, roc_auc_score,
                             roc_curve)


def get_metrics(y_true, y_pred, y_pred_prob, res, gamma, lr, group):
    """Calculate and return a dataframe of various metrics."""
    
    y_pred_rounded  = np.round(y_pred).squeeze()

    acc             = accuracy_score(y_true, y_pred_rounded)
    balanced_acc    = balanced_accuracy_score(y_true, y_pred_rounded)
    phi_coef        = matthews_corrcoef(y_true, y_pred_rounded)
    conf_matrix     = confusion_matrix(y_true, y_pred_rounded)
    fpr, tpr, th    = roc_curve(y_true, y_pred_prob)
    roc             = roc_auc_score(y_true, y_pred_prob)
    
    
    rng = np.random.RandomState(42)
    roc_boot        = []
    n_boot          = 1000
    for _ in range(n_boot):
        indices = rng.randint(0, len(y_true), len(y_true))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred_prob[indices])
        roc_boot.append(score)
        
    sorted_scores = np.array(roc_boot)
    sorted_scores.sort()
    
    roc_boot_mean = np.mean(sorted_scores)
    roc_boot_ci = [sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]]
    
    null_auc = 0.5
    _, roc_p_value = wilcoxon(sorted_scores - null_auc)
        

    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    metrics_dict = {
        "Res": [res],
        "Gamma": [gamma],
        "Lr": [lr],
        "Group": [group],
        "Accuracy": [acc],
        "Threshold": [str(th)],
        "Balanced Accuracy": [balanced_acc],
        "Phi Coefficient": [phi_coef],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "ROC": [roc],
        "ROC Boot Mean": [roc_boot_mean],
        "ROC Boot CI 2.5": [roc_boot_ci[0]],
        "ROC Boot CI 97.5": [roc_boot_ci[1]],
        "ROC Boot p": [roc_p_value],
        "Conf": [str(conf_matrix)],
        "TPR": [str(tpr)],
        "FPR": [str(fpr)]
    }

    df_metrics = pd.DataFrame(metrics_dict)
    return df_metrics


def get_cnn_result(cnn_dir, folds=5, gamma=0, resolution=0.25, learning_rate=0.001, return_predictions=False):
    all_y_tests, all_y_preds = [], []
    all_y_valids, all_y_valid_preds = [], []
    all_y_trains, all_y_train_preds = [], []
    
    for i in range(folds):
        file_path = glob.glob(os.path.join(cnn_dir,
            f"z_{resolution}_g_{gamma}_l_{learning_rate}/"
            f"z_{resolution}_f_{i}_g_{gamma}_l_{learning_rate}.pickle")
        )[0]
        with open(file_path, "rb") as handle:
            y_test, y_pred, y_train, y_pred_train, y_valid, y_pred_valid = [pickle.load(handle) for _ in range(6)]
        
        all_y_tests.append(y_test)
        all_y_preds.append(y_pred.squeeze())
        all_y_valids.append(y_valid)
        all_y_valid_preds.append(y_pred_valid.squeeze())
        all_y_trains.append(y_train)
        all_y_train_preds.append(y_pred_train.squeeze())
    
    all_y_tests_combined = np.concatenate(all_y_tests)
    all_y_preds_combined = np.concatenate(all_y_preds)
    all_y_valids_combined = np.concatenate(all_y_valids)
    all_y_valid_preds_combined = np.concatenate(all_y_valid_preds)
    all_y_trains_combined = np.concatenate(all_y_trains)
    all_y_train_preds_combined = np.concatenate(all_y_train_preds)
    
    if return_predictions:
        return (all_y_tests, all_y_preds, all_y_valids, all_y_valid_preds, all_y_trains, all_y_train_preds)
    
    df_metrics_valids = get_metrics(
        all_y_valids_combined, all_y_valid_preds_combined.squeeze(), all_y_valid_preds_combined.squeeze(), 
        resolution, gamma, learning_rate, group="Valid"
    )
    df_metrics_tests = get_metrics(
        all_y_tests_combined, all_y_preds_combined.squeeze(), all_y_preds_combined.squeeze(), 
        resolution, gamma, learning_rate, group="Test"
    )
    df_metrics_trains = get_metrics(
        all_y_trains_combined, all_y_train_preds_combined.squeeze(), all_y_train_preds_combined.squeeze(), 
        resolution, gamma, learning_rate, group="Train"
    )
    df_metrics = pd.concat((df_metrics_tests, df_metrics_valids, df_metrics_trains))

    return df_metrics


def biomarker_data(data_dir, file_list, l1=None, l2=None, l3=None, l4=None, LAA=False):
    # Loading and preprocessing CSV data
    DF = pd.read_csv(os.path.join(data_dir, "severity_LAA.csv"))

    # Replacing 'Not Found' entries with NaN
    DF.replace("Not Found", np.nan, inplace=True)

    # Converting certain columns to numeric and handling missing values
    DF["EOS"] = pd.to_numeric(DF["EOS"], errors="coerce")
    DF["TAD"] = pd.to_numeric(DF["TAD"], errors="coerce")
    DF["LAA"] = pd.to_numeric(DF["LAA"], errors="coerce")

    # Filling missing values with median values
    DF["EOS"].fillna(DF["EOS"].median(), inplace=True)
    DF["TAD"].fillna(DF["TAD"].median(), inplace=True)
    DF["LAA"].fillna(DF["LAA"].median(), inplace=True)
    
    # Extracting IDs from file paths
    extracted_ids = [path.split('/')[-1].split('-V')[0][4:] for path in file_list]

    # Filtering the dataframe based on the extracted IDs
    filtered_df = DF[DF['ID'].isin(extracted_ids)].copy()

    # Initializing LabelEncoders if not provided
    l1 = LabelEncoder() if l1 is None else l1
    l2 = LabelEncoder() if l2 is None else l2
    l3 = LabelEncoder() if l3 is None else l3
    l4 = LabelEncoder() if l4 is None else l4
        
    # Encoding categorical variables
    filtered_df['SIN'] = l1.fit_transform(filtered_df['SIN'])
    filtered_df['GERD'] = l2.fit_transform(filtered_df['GERD'])
    filtered_df['GENDER'] = l3.fit_transform(filtered_df['GENDER'])
    filtered_df['Group'] = l4.fit_transform(filtered_df['Group'])

    # Selecting columns based on the LAA flag
    if LAA:
        result_df = filtered_df[["SIN", "GERD", "BMI", "MRV", "EOS", "GENDER", "AGE", "TAD", "LAA"]]
    else:
        result_df = filtered_df[["SIN", "GERD", "BMI", "MRV", "EOS", "GENDER", "AGE", "TAD"]]

    label_df = filtered_df[['Group']]

    # Returning the processed data and label encoders
    return result_df.values.squeeze(), label_df.values.squeeze(), l1, l2, l3, l4