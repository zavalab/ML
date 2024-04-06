import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from result_utils import biomarker_data, get_metrics, get_cnn_result


def run_svc(DATA_DIR, fold_file, LAA=False):
    # Initializing lists to store results
    y_tests_svm = []
    y_preds_svm = []
    y_preds_svm_prob = []
    clfs_svm = []

    # Looping through each fold
    for i in range(5):
        # Biomarker data processing
        if i == 0:
            x_train, y_train, l1, l2, l3, l4 = biomarker_data(DATA_DIR,
                                                              fold_file[i][0],
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              LAA)
            l5 = MinMaxScaler()
        else:
            x_train, y_train, l1, l2, l3, l4 = biomarker_data(DATA_DIR, 
                                                              fold_file[i][0], 
                                                              l1, l2, l3, l4,
                                                              LAA)
        x_valid, y_valid, _, _, _, _ = biomarker_data(DATA_DIR, 
                                                      fold_file[i][1], 
                                                      l1, l2, l3, l4,
                                                      LAA)
        x_test, y_test, _, _, _, _ = biomarker_data(DATA_DIR, 
                                                    fold_file[i][2], 
                                                    l1, l2, l3, l4,
                                                    LAA)

        # Normalizing data
        x_train = l5.fit_transform(x_train)
        x_valid = l5.transform(x_valid)
        x_test = l5.transform(x_test)

        # SVM model training and validation
        parameters = {'C': 10 ** np.arange(-10.0, 3.1, 1.0), 
                      'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
                      'gamma': ['auto', 'scale']}
        model = SVC(class_weight='balanced', 
                    random_state=0, 
                    probability=True)
        
        scoring = "balanced_accuracy"
        x_train = np.concatenate((x_train, x_valid), axis=0)
        y_train = np.concatenate([y_train, y_valid])
        
        clf = GridSearchCV(model, 
                           parameters, 
                           scoring=scoring, 
                           cv=5, 
                           n_jobs=-1)
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Storing results
        y_tests_svm.append(y_test)
        y_preds_svm.append(y_pred)
        y_preds_svm_prob.append(clf.predict_proba(x_test))

        # Linear SVM for importance
        model2 = SVC(class_weight='balanced', 
                     random_state=0, 
                     kernel='linear')
        
        parameters2 = {'C': 10 ** np.arange(-10.0, 3.1, 1.0), 
                       'gamma': ['auto', 'scale']}
        
        clf2 = GridSearchCV(model2, 
                            parameters2, 
                            scoring=scoring, 
                            cv=5, 
                            n_jobs=-1)
        
        clf2.fit(x_train, y_train)
        clf3 = clf2.best_estimator_
        clfs_svm.append(clf3)

    # Aggregating and calculating metrics
    y_tests_svm2 = np.concatenate(y_tests_svm)
    y_preds_svm2 = np.concatenate(y_preds_svm)
    y_preds_svm_prob = np.concatenate(y_preds_svm_prob)[:, 1]
    metric_test = balanced_accuracy_score(y_tests_svm2, np.round(y_preds_svm2).squeeze())
    print(f"Balanced Accuracy: {metric_test:0.4f}")

    # Preparing metrics dataframe
    res, lr, gamma = None, None, None
    df_metrics_svm = get_metrics(y_tests_svm2, np.round(y_preds_svm2).squeeze(), y_preds_svm_prob, res, gamma, lr, group="SVM")
    coefs = [str(clf.coef_.squeeze()) for clf in clfs_svm]
    for i, coef in enumerate(coefs):
        df_metrics_svm[f"coef {i}"] = [coef]
    return df_metrics_svm, y_preds_svm_prob


def run_cascade(DATA_DIR, CNN_DIR, fold_file, LAA=False):
    # Receiving CNN predictions
    (_, y_preds_cnn, _, y_pred_valid_cnn, _, y_pred_train_cnn) = get_cnn_result(CNN_DIR,
        folds=5, gamma=2, resolution=0.25, learning_rate=0.001, return_predictions=True
    )

    # Initializing lists for cascade hybrid model results
    y_tests_cas = [] # Cascade hybrid
    y_preds_cas = []
    y_preds_cas_prob = []
    clfs_cas = []

    # Looping through each fold for the cascade model
    for i in range(5):
        # Data processing with biomarker_data function
        if i == 0:
            x_train, y_train, l1, l2, l3, l4 = biomarker_data(DATA_DIR, 
                                                              fold_file[i][0],
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              LAA)
            l5 = MinMaxScaler()
        else:
            x_train, y_train, l1, l2, l3, l4 = biomarker_data(DATA_DIR, 
                                                              fold_file[i][0], 
                                                              l1, l2, l3, l4,
                                                              LAA)
        x_valid, y_valid, _, _, _, _ = biomarker_data(DATA_DIR, 
                                                      fold_file[i][1], 
                                                      l1, l2, l3, l4,
                                                      LAA)
        x_test, y_test, _, _, _, _ = biomarker_data(DATA_DIR, 
                                                    fold_file[i][2], 
                                                    l1, l2, l3, l4,
                                                    LAA)

        # Combining CNN predictions with training, validation, and test data
        x_train = np.concatenate((x_train, np.round(y_pred_train_cnn[i][..., None])), axis=1)
        x_valid = np.concatenate((x_valid, np.round(y_pred_valid_cnn[i][..., None])), axis=1)

        x_test = np.concatenate((x_test, np.round(y_preds_cnn[i][..., None])), axis=1)

        # Normalizing the data
        x_train = l5.fit_transform(x_train)
        x_valid = l5.transform(x_valid)
        x_test = l5.transform(x_test)

        # Setting parameters for SVM model
        parameters = {'C': 10 ** np.arange(-10.0, 3.1, 1.0), 
                      'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 
                      'gamma': ['auto', 'scale']}
        model = SVC(class_weight='balanced', 
                    random_state=0, 
                    probability=True)
        
        scoring = "balanced_accuracy"

        # Combining training and validation data for training
        x_train = np.concatenate((x_train, x_valid), axis=0)
        y_train = np.concatenate([y_train, y_valid])

        clf = GridSearchCV(model, 
                           parameters, 
                           scoring=scoring, 
                           cv=5, 
                           n_jobs=-1)
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Linear SVM for importance
        model2 = SVC(class_weight='balanced', 
                     random_state=0, 
                     kernel='linear')
        
        parameters2 = {'C': 10 ** np.arange(-10.0, 3.1, 1.0), 
                       'gamma': ['auto', 'scale']}
        clf2 = GridSearchCV(model2, 
                            parameters2, 
                            scoring=scoring, 
                            cv=5, 
                            n_jobs=-1)
        
        clf2.fit(x_train, y_train)
        clf3 = clf2.best_estimator_

        # Storing the results
        y_tests_cas.append(y_test)
        y_preds_cas.append(y_pred)
        y_preds_cas_prob.append(clf.predict_proba(x_test))
        clfs_cas.append(clf3)

    # Aggregating and calculating metrics
    y_tests_cas2 = np.concatenate(y_tests_cas)
    y_preds_cas2 = np.concatenate(y_preds_cas)
    y_preds_cas_prob = np.concatenate(y_preds_cas_prob)[:, 1]
    metric_test = balanced_accuracy_score(y_tests_cas2, np.round(y_preds_cas2).squeeze())

    print(f"Balanced Accuracy: {metric_test:0.4f}")

    # Preparing metrics dataframe
    res, lr, gamma = None, None, None
    df_metrics_cas = get_metrics(
    y_tests_cas2, np.round(y_preds_cas2).squeeze(), y_preds_cas_prob, res, gamma, lr, group="CAS"
    )
    coefs = [str(clf.coef_.squeeze()) for clf in clfs_cas]
    for i, coef in enumerate(coefs):
        df_metrics_cas[f"coef {i}"] = [coef]
        
    return df_metrics_cas, y_tests_cas2, coefs


def run_and(CNN_DIR, y_preds_svm_prob, y_tests, coefs):
    # Receiving CNN predictions
    (_, y_preds_cnn, _, _, _, _) = get_cnn_result(CNN_DIR,
        folds=5, gamma=2, resolution=0.25, learning_rate=0.001, return_predictions=True
    )
    # Combining CNN and SVM predictions
    y_preds_cnn_concat = np.concatenate(y_preds_cnn)
    y_preds_svm_prob_concat = y_preds_svm_prob

    # Applying the 'AND' voting mechanism
    y_pred_min = np.minimum(y_preds_cnn_concat, y_preds_svm_prob_concat)

    # Calculating metrics for the 'AND' voting model
    res, lr, gamma = None, None, None
    df_metrics_and = get_metrics(
        y_tests, np.round(y_pred_min).squeeze(), y_pred_min, res, gamma, lr, group="AND"
    )
    metric_test = balanced_accuracy_score(y_tests, np.round(y_pred_min).squeeze())
    
    print(f"Balanced Accuracy: {metric_test:0.4f}")

    # Appending coefficient columns with None values
    for i, _ in enumerate(coefs):
        df_metrics_and[f"coef {i}"] = [None]

    # Displaying the final dataframe
    return df_metrics_and


def run_or(CNN_DIR, y_preds_svm_prob, y_tests, coefs):
    # Receiving CNN predictions
    (_, y_preds_cnn, _, _, _, _) = get_cnn_result(CNN_DIR,
        folds=5, gamma=2, resolution=0.25, learning_rate=0.001, return_predictions=True
    )
    # Combining CNN and SVM predictions
    y_preds_cnn_concat = np.concatenate(y_preds_cnn)
    y_preds_svm_prob_concat = y_preds_svm_prob

    # Applying the 'OR' voting mechanism
    y_pred_max = np.maximum(y_preds_cnn_concat, y_preds_svm_prob_concat)

    # Calculating metrics for the 'OR' voting model
    res, lr, gamma = None, None, None
    df_metrics_or = get_metrics(
        y_tests, np.round(y_pred_max).squeeze(), y_pred_max, res, gamma, lr, group="OR"
    )
    
    metric_test = balanced_accuracy_score(y_tests, np.round(y_pred_max).squeeze())
    
    print(f"Balanced Accuracy: {metric_test:0.4f}")

    # Appending coefficient columns with None values
    for i, _ in enumerate(coefs):
        df_metrics_or[f"coef {i}"] = [None]

    # Displaying the final dataframe
    return df_metrics_or


def run_avg(CNN_DIR, y_preds_svm_prob, y_tests, coefs):
    # Receiving CNN predictions
    (_, y_preds_cnn, _, _, _, _) = get_cnn_result(CNN_DIR,
        folds=5, gamma=2, resolution=0.25, learning_rate=0.001, return_predictions=True
    )
    # Combining CNN and SVM predictions
    y_preds_cnn_concat = np.concatenate(y_preds_cnn)
    y_preds_svm_prob_concat = y_preds_svm_prob

    # Calculating the average of CNN and SVM predictions
    y_pred_avg = (y_preds_cnn_concat + y_preds_svm_prob_concat) / 2

    # Calculating metrics for the averaged predictions
    res, lr, gamma = None, None, None
    df_metrics_avg = get_metrics(
        y_tests, np.round(y_pred_avg).squeeze(), y_pred_avg, None, None, None, group="AVG"
    )
    
    
    metric_test = balanced_accuracy_score(y_tests, np.round(y_pred_avg).squeeze())
    
    print(f"Balanced Accuracy: {metric_test:0.4f}")

    # Appending coefficient columns with None values
    for i, _ in enumerate(coefs):
        df_metrics_avg[f"coef {i}"] = [None]

    # Displaying the final dataframe
    return df_metrics_avg