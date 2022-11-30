import pickle
import itertools

import numpy as np
from time import gmtime, strftime
from timeit import default_timer as timer

# sklearn import
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR


# default directories
DATA_DIR_SO2 = '../data/so2rh/endpoint_so2.pickle'
DATA_DIR_RH = '../data/so2rh/endpoint_rh.pickle'
RESULT_DIR = '../result/tda/'


def svm_grid_search(x, y, clf):
    '''
    A function to perform svm grid search.
    Args:
        x: np.array
        y: np.array
        clf: sklearn SVC or SVR

    Returns:
        best estimator, best score, best parameters
    '''
    grid = {'C': 10.0 ** np.arange(-4, 4.5, 0.5)}

    rs = GridSearchCV(clf, grid, cv=5, verbose=0, n_jobs=-1)
    rs.fit(x, y)

    return rs.best_estimator_, rs.best_score_, rs.best_params_


def grid_search_pca(x_train, x_test, r_train, c_train, pca_ratio=0.9):
    '''
    A function to perform grid search for SVM after PCA.
    Args:
        x_train: np.array, training data
        x_test: np.array, test data
        r_train: np.array, regression label
        c_train: np.array, classification label
        pca_ratio: float or None, the percentage of PCA variance to keep

    Returns:
        predicted regression label
        predicted classification label
        classification time
        regression time
    '''
    # Step 1: standardization for training and test data.
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # Step 2: perform PCA with certain ratio
    if pca_ratio is not None:
        pca = PCA(n_components=pca_ratio, random_state=0, whiten=True)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    # Step 3: classification
    t0 = timer()
    svc = SVC(random_state=0)
    best_svc, score_c, param_c = svm_grid_search(x_train, c_train, svc)
    c_pred = best_svc.predict(x_test)
    t1 = timer()

    # Step 4: regression
    svr = SVR()
    best_svr, score_r, param_r = svm_grid_search(x_train, r_train, svr)
    r_pred = best_svr.predict(x_test)
    t2 = timer()

    # Step 5: record time
    t_c = t1 - t0
    t_r = t2 - t1
    return c_pred, r_pred, t_c, t_r


def load_data(data_dir):
    '''
    A function to load SO2/RH endpoint TDA data.
    Args:
        data_dir: str, data directory
    Returns:
        x: np.array, data
        y: np.array, regression label
        yc: np.array, classification label
    '''
    # Step 1: locate TDA data file
    with open(data_dir, 'rb') as handle:
        x = pickle.load(handle)
        y = pickle.load(handle)

    # Step 2: random permutation
    x = np.random.RandomState(0).permutation(x)
    y = np.random.RandomState(0).permutation(y)

    # Step 3: encode labels for classification
    le = LabelEncoder()
    yc = le.fit_transform(y)
    return x, y, yc


def main(analyte, pca_ratio, color, descriptor):
    '''
    The main tda training function.
    Args:
        analyte: str, rh or so2
        pca_ratio: float or int or None, if None do not perform PCA
        color: str, color space to keep
        descriptor: int or str, if combined use all; int indicates index

    Returns:
        storing result in a pickle file
    '''
    if analyte == 'rh':
        data_dir = DATA_DIR_RH
    elif analyte == 'so2':
        data_dir = DATA_DIR_SO2
    # Step 1: five fold CV split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Step 2: load data
    x, y, yc = load_data(data_dir)

    # Step 3: select data based on color
    # color orders:
    # 0: A*, 1: B*, 2: blue, 3: gray, 4: green, 5: L*, 6: red
    if color == 'gray':
        x = x[..., 3]
    elif color == 'r':
        x = x[..., 6]
    elif color == 'g':
        x = x[..., 4]
    elif color == 'b':
        x = x[..., 2]
    elif color == 'L':
        x = x[..., 5]
    elif color == 'A':
        x = x[..., 0]
    elif color == 'BB':
        x = x[..., 1]
    elif color == 'combined':
        x = np.concatenate(x, axis=-1)

    # Step 4: select color based on descriptors
    if descriptor == 'combined':
        x = x
    elif type(descriptor) is int:
        x = x[..., descriptor::5]

    # Step 5: five fold CV to get regression and classification results
    c_result = []
    r_result = []
    time = []
    for train_ix, test_ix in cv.split(x, yc):
        x_train, x_test = x[train_ix], x[test_ix]
        r_train, r_test = y[train_ix], y[test_ix]
        c_train, c_test = yc[train_ix], yc[test_ix]

        c_pred, r_pred, t_c, t_r = grid_search_pca(
            x_train, x_test, r_train, c_train, pca_ratio)
        c_result.append([c_test, c_pred])
        r_result.append([r_test, r_pred])
        time.append([t_c, t_r])
    c_result = np.concatenate(c_result, axis=-1)
    r_result = np.concatenate(r_result, axis=-1)
    time = np.array(time)

    # Step 6: store result
    result_name = f'{analyte}_{color}_{descriptor}_{pca_ratio}.pickle'

    with open(RESULT_DIR + result_name, 'wb') as handle:
        pickle.dump(c_result, handle)
        pickle.dump(r_result, handle)
        pickle.dump(time, handle)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +
          ': ' + result_name + ' finished...')


if __name__ == "__main__":
    analyte = ['so2', 'rh']
    pca_ratio = [None, 100, 50, 20, 10, 5, 2]
    color = ["gray", "r", "g", "b", "L", "A", "BB", "combined"]
    descriptor = [0, 1, 2, 3, 4, "combined"]

    for param in list(itertools.product(analyte, pca_ratio, color, descriptor)):
        main(*param)
