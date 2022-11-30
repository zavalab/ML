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
DATA_DIR = '../data/o3cl2/tda_3d.pickle'
RESULT_DIR = '../result/tda/'


def svm_grid_search(x, y, clf):
    """A function to perform svm grid search.

    Args:
        x (numpy.ndarray): data.
        y (numpy.ndarray): label.
        clf (estimator): SVC or SVM.

    Returns:
        tuple: best estimator, best score, best parameters.
    """
    grid = {'C': 10.0 ** np.arange(-4, 4.5, 0.5)}

    rs = GridSearchCV(clf, grid, cv=5, verbose=0, n_jobs=-1)
    rs.fit(x, y)

    return rs.best_estimator_, rs.best_score_, rs.best_params_


def grid_search_pca(x_train, x_test, r_train, c_train, pca_ratio=100):
    """A function to perform grid search for SVM after PCA.

    Args:
        x_train (numpy.ndarray): training data.
        x_test (numpy.ndarray): test data.
        r_train (numpy.ndarray): regression label.
        c_train (numpy.ndarray): classification label.
        pca_ratio (int, optional): PCA ratio. Defaults to 100.

    Returns:
        tuple: predicted regression label, predicted classification label, classification time, regression time.
    """
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


def load_data():
    """A function to load O3/Cl2 endpoint TDA data.

    Returns:
        tuple: data, ozone regression label, chlorine regression label, ozone classification label, chlorine regression label.
    """
    # Step 1: locate TDA data file
    with open(DATA_DIR, 'rb') as handle:
        x = pickle.load(handle)
        y = pickle.load(handle)

    # Step 2: select only the main data (the other two are unseen and pure cl2)
    x = np.concatenate([x[i] for i in range(7)], axis=-1)
    y_o3 = y[:, 0]
    y_cl2 = y[:, 1]

    n = int(x.shape[-1])
    for i in range(n):
        x[..., i] /= np.max(x[..., i])

    idx = np.where((y_o3 != 0) & (y_cl2 != 3.5))[0]
    x = x[idx]
    y_o3 = y_o3[idx]
    y_cl2 = y_cl2[idx]

    # Step 3: random permutation
    x = np.random.RandomState(0).permutation(x)
    y_o3 = np.random.RandomState(0).permutation(y_o3)
    y_cl2 = np.random.RandomState(0).permutation(y_cl2)

    # Step 4: categorize labels
    le = LabelEncoder()
    c_o3 = le.fit_transform(y_o3)

    le = LabelEncoder()
    c_cl2 = le.fit_transform(y_cl2)

    return x, y_o3, y_cl2, c_o3, c_cl2


def single_chemical_prediction(x, y, yc, pca_ratio=100):
    """A function to perform single chemical prediction.

    Args:
        x (numpy.ndarray): data.
        y (numpy.ndarray): regression label.
        yc (numpy.ndarray): classification label.
        pca_ratio (int, optional): PCA ratio. Defaults to 100.

    Returns:
        tuple: classification results, regression results, training time.
    """
    # Step 1: five fold CV split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Step 2: five fold CV to get regression and classification results
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

    return c_result, r_result, time


def main(pca_ratio=100, color='combined', descriptor='combined'):
    """The main tda training function.

    Args:
        pca_ratio (int, optional): PCA ratio. Defaults to 100.
        color (str, optional): color channel. Defaults to 'combined'.
        descriptor (str, optional): topological descritpro. Defaults to 'combined'.
    """
    result_name = f"o3cl2_{color}_{descriptor}_{pca_ratio}.pickle"

    # Step 1: load data
    x, y_ozone, y_chlorine, c_ozone, c_chlorine = load_data()

    # Step 2: select data based on color
    if color == "gray":
        x = np.concatenate([x[..., i] for i in range(18, 24)], axis=-1)
    if color == "r":
        x = np.concatenate([x[..., i] for i in range(36, 42)], axis=-1)
    if color == "g":
        x = np.concatenate([x[..., i] for i in range(24, 30)], axis=-1)
    if color == "b":
        x = np.concatenate([x[..., i] for i in range(12, 18)], axis=-1)
    if color == "L":
        x = np.concatenate([x[..., i] for i in range(30, 36)], axis=-1)
    if color == "A":
        x = np.concatenate([x[..., i] for i in range(6)], axis=-1)
    if color == "BB":
        x = np.concatenate([x[..., i] for i in range(6, 12)], axis=-1)
    elif color == "combined":
        x_ = []
        for i in [36, 30, 24, 18, 12, 6, 0]:
            x_ += [x[..., j] for j in range(i, i+6)]
        x = np.concatenate(x_, axis=-1)

    # Step 3: select color based on descriptors
    if descriptor == "combined":
        x = x
    elif type(descriptor) is int:
        x = x[..., descriptor::6]

    # Step 4: ozone prediction
    ozone_c_result, ozone_r_result, ozone_time = single_chemical_prediction(
        x, y_ozone, c_ozone, pca_ratio)

    # Step 5: chlorine prediction
    chlorine_c_result, chlorine_r_result, chlorine_time = single_chemical_prediction(
        x, y_chlorine, c_chlorine, pca_ratio)

    # Step 6: store result
    with open(RESULT_DIR + result_name, "wb") as handle:
        pickle.dump(ozone_c_result, handle)
        pickle.dump(ozone_r_result, handle)
        pickle.dump(ozone_time, handle)
        pickle.dump(chlorine_c_result, handle)
        pickle.dump(chlorine_r_result, handle)
        pickle.dump(chlorine_time, handle)

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +
          ': ' + result_name + ' finished...')


if __name__ == "__main__":
    pca_ratio = [None, 100, 50, 20, 10, 5, 2]
    color = ["gray", "r", "g", "b", "L", "A", "BB", "combined"]
    descriptor = [0, 1, 2, 3, 4, 5, "combined"]

    for param in list(itertools.product(pca_ratio, color, descriptor)):
        main(*param)
