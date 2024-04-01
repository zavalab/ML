import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gnn_uq.data_utils import split_data, get_data

DATA_DIR = '/some/data/dir'

def load_data(dataset='qm7', split_type='random', seed=0, verbose=0, test=0, norm=1, sizes=(0.5, 0.2, 0.3)):
    """
    Load and preprocess the specified dataset.

    Args:
        dataset (str, optional): Name of the dataset. Defaults to 'qm7'.
        split_type (str, optional): Type of data split. Defaults to 'random'.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        verbose (int, optional): Verbosity level. Defaults to 0.
        test (int, optional): Flag indicating whether to include test data. Defaults to 0.
        norm (int, optional): Flag indicating whether to perform normalization. Defaults to 1.
        sizes (tuple, optional): Tuple containing proportions for train, validation, and test sets. Defaults to (0.5, 0.2, 0.3).

    Returns:
        tuple: Tuple containing train, validation, and optionally test data along with mean and standard deviation values.
    """
    if dataset == "qm7":
        tasks = ["u0_atom"]
    elif dataset == "lipo":
        tasks = ["lipo"]
    elif dataset == "freesolv":
        tasks = ["freesolv"]
    elif dataset == "delaney":
        tasks = ["logSolubility"]
    
    data = get_data(os.path.join(DATA_DIR, f'{dataset}.csv'), tasks, max_data_size=None)
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(data, split_type=split_type, sizes=sizes, show_mol=False, seed=seed)
    
    if len(y_train.shape) == 1:
        y_train = y_train[..., None]
        y_valid = y_valid[..., None]
        y_test = y_test[..., None]
    
    if norm:
        ss = StandardScaler()
        y_train = ss.fit_transform(y_train)
        y_valid = ss.transform(y_valid)
        y_test = ss.transform(y_test)
        mean = ss.mean_
        std = ss.var_ ** 0.5
    else:
        mean = 0
        std = 1
    
    ss2 = MinMaxScaler()
    d = x_train[0]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss2.fit_transform(d_)
    d_ = d_.reshape(d.shape)
    x_train[0] = d_
    
    d = x_valid[0]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss2.transform(d_)
    d_ = d_.reshape(d.shape)
    x_valid[0] = d_
    
    d = x_test[0]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss2.transform(d_)
    d_ = d_.reshape(d.shape)
    x_test[0] = d_
    
    ss3 = MinMaxScaler()
    d = x_train[2]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss3.fit_transform(d_)
    d_ = d_.reshape(d.shape)
    x_train[2] = d_
    
    d = x_valid[2]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss3.transform(d_)
    d_ = d_.reshape(d.shape)
    x_valid[2] = d_
    
    d = x_test[2]
    d_ = d.reshape(-1, d.shape[-1])
    d_ = ss3.transform(d_)
    d_ = d_.reshape(d.shape)
    x_test[2] = d_
    

    if verbose:
        print(f'x_train shape: {[x.shape for x in x_train]}')
        print(f'y_train shape: {y_train.shape}')
        print(f'x_valid shape: {[x.shape for x in x_valid]}')
        print(f'y_valid shape: {y_valid.shape}')
        print(f'x_test shape: {[x.shape for x in x_test]}')
        print(f'y_test shape: {y_test.shape}')

    if test:
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (mean, std)
    else:
        return (x_train, y_train), (x_valid, y_valid)



if __name__ == '__main__':
    _, _ = load_data(verbose=1)
