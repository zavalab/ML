from sklearn.preprocessing import StandardScaler
from gnn_uq.data_utils import split_data, get_data

def load_data(dataset='qm7', split_type='random', seed=0, verbose=0, test=0):
    
    data = get_data(path=f'../data/{dataset}.csv')
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(data, split_type=split_type, show_mol=False, seed=seed)
    
    if len(y_train.shape) == 1:
        y_train = y_train[..., None]
        y_valid = y_valid[..., None]
        y_test = y_test[..., None]
        
    ss = StandardScaler()
    y_train = ss.fit_transform(y_train)
    y_valid = ss.transform(y_valid)
    y_test = ss.transform(y_test)

    if verbose:
        print(f'x_train shape: {[x.shape for x in x_train]}')
        print(f'y_train shape: {y_train.shape}')
        print(f'x_valid shape: {[x.shape for x in x_valid]}')
        print(f'y_valid shape: {y_valid.shape}')
        print(f'x_test shape: {[x.shape for x in x_test]}')
        print(f'y_test shape: {y_test.shape}')

    if test:
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (ss.mean_, ss.var_ ** 0.5)
    else:
        return (x_train, y_train), (x_valid, y_valid)


if __name__ == '__main__':
    _, _ = load_data(verbose=1)
