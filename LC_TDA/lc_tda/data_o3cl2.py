import pickle
import numpy as np
from skimage.color import rgb2lab
from lc_tda.topology import minkowski, fractal_minkowski


def topo_feature(x):
    """Topological feature of an endpoint image from O3/Cl2 dataset.

    Args:
        x (np.ndarray): 3D RGB video.

    Returns:
        np.ndarray: topological descriptor curves
    """
    gray = x.mean(axis=-1)
    red = x[..., 0]
    green = x[..., 1]
    blue = x[..., 2]
    lab = rgb2lab(x)
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    colors = [a, b, blue, gray, green, l, red]

    for c in colors:
        x_ = c
        t = np.linspace(-4, 4, 100)

        x_ -= x_.mean()
        x_ /= x_.std()
        m = minkowski(x_, t)
        f = fractal_minkowski(x_, t)
        tdc = np.concatenate((m, f), axis=-1)  # curves
    return tdc


if __name__ == "__main__":
    with open(f'../data/o3cl2/video.pickle', 'rb') as handle:
        x = pickle.load(handle)
        y = pickle.load(handle)

    out = []
    for x_ in x:
        tdc = topo_feature(x_)
        out.append(tdc)
        print('c')
    out = np.array(out)
    with open(f'../data/o3cl2/tda_3d.pickle', 'wb') as handle:
        pickle.dump(out)
        pickle.dump(y)
