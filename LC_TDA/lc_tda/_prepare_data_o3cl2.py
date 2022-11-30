from multiprocessing import Pool, process

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle
from skimage.color import rgb2lab
from lc_tda.topology import minkowski, fractal_minkowski
from timeit import default_timer as timer


with open('/Users/sjiang87/data_weight/lc_tda/o3cl2/video.pickle', 'rb') as handle:
    x = pickle.load(handle)
    y = pickle.load(handle)

def double(i):
    gray = x[i].mean(axis=-1)
    red = x[i, ..., 0]
    green = x[i, ..., 1]
    blue = x[i, ..., 2]
    lab = rgb2lab(x[i])
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    colors = [gray, red, green, blue, l, a, b]
    
    for c in colors:
        x_ = c
        t = np.linspace(-4, 4, 100)

        x_ -= x_.mean()
        x_ /= x_.std()
        m = minkowski(x_, t)
        f = fractal_minkowski(x_, t)

    print(i)







if __name__ == "__main__":
    t0 = timer()
    # Pick the amount of workers that works best for you.
    # Most likely equal to the amount of threads of your machine.
    processes = 8

    # with ThreadPoolExecutor(workers) as pool:
    #     processed = pool.map(double, range(len(x)))
    with Pool(processes) as pool:
        processed = pool.map(double, range(800))

    t1 = timer()
    print(f'Time: {t1-t0:0.4f}')