import numpy as np
from quantimpy.minkowski import functionals


def fractal_minkowski(x, thresholds):
    """Calculate fractal dimension and lacunarity of a 2D or 3D array.
    Adapted from https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1#file-fractal-dimension-py

    Args:
        x (numpy.ndarray): data.
        thresholds (numpy.ndarray): thresholds.

    Returns:
        numpy.ndarray: fractal dimension and lacunarity.
    """
    dimension = len(x.shape)
    assert (dimension in [2, 3]), "Only 2D or 3D"

    # From https://github.com/rougier/numpy-100 (#87)
    if dimension == 2:
        def boxcount(z, k):
            s = np.add.reduceat(
                np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
                np.arange(0, z.shape[1], k), axis=1)

            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((s > 0) & (s < k * k))[0])
    elif dimension == 3:
        def boxcount(z, k):
            s = np.add.reduceat(
                np.add.reduceat(np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0), np.arange(0, z.shape[1], k),
                                axis=1), np.arange(0, z.shape[2], k), axis=2)

            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((s > 0) & (s < k * k * k))[0])

    # Minimal dimension of image
    p = min(x.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    fd = []
    la = []
    for threshold in thresholds:
        # Transform Z into a binary array
        image_binary = (x <= threshold)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(image_binary, size))

        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        fd_temp = -coeffs[0]
        la_temp = np.exp(coeffs[1])
        if (fd_temp <= dimension - 1) or (np.isnan(fd_temp)):
            fd.append(0)
            la.append(0)
        else:
            fd.append(fd_temp)
            la.append(la_temp)

    fd = np.array(fd)[..., None]
    la = np.array(la)[..., None]
    return np.concatenate((fd, la), axis=1)


def minkowski(x, thresholds):
    """A function to calculate Minkowski functionals.

    Args:
        x (numpy.ndarray): data.
        thresholds (numpy.ndarray): thresholds.

    Returns:
        numpy.ndarray: Minkowski functionals.
    """
    dimension = len(x.shape)
    assert (dimension in [2, 3]), "Only 2D or 3D"

    out = []
    for threshold in thresholds:
        x_temp = x <= threshold
        x_temp = x_temp.astype('bool')
        out.append(functionals(x_temp))
    out = np.array(out)
    out = np.flip(out, axis=0)
    return out
