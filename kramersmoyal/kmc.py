import numpy as np
from scipy.signal import convolve
from itertools import product

from .binning import histogramdd


def kmc_kernel_estimator(timeseries: np.ndarray, kernel: callable, bw=bw,
                         bins: np.ndarray, powers: np.ndarray):
    """
    Estimates Kramers-Moyal coefficients from a timeseries using a kernel
    estimator method.
    """
    # Calculate derivatives and its powers
    grads = np.diff(timeseries)
    weights = np.prod(np.power(grads[..., None], powers), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries, bins=bins,
                              weights=weights, density=True)

    # Generate kernel
    mesh = np.asarray(list(product(*edges)))
    kernel_ = kernel(mesh, bw=bw).reshape(*(edge.size for edge in edges))
    kernel_ /= np.sum(kernel_)

    kmc = list()
    # Convolve with kernel for all powers
    for p in range(powers.shape[1]):
        kmc.append(convolve(hist[..., p], kernel_, mode='full'))

    return np.stack(kmc, axis=-1)
