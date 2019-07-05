import numpy as np
from scipy.signal import convolve
from itertools import product

from .binning import histogramdd


def kmc_kernel_estimator(timeseries: np.ndarray, kernel: callable, bw: float,
                         bins: np.ndarray, powers: np.ndarray):
    """
    Estimates Kramers-Moyal coefficients from a timeseries using an kernel
    estimator method.
    """
    # Calculate derivatives and its powers
    grads = np.diff(timeseries)
    weights = np.prod(np.power(grads[..., None], powers), axis=1)

    # Get weighted histogram
    kmc, edges = histogramdd(timeseries, bins=bins,
                             weights=weights, density=True)

    # Generate kernel
    mesh = np.asarray(
        list(product(*edges))).reshape((*(edge.size for edge in edges), -1))
    kernel_ = kernel(mesh / bw)
    kernel_ /= np.sum(kernel_)

    # Convolve with kernel for all powers
    for p in range(powers.shape[1]):
        kmc[..., p] = convolve(kmc[..., p], kernel_, mode='same')

    return kmc
