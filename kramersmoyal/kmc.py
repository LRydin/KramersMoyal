import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from .binning import histogramdd


def kmc_kernel_estimator(timeseries: np.ndarray, bins: np.ndarray,
                         kernel: callable, bw: float,
                         powers: np.ndarray, eps=1e-12):
    """
    Estimates Kramers-Moyal coefficients from a timeseries using a kernel
    estimator method.

    Parameters
    ----------
    timeseries: np.ndarray
        The D-dimensional timeseries (N, D)

    bins: np.ndarray
        The number of bins for each dimension

    kernel: callable
        Kernel used to calculate the Kramers-Moyal coefficients

    bw: float
        Desired bandwidth of the kernel

    Returns
    -------
    kmc: np.ndarray
        The calculated Kramers-Moyal coefficients

    edges: np.ndarray
        The bin edges of the calculated Kramers-Moyal coefficients
    """
    def add_bandwidth(edges: list, bw: float, eps=1e-12):
        new_edges = list()
        for edge in edges:
            dx = edge[1] - edge[0]
            min = edge[0] - bw
            max = edge[-1] + bw
            new_edge = np.arange(min, max + eps, dx)
            new_edges.append(new_edge)
        return new_edges

    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    # Calculate derivative and the product of its powers
    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads[..., None], powers), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries[:-1, ...], bins=bins,
                              weights=weights, density=False)

    # Generate kernel
    edges_k = add_bandwidth(edges, bw, eps=eps)
    mesh_k = cartesian_product(edges_k)
    kernel_ = kernel(mesh_k, bw=bw).reshape(*(edge.size for edge in edges_k))
    kernel_ /= np.sum(kernel_)

    # Convolve weighted histogram with kernel
    kmc = convolve(hist, kernel_[..., None], mode='same')

    # Normalize
    mask = np.abs(kmc[..., 0]) < eps
    kmc[mask, 0:] = 0.0
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[~mask, 1:] /= np.tensordot(kmc[~mask, 0], taylors, axes=0)

    return kmc, [edge[:-1] + 0.5 * (edge[1] - edge[0]) for edge in edges]
