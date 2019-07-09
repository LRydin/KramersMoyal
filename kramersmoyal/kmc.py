import numpy as np
from scipy.signal import convolve
from .binning import histogramdd
# from scipy.special import factorial


def kmc_kernel_estimator(timeseries: np.ndarray, bins: np.ndarray,
                         kernel: callable, bw: float,
                         powers: np.ndarray):
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

    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    # Calculate derivatives and its powers
    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads[..., None], powers), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries[:-1, ...], bins=bins,
                              weights=weights, density=False)

    # Generate kernel
    mesh = cartesian_product(edges)
    kernel_ = kernel(mesh, bw=bw).reshape(*(edge.size for edge in edges))
    kernel_ /= np.sum(kernel_)

    # Convolve with kernel for all powers
    kmc = np.stack([convolve(kernel_, hist[..., p], mode='same')
                    for p in range(powers.shape[1])], axis=-1)

    # normalization = np.prod(factorial(2 * powers) /
    #                         (np.power(2, powers) * factorial(powers)), axis=0)

    # Normalize
    kmc[..., 1:] /= kmc[..., 0, None]  # * normalization[1:]

    return kmc, edges
