import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from .binning import histogramdd
from .kernels import silvermans_rule, epanechnikov, _kernels


def km(timeseries: np.ndarray, bins: np.ndarray, powers: np.ndarray,
        kernel=None, bw=None, eps=1e-12, conv_method='auto'):
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

    conv_method: str
        A string indicating which method to use to calculate the convolution.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    Returns
    -------
    kmc: np.ndarray
        The calculated Kramers-Moyal coefficients

    edges: np.ndarray
        The bin edges of the calculated Kramers-Moyal coefficients
    """
    timeseries = np.asarray_chkfinite(timeseries, dtype=float)
    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape(-1, 1)

    assert len(timeseries.shape) == 2, "Timeseries must (n, dims) shape"
    assert timeseries.shape[0] > 0, "No data in timeseries"

    n, dims = timeseries.shape

    powers = np.asarray_chkfinite(powers, dtype=float)
    if len(powers.shape) == 1:
        powers = powers.reshape(-1, 1)

    assert (powers[0] == [0] * dims).all(), "First power must be zero"
    assert dims == powers.shape[1], "Powers not matching timeseries' dimension"

    assert dims == bins.shape[0], "Bins not matching timeseries' dimension"

    if bw is None:
        bw = silvermans_rule(timeseries)
    elif callable(bw):
        bw = bw(timeseries)
    assert bw > 0.0, "Bandwidth must be > 0"

    if kernel is None:
        kernel = epanechnikov
    assert kernel in _kernels, "Kernel not found"

    return _km(timeseries, bins, powers, kernel, bw, eps, conv_method)


def _km(timeseries: np.ndarray, bins: np.ndarray, powers: np.ndarray,
        kernel: callable, bw: float, eps: float, conv_method: str):
    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    # Calculate derivative and the product of its powers
    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads[..., None], powers.T), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries[:-1, ...], bins=bins,
                              weights=weights, bw=2*bw)

    # Generate kernel
    mesh = cartesian_product(edges)
    kernel_ = kernel(mesh, bw=bw).reshape(*(edge.size for edge in edges))
    kernel_ /= np.sum(kernel_)

    # Convolve weighted histogram with kernel and trim it
    kmc = convolve(hist, kernel_[..., None], mode='same', method=conv_method)

    # Normalize
    mask = np.abs(kmc[..., 0]) < eps
    kmc[mask, 0:] = 0.0
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[~mask, 1:] /= kmc[~mask, 0, None] * taylors

    return kmc, [edge[:-1] + 0.5 * (edge[1] - edge[0]) for edge in edges]
