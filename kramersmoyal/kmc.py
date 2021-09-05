import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

from .binning import histogramdd
from .kernels import silvermans_rule, epanechnikov

def km(timeseries: np.ndarray, bins: np.ndarray, powers: np.ndarray,
        kernel: callable=epanechnikov, bw: float=None, tol: float=1e-10,
        conv_method: str='auto', center_edges: bool=True) -> np.ndarray:
    """
    Estimates the Kramers─Moyal coefficients from a timeseries using a kernel
    estimator method. ``km`` can calculate the Kramers─Moyal coefficients for a
    timeseries of any dimension, up to any desired power.

    Parameters
    ----------
    timeseries: np.ndarray
        The D-dimensional timeseries ``(N, D)``. The timeseries of length ``N``
        and dimensions ``D``.

    bins: np.ndarray
        The number of bins for each dimension. This is the underlying space for
        the Kramers-Moyal coefficients. In 1-dimension a choice as
            ``bins = np.array([6000])``
        is recommended. In 2-dimensions
            ``bins = np.array([300,300])``
        is recommended.

    powers: np.ndarray
        Powers for the operation of calculating the Kramers─Moyal coefficients,
        which need to match dimensions of the timeseries. In 1-dimension the
        first four Kramers-Moyal coefficients can be found via
            ``powers = np.array([0],[1],[2],[3],[4])``.
        In 2 dimensions take into account each dimension, as
        ::

            powers = np.array([0,0],[0,1],[1,0],[1,1],[0,2],[2,0],[2,2],
                              [0,3],[3,0],[3,3],[0,4],[4,0],[4,4])


    kernel: callable (default ``epanechnikov``)
        Kernel used to convolute with the Kramers-Moyal coefficients. To select
        for example a Gaussian kernel use

            ``kernel = kernels.gaussian``

    bw: float (default ``None``)
        Desired bandwidth of the kernel. A value of 1 occupies the full space of
        the bin space. Recommended are values ``0.005 < bw < 0.5``.

    tol: float (default ``1e-10``)
        Round to zero absolute values smaller than ``tol``, after the
        convolutions.

    conv_method: str (default ``auto``)
        A string indicating which method to use to calculate the convolution.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    center_edges: bool (default ``True``)
        Whether to center the bin edges, since for `n` bins, `n+1` edges must 
        have been defined.

    Returns
    -------
    kmc: np.ndarray
        The calculated Kramers-Moyal coefficients in accordance to the
        timeseries dimensions in (D,bins.shape) shape. To extract the selected
        orders of the kmc, use kmc[i,...], with i the order according to powers

    edges: np.ndarray
        The bin edges with shape (D,bins.shape) of the calculated Kramers-Moyal
        coefficients
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

    if not (powers[0] == [0] * dims).all():
        powers = np.array([[0] * dims, *powers])
        trim_output = True
    else:
        trim_output = False

    assert (powers[0] == [0] * dims).all(), "First power must be zero"
    assert dims == powers.shape[1], "Powers not matching timeseries' dimension"
    assert dims == bins.shape[0], "Bins not matching timeseries' dimension"

    if bw is None:
        bw = silvermans_rule(timeseries)
    elif callable(bw):
        bw = bw(timeseries)
    assert bw > 0.0, "Bandwidth must be > 0"

    kmc, edges =  _km(timeseries, bins, powers, kernel, bw, tol, conv_method)

    if center_edges:
        edges = [edge[:-1] + 0.5 * (edge[1] - edge[0]) for edge in edges]

    return (kmc, edges) if not trim_output else (kmc[1:], edges)


def _km(timeseries: np.ndarray, bins: np.ndarray, powers: np.ndarray,
        kernel: callable, bw: float, tol: float, conv_method: str) -> np.ndarray:
    """
    Helper function for km that does the heavy lifting and actually estimates
    the Kramers─Moyal coefficients from the timeseries.
    """
    def cartesian_product(arrays: np.ndarray):
        # Taken from https://stackoverflow.com/questions/11144513
        la = len(arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr

    # Calculate derivative and the product of its powers
    grads = np.diff(timeseries, axis=0)
    weights = np.prod(np.power(grads.T, powers[..., None]), axis=1)

    # Get weighted histogram
    hist, edges = histogramdd(timeseries[:-1, ...], bins=bins,
                              weights=weights, bw=bw)

    # Generate centred kernel on a larger grid (fft'ed convolutions are circular)
    edges_k = [(e[1] - e[0]) * np.arange(-e.size, e.size+1) for e in edges]
    kernel_ = kernel(cartesian_product(edges_k), bw=bw)

    # Convolve weighted histogram with kernel and trim it
    kmc = convolve(hist, kernel_[None, ...], mode='same', method=conv_method)

    # Normalise
    mask = np.abs(kmc[0]) < tol
    kmc[0:, mask] = 0.0
    taylors = np.prod(factorial(powers[1:]), axis=1)
    kmc[1:, ~mask] /= taylors[..., None] * kmc[0, ~mask]

    return kmc, edges
