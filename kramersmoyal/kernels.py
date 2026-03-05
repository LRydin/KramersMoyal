import numpy as np
from functools import wraps
from scipy.special import gamma


def _volume_unit_ball(dims: int) -> float:
    """
    Volume of a unit ball in `dims` dimensions.
    """
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)


def kernel(kernel_func):
    """
    Transforms a kernel function into a scaled kernel function
    (for a certain bandwidth ``bw``)

    Currently implemented kernels are:
        Epanechnikov, Gaussian, Uniform, Triangular, Quartic

    For a good overview of various kernels see
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """

    @wraps(kernel_func)
    def decorated(x: np.ndarray, bw: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        dims = x.shape[-1]
        dist = np.sqrt((x * x).sum(axis=-1))  # Euclidean norm

        return kernel_func(dist / bw, dims) / (bw**dims) / _volume_unit_ball(dims)

    return decorated

@kernel
def epanechnikov(x: np.ndarray, dims: int) -> np.ndarray:
    """
    The Epanechnikov kernel in dimensions dims.
    """
    normalisation = 2.0 / (dims + 2.0)
    x2 = (x ** 2)
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = (1.0 - x2[mask]) / normalisation
    return kernel

@kernel
def gaussian(r: np.ndarray, dims: int) -> np.ndarray:
    """
    Gaussian kernel in `dims` dimensions.
    """
    normalisation = dims * (2 ** ((dims - 2) / 2.0)) * gamma(dims / 2.0)
    return np.exp(-(r**2) / 2.0) / normalisation


@kernel
def uniform(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Uniform, or rectangular kernel in dimensions dims.
    """
    mask = x < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = 1.0
    return kernel

@kernel
def triagular(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Triagular kernel in dimensions dims.
    """
    normalisation = 1.0 / 2.0
    mask = x < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = (1.0 - np.abs(x[mask])) / normalisation
    return kernel

@kernel
def quartic(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Quartic, or biweight kernel in dimensions dims.
    """
    normalisation = 2.0 / (dims + 2.0)
    x2 = (x ** 2)
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = ((1.0 - x2[mask]) ** 2) / normalisation
    return kernel

def silvermans_rule(timeseries: np.ndarray) -> float:
    n, dims = timeseries.shape
    sigma = np.std(timeseries, axis=0)
    sigma = sigma.max()
    return  ((4.0 * sigma ** 5) / (3 * n)) ** (1 / 5)
