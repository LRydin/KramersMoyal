import numpy as np
from functools import wraps
from scipy.special import gamma, factorial2
from scipy.stats import norm


def kernel(kernel_func):
    """
    Transforms a kernel function into a scaled kernel function
    (for a certain bandwidth `bw`)

    Currently implemented kernels are:
        Epanechnikov, Gaussian, Uniform, Triangular, Quartic

    For a good overview of various kernels see
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    @wraps(kernel_func)  # just for naming
    def decorated(x, bw):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        dims = x.shape[-1]

        # Euclidean norm
        dist = np.sqrt((x * x).sum(axis=-1))

        return kernel_func(dist / bw, dims) / (bw ** dims)
    return decorated


def volume_unit_ball(dims: int) -> float:
    """
    Returns the volume of a unit ball in dimensions dims
    """
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)

@kernel
def epanechnikov(x: np.ndarray, dims: int) -> np.ndarray:
    """
    The Epanechnikov kernel in dimensions dims
    """
    x2 = (x ** 2)
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = (1.0 - x2[mask])
    normalisation = 2.0 / (dims + 2.0) * volume_unit_ball(dims)
    return kernel / normalisation

@kernel
def gaussian(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Gaussian kernel in dimensions dims
    """
    def gaussian_integral(n):
        if n % 2 == 0:
            return np.sqrt(np.pi * 2) * factorial2(n - 1) / 2
        elif n % 2 == 1:
            return np.sqrt(np.pi * 2) * factorial2(n - 1) * norm.pdf(0)
    kernel = np.exp(-x ** 2 / 2.0)
    normalisation = dims * gaussian_integral(dims - 1) * volume_unit_ball(dims)
    return kernel / normalisation

@kernel
def uniform(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Uniform, or rectangular kernel in dimensions dims
    """
    mask = x < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = 1.0
    normalisation = volume_unit_ball(dims)
    return kernel / normalisation

@kernel
def triagular(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Triagular kernel in dimensions dims
    """
    mask = x < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = 1.0 - np.abs(x[mask])
    normalisation = volume_unit_ball(dims) / 2.0
    return kernel / normalisation


@kernel
def quartic(x: np.ndarray, dims: int) -> np.ndarray:
    """
    Quartic, or biweight kernel in dimensions dims
    """
    x2 = (x ** 2)
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = ((1.0 - x2[mask]) ** 2)
    normalisation = 2.0 / (dims + 2.0) * volume_unit_ball(dims)
    return kernel / normalisation


_kernels = {epanechnikov, gaussian, uniform, triagular, quartic}


def silvermans_rule(timeseries: np.ndarray) -> float:
    n, dims = timeseries.shape
    sigma = np.std(timeseries, axis=0)
    sigma = sigma.max()

    return sigma * (4.0 / (3 * n)) ** (1 / 5)

#TODO do we need dims in the kernel functions? Can't se just read that off the
#     the x (ndarray)?
