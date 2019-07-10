import numpy as np
from functools import wraps
from scipy.special import gamma, factorial2
from scipy.stats import norm


def kernel(kernel_func):
    """
    Transforms a kernel function into a scaled kernel function
    (for a certain bandwidth `bw`)
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


def volume_unit_ball(dims):
    return np.pi ** (dims / 2.0) / gamma(dims / 2.0 + 1.0)


@kernel
def epanechnikov(x, dims):
    x2 = (x ** 2)
    mask = x2 < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = (1.0 - x2[mask])
    normalization = 2.0 / (dims + 2.0) * volume_unit_ball(dims)
    return kernel / normalization


@kernel
def gaussian(x, dims):
    def gaussian_integral(n):
        if n % 2 == 0:
            return np.sqrt(np.pi * 2) * factorial2(n - 1) / 2
        elif n % 2 == 1:
            return np.sqrt(np.pi * 2) * factorial2(n - 1) * norm.pdf(0)
    kernel = np.exp(-x ** 2 / 2.0)
    normalization = dims * gaussian_integral(dims - 1) * volume_unit_ball(dims)
    return kernel / normalization


@kernel
def uniform(x, dims):
    mask = x < 1.0
    kernel = np.zeros_like(x)
    kernel[mask] = 1.0
    normalization = volume_unit_ball(dims)
    return kernel / normalization

_kernels = {epanechnikov, gaussian, uniform}


def silvermans_rule(timeseries):
    n, dims = timeseries.shape
    sigma = np.std(timeseries, axis=0)
    sigma = sigma.max()

    return sigma * (4.0 / (3 * n)) ** (1 / 5)
