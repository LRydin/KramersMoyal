import numpy as np
from scipy.signal import convolve


def to_grid_index(dataseries: np.ndarray, bin_size: np.ndarray, bw: float):
    assert len(bin_size) == dataseries.shape[
        1], "N-points not matching dimension"

    coords = np.zeros_like(dataseries, dtype=int)

    mins, maxs = dataseries.min(axis=0), dataseries.max(axis=0)
    V = maxs - mins

    for i, point in enumerate(dataseries):
        coords[i, ...] = (
            (point - mins) * (bin_size - 1 + 1e-8) / V).astype(int)

    return coords


def gen_slices(dataseries: np.ndarray, bin_size: np.ndarray, bw: float):
    assert len(bin_size) == dataseries.shape[
        1], "N-points not matching dimension"

    mins, maxs = dataseries.min(axis=0), dataseries.max(axis=0)

    slices = list()
    for i, (min, max, points) in enumerate(zip(mins, maxs, bin_size)):
        step = (max - min) / (points - 1 + 1e-8)
        slices.append(slice(min, max, step))
    return slices


def kmc_kernel_estimator(dataseries: np.ndarray, kernel: callable, bw: float,
                         bin_size: np.ndarray, powers: np.ndarray):
    """
    Estimates Kramers-Moyal coefficients from a timeseries using an kernel
    estimator method.
    """
    # Dataseries to coordinates
    coords = dataseries_to_grid_index(dataseries, bw=bw, bin_size=bin_size)

    # Calculate 1st gradient
    grads = np.gradient(dataseries, axis=0)

    # Kramers-Moyal coefficients
    kmc = np.zeros((*bin_size, np.size(powers, 1)))

    for (coord, grad) in zip(coords, grads):
        kmc[coord] += np.sum(np.power(grad[0], powers[0, :]) *
                             np.power(grad[1], powers[1, :]))

    # Get kernel
    slices = gen_slices(dataseries, n_points=bin_size, border=bw)
    x = np.mgrid[slices].reshape(len(slices), -1).T
    kernel = kernel(x * bw)

    kmc = convolve(kmc, kernel, mode='same', boundary='fill')

    return kmc
