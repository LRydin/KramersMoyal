import numpy as np
from scipy.signal import convolve


def to_grid_index(timeseries: np.ndarray, bin_size: np.ndarray, bw: float):
    assert len(bin_size) == timeseries.shape[
        1], "Bins not matching timeseries dimension"

    coords = np.zeros_like(timeseries, dtype=int)

    mins, maxs = timeseries.min(axis=0), timeseries.max(axis=0)

    for dim, (min, max, size) in enumerate(zip(mins, maxs, bin_size)):
        coords[:, dim] = ((timeseries[:, dim] - min) *
                          (size - 1 + 1e-8) / (max - min)).astype(int)

    return coords


def gen_slices(timeseries: np.ndarray, bin_size: np.ndarray, bw: float):
    assert len(bin_size) == timeseries.shape[
        1], "Bins not matching timeseries dimension"

    mins, maxs = timeseries.min(axis=0), timeseries.max(axis=0)

    slices = list()
    for (min, max, size) in zip(mins, maxs, bin_size):
        step = (max - min) / (size - 1 + 1e-8)
        slices.append(slice(min, max, step))
    return slices


def kmc_kernel_estimator(timeseries: np.ndarray, kernel: callable, bw: float,
                         bin_size: np.ndarray, powers: np.ndarray):
    """
    Estimates Kramers-Moyal coefficients from a timeseries using an kernel
    estimator method.
    """
    # Kramers-Moyal coefficients
    kmc = np.zeros((*bin_size, np.size(powers, 1)))

    # Calculate gradient
    grads = np.gradient(timeseries, axis=0)

    # Timeseries to coordinates
    coords = to_grid_index(timeseries, bin_size=bin_size, bw=bw)

    # Binning NOTE: Hardcoded for 2D...
    for (coord, grad) in zip(coords, grads):
        kmc[coord] += np.multiply(*np.power(grad[:, None], powers))

    # Generate kernel
    slices = gen_slices(timeseries, bin_size=bin_size, bw=bw)
    x = np.mgrid[slices].reshape(len(slices), -1).T
    kernel = kernel(x * bw)
    kernel = kernel / np.sum(kernel)

    # Convolution
    for p in range(np.size(powers, 1)):
        kmc[..., p] = convolve(kmc[..., p], kernel,
                               mode='same')

    # Normalisation
    kmc[:, :, 0][kmc[:, :, 0] == 0.] = 1.
    kmc[:, :, 1:] = np.divide(kmc[:, :, 1:], (kmc[:, :, 0, np.newaxis]))

    return kmc
