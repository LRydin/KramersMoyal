import numpy as np


def epanechnikov_1d(n_points):
    """
    Generates the Epanechnikov kernel in 1 dimension

    Parameters
    ----------
    n_points  : integer
        Number ( >= 1) total array size of the kernel. Must be 
        either smaller than the total size of the phase space,
        for bounded compact support kernels, or equivalent of 
        the size of the phase space.

    Returns
    -------
    kernel  : array
        The specified kernel.
    """
    kernel = 1 - np.linspace(-1, 1, n_points, endpoint=True) ** 2

    normalisation = 3 / 4

    return kernel * normalisation


def Epanechnikov_2d(n_points):
    """
    Generates the symmetric Epanechnikov kernel in 2 dimension

    Parameters
    ----------
    n_points  : integer
        Number ( >= 1) total array size of the kernel. Must be 
        either smaller than the total size of the phase space,
        for bounded compact support kernels, or equivalent of 
        the size of the phase space.

    Returns
    -------
    kernel  : array
        The specified kernel.
    """

    x1 = np.linspace(-1, 1, n_points, endpoint=True)
    x1_2D, y1_2D = np.meshgrid(x1, x1, sparse=True)

    kernel = 1 - (x1_2D ** 2 + y1_2D ** 2)
    kernel[kernel < 0.] = 0.

    normalisation = 2 / np.pi

    return kernel * normalisation
