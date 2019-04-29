import numpy as np


def epanechnikov_1d(n_points):
    """
        Generates the Epanechnikov kernel in 1 dimension

        Arguments
            n_points (float):
        Returns
    """
    kernel = 1 - np.linspace(-1, 1, n_points, endpoint=True) ** 2

    normalization = 3 / 4

    return kernel * normalization


def epanechnikov_2d(n_points):
    """
        Generates the Epanechnikov kernel in 1 dimension

        Arguments
            n_points (float):
        Returns
    """
    x1 = np.linspace(-1, 1, n_points, endpoint=True)
    x1_2D, y1_2D = np.meshgrid(x1, x1, sparse=True)

    kernel = 1 - (x1_2D ** 2 + y1_2D ** 2)
    kernel[kernel < 0.] = 0.

    normalization = 2 / np.pi

    return kernel * normalization
