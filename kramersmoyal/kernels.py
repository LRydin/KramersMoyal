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


def Epanechnikov_2d(n_points, bandwidth = 0, data=False, bounds=False, symmetric=True):
    """
    Generates the symmetric Epanechnikov kernel in 2 dimension

    Parameters
    ----------
    n_points  : integer
        Number ( >= 1) total array size of the kernel. Must be 
        either smaller than the total size of the phase space,
        for bounded compact support kernels, or equivalent of 
        the size of the phase space.

    bandwidth  : float
        Number ( >= 0) indicating the bandwidth of the kernel.
        If unspecified, the optimal bandwidth according to 
	Silverman will be used.
	Can require the argument  : data

    data  : array (2D)
	Data where to apply the kernel. Needed to calculate the
	extrema, if not given by argument bounds, or to
	calculate the optimal bandwidth	
	
    n_points_data  : integer
        Required to calculate the optimal bandwidth

    symmetric  : boolean
        The Epanechnikov kernel in two dimensional has a 
        symmetric and a non-symmetric version

    Returns
    -------
    kernel  : array (float)
        The specified kernel.
    """

    x1 = np.linspace(-1, 1, n_points, endpoint=True)
    x1_2D, y1_2D = np.meshgrid(x1, x1, sparse=True)


    if symmetric == True:
	# Epanechnikov kernel:
	#   (8/3*pi)*3/4(1 - (x² + y²), x=-1 to x=1 CHECK!
    	kernel = 1 - (np.power(x1_2D,2) + np.power(y1_2D,2)) / (np.power(bandwidth,2))
    	kernel[kernel < 0.] = 0.	#Remove <0 values

    	normalisation = 2 / (bandwidth * np.pi)

    elif symmetric == False:
        

    return kernel * normalisation
