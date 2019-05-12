import numpy as np

class Error(Exception):
	"""
	Error class pass
	"""
	pass

def _assertDim(dim,array):
	if array.ndim != dim:
		raise Error('%d-dimensional array given. Array must be'
							'%d-dimensional' % array.ndim % dim)

def _assertDimRange(array):
	if array.ndim > 2:
		raise Error('%d-dimensional array given. Array must be either a 1- or '
							'2-dimensional array' % array.ndim % dim)
def _assertType(typedesired,array):
	if type(array) != typedesired:
		raise Error('A ' + str(typedesired) +' is required. A ' +
						str(type(array))  + ' was given.')

def OptimalBandwidth(array):
	"""
	Calculates the optimal bandwidth from the data according to Silverman's
	optimal bandwidth.

	Parameters
	----------
	data  : array
		Either 1D or 2D array

	Returns
	-------
	bandwidth  : float
		Returns optimal value of the bandwidth
	"""
	_assertType(np.ndarray, array)
	_assertDimRange(array)

	if array.ndim == 2:
		std_1 = np.std(array[:,0])
		std_2 = np.std(array[:,1])
		std = std_2 if std_1 < std_2 else std_1
		n_size = array[:,0].size
	elif array.ndim == 1:
		std = np.std(array)
		n_size = array.size

	return np.power(( (4 * std) / (5 * n_size) ),1/5)

def Space(n_points,bandwidth):
	"""
	Generates underlying space

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel.
	Returns
	-------
	space  : array
		The underlying space
	"""
	# For a bandwidth of 1, the kernel size is one tenth of the phase space
	kernel_size = int(n_points * bandwidth / 10)
	# linear space
	space = np.linspace(-1 * bandwidth, 1 * bandwidth, kernel_size,
	 					endpoint=False)
	space = space + (space[1] - space[0])
	return space

def Grid(space, bounds=0):
	"""
	Generates underlying space

	Parameters
	----------
	Space  : array
		Requires a 1-dimensional underlying space. Run Space().

	bounds  : array
		Not implemented yet!
		Either an array of two entries, mininum and maximum, or an 2D array of
		minimum and maximum in each dimension. If unspecified and argument data
		is passed as argument, extrema will be taken from there.
	Returns
	-------
	space  : array
		The underlying space
	"""
	# For a bandwidth of 1, the kernel size is one tenth of the phase space

	# 2d linear space for Kernel generator

	# if data != False:
	# 	bounds = np.array([[data[:,0].min(),data[:,0].max()],
	# 			   [data[:,1].min(),data[:,1].max()]])
	# elif bounds.size == 2:
	# 	bounds = np.array([[bounds[0],bounds[1]],
	# 			   [bounds[0],bounds[1]]])

	x_2D, y_2D = np.meshgrid(space, space, sparse=True)
	return x_2D, y_2D

# 1 Dimensional Kernels

def Epanechnikov_1d(n_points, bandwidth='optimal'):
	"""
	Generates the Epanechnikov kernel in 1 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Requires the argument  : data
	Returns
	-------
	kernel  : array
		The specified kernel.
	"""
	#Produce underlying space
	space = Space(n_points,bandwidth)
	# Epanechnikov kernel
	kernel = 1 - np.power((space / bandwidth) , 2)

	# tTheoretical normalisation
	normalisation = 3 / (4 * bandwidth)
	# Numerical re-normalisation to ensure integral is 1
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * (space[1] - space[0]))

	return kernel

def Uniform_1d(n_points, bandwidth='optimal'):
	"""
	Generates a uniform kernel in 1 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Requires the argument  : data
	Returns
	-------
	kernel  : array
		The specified kernel.
	"""
	#Produce underlying space
	space = Space(n_points,bandwidth)
	# Uniform kernel
	kernel = (space*0. + 1.) / (2. * bandwidth)

	# Theoretical normalisation
	normalisation = 1.
	# Numerical re-normalisation to ensure integral is 1
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * (space[1] - space[0]))

	return kernel

def Quartic_1d(n_points, bandwidth='optimal'):
	"""
	Generates a quartic, or biweight, kernel in 1 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Requires the argument  : data
	Returns
	-------
	kernel  : array
		The specified kernel.
	"""
	#Produce underlying space
	space = Space(n_points,bandwidth)
	# Quartic kernel
	kernel = np.power((1 - np.power((space / bandwidth) , 2) ) , 2)

	# tTheoretical normalisation
	normalisation = 15 / (16 * bandwidth)
	# Numerical re-normalisation to ensure integral is 1
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * (space[1] - space[0]))

	return kernel

def Triweight_1d(n_points, bandwidth='optimal'):
	"""
	Generates a triweight kernel in 1 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Requires the argument  : data
	Returns
	-------
	kernel  : array
		The specified kernel.
	"""
	#Produce underlying space
	space = Space(n_points,bandwidth)
	# Triweight kernel
	kernel = np.power((1 - np.power((space / bandwidth) , 2) ) , 3)

	# Theoretical normalisation
	normalisation = 35 / (32 * bandwidth)
	# Numerical re-normalisation to ensure integral is 1
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * (space[1] - space[0]))

	return kernel

def Gaussian_1d(n_points, bandwidth='optimal'):
	"""
	Generates a Gaussian, or normal, kernel in 1 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the kernel. Must be either smaller
		than the total size of the phase space, for bounded compact support
		kernels, or equivalent of the size of the phase space.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Requires the argument  : data
	Returns
	-------
	kernel  : array
		The specified kernel.
	"""
	#Produce underlying space
	space = Space(n_points,bandwidth)
	# Triweight kernel
	kernel = np.exp(- np.power((space / bandwidth), 2) / 2)

	# Theoretical normalisation
	normalisation = 1 / (np.sqrt(np.pi* bandwidth) )
	# Numerical re-normalisation to ensure integral is 1
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * (space[1] - space[0]))

	return kernel

def Epanechnikov_2d(n_points, bandwidth = 'optimal', data=False, bounds=np.array([]),
					symmetric=True):
	"""
	Generates the symmetric Epanechnikov kernel in 2 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the whole phase space. The kernel
		will be of a reduced size in this 2D space of size n_points x n_point.
		Suggested values is of order n_points = 1000.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Can require the argument  : data

	data  : array (2D)
		Data where to apply the kernel. Needed to calculate the extrema, if not
		given by argument bounds, or to	calculate the optimal bandwidth.

	bounds  : array
		Either an array of two entries, mininum and maximum, or an 2D array of
   		minimum and maximum in each dimension. If unspecified and argument data
	   	is passed as argument, extrema will be taken from there.

	symmetric  : boolean
		The Epanechnikov kernel in two dimensional has a symmetric and a
		non-symmetric version.

	Returns
	-------
	kernel  : array (float)
		The specified kernel.
	"""

	#Produce underlying space
	space = Space(n_points,bandwidth)
	# 2d linear space for Kernel generator
	x_2D, y_2D = Grid(space)


	if symmetric == True:
		# Epanechnikov kernel:
		#   (8/3*pi)*3/4(1 - (x² + y²), x=-1 to x=1 CHECK!
		kernel = 1 - (np.power(x1_2D,2) + np.power(y1_2D,2))
							/ (np.power(bandwidth,2))

	elif symmetric == False:
		# Epanechnikov kernel:
		#   (8/3*pi)*3/4(1 - (x + y)², x=-1 to x=1 CHECK!
		kernel = (1 - (np.power(x1_2D,2)))*(1 - (np.power(y1_2D,2)))
							/ (np.power(bandwidth,2))
	#Remove <0 values
	kernel[kernel < 0.] = 0.0

	#normalisation
	normalisation = 2 / (bandwidth * np.pi)
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * np.power((space[1] - space[0]),2))
	return kernel

def Uniform_2d(n_points, bandwidth = 'optimal', data=False, bounds=np.array([]),
					symmetric=True):
	"""
	Generates the symmetric Epanechnikov kernel in 2 dimension

	Parameters
	----------
	n_points  : integer
		Number ( >= 1) total array size of the whole phase space. The kernel
		will be of a reduced size in this 2D space of size n_points x n_point.
		Suggested values is of order n_points = 1000.

	bandwidth  : float
		Number ( >= 0) indicating the bandwidth of the kernel. If unspecified,
		the optimal bandwidth according to Silverman will be used.
		Can require the argument  : data

	data  : array (2D)
	   Data where to apply the kernel. Needed to calculate the extrema, if not
	   given by argument bounds, or to	calculate the optimal bandwidth.

	bounds  : array
	   Either an array of two entries, mininum and maximum, or an 2D array of
	   minimum and maximum in each dimension. If unspecified and argument data
	   is passed as argument, extrema will be taken from there.

	symmetric  : boolean
		Non-existent. Left here for conformity with other kernels. The uniform
		kernel is symmetric

	Returns
	-------
	kernel  : array
		The specified kernel.
	"""

	#Produce underlying space
	space = Space(n_points,bandwidth)
	# 2d linear space for Kernel generator
	x_2D, y_2D = Grid(space)


	# Uniform kernel in 2d:
	#   (8/3*pi)*3/4(1 - (x² + y²), x=-1 to x=1 CHECK!
	kernel = (x_2D*0.  + 1.) / (2. * bandwidth)

	#Remove <0 values
	kernel[kernel < 0.] = 0.0

	#Normalisation
	normalisation = 1.
	kernel = kernel * normalisation
	kernel = kernel / (np.sum(kernel) * np.power((space[1] - space[0]),2))

	return kernel
