import numpy as np


_range = range


def bincount(x, weights, minlength=0):
    return np.array(
        [np.bincount(x, w, minlength=minlength) for w in weights])

# # Prefer for very big bins
# from scipy.sparse import csr_matrix
# def bincount(x, weights, minlength=0):

#     assert len(x.shape) == 1

#     ans_size = x.max() + 1

#     if (ans_size < minlength):
#         ans_size = minlength

#     csr = csr_matrix((np.ones(x.shape[0]), (x, np.arange(x.shape[0]))), shape=[
#                      ans_size, x.shape[0]])
#     return csr * weights


def _get_outer_edges(a, range, bw):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min() - bw, a.max() + bw
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


# An alternative to Numpy's histogramdd, supporting a weights matrix
def histogramdd(sample, bins=10, range=None, normed=None, weights=None, density=None, bw=0.0):

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i], bw)
            edges[i] = np.linspace(smin, smax, bins[i] + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    if weights.ndim == 1:
        hist = hist.reshape(nbin)
    else:
        hist = hist.reshape((weights.shape[0], *nbin))

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D * (slice(1, -1),)
    hist = hist[(...,) + core]

    # handle the aliasing normed argument
    if normed is None:
        if density is None:
            density = False
    elif density is None:
        # an explicit normed argument was passed, alias it to the new name
        density = normed
    else:
        raise TypeError("Cannot specify both 'normed' and 'density'")

    if density:
        if weights.ndim == 1:
            # calculate the probability density function
            s = hist.sum()
            for i in _range(D):
                shape = np.ones(D, int)
                shape[i] = nbin[i] - 2
                hist = hist / dedges[i].reshape(shape)
            hist /= s
        else:
            for d in _range(weights.shape[1]):
                s = hist[..., d].sum()
                for i in _range(D):
                    shape = np.ones(D, int)
                    shape[i] = nbin[i] - 2
                    hist[..., d] = hist[..., d] / dedges[i].reshape(shape)
                hist[..., d] /= s

    if weights.ndim == 1:
        if (hist.shape != nbin - 2).any():
            raise RuntimeError(
                "Internal Shape Error")
    else:
        if (hist.shape != np.array([weights.shape[0], *(nbin - 2)])).any():
            raise RuntimeError(
                "Internal Shape Error")
    return hist, edges
