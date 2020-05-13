import numpy as np

from kramersmoyal import km
from kramersmoyal import kernels

def test_kmc():
    for t in [1,0.1,0.01,0.001]:
        for lag in [None, [1,2,3]]:

            X = np.random.normal(loc = 0, scale = np.sqrt(t), size = 10000)

            bins = np.array([5000])

            powers = np.array([[1], [2]])

            bw = 0.15

            # The kmc holds the results, where edges holds the binning space
            kmc, edges = km(X, kernel = kernels.epanechnikov, bw = bw,
                    bins = bins, powers = powers)

            assert isinstance(kmc, np.ndarray)
            assert isinstance(edges[0], np.ndarray)

            kmc, edges = km(X, kernel = kernels.epanechnikov, bins = bins,
                        powers = powers)

            assert isinstance(kmc, np.ndarray)
            assert isinstance(edges[0], np.ndarray)


test_kmc()
