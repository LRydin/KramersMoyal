import numpy as np

from kramersmoyal.binning import histogramdd

N = 1000000

def test_binning():
    for dim in [1, 2, 3]:
        bins = np.array([30] * dim)
        timeseries = np.random.rand(N, dim)

        Nw = 10
        weights = np.random.rand(N, Nw)

        hist1 = [np.histogramdd(timeseries, bins=bins,
                                weights=w, density=True)[0] for w in weights.T]

        hist2 = histogramdd(timeseries, bins=bins,
                            weights=weights.T, density=True)[0]

        assert np.array(
            list(map(lambda i: (hist1[i] == hist2[i, ...]), range(Nw)))).all()
