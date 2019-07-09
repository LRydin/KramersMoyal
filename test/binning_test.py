import numpy as np

import sys
sys.path.append("../")
from kramersmoyal.binning import histogramdd

N = 1_000_000

for dim in [1, 2, 3]:
    bins = np.array([30] * dim)
    timeseries = np.random.rand(N, dim)

    Nw = 10
    weights = np.random.rand(N, Nw)

    hist1 = list()
    for w in weights.T:
        hist1.append(np.histogramdd(
            timeseries, bins=bins, weights=w, density=True)[0])

    hist2 = histogramdd(timeseries, bins=bins,
                        weights=weights, density=True)[0]

    assert np.array(
        list(map(lambda i: (hist1[i] == hist2[..., i]), range(Nw)))).all()
    print("Binning in {0}D: passed".format(dim))
