import numpy as np

from kramersmoyal import km
from kramersmoyal import kernels

N = 10000

def test_kmc():

    # Naïve runs, 1-D and 2-D no bins or powers given.
    km(np.random.normal(loc=0, scale=np.sqrt(1), size=10000))
    km(np.random.normal(loc=0, scale=np.sqrt(1), size=(10000, 2)))

    for t in [1, 0.1, 0.01, 0.001]:
        timeseries = np.random.normal(loc=0, scale=np.sqrt(t), size=10000)

        bins = np.array([5000])
        powers = np.array([[1], [2]])
        bw = 0.15

        # The kmc holds the results, where edges holds the binning space
        kmc, edges = km(timeseries, kernel=kernels.epanechnikov, bw=bw,
                bins=bins, powers=powers)

        assert isinstance(kmc, np.ndarray)
        assert isinstance(edges[0], np.ndarray)

        kmc, edges = km(timeseries, kernel=kernels.epanechnikov, bins=bins,
                    powers=powers)

        assert isinstance(kmc, np.ndarray)
        assert isinstance(edges[0], np.ndarray)

    # test powers
    for dim in [1, 2, 3]:
        timeseries = np.random.normal(loc=0, scale=1, size=(N, dim))

        kmc, edges = km(timeseries, powers=4)
        assert isinstance(kmc, np.ndarray)
        assert len(edges) == dim

    powers = np.array([[0, 0, 1, 1, 0, 1, 2, 2, 2],
                       [0, 1, 0, 1, 2, 2, 0, 1, 2]]).T

    timeseries = np.random.normal(loc=0, scale=1, size=(N, 2))
    kmc, edges = km(timeseries, powers=powers)
    assert isinstance(kmc, np.ndarray)
    assert len(edges) == 2
    assert powers.shape[0] == kmc.shape[0]

    powers = np.array([[0, 0, 1, 1, 0, 1, 2, 2, 2],
                       [0, 1, 0, 1, 2, 2, 0, 1, 2],
                       [0, 1, 0, 1, 2, 2, 0, 1, 2]]).T

    timeseries = np.random.normal(loc=0, scale=1, size=(N, 3))
    kmc, edges = km(timeseries, powers=powers)
    assert isinstance(kmc, np.ndarray)
    assert len(edges) == 3
    assert powers.shape[0] == kmc.shape[0]

    # test bins
    for dim in [1, 2, 3]:
        timeseries = np.random.normal(loc=0, scale=1, size=(N, dim))

        kmc, edges = km(timeseries, powers=4, bins=10000)
        assert isinstance(kmc, np.ndarray)
        assert len(edges) == dim

        kmc, edges = km(timeseries, powers=4, bins=[20]*dim)
        assert isinstance(kmc, np.ndarray)
        assert len(edges) == dim

        kmc, edges = km(timeseries, powers=4, bins=(20,)*dim)
        assert isinstance(kmc, np.ndarray)
        assert len(edges) == dim

        kmc, edges = km(timeseries, powers=4, bins=np.array([20]*dim))
        assert isinstance(kmc, np.ndarray)
        assert len(edges) == dim

    timeseries = np.random.normal(loc=0, scale=1, size=(N, 1))
    x = np.linspace(timeseries.min(), timeseries.max(), 10000)
    kmc, edges = km(timeseries, powers=4, bins=[x])
    assert isinstance(kmc, np.ndarray)
    assert len(edges) == 1

    timeseries = np.random.normal(loc=0, scale=1, size=(N, 2))
    x = np.linspace(timeseries.min(), timeseries.max(), 100)
    kmc, edges = km(timeseries, powers=4, bins=[x, x])
    assert isinstance(kmc, np.ndarray)
    assert len(edges) == 2

    timeseries = np.random.normal(loc=0, scale=1, size=(N, 2))
    kmc, edges = km(timeseries, powers=4, bins=[25, 35])
    assert isinstance(kmc, np.ndarray)
    assert len(edges) == 2
