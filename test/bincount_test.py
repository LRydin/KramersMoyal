import numpy as np

from kramersmoyal.binning import bincount1, bincount2

def test_bincount():
    for N in [10000, 100000, 1000000]:
        for Nw in [1, 5, 10, 20, 40]:
            xy = np.random.randint(100, size=(N))
            weights = np.random.rand(N, Nw).T

            assert (bincount1(xy, weights) == bincount2(xy, weights)).all()
