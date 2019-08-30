from timeit import timeit
import matplotlib.pyplot as plot
import numpy as np

import sys
sys.path.append("../")
from kramersmoyal.binning import bincount1, bincount2

b1 = list()
b2 = list()

for N in [1_000_000, 10_000_000]:
    for Nw in [1, 5, 10, 20, 40]:
        xy = np.random.randint(100, size=(N))
        weights = np.random.rand(N, Nw).T

        b1.append(timeit(lambda: bincount1(xy, weights), number=5))
        b2.append(timeit(lambda: bincount2(xy, weights), number=5))

        assert (bincount1(xy, weights) == bincount2(xy, weights)).all()

print(b1)
print(b2)