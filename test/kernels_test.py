import numpy as np
from itertools import product

# import sys
# sys.path.append("..")
from kramersmoyal.kernels import *

def test_kernels():
    for dim in [1, 2, 3]:
        edges = [np.linspace(-10, 10, 100000 // 10**dim, endpoint=True)] * dim
        mesh = np.asarray(list(product(*edges)))
        dx = (edges[0][1] - edges[0][0]) ** dim
        for kernel in [epanechnikov, gaussian, uniform, triagular]:
            for bw in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
                kernel_ = kernel(mesh, bw=bw).reshape(
                    *(edge.size for edge in edges))
                passed = np.allclose(kernel_.sum() * dx, 1, atol=1e-2)
                print("Kernel {0:10s}\t with {1:.2f} bandwidth at {2}D passed: {3}".format(
                    kernel.__name__, bw, dim, passed))
            print()


test_kernels()
