import sys
sys.path.append("..")

from kramersmoyal import kernels
import numpy as np

N = 200
grid = np.linspace(-1, 1, N, endpoint=True)
dx = grid[1] - grid[0]

epanechnikov_1d = kernels.epanechnikov_1d(N)
epanechnikov_2d = kernels.epanechnikov_2d(N)

integrate_e1d = np.sum(epanechnikov_1d) * dx
integrate_e2d = np.sum(epanechnikov_2d) * (dx ** 2)

print("Epanechnikov integral in 1D: {}".format(integrate_e1d))
print("Epanechnikov integral in 2D: {}".format(integrate_e2d))

assert np.allclose(integrate_e1d, 1) "Not enough close in 1D"
assert np.allclose(integrate_e2d, 1) "Not enough close in 2D"

# import matplotlib.pyplot as plt
# plt.plot(range(N), epanechnikov_2d)
# plt.show()
