import numpy as np
from u import u
from numba import njit, prange
from scipy.optimize import minimize
from scipy.spatial import distance


@njit(parallel=True)
def u_jit(x):
    ret = 0
    for i in prange(x.shape[0] - 1):
        xi = x[i]
        for j in prange(i + 1, x.shape[0]):
            ret += 1 / np.linalg.norm(xi - x[j])
    return ret


def u_py(x):
    ret = 0
    for i in range(x.shape[0] - 1):
        xi = x[i]
        for j in range(i + 1, x.shape[0]):
            ret += 1 / np.linalg.norm(xi - x[j])
    return ret


def n1(x):
    x = x.reshape(-1, 3)
    x = x / np.linalg.norm(x, axis=1)[:, None]
    return u_jit(x)


def n2(x):  # fastest while N > 1000 particles, ~1 ms for N=1000
    x = x.reshape(-1, 3)
    x = x / np.linalg.norm(x, axis=1)[:, None]
    return u(x)


def n3(x):  # fastest, if N is small, i.e. N << 1000 particles
    x = x.reshape(-1, 3)  # reshape to 3-D
    # transfer to unit vectors to calculate the energy.
    x = x / np.linalg.norm(x, axis=1)[:, None]
    return np.sum(1 / distance.pdist(x))


# x0 = (N\times 3,) points, for scipy.optimize.minimize takes 1-D array as input.
res = minimize(n3, x0=np.random.random((300,)))
ret = res.x.reshape(-1, 3)
np.savetxt('out.txt', ret / np.linalg.norm(ret, axis=1)[:, None], fmt='%.6f')
