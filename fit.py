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

# x, y, z -> x/a, y/b, z/c
# let x' = (x/a) / sqrt((x/a)^2+(y/b)^2+(z/c)^2) then
# a x' = x / r with r = sqrt((x/a)^2+(y/b)^2+(z/c)^2)
# The function first map x -> x / ratio then
# normalize the (x/a, y/b, z/c) vector -> x', y', z'; afterwards
# energy is calculated as u((x', y',z') \times ratio). 
# for (x', y', z') is normalized, (x', y', z')\times ratio is
# on the ellipsoid surface.
# Therefore the derivative is calculated as
# D[1/sqrt((u-a x')^2+(v-b y')^2+(w-c z')^2), x]
# let f = 1/sqrt((u-a x')^2+(v-b y')^2+(w-c z')^2) then
# df/dx = df/dx' dx'/dx + df/dy' dy'/dx + df/dz' dz'/dx


def n4(x, ratio):  # Ellipsoid support
    x = x.reshape(-1, 3)
    norm = np.linalg.norm(x/ratio, axis=1)
    #x = x / np.linalg.norm(x, axis=1)[:, None]
    #x = x * ratio
    return uf(x/norm[:,None], ratio)



ratio = np.array([1.,1,2])

# x0 = (N\times 3,) points, for scipy.optimize.minimize takes 1-D array as input.
res = minimize(n4, x0=np.random.random((600,)), args=(ratio,), jac=True, tol=1e-3, options=dict(maxiter=1000, disp=True))
#print(res)
ret = res.x.reshape(-1, 3)/ratio
ret = ret / np.linalg.norm(ret, axis=-1)[:, None]
ret = ret * ratio
np.savetxt('out.txt', ret, fmt='%.6f')
