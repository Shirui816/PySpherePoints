cimport cython
from cython.parallel cimport prange
from cython.parallel cimport parallel
from libc.math cimport sqrt
from libc.math cimport pow


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
def u(double[:,:] x):
	cdef long i
	cdef long j,k
	cdef double ret=0,tmp
	with nogil, parallel():
		for i in prange(x.shape[0]-1,  schedule='static'):
			for j in range(i + 1, x.shape[0]):
				tmp = 0
				for k in range(x.shape[1]):
					tmp = tmp + pow(x[i,k]-x[j,k],2)
				ret += 1 / sqrt(tmp)
				#ret = ret + 1 / sqrt(tmp)
				# do not make it this way: reduction variable & thread-private
				# ret = ret + ... produces nan results
	return ret


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
def uf(double[:,:] x, double[:] norm, double[:] ratio):
	r"""Ellipsoid surface, norm = np.linalg.norm(x/ratio, axis=-1)
	ratio = np.asarray([1,2,3], dtype=np.float) for example.
	"""
        cdef long i
        cdef long j,k
        cdef double l, m, n
        cdef double ret=0,d, a,b,c,r, xp, yp, zp
        cdef np.ndarray[double, ndim=2] grad
        l = ratio[0]
        m = ratio[1]
        n = ratio[2]
        grad = np.zeros((x.shape[0], x.shape[1]), dtype=np.float)
        with nogil, parallel():
                for i in prange(x.shape[0],  schedule='static'):
                        xp = x[i, 0] / norm[i]
                        yp = x[i, 1] / norm[i]
                        zp = x[i, 2] / norm[i]
                        r = norm[i]
                        for j in range(0, x.shape[0]):
                                if i == j:
                                        continue
                                a = x[j, 0]/norm[j]
                                b = x[j, 1]/norm[j]
                                c = x[j, 2]/norm[j]
                                d = (a-xp)**2 + (b-yp)**2 + (c-zp)**2
                                d = sqrt(d)
                                grad[i, 0] += - ((xp**2/(l**2*r)-1/r)*(a-xp) + (xp*yp*(b-yp)+xp*zp*(c-zp))/(l**2*r))/d**3
                                grad[i, 1] += - ((yp**2/(m**2*r)-1/r)*(b-yp) + (xp*yp*(a-xp)+yp*zp*(c-zp))/(m**2*r))/d**3
                                grad[i, 2] += - ((zp**2/(n**2*r)-1/r)*(c-zp) + (xp*zp*(a-xp)+yp*zp*(b-yp))/(n**2*r))/d**3
                                ret += 1 / d
                                #ret = ret + 1 / sqrt(tmp)
                                # do not make it this way: reduction variable & thread-private
                                # ret = ret + ... produces nan results
        return ret, grad.reshape(-1)
