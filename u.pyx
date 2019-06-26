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
	return ret
