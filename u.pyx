# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
from cython.parallel cimport prange
from cython.parallel cimport parallel
from libc.math cimport sqrt
from libc.math cimport pow
import multiprocessing
cimport openmp
import numpy as np
cimport numpy as np


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


def uf(double[:,:] x,  double[:] ratio):
        r"""Ellipsoid surface, norm = np.linalg.norm(x/ratio, axis=-1)
        ratio = np.asarray([1,2,3], dtype=np.float) for example.
        """
        cdef long i
        cdef long j,k
        cdef double l, m, n, r=1
        cdef double ret=0,d, a,b,c, xp, yp, zp
        cdef np.ndarray[double, ndim=3] grad
        cdef int nths, thn
        l = ratio[0]
        m = ratio[1]
        n = ratio[2]
        nths = multiprocessing.cpu_count()
        grad = np.zeros((nths, x.shape[0], 3), dtype=np.float)
        with nogil, parallel(num_threads=nths):
                for i in prange(x.shape[0]-1,  schedule='static'):
                        xp = x[i, 0]
                        yp = x[i, 1]
                        zp = x[i, 2]
                        thn = openmp.omp_get_thread_num()
                        for j in range(i+1, x.shape[0]):
                                a = x[j, 0]
                                b = x[j, 1]
                                c = x[j, 2]
                                d = (a-xp)**2 + (b-yp)**2 + (c-zp)**2
                                d = sqrt(d)
                                grad[thn, i, 0] += - ((xp**2/(l**2)-1)*(a-xp) + (xp*yp*(b-yp)+xp*zp*(c-zp))/(l**2))/d**3
                                grad[thn, i, 1] += - ((yp**2/(m**2)-1)*(b-yp) + (xp*yp*(a-xp)+yp*zp*(c-zp))/(m**2))/d**3
                                grad[thn, i, 2] += - ((zp**2/(n**2)-1)*(c-zp) + (xp*zp*(a-xp)+yp*zp*(b-yp))/(n**2))/d**3
                                grad[thn, j, 0] += - ((a**2/(l**2)-1)*(xp-a) + (a*b*(yp-b)+a*c*(zp-c))/(l**2))/d**3
                                grad[thn, j, 1] += - ((b**2/(m**2)-1)*(yp-b) + (a*b*(xp-a)+b*c*(zp-c))/(m**2))/d**3
                                grad[thn, j, 2] += - ((c**2/(n**2)-1)*(zp-c) + (a*c*(xp-a)+b*c*(yp-b))/(n**2))/d**3
                                ret += 1 / d
                                #ret = ret + 1 / sqrt(tmp)
                                # do not make it this way: reduction variable & thread-private
                                # ret = ret + ... produces nan results
        return ret, grad.sum(axis=0).reshape(-1)
