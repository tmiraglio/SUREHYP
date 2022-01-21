cimport numpy as cnp
import numpy as np
from libc.math cimport floor
from cython cimport boundscheck, wraparound, nonecheck, cdivision
from cython.parallel import prange

@boundscheck(False)
@cdivision(True)
cpdef cnp.ndarray[cnp.float_t, ndim=2] _interp3D(cnp.float_t[:,:,:,::1] v, cnp.float_t[:] x, cnp.float_t[:] y, cnp.float_t[:] z, int nelem, int X, int Y, int Z, int L):

    cdef int i, j, x0, x1, y0, y1, z0, z1, dim
    cdef cnp.float_t xd, yd, zd
    
    c00=np.zeros(L,dtype=float)
    c01=np.zeros(L,dtype=float)
    c10=np.zeros(L,dtype=float)
    c11=np.zeros(L,dtype=float)
    c0=np.zeros(L,dtype=float)
    c1=np.zeros(L,dtype=float)
    c=np.zeros((nelem,L),dtype=float)

    cdef cnp.float_t[:] c00v = c00
    cdef cnp.float_t[:] c01v = c01
    cdef cnp.float_t[:] c10v = c10
    cdef cnp.float_t[:] c11v = c11
    cdef cnp.float_t[:] c0v = c0
    cdef cnp.float_t[:] c1v = c1
    cdef cnp.float_t[:,:] cv = c
    cdef cnp.float_t *v_c

    v_c = &v[0,0,0,0]
    for j in prange(nelem,nogil=True,num_threads=5):
        x0 = <int>floor(x[j])
        x1 = x0 + 1
        y0 = <int>floor(y[j])
        y1 = y0 + 1
        z0 = <int>floor(z[j])
        z1 = z0 + 1
        xd = (x[j]-x0)/(x1-x0)
        yd = (y[j]-y0)/(y1-y0)
        zd = (z[j]-z0)/(z1-z0)
    
        if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 < X and y1 < Y and z1 < Z:
            for i in prange(L):
                c00v[i] = v_c[Y*Z*L*x0+Z*L*y0+L*z0+i]*(1-xd) + v_c[Y*Z*L*x1+Z*L*y0+L*z0+i]*xd
                c01v[i] = v_c[Y*Z*L*x0+Z*L*y0+L*z1+i]*(1-xd) + v_c[Y*Z*L*x1+Z*L*y0+L*z1+i]*xd
                c10v[i] = v_c[Y*Z*L*x0+Z*L*y1+L*z0+i]*(1-xd) + v_c[Y*Z*L*x1+Z*L*y1+L*z0+i]*xd
                c11v[i] = v_c[Y*Z*L*x0+Z*L*y1+L*z1+i]*(1-xd) + v_c[Y*Z*L*x1+Z*L*y1+L*z1+i]*xd

                c0v[i] = c00v[i]*(1-yd) + c10v[i]*yd
                c1v[i] = c01v[i]*(1-yd) + c11v[i]*yd

                cv[j,i] = c0v[i]*(1-zd) + c1v[i]*zd

    return c
