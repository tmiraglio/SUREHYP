cimport numpy as cnp
#cpdef np.float_t[:,:] _interp3D(np.float_t[:,:,:,::1] v, np.float_t[:] x, np.float_t[:] y, np.float_t[:] z, int nelem, int X, int Y, int Z, int L)
cpdef cnp.ndarray[cnp.float_t, ndim=2] _interp3D(cnp.float_t[:,:,:,::1] v, cnp.float_t[:] x, cnp.float_t[:] y, cnp.float_t[:] z, int nelem, int X, int Y, int Z, int L)

