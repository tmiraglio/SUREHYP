from .interp import _interp3D
import numpy as np

class Interp3D(object):
    def __init__(self, v, x, y, z):
        self.v = v
        self.min_x, self.max_x = x[0], x[-1]
        self.min_y, self.max_y = y[0], y[-1]
        self.min_z, self.max_z = z[0], z[-1]
        self.delta_x = (self.max_x - self.min_x)/(x.shape[0]-1)
        self.delta_y = (self.max_y - self.min_y)/(y.shape[0]-1)
        self.delta_z = (self.max_z - self.min_z)/(z.shape[0]-1)

    def __call__(self, t):
        X,Y,Z,L = self.v.shape[0], self.v.shape[1], self.v.shape[2],self.v.shape[3]

        x = (t[:,0]-self.min_x)/self.delta_x
        y = (t[:,1]-self.min_y)/self.delta_y
        z = (t[:,2]-self.min_z)/self.delta_z

        nelem=t.shape[0]
        out=np.zeros((nelem,L))
        out=_interp3D(self.v, x, y, z, nelem, X, Y, Z, L)
        return out
        #return _interp3D(self.v, x, y, z, nelem, X, Y, Z, L)

