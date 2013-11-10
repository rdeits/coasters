import numpy as np
from numpy.linalg import norm

def axis2quat(axis):
    axis[:3] = axis[:3] / norm(axis[:3])
    return np.hstack([np.cos(axis[3]/2), axis[:3]*np.sin(axis[3]/2)])

def quat2rotmat(q):
    q = q / norm(q)
    w,x,y,z = q[0], q[1], q[2], q[3]
    return np.array([[w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
          [2*x*y + 2*w*z,  w*w + y*y - x*x - z*z, 2*y*z - 2*w*x],
          [2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w + z*z - x*x - y*y]])

def axis2rotmat(axis):
    """
    axis in [x,y,z,theta] form
    """
    return quat2rotmat(axis2quat(axis))
