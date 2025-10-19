
import numpy as np
from ..QuadrotorKoopman import UtilsFunctions
from scipy.spatial.transform import Rotation as Rot


utilsf = UtilsFunctions(M=3,N=2)

"""
## Nonlinear Plant Model of Quadrotor with additive uncertainty,
"""


def quadrotor_nl_plant(t,x,u,quad_params, noise_bound=0.001):
    mass, J, g, e3 = quad_params['mass'], quad_params['J'], quad_params['g'], quad_params['e3']
    f=u[0]
    tau=u[1:]

    R = x[6:15].reshape((3, 3), order='F')
    omega = x[15:18]
    omega_x = utilsf.skew(omega)
    sdot = x[3:6] 
    vdot = -g * e3 + (f/mass) * R @ e3
    Rdot = R @ omega_x
    omega_dot = np.linalg.inv(J) @ (-omega_x @ (J @ omega) + tau)
    xdot = np.concatenate([sdot, vdot.flatten(), Rdot.flatten(order='F'), omega_dot.flatten()]) + np.random.uniform(-noise_bound,noise_bound)
    return xdot

def rand_state():
    s = np.random.uniform(-1, 1, size=(3, 1))
    v = np.random.uniform(-0.5, 0.5, size=(3, 1))
    omega = np.random.uniform(-0.01, 0.01, size=(3, 1))
    euler_angles_deg = np.random.uniform(-10, 10, size=3)
    R = Rot.from_euler('xyz', euler_angles_deg, degrees=True).as_matrix()
    R = R.flatten(order='F').reshape(-1,1)

    x = np.vstack([s,v,R,omega])

    return x

