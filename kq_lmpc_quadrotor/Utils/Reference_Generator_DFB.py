"""
Created on 08/28/2025

@author: Santosh Rajkumar

Go Buckeyes! 
"""

"""
Necessary imports
"""
import numpy as np
from ..QuadrotorKoopman import UtilsFunctions
from .quadrotor_util import minimum_snap_trajectory_generator, q_to_rot_mat
#######################################################

class Gen_ref_dfb:

    """
    task_ids:
        1 - hover (move to sd and stay there)
        2 - move to a point following a polynomial trajectory
        3 - Follow Bernouli Lemniscate trajectory
        4 - Move to a point and Make Bernoulli Lemniscate trajectory
        5 - Helical trajectory
        6 - Torus knot trajectory
    
    sd: desired x,y,z position (inertial frame)
    s0: initial x,y,z position (inertial frame)
    tsim: total simulation time
    tsim_ind: time at which hover starts (only for task_id 1 and task_id 2)
    g: gravitational acceleration
    v0: initial velocity (inertial frame)
    vd: desired velocity (inertial frame)
    """
    def __init__(self, task_id=1, sd=np.array([0,0,1.0]), 
                 s0=np.array([0,0,0.0]), tsim=10, no_hov=0.8,
                 g=9.81, dt=0.01, v0=np.zeros((3,1)), vd = np.zeros((3,1))):
        
        self.task_id = task_id
        self.s0 = s0.reshape(-1,1)
        self.sd = sd.reshape(-1,1)
        self.v0 = v0.reshape(-1,1)# considering zero initial velocity
        self.vd = vd.reshape(-1,1)# considering zero desired velocity
        self.tsim = tsim
        self.dt = dt # control sampling time
        utilsf = UtilsFunctions(M=4,N=2)
        self.quad_params,_,_ = utilsf.get_quad_params()
        self.g = g
        self.f_hover = utilsf.params['mass']*g+0.1
        self.no_hov = no_hov
        self.coeff_poly5 = self.compute_poly5_coeff(self.s0, self.v0, self.sd, self.vd)

        if task_id == 3:
            self.omega_lem = 1.5
            self.rad_lem = 1.0
            self.h_lem = 1.0
        
        if task_id == 4 or task_id == 5:
            self.rad_hel = 1
            self.omega_hel = 1.3
            self.helix_scl = 5
        if task_id == 6:
            self.omega_knot = 1.0
            self.scale_knot = 0.3



    def gen_ref(self, t):

        if self.task_id == 1:
            xd = self.static_ref_ic(self.sd)
            ud = np.zeros((4,1))
            ud[0,:] = self.f_hover
            
        elif self.task_id == 2:
                line_flat_t,_ = self.line_follow_flat_vars(t)
                line_flat_tp,xd = self.line_follow_flat_vars(t+self.dt)
                line_flat_all = np.stack((line_flat_t,line_flat_tp),axis=2)
                xrefq,_,urefq = minimum_snap_trajectory_generator(line_flat_all,
                                                                np.zeros((2,2)), [t,t+self.dt],
                                                                    self.quad_params)


        elif self.task_id == 3:
            lem_flat_t, _  = self.lemniscate_flat_vars(t)
            lem_flat_tp, xd = self.lemniscate_flat_vars(t+self.dt)
            lem_flat_all = np.stack((lem_flat_t, lem_flat_tp), axis=2)
            xrefq,_,urefq = minimum_snap_trajectory_generator(lem_flat_all,
                                                              np.zeros((2,2)), [t,t+self.dt],
                                                                self.quad_params)
        elif self.task_id == 4:
            hel_flat_t, _ = self.helics_flat_vars_4(t)
            hel_flat_tp, xd = self.helics_flat_vars_4(t)
            hel_flat_all  = np.stack((hel_flat_t, hel_flat_tp), axis=2)
            xrefq,_,urefq = minimum_snap_trajectory_generator(hel_flat_all,
                                                              np.zeros((2,2)), [t,t+self.dt],
                                                                self.quad_params)
        elif self.task_id == 5:
            hel_flat_t, _ = self.helics_flat_vars_5(t)
            hel_flat_tp, xd = self.helics_flat_vars_5(t)
            hel_flat_all  = np.stack((hel_flat_t, hel_flat_tp), axis=2)
            xrefq,_,urefq = minimum_snap_trajectory_generator(hel_flat_all,
                                                              np.zeros((2,2)), [t,t+self.dt],
                                                                self.quad_params)
        elif self.task_id == 6:
            torus_flat_t, _ = self.torus_knot_flat_vars(t)
            torus_flat_tp, xd = self.torus_knot_flat_vars(t+self.dt)
            torus_flat_all = np.stack((torus_flat_t, torus_flat_tp), axis=2)
            xrefq,_,urefq = minimum_snap_trajectory_generator(torus_flat_all,
                                                            np.zeros((2,2)), [t,t+self.dt],
                                                                self.quad_params)
        if self.task_id > 1:
            if self.task_id == 2 and t > self.tsim*self.no_hov:
                xd = self.static_ref_ic(self.sd)
                ud = np.zeros((4,1))
                ud[0,:] = self.f_hover
            else:
                s = xd[0:3, :].reshape(-1,1)
                v = xrefq[1, 3:6].reshape(-1,1)
                omega = xrefq[1, 10:13].reshape(-1,1)
                R = q_to_rot_mat(xrefq[1, 6:10].flatten())
                xd = self.static_ref_ic(s, v, R, omega)
                ud = urefq[1, :].reshape(-1,1)
        
        return xd, ud
        

    def line_follow_flat_vars(self,t):

        a0, a1, a3, a4, a5 = self.coeff_poly5
        
        # Compute position using the quintic polynfloat(self.s0[2, :]) + omial
        s = a0 + a1*t + a3*t**3 + a4*t**4 + a5*t**5
        # Compute velocity as the first derivative of s(t)
        v = a1 + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        
        # Compute acceleration as the second derivative of s(t)
        acc = 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        
        # Compute jerk as the third derivative of s(t)
        jr = 6*a3 + 24*a4*t + 60*a5*t**2

        # Combine into 4x3 array
        flat_vars = np.zeros((4, 3))
        flat_vars[0, :] = s.flatten()+self.s0.flatten()
        flat_vars[1, :] = v.flatten()
        flat_vars[2, :] = acc.flatten()
        flat_vars[3, :] = jr.flatten()
        
        return flat_vars, self.static_ref_ic(s,v)

    def lemniscate_flat_vars(self, t):
        # Trajectory parameters
        a = self.rad_lem  # size of the 8
        b = self.h_lem       # height parameter
        omega = self.omega_lem  # angular frequency
        
        # Position vector (3D lemniscate of Bernoulli)
        s = np.array([
            a * np.sin(omega * t),
            a * np.sin(omega * t) * np.cos(omega * t),
            b   # Constant altitude
        ])
        
        # Velocity vector
        v = np.array([
            a * omega * np.cos(omega * t),
            a * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2),
            0.0
        ])
        
        # Acceleration vector
        acc = np.array([  # Changed variable name to avoid conflict
            -a * omega**2 * np.sin(omega * t),
            -4 * a * omega**2 * np.sin(omega * t) * np.cos(omega * t),
            0.0
        ])
        
        # Jerk vector
        jr = np.array([
            -a * omega**3 * np.cos(omega * t),
            -4 * a * omega**3 * (np.cos(omega * t)**2 - np.sin(omega * t)**2),
            0.0
        ])
        
        # Combine into 4x3 array
        flat_vars = np.zeros((4, 3))
        flat_vars[0, :] = s
        flat_vars[1, :] = v
        flat_vars[2, :] = acc
        flat_vars[3, :] = jr
        
        return flat_vars, self.static_ref_ic(s,v)
    

    
    def helics_flat_vars_4(self, time):
        w = self.omega_hel
        rad = self.rad_hel
        scl = self.helix_scl
        
        # Position (inertial frame) - Helix in y-z plane, linear in x
        s = np.array([
            time / scl,                          # Linear motion in x (v_linear = 1/scl)
             rad * np.cos(w * time),              # Helix in y
            rad * np.sin(w * time)  # Helix in z
        ])
        
        # Velocity (inertial frame)
        v = np.array([
            1 / scl,                            # Constant velocity in x (v_linear = 1/scl)
            -rad * w * np.sin(w * time),        # Derivative of y
            rad * w * np.cos(w * time)          # Derivative of z
        ])
        
        # Acceleration (inertial frame)
        acc = np.array([
            0,                                  # Derivative of vx
            -rad * w**2 * np.cos(w * time),     # Derivative of vy
            -rad * w**2 * np.sin(w * time)      # Derivative of vz
        ])
        
        # Jerk (inertial frame)
        jr = np.array([
            0,                                  # Derivative of ax
            rad * w**3 * np.sin(w * time),      # Derivative of ay
            -rad * w**3 * np.cos(w * time)      # Derivative of az
        ])

                # Combine into 4x3 array
        flat_vars = np.zeros((4, 3))
        flat_vars[0, :] = s
        flat_vars[1, :] = v
        flat_vars[2, :] = acc
        flat_vars[3, :] = jr
        
        return flat_vars, self.static_ref_ic(s,v)
    
    def helics_flat_vars_5(self, time):
        w = self.omega_hel
        rad = self.rad_hel
        scl = self.helix_scl
        
        # Position (inertial frame) - Helix in x-y plane, linear in z
        s = np.array([
            rad * np.cos(w * time),              # Helix in x
            rad * np.sin(w * time),              # Helix in y  
            time / scl    # Linear motion in z (v_linear = 1/scl)
        ])
        
        # Velocity (inertial frame)
        v = np.array([
            -rad * w * np.sin(w * time),        # Derivative of x
            rad * w * np.cos(w * time),         # Derivative of y
            1 / scl                             # Constant velocity in z (v_linear = 1/scl)
        ])
        
        # Acceleration (inertial frame)
        acc = np.array([
            -rad * w**2 * np.cos(w * time),     # Derivative of vx
            -rad * w**2 * np.sin(w * time),     # Derivative of vy
            0                                   # Derivative of vz
        ])
        
        # Jerk (inertial frame)
        jr = np.array([
            rad * w**3 * np.sin(w * time),      # Derivative of ax
            -rad * w**3 * np.cos(w * time),     # Derivative of ay
            0                                   # Derivative of az
        ])

        # Combine into 4x3 array
        flat_vars = np.zeros((4, 3))
        flat_vars[0, :] = s
        flat_vars[1, :] = v
        flat_vars[2, :] = acc
        flat_vars[3, :] = jr
        
        return flat_vars, self.static_ref_ic(s, v)
    
    def torus_knot_flat_vars(self, t):
        # Torus knot parameters
        omega = self.omega_knot # Angular speed
        scale = self.scale_knot  # Scaling factor: 1/3 to ensure [-1, 1] range
        
        # Position
        s = np.array([
            np.sin(omega * t) + 2 * np.sin(2 * omega * t),
            np.cos(omega * t) - 2 * np.cos(2 * omega * t), 
            4+np.sin(3 * omega * t)  # Removed +1 to center around 0
        ]) * scale # Apply scaling factor
        
        # Velocity
        v = np.array([
            omega * np.cos(omega * t) + 4 * omega * np.cos(2 * omega * t),
            -omega * np.sin(omega * t) + 4 * omega * np.sin(2 * omega * t),
            3 * omega * np.cos(3 * omega * t)
        ]) * scale  # Apply same scaling factor
        
        # Acceleration
        acc = np.array([
            -omega**2 * np.sin(omega * t) - 8 * omega**2 * np.sin(2 * omega * t),
            -omega**2 * np.cos(omega * t) + 8 * omega**2 * np.cos(2 * omega * t),
            -9 * omega**2 * np.sin(3 * omega * t)
        ]) * scale  # Apply same scaling factor
        
        # Jerk
        jr = np.array([
            -omega**3 * np.cos(omega * t) - 16 * omega**3 * np.cos(2 * omega * t),
            omega**3 * np.sin(omega * t) - 16 * omega**3 * np.sin(2 * omega * t),
            -27 * omega**3 * np.cos(3 * omega * t)
        ]) * scale  # Apply same scaling factor

        # Combine into 4x3 array
        flat_vars = np.zeros((4, 3))
        flat_vars[0, :] = s
        flat_vars[1, :] = v
        flat_vars[2, :] = acc
        flat_vars[3, :] = jr
        
        return flat_vars, self.static_ref_ic(s, v)
    
    def compute_poly5_coeff(self, s0, v0, sd, vd):
        T = self.tsim*self.no_hov
        a0 = s0.reshape(-1,1)
        a1 = v0.reshape(-1,1)
        # Since s''(0)=2*a2 = 0, we set a2 = 0.
        # Define helper variables
        A = sd - s0 - v0 * T   # difference between desired and initial conditions
        D = v0 - vd            # difference in velocities (note: v0 - vd)
        
        a3 = (10 * A + 4 * D * T) / (T**3)
        a4 = - (15 * A + 7 * D * T) / (T**4)
        a5 = (3 * (2 * A + D * T)) / (T**5)
        
        return (a0, a1, a3, a4, a5)

    @staticmethod
    def static_ref_ic(s=np.zeros((3,1)), v = np.zeros((3,1)), R = np.eye(3), omega = np.zeros((3,1))):
        xd = np.vstack([s.reshape(-1,1), v.reshape(-1,1), 
                        R.flatten(order='F').reshape(-1,1), omega.reshape(-1,1)])
        return xd
    
    def hover_controls(self):
        f = self.quad_params['mass'] * self.g  # Hover thrust
        u = np.zeros((4,1))
        u[0,:] = f
        return u





## Static Functions
# def init_moving_pos(task_id, init_pos=np.array([0,0,1]), rad_lem=1.0, omega_lem=1.0, h_lem=1.0,
#                      rad_hel=1.0, omega_hel=1.0, helix_scl = 5,
#                        omega_knot=1.0, scale_knot=1.0, init_vel_zero=False):

def setup_ref(task_id, traj_params, tsim, s0, v0, sd, vd=np.zeros((1,3)), no_hov = 0.8):

    gref = Gen_ref_dfb(task_id = task_id, s0=s0, v0=v0, tsim=tsim, sd = sd, vd=vd, no_hov=no_hov)

    hover_init_pos, rad_lem, omega_lem, h_lem, rad_hel, omega_hel, helix_scl, omega_knot, scale_knot = traj_params

    if task_id  == 3:
        gref.rad_lem = rad_lem
        gref.omega_lem = omega_lem
        gref.h_lem = h_lem

    elif task_id  == 4 or task_id == 5:
        gref.rad_hel = rad_hel
        gref.omega_hel = omega_hel
        gref.helix_scl = helix_scl
    else:
        gref.omega_knot = omega_knot
        gref.scale_knot = scale_knot

    return gref
 

def init_moving_pos(task_id, traj_params, init_vel_zero=False):
    
    gref = Gen_ref_dfb(task_id = task_id)

    hover_init_pos, rad_lem, omega_lem, h_lem,\
        rad_hel, omega_hel, helix_scl, omega_knot, scale_knot = traj_params
    
    
    if task_id < 3:
        gref.sd = hover_init_pos
        x0 = gref.static_ref_ic(s=hover_init_pos)
    elif task_id  == 3:
        gref.rad_lem = rad_lem
        gref.omega_lem = omega_lem
        gref.h_lem = h_lem
        if init_vel_zero:
            _, x0 = gref.lemniscate_flat_vars(0.0)
            x0[3:6,:] = np.zeros((1,3))
        else:
          _, x0 = gref.lemniscate_flat_vars(0.0)  

    elif task_id  == 4:
        gref.rad_hel = rad_hel
        gref.omega_hel = omega_hel
        gref.helix_scl = helix_scl
        if init_vel_zero:
            _, x0 = gref.helics_flat_vars_4(0.0)
            x0[3:6,:] = np.zeros((1,3))
        else:
          _, x0 = gref.helics_flat_vars_4(0.0)  

    elif task_id  == 5:
        gref.rad_hel = rad_hel
        gref.omega_hel = omega_hel
        gref.helix_scl = helix_scl
        if init_vel_zero:
            _, x0 = gref.helics_flat_vars_5(0.0)
            x0[3:6,:] = np.zeros((1,3))
        else:
          _, x0 = gref.helics_flat_vars_5(0.0) 
    else:
        gref.omega_knot = omega_knot
        gref.scale_knot = scale_knot
        if init_vel_zero:
            _, x0 = gref.torus_knot_flat_vars(0.0)
            x0[3:6,:] = np.zeros((1,3))
        else:
          _, x0 = gref.torus_knot_flat_vars(0.0) 

    return x0[0:3], x0[3:6]
