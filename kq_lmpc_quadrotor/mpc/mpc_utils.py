"""
@author: Santosh Rajkumar

Go Buckeyes!
"""

import numpy as np



def get_state_input_bounds(conservative=True):

    u_max = np.array([30.56, 0.764, 0.764, 0.0378]) 
    u_min = np.array([0, -0.764, -0.764, -0.0378]) 
    s_max = np.array([2.5,2.5,2.5])
    v_max = np.array([2,2,2])
    omega_max = np.array([0.706, 0.706,0.706])



    return u_max, u_min, s_max, v_max, omega_max



def init_guide(koop, mpc_div, gref, dt, t=0, nmpc_flag=False):
    if nmpc_flag:
        x_traj_init = np.zeros((18,mpc_div+1))
    else:
        x_traj_init = np.zeros((9*koop.M+9*koop.N,mpc_div+1))
    u_traj_init = np.zeros((4,mpc_div))

    for i in range(mpc_div+1):
        xref, uref = gref.gen_ref(t+i*dt)
        if nmpc_flag:
            x_traj_init[:,i] = xref.flatten()
        else:
            x_traj_init[:,i] = koop.fcn_gen_koopman_states_se3(xref).flatten()
        if i < mpc_div:
            u_traj_init[:,i] = uref.flatten()

    return x_traj_init, u_traj_init



def shift_guide(x_traj_init, u_traj_init, gref,
                 koop, mpc_div,t,dt, ocp_solver,
                   nmpc_flag=False):

    for i in range(mpc_div+1):
        if i < mpc_div:
            x_traj_init[:,i] = ocp_solver.get(i+1,'x')
        else:
            xref, uref = gref.gen_ref(t+i*dt)
            if nmpc_flag:
                x_traj_init[:,i] = xref.flatten()
            else:
                x_traj_init[:,i] = koop.fcn_gen_koopman_states_se3(xref).flatten()
        if i < mpc_div-1:
            u_traj_init[:,i] = ocp_solver.get(i+1,'u')
        
    return x_traj_init, u_traj_init


