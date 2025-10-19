"""
@author: Santosh Rajkumar

Go Buckeyes!
"""

import numpy as np

## Quad_Koop imports
from scipy.integrate import solve_ivp

## Koopman Lifting Aid + Plant
from kq_lmpc_quadrotor import KoopmanLift, UtilsFunctions, quadrotor_nl_plant
# Reference generation aid
from kq_lmpc_quadrotor import Gen_ref_dfb, setup_ref, init_moving_pos
# Import LQR Componenets
from kq_lmpc_quadrotor  import lqr_init, compute_lqr_control
# import Visulaization aid
from kq_lmpc_quadrotor import pltt



def lqr_demo(tsim=15, dt=0.01, start=[0,0,0], final_pos=[1,1,1], task_id=3):

    """      
    Initialize Koopman Lifting and Utilities
    """
    # initialize the utilities functions
    utilsf = UtilsFunctions(M=3,N=3)
    # get the quadrotor parameters and the # of observables
    params, M, N = utilsf.get_quad_params() # you can manually set up M, N as well
    # Initialize KoopmanLift class for Koopman-based state lifting
    koop = KoopmanLift(params, M, N)

    """"
        #############  Set Trajectories' Parameters ###################
        hover_init_pos: initial x,y,z position from where the quadrotor starts
        Lemniscate parameters:
            rad_lem: radius (m), omega_lem: ang vel (rad/s), h_lem: starting height
        Helics parameters:
            rad_hel: radius (m), omega_hel: ang vel (rad/s),
            helix_scaling: scaling of linearly growing term (x-direction for task_id 4, z-direction for task_id 5)
        Torus Knot Parameters:
            omega_knot = ang vel (rad/s), 
            scale_knot = scaling to bound x,y,z positions (0.33 bounds in a box of unit volume)
        traj_params: tuple of all the above parameters
        init_pos: x,y,z position (to which onboard controller will work/from which MPC will begin)
        init_vel: velocities (at which onboard controller leaves/from which MPC will begin)
    """
    hover_init_pos =  np.array(start)
    rad_lem, omega_lem, h_lem = 1.0, 0.6, 1.0 # Lemniscate Parameters
    rad_hel, omega_hel, helix_scl = 1.0, 0.6, 10 # helics parameters
    omega_knot, scale_knot = 0.6, 0.3 # knot parameters
    traj_params = (hover_init_pos, rad_lem, omega_lem, h_lem, rad_hel, omega_hel, helix_scl, omega_knot, scale_knot)
    init_pos, init_vel = init_moving_pos(task_id, traj_params)


    """
    ctrdDt : Control update interval
    """
    ctrlDt = dt  # sec, time interval of control application



    time_sim = [0] # storing the time vector as per plant sampling
    comptime = [] # storage variable for computing MPC computation time at each step
    time_mpc = 0 # sec, current timetime as per MPC sampling
    iter_t = 0 # time step iteration for MPC sampling

    ### Specify final hover pos (placeholder for task_id > 2)
    hov_final_pos = np.array(final_pos)
    X = Gen_ref_dfb.static_ref_ic(s=init_pos, v=init_vel)
    gref = setup_ref(task_id, traj_params, tsim, s0 = init_pos, v0=init_vel, no_hov=0.95, sd=hov_final_pos)
    Xd,_ = gref.gen_ref(time_mpc) # storing the reference trajectory
    u_act = np.zeros((1,4)) # initializing control input storage


    # -------------------------------
    # Simulation loop
    # -------------------------------
    Bbar = koop.fcn_Bbar()
    K = lqr_init(koop.fcn_A_lifted(),Bbar,M,N)

    U_bounds = np.array([30.56, 0.764, 0.764, 0.0378])  # From paper
    dt = 0.01  # Control sampling time


    while time_mpc < tsim-ctrlDt:
        
        # current 18x1 state, the last column of X
        X_current = X[:,-1].reshape(-1,1)
        X_ref = Xd[:,-1].reshape(-1,1)
        u_lqr = compute_lqr_control(X_current,X_ref,Bbar,K,koop, emp_thrust_comp=False)
        u_act = np.vstack([u_act, u_lqr.T])
        # apply control to the plant
        sol = solve_ivp(quadrotor_nl_plant, (0,ctrlDt), X_current.flatten(), args=(u_lqr.flatten(),params), 
                        method='RK45', t_eval = np.linspace(0, ctrlDt, num=3), dense_output=True)
        
    
        X = np.hstack([X, sol.y[:,-1].reshape(-1,1)])

        Xd_new,_ = gref.gen_ref(time_mpc+ctrlDt)
        Xd = np.hstack([Xd, Xd_new])
        
        time_mpc += ctrlDt # advance the time of control
        iter_t += 1
        time_sim.append(time_sim[-1] + ctrlDt)

    fig1= pltt.fcn_plot_traj_track(X,Xd,X[0:3,0],task_id=task_id)


    

