"""
@author: Santosh Rajkumar

Go Buckeyes!
"""

# Regular Imports
import numpy as np
from scipy.integrate import solve_ivp

## Koopman Lifting Aid + Plant
from kq_lmpc_quadrotor import KoopmanLift, UtilsFunctions, quadrotor_nl_plant
# Reference generation aid
from kq_lmpc_quadrotor import Gen_ref_dfb, setup_ref, init_moving_pos
# Import LQR Componenets
from kq_lmpc_quadrotor  import lqr_init, compute_lqr_control
# import MPC utils
from kq_lmpc_quadrotor import init_guide,shift_guide, get_state_input_bounds
from kq_lmpc_quadrotor.mpc.kq_lmpc import  setup_ocp_kqlmpc, lift_constr   
from kq_lmpc_quadrotor.mpc.nmpc import  setup_ocp_nmpc                                                         
# Visualization aid
from kq_lmpc_quadrotor import pltt, animate_quadrotor



def kqlmpc_demo(tsim=15, task_id = 3, nmpc_flag = False, meas_noise=False):

    """      
    Koopman Lifting Init
    """
    # initialize the utilities functions
    utilsf = UtilsFunctions(M=3,N=2)
    # get the quadrotor parameters and the # of observables
    params, M, N = utilsf.get_quad_params()
    # Initialize KoopmanLift class for Koopman-based state lifting
    koop = KoopmanLift(params, M, N)


    """
    Set task_id and time of simulation
    """ 
    #task_id, tsim = 6, 20.0
    #nmpc_flag = 0 # 0 for KQ-LMPC, 1 for NMPC
    #meas_noise = True

    """
    Reference Trajectory Parameters
    """
    hover_init_pos =  np.array([0,0,0.0]) # initial start point
    rad_lem, omega_lem, h_lem = 1.0, 0.6, 1.0 # Lemniscate Parameters
    rad_hel, omega_hel, helix_scl = 1.0, 0.6, 10 # helics parameters
    omega_knot, scale_knot = 0.6, 0.3 # knot parameters

    traj_params = (hover_init_pos, rad_lem, omega_lem, h_lem, rad_hel, omega_hel, helix_scl, omega_knot, scale_knot)
    # compute start point and velocity
    init_pos, init_vel = init_moving_pos(task_id, traj_params)


    """
    OCP & Integrator parameters + Setup
    """
    ctrlDt, mpc_horizon, mpc_div, simDt = 0.02, 1.0, 10, 0.01  # sec, time interval of control application
    delt = mpc_horizon/mpc_div # delt: MPC time interval b/w two divisions

    if nmpc_flag:
        ocp_solver = setup_ocp_nmpc(mpc_div, mpc_horizon, koop)
    else:
        ocp_solver = setup_ocp_kqlmpc(mpc_div, mpc_horizon, koop)

    time_sim = [0] # storing the time vector as per plant sampling
    comptime = [] # storage variable for computing MPC computation time at each step
    time_mpc = 0 # sec, current timetime as per MPC sampling
    iter_t = 0 # time step iteration for MPC sampling

    ### Specify final hover pos (placeholder for task_id > 2)
    hov_final_pos = np.array([1,1,1])
    X = Gen_ref_dfb.static_ref_ic(s=init_pos, v=init_vel)
    gref = setup_ref(task_id, traj_params, tsim, s0 = init_pos, v0=init_vel, no_hov=0.95, sd=hov_final_pos)
    Xd,_ = gref.gen_ref(time_mpc) # storing the reference trajectory
    u_act = np.zeros((1,4)) # initializing control input storage


    Bbar = koop.fcn_Bbar() # LTI B matrix
    K = lqr_init(koop.fcn_A_lifted(),Bbar,M,N) # lqr feedback gain
    ## initialize guide or PPOT
    x_traj_init, u_traj_init = init_guide(koop, mpc_div, gref, delt, t=0, nmpc_flag=nmpc_flag)

    # get constraints
    u_max, u_min, s_max, v_max, omega_max = get_state_input_bounds(conservative=False)

    # -------------------------------
    # Simulation loop
    # -------------------------------

    while time_mpc < tsim-ctrlDt:

        X_t = X[:,-1].reshape(-1,1)

        if meas_noise:
            X_t = utilsf.fcn_meas_noise(X_t, noise_power=0.005)

        if not nmpc_flag:
            X_tl = koop.fcn_gen_koopman_states_se3(X_t)
            x_traj_init[:,0] = X_tl.flatten()
            ocp_solver.set(0, "lbx", X_tl)
            ocp_solver.set(0, "ubx", X_tl)
        else:
            ocp_solver.set(0, "lbx", X_t)
            ocp_solver.set(0, "ubx", X_t)
            x_traj_init[:,0] = X_t.flatten()


        for i in range(mpc_div+1):
                    # generate the reference trajectory for the MPC
            xref, uref = gref.gen_ref(time_mpc+i*mpc_horizon/mpc_div)
            if not nmpc_flag:
                x_star = koop.fcn_se3_states_to_actual(x_traj_init[:,i].reshape(-1,1))
                mpc_par = np.vstack([xref, uref, x_star])

                if i>0 and i<mpc_div:
                    Xlift_max, Xlift_min = lift_constr((s_max, v_max, omega_max), x_star[0:18])
                    ocp_solver.constraints_set(i, 'ubx', Xlift_max)
                    ocp_solver.constraints_set(i, 'lbx', Xlift_min)
            else:
                mpc_par = np.vstack([xref, uref])
                if i>0 and i < mpc_div:
                    ocp_solver.constraints_set(i, 'ubx', np.concatenate([s_max, v_max, omega_max]))
                    ocp_solver.constraints_set(i, 'lbx', -np.concatenate([s_max, v_max, omega_max]))
        
            ocp_solver.set(i, "p", mpc_par) 
            ocp_solver.set(i, "x", x_traj_init[:,i].reshape(-1,1))

            if i < mpc_div:
                ocp_solver.set(i, "u", u_traj_init[:,i].reshape(-1,1))

                ocp_solver.constraints_set(i, 'ubu', u_max)
                ocp_solver.constraints_set(i, 'lbu', u_min)

        
        status = ocp_solver.solve()
        if status == 0:
            comptime =  np.append(comptime, ocp_solver.get_stats('time_tot'))
            u_mpc= ocp_solver.get(0, "u")
            
            x_traj_init, u_traj_init = shift_guide(x_traj_init, u_traj_init, gref,
                        koop, mpc_div,time_mpc,delt, ocp_solver,
                        nmpc_flag=nmpc_flag)
        else:
            print('Fallback LQR')
            u_mpc = compute_lqr_control(X_tl, koop.fcn_gen_koopman_states_se3(Xd[:,-1]),Bbar,K,koop, emp_thrust_comp=True)
            x_traj_init, u_traj_init = init_guide(koop, mpc_div, gref, delt, t=time_mpc, nmpc_flag=nmpc_flag)
        u_act = np.vstack([u_act, u_mpc.T])

        sol = solve_ivp(quadrotor_nl_plant, (0,ctrlDt), X_t.flatten(), args=(u_mpc.flatten(),params), 
                        method='RK45', t_eval = np.linspace(0, ctrlDt, num=3), dense_output=True)
        X = np.hstack([X, sol.y[:,-1].reshape(-1,1)])
        Xd_new,_ = gref.gen_ref(time_mpc+ctrlDt)
        Xd = np.hstack([Xd, Xd_new])
        time_mpc += ctrlDt # advance the time of control
        iter_t += 1
        time_sim.append(time_sim[-1] + ctrlDt)
                        

    fig1= pltt.fcn_plot_traj_track(X,Xd,X[0:3,0], task_id=task_id)
    if meas_noise:
        print(f'Mean optimization time {np.mean(comptime)*1e3} ms')
        print(f'Wors case computation time {np.max(comptime)*1e3} ms')

  


