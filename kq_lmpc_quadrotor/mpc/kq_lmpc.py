
"""
@author: Santosh Rajkumar

Go Buckeyes!
"""

import numpy as np
from ..Utils.lqr_utils import lqr_riccati, cost_weightings_koopman
import casadi as ca
try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except:
    print('acados is not set up, MPC demo will not run')
from ..Utils.Reference_Generator_DFB import Gen_ref_dfb


def cost_lmpc(koop):
    nstates, nctrl = 9*koop.M+9*koop.N, 4
    Q_x = np.zeros((nstates, nstates))
    m=koop.M
    R_u = 0.001*np.eye(nctrl)
    R_u[1, 1] = 0.0001

    Q_x[0:3, 0:3] = 1e3 * np.eye(3)
    Q_x[3:6, 3:6] = 500 * np.eye(3)
    Q_x[3 * m:3 * m + 3, 3 * m:3 * m + 3] = 500 * np.eye(3)
    Q_x[3 * m + 3:3 * m + 6, 3 * m + 3:3 * m + 6] = 500 * np.eye(3)
    Q_x[9 * m:9 * m + 9, 9 * m:9 * m + 9] = 500* np.eye(9)
    Q_x[9 * m + 9:9 * m + 18, 9 * m + 9:9 * m + 18] = 200* np.eye(9)
    return Q_x, R_u



def quad_lifted_ql(koop, use_lqr_wt=False, term_cost=False):

    # Get LQR weightings
    Q_lqr, R_lqr = cost_weightings_koopman(koop.M,koop.N)
    A = koop.fcn_A_lifted() # Lifted A
    B = koop.fcn_Bbar()  # LTI Bbar
    K, P  = lqr_riccati(Q_lqr,R_lqr,A,B) # LQR Gain & Lyapunov matrix

    # state and input dimension in the lifted space
    nstates, nctrl = 9*koop.M+9*koop.N, 4

    X = ca.SX.sym('x', nstates, 1) # state vector symbolic
    u = ca.SX.sym('U', nctrl, 1) # control vector symbolic
    p = ca.SX.sym('p', 18+4+18) # parameter vector, dim(ref x + ref u + seed/ppot+scaling of omega)

    ## extract references
    Xref = koop.casadi_gen_koopman_states_se3(p[0:18]) # lifted reference X_r
    uref = p[18:22] # input ref u_r
    xstar = p[22:40] # ppot (actual, not lifted)

    ## construct state depndent B matrix
    calB = koop.casadi_calB(xstar)
    # construct modified control
    u_tilde = koop.casadi_u2u_tilde(xstar, u, sc=1)

    # lifted nominal dynamics
    dyn_expr_f_expl = A @ X + calB @ u_tilde

    # get LMPC weights
    Q_x, R_u = cost_lmpc(koop)

    if use_lqr_wt:
        if term_cost:
            cost_expr_ext_cost_e = ca.mtimes(ca.transpose(X - Xref), ca.mtimes(P, (X - Xref)))
            cost_expr_ext_cost = (  ca.mtimes(ca.transpose(X - Xref), ca.mtimes(Q_lqr, (X - Xref))) +
                                     ca.mtimes(ca.transpose(u - uref), ca.mtimes(R_u, (u - uref)))
                                 )
        else:
            cost_expr_ext_cost_e = ca.mtimes(ca.transpose(X - Xref), ca.mtimes(Q_lqr, (X - Xref)))
            cost_expr_ext_cost = (  cost_expr_ext_cost_e +
                                ca.mtimes(ca.transpose(u - uref), ca.mtimes(R_u, (u - uref)))
                             )
    else:

        if term_cost:
            cost_expr_ext_cost_e = ca.mtimes(ca.transpose(X - Xref), ca.mtimes(P, (X - Xref)))
            cost_expr_ext_cost = (  ca.mtimes(ca.transpose(X - Xref), ca.mtimes(Q_x, (X - Xref))) +
                                     ca.mtimes(ca.transpose(u - uref), ca.mtimes(R_u, (u - uref)))
                                 )
        else:
            cost_expr_ext_cost_e = ca.mtimes(ca.transpose(X - Xref), ca.mtimes(Q_x, (X - Xref)))
            cost_expr_ext_cost = (  cost_expr_ext_cost_e +
                                    ca.mtimes(ca.transpose(u - uref), ca.mtimes(R_u, (u - uref)))
                                 )
   
    ## Acados etup model
    model = AcadosModel()
    model.name = 'quad_lpv'
    model.nstates = nstates
    model.nctrl = nctrl
    model.x = X
    model.u = u
    model.f_expl_expr = dyn_expr_f_expl
    model.cost_expr_ext_cost = cost_expr_ext_cost
    model.cost_expr_ext_cost_e = cost_expr_ext_cost_e
    model.p = p

    return model



## Setup the OCP
def setup_ocp_kqlmpc(mpc_div, mpc_horizon, koop):
    
    """"
    Inputs:
        mpc_div: number of horizon divisions
        mpc_horizon: length of the mpc horizon
    """
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    
    
    model = quad_lifted_ql(koop)
    ocp.model = model  # attach the model to the ocp
    ocp.dims.N = mpc_div  # horizon divisions
    ocp.parameter_values = np.zeros((40,1))
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.tf = mpc_horizon  # mpc_horizon length
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.x0 = koop.fcn_gen_koopman_states_se3(Gen_ref_dfb.static_ref_ic())

    
    # set the input constraints (just initializing)
    ocp.constraints.ubu = np.array([1e3,1e3,1e4,1e4])
    ocp.constraints.lbu = np.array([0,-1e3,-1e4,-1e4])
    ocp.constraints.idxbu = np.arange(4)

    # set the state constraints (just initializing)
    ocp.constraints.ubx = 1e3*np.ones((1,15))
    ocp.constraints.lbx = -1e3*np.ones((1,15))
    M = koop.M
    ocp.constraints.idxbx = np.array([0, 1, 2, 3*M, 3*M+1, 3*M+2,
                            9*(M+1),9*(M+1)+1, 9*(M+1)+2, 9*(M+1)+3,
                                9*(M+1)+4, 9*(M+1)+5, 9*(M+1)+6, 9*(M+1)+7, 9*(M+1)+8])

    
    # Define cost function in OCP
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.cost.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.cost.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e
    
    # Save OCP solver as JSON file
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)
    
    return acados_ocp_solver



# lift the constraints
def lift_constr(max_vals, x_star):
        s_max, v_max, omega_max = max_vals
        s_min, v_min, omega_min = -s_max, -v_max, -omega_max
        R = x_star[6:15].reshape((3,3), order='F')
        Rp = np.maximum(R,0)
        Rn = np.minimum(R,0)

        K = np.array([
            [0, -R[0, 2], R[0, 1]],      # row 1
            [0, -R[1, 2], R[1, 1]],      # row 2
            [0, -R[2, 2], R[2, 1]],      # row 3
            [R[0, 2], 0, -R[0, 0]],      # row 4
            [R[1, 2], 0, -R[1, 0]],      # row 5
            [R[2, 2], 0, -R[2, 0]],      # row 6
            [-R[0, 1], R[0, 0], 0],      # row 7
            [-R[1, 1], R[1, 0], 0],      # row 8
            [-R[2, 1], R[2, 0], 0]       # row 9
        ])

        Kp = np.maximum(K,0)
        Kn = np.minimum(K,0)

        p1_min = Rp @ s_min + Rn @ s_max
        p1_max = Rp @ s_max + Rn @ s_min
        y1_min = Rp @ v_min + Rn @ v_max
        y1_max = Rp @ v_max + Rn @ v_min
        z2_min = Kp @ omega_min + Kn @ omega_max
        z2_max = Kp @ omega_max + Kn @ omega_min

        Xlift_max = np.concatenate([p1_max, y1_max, z2_max])
        Xlift_min = np.concatenate([p1_min, y1_min, z2_min])

        return Xlift_max, Xlift_min