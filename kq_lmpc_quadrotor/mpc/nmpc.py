
import numpy as np
from ..Utils.lqr_utils import lqr_riccati, cost_weightings_koopman
import casadi as ca
try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except:
    print('acados is not set up, MPC demo will not run')
from ..Utils.Reference_Generator_DFB import Gen_ref_dfb
from ..QuadrotorKoopman import UtilsFunctions

utilsf = UtilsFunctions(M=3,N=2)




def quad_nl(koop):


    # state and input dimension in the lifted space
    nstates, nctrl = 18, 4

    x = ca.SX.sym('x', nstates, 1)  # State vector
    u = ca.SX.sym('u', nctrl, 1)  # Control inputself.
    p = ca.SX.sym('p', 18+4)
    xref = p[0:18]
    uref = p[18:22]

        # extract the relavent variables
    R = x[6:15].reshape((3,3))
    omega = x[15:18]

    omega_x = utilsf.casadi_skew(omega)
    
    f=u[0]
    M=u[1:]
    
    sdot = x[3:6]
    vdot = -koop.g * koop.e3 + (f/koop.mass) * R @ koop.e3
    # -mtimes(g,e3)+mtimes((f/mass), mtimes(R,e3))
    Rdot = ca.mtimes(R, omega_x)
    omega_dot = np.linalg.inv(koop.J) @ (-ca.mtimes(omega_x, ca.mtimes(koop.J, omega)) + M)
    
    dyn_expr_f_expl = ca.vertcat(sdot, vdot, Rdot.reshape((9,1)), omega_dot)

    Qx_nmpc = 1e3*np.eye(18)
    R_u = 0.001*np.eye(nctrl)

    # Cost expressions
    cost_expr_ext_cost_e = ca.mtimes(ca.transpose(x-xref), ca.mtimes(Qx_nmpc, (x-xref)))  # state cost
    cost_expr_ext_cost = cost_expr_ext_cost_e + ca.mtimes(ca.transpose(u-uref), ca.mtimes(R_u, (u-uref)))

    ## Acados etup model
    model = AcadosModel()
    model.name = 'quad_nl'
    model.nstates = nstates
    model.nctrl = nctrl
    model.x = x
    model.u = u
    model.f_expl_expr = dyn_expr_f_expl
    model.cost_expr_ext_cost = cost_expr_ext_cost
    model.cost_expr_ext_cost_e = cost_expr_ext_cost_e
    model.p = p

    return model



## Setup the OCP
def setup_ocp_nmpc(mpc_div, mpc_horizon, koop):
    
    """"
    Inputs:
        mpc_div: number of horizon divisions
        mpc_horizon: length of the mpc horizon
    """
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    
    
    model = quad_nl(koop)
    ocp.model = model  # attach the model to the ocp
    ocp.dims.N = mpc_div  # horizon divisions
    ocp.parameter_values = np.zeros((22,1))
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.tf = mpc_horizon  # mpc_horizon length
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.x0 = Gen_ref_dfb.static_ref_ic()
    ocp.solver_options.nlp_solver_exact_hessian = True
    ocp.solver_options.regularize_method = 'PROJECT_REDUC_HESS'
    ocp.solver_options.tol = 1e-6
    
    # set the input constraints (just initializing)
    ocp.constraints.ubu = np.array([1e3,1e3,1e4,1e4])
    ocp.constraints.lbu = np.array([0,-1e3,-1e4,-1e4])
    ocp.constraints.idxbu = np.arange(4)

    # set the state constraints (just initializing)
    ocp.constraints.ubx = 1e3*np.ones((1,9))
    ocp.constraints.lbx = 1e3*np.ones((1,9))
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 15, 16, 17])

    
    # Define cost function in OCP
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.cost.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.cost.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e
    
    # Save OCP solver as JSON file
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)
    
    return acados_ocp_solver



