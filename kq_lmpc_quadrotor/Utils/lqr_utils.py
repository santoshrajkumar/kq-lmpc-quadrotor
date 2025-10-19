
import numpy as np
from scipy.linalg import solve_continuous_are, pinv

def lqr_riccati(Q,R,A,B):
    # Solve the continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Calculate LQR gain K = R^(-1) * B^T * P
    K = np.linalg.inv(R) @ B.T @ P
    
    return K, P


def cost_weightings_koopman(M,N):
    Q_x = 5*np.eye(9*M+9*N)
    m= M
    R_u = 0.1*np.eye(9*M+9*N-17)
 

    Q_x[0:3, 0:3] = 2e3 * np.eye(3)
    Q_x[3:6, 3:6] = 1e3 * np.eye(3)
    Q_x[3 * m:3 * m + 3, 3 * m:3 * m + 3] = 200* np.eye(3)
    Q_x[3 * m + 3:3 * m + 6, 3 * m + 3:3 * m + 6] = 100 * np.eye(3)
    Q_x[9 * m:9 * m + 9, 9 * m:9 * m + 9] = 1e3* np.eye(9)
    Q_x[9 * m + 9:9 * m + 18, 9 * m + 9:9 * m + 18] = 1e3* np.eye(9)
    return Q_x, R_u

def lqr_init(A,B,M=3,N=3):
    Q,R = cost_weightings_koopman(M,N)
    K,_ = lqr_riccati(Q,R,A,B)
    return K

def compute_lqr_control(X,X_ref,Bbar,K,koop, emp_thrust_comp=True, f_comp=8.5):
    calB = koop.fcn_calB(X)
    U_lifted = -K @ (koop.fcn_gen_koopman_states_se3(X)-koop.fcn_gen_koopman_states_se3(X_ref))
    u_tilde = pinv(calB) @ Bbar @ U_lifted

    u_lqr = koop.fcn_u_tilde2u(X, u_tilde)
    # if empirical thrust compensation needed
    if emp_thrust_comp:
        u_lqr[0,:]=u_lqr[0,:]+f_comp
    return u_lqr