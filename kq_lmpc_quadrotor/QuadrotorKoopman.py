import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as rot
import casadi as ca
import matplotlib.pyplot as plt


class KoopmanLift:
    
    def __init__(self, params, M, N):
        # Initialize properties
        self.mass = params['mass']
        self.J = params['J']
        self.d = params['d']
        self.M = M
        self.N = N
        self.g = params['g']
        self.e3 = params['e3']
        self.utilsf = UtilsFunctions(M=self.M, N=self.N)  # Initialize the UtilityFunctionClass
        self.num_states_lifted = 9*M+9*N
        self.num_ctrl_lifted = 9*M+9*N-17
        
        
    
    def fcn_gen_koopman_states_se3(self, Xin):
        """
        Parameters
        ----------
        Xin : 18x1 vector containing the quadrotor state
        Xin = [x, v, R.flatten(order='F'), omega]^T

        Returns
        -------
        X_out : (9M+9N)x1 vector - the lifted state 
        X_out = [p1,p2,...,y1,y2,...,h1,h2,..., z1vec, z2vec,...]
        

        """
        # extract the components
        R = Xin[6:15].reshape((3,3),order='F') # rotation matrix
        x = Xin[0:3] # s , m
        v = Xin[3:6] # v, m/sec
        w = Xin[15:18] # omega, rad/sec
        
        gvec = np.array([[0], [0], [self.g]])  # g*e3
        wx = self.utilsf.skew(w)  # Skewed omega
        
        hs = [] # h1,h2,....
        ys = [] # y1,y2,....
        ps = [] # p1,p2,....
        
        for i in range(self.M):
            hs.append(-np.linalg.matrix_power(wx.T, i) @ R.T @ gvec)
            ys.append(np.linalg.matrix_power(wx.T, i) @ R.T @ v)
            ps.append(np.linalg.matrix_power(wx.T, i) @ R.T @ x)
        
        hs = np.concatenate(hs)
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        
        X_out = np.concatenate([ps.reshape(-1, 1), ys.reshape(-1, 1), hs.reshape(-1, 1),\
                                self.fcn_gen_koopman_states_so3(R, w).reshape(-1,1)])
        return X_out
    
    def fcn_gen_koopman_states_so3(self, R, w):
        """

        Parameters
        ----------
        R : 3x3 rotation matrix
        w : 3x1 vector of the angular rates (omega)

        Returns
        -------
        X : 9Nx1 vector of the lifted states for the attitude dynamics

        """
        
        n = self.N
        X = np.zeros(9 * n)
        wx = self.utilsf.skew(w)
        
        for k in range(n):
            z_nvec = R @ np.linalg.matrix_power(wx, k)
            X[9*k:9*(k+1)] = z_nvec.flatten(order='F')
        
        return X
    
    def fcn_se3_states_to_actual(self, Xin):
        """

        Parameters
        ----------
        Xin : (9M+9N)x1 vector - the lifted state
            Xin = [p1,p2,...,y1,y2,...,h1,h2,..., z1vec, z2vec,...]
        Returns
        -------
        X_out : 18x1 vector containing the quadrotor state
                Xout = [x, v, R.flatten(order='F'), omega]^T

        """
      
        R = Xin[9*self.M:9*(self.M+1)].reshape((3,3), order='F')
        omega_x = R.T @ np.reshape(Xin[9*(self.M+1):9*(self.M+1)+9], (3, 3), order='F')
        
        x = R @ Xin[0:3]
        v = R @ Xin[3*self.M:3*(self.M+1)]
        
        X_out = np.concatenate([x.reshape(-1, 1), v.reshape(-1, 1), R.flatten(order='F').reshape(-1, 1), self.utilsf.fcn_vee(omega_x).reshape(-1, 1)])
        return X_out



    def casadi_gen_koopman_states_se3(self, Xin):
        """
        Parameters
        ----------
        Xin : 18x1 vector containing the quadrotor state
        Xin = [x, v, R.flatten(order='F'), omega]^T
    
        Returns
        -------
        X_out : (9M+9N)x1 vector - the lifted state 
        X_out = [p1,p2,...,y1,y2,...,h1,h2,..., z1vec, z2vec,...]
        """
        # Extract the components
        R = Xin[6:15].reshape((3, 3))  # rotation matrix
        x = Xin[0:3]  # s , m
        v = Xin[3:6]  # v, m/sec
        w = Xin[15:18]  # omega, rad/sec
        
    
        gvec = ca.DM([[0], [0], [self.g]])  # g*e3
        wx = self.utilsf.casadi_skew(w)  # Skewed omega
    
        hs = []  # h1,h2,....
        ys = []  # y1,y2,....
        ps = []  # p1,p2,....
    
        for i in range(self.M):
            hs.append(-ca.mpower(wx.T, i) @ R.T @ gvec)
            ys.append(ca.mpower(wx.T, i) @ R.T @ v)
            ps.append(ca.mpower(wx.T, i) @ R.T @ x)
        

        hs = ca.vertcat(*hs)
        ys = ca.vertcat(*ys)
        ps = ca.vertcat(*ps)
    
        X_out = ca.vertcat(ps, ys, hs, self.casadi_gen_koopman_states_so3(R, w))
        return X_out
    
    def casadi_gen_koopman_states_so3(self, R, w):
        """
        Parameters
        ----------
        R : 3x3 rotation matrix
        w : 3x1 vector of the angular rates (omega)
    
        Returns
        -------
        X : 9Nx1 vector of the lifted states for the attitude dynamics
        """
        n = self.N
        X = []
        wx = self.utilsf.casadi_skew(w)
    
        for k in range(n):
            z_nvec = R @ ca.mpower(wx, k)
            X.append(z_nvec.reshape((9, 1)))
        X = ca.vertcat(*X)
        return X

    
    def casadi_se3_to_actual(self,Xin):
        M, N = self.M, self.N
        R = Xin[9 * M:9 * (M + 1)].reshape((3, 3))
        # Extract omega_x
        omega_x = R.T @ Xin[9 * (M + 1):9 * (M + 1) + 9].reshape((3, 3))
        # Compute x and v
        s = R @ Xin[0:3]
        v = R @ Xin[3 * M:3 * (M + 1)]
        omega = self.utilsf.casadi_vee(omega_x)
        Xout = ca.vertcat(s,v, R.reshape((9,1)), omega)
        return Xout
    
    def fcn_A_lifted(self):
        
        """
            A: square matrix of dimension 9M+9N, system matrix
        """

        
        m, n = self.M, self.N
        A1 = np.block([
            [np.zeros((3*(m-1), 3)), np.eye(3*(m-1)), np.eye(3*(m-1)), np.zeros((3*(m-1), 3))],
            [np.zeros((3, 3)), np.zeros((3, 3*(m-1))), np.zeros((3, 3*(m-1))), np.zeros((3, 3))]
        ])
        
        A2 = A1
        A3 = np.block([
            [np.zeros((3*(m-1), 3)), np.eye(3*(m-1))],
            [np.zeros((3, 3)), np.zeros((3, 3*(m-1)))]
        ])
        
        Aa = np.block([
            [np.zeros((9*(n-1), 9)), np.eye(9*(n-1))],
            [np.zeros((9, 9)), np.zeros((9, 9*(n-1)))]
        ])
        
        Ap = np.block([
            [A1, np.zeros((3*m, 3*m))],
            [np.zeros((3*m, 3*m)), A2],
            [np.zeros((3*m, 6*m)), A3]
        ])
        
        A = block_diag(Ap, Aa)
        
        return A
    
    def fcn_Bbar(self):
        """
            Bbar: Control input matrix for the lifted LTI model
            (9M+9N)x(9M+(n-17))
        """
        m = self.M
        n = self.N
        e3 = np.array([0, 0, 1]).reshape(-1,1)
        B1 = np.zeros((3 * (m-1), 3))
        B1 = np.hstack([B1, np.eye(3 * (m-1))]).T
        B3 = B1.copy()
        B2 = np.vstack([np.hstack([e3, np.zeros((3, 3 * (m-1)))]), np.hstack([np.zeros((3 * (m-1), 1)), np.eye(3 * (m-1))])])
        B4 = np.hstack([np.zeros((9 * (n-1), 9)), np.eye(9 * (n-1))]).T
        #Bbar_SO3 = B4
        Bbar = block_diag(B1, B2, B3, B4)
        return Bbar
    
    def fcn_calB(self, x_act, sc=1):
        """
        

        Parameters
        ----------
        x_act : 18x1 vector containing the quadrotor state
                x_act = [x, v, R.flatten(order='F'), omega]^T

        Returns
        -------
        calB : state dependent B matrix
                (9M+9N)x4

        """

        
        m = self.M
        n = self.N
        X_lifted = self.fcn_gen_koopman_states_se3(x_act)
        omega_x = self.utilsf.skew(x_act[15:18])
        omega_xT = omega_x.T
        R = np.reshape(x_act[6:15], (3, 3))
        lN = 3 * m+n
        p1 = X_lifted[0:3]
        y1 = X_lifted[3 * m:3 * m + 3]
        h1 = X_lifted[6 * m:6 * m + 3]
        e3 = np.array([0, 0, 1]).reshape(-1,1)
        
        Bp_temp = np.zeros((3,4,3*m))
        Ba_temp = np.zeros((9,4,n))
        
        for l in range(1, lN + 1):
            if l == 1 or l == 2 * m + 1:
                Btemp = np.zeros((3, 4))
            elif l > 1 and l <= m:
                Btemp = np.hstack([np.zeros((3, 1)), self.fcn_Psi(l, p1, omega_xT, sc=sc)])
            elif l == m + 1:
                Btemp = np.hstack([e3 / self.mass, np.zeros((3, 3))])
            elif l > m + 1 and l <= 2 * m:
                Btemp = np.hstack([np.linalg.matrix_power(omega_xT, l - m - 1) @ e3 / self.mass, self.fcn_Psi(l - m, y1, omega_xT, sc=sc)])
            elif l > 2 * m + 1 and l <= 3 * m:
                Btemp = np.hstack([np.zeros((3, 1)), self.fcn_Psi(l - 2 * m, h1, omega_xT, sc=sc)])
            elif l == 3 * m + 1:
                Btemp = np.zeros((9, 4))
            else:
                Btemp = np.hstack([np.zeros((9, 1)), np.kron(np.eye(3), R) @ self.fcn_G(l - 3 * m, omega_xT, sc=sc)])
            
            if l <= 3*m:
                Bp_temp[:,:,l-1] = Btemp
                
            else:
                Ba_temp[:,:,l-3*m-1]=Btemp
                
        Bp=Bp_temp[:,:,0]   
        Ba=Ba_temp[:,:,0]
        
        for i in range(1,3*m):
            Bp = np.vstack([Bp, Bp_temp[:,:,i]])
        for i in range(1,n):
            Ba = np.vstack([Ba,Ba_temp[:,:,i]])
        calB = np.vstack([Bp, Ba])
        
        return calB
    
    # Helper function for calB
    def fcn_Psi(self, l, a, omega_xT, sc=1):

        Psi_l = np.zeros((3, 3))
        J_i = np.linalg.inv(self.J)
        for i in range(1, l):
            Psi_l += np.linalg.matrix_power(omega_xT, i - 1) @ self.utilsf.skew(np.linalg.matrix_power(omega_xT, l - 1 - i) @ a) @ J_i
        return Psi_l
    
    # helper function for calB
    def fcn_G(self, l, omega_xT, sc=1):

        g1 = np.zeros((3, 3))
        g2 = np.zeros((3, 3))
        g3 = np.zeros((3, 3))
        J_i = np.linalg.inv(self.J)
        for i in range(1, l):
            g1 += np.linalg.matrix_power(omega_xT, i - 1) @ self.utilsf.skew(J_i[:, 0]) @ np.linalg.matrix_power(omega_xT, l - 1 - i)
            g2 += np.linalg.matrix_power(omega_xT, i - 1) @ self.utilsf.skew(J_i[:, 1]) @ np.linalg.matrix_power(omega_xT, l - 1 - i)
            g3 += np.linalg.matrix_power(omega_xT, i - 1) @ self.utilsf.skew(J_i[:, 2]) @ np.linalg.matrix_power(omega_xT, l - 1 - i)
        G_l = np.hstack([g1.flatten(order='F').reshape(-1,1), g2.flatten(order='F').reshape(-1,1), g3.flatten(order='F').reshape(-1,1)])
        return G_l
    
    
    def casadi_calB(self, x_act, sc=1):
        """
        Calculate the state-dependent B matrix using CasADi.
    
        Parameters
        ----------
        x_act : casadi.DM or casadi.MX, 18x1 vector containing the quadrotor state
                x_act = [x, v, R.flatten(order='F'), omega]^T
    
        Returns
        -------
        calB : casadi.MX or casadi.DM, state-dependent B matrix (9M+9N)x4
        """

        m, n = self.M, self.N
        X_lifted = self.casadi_gen_koopman_states_se3(x_act)
        omega_x = self.utilsf.casadi_skew(x_act[15:18])
        omega_xT = omega_x.T
        R = x_act[6:15].reshape((3, 3))
    
        lN = 3 * m + n
        p1 = X_lifted[0:3]
        y1 = X_lifted[3 * m:3 * m + 3]
        h1 = X_lifted[6 * m:6 * m + 3]
        e3 = ca.DM([0, 0, 1])
    
        Bp_temp = []
        Ba_temp = []
    
        for l in range(1, lN + 1):
            if l == 1 or l == 2 * m + 1:
                Btemp = ca.DM.zeros(3, 4)
            elif l > 1 and l <= m:
                Btemp = ca.horzcat(ca.DM.zeros(3, 1), self.casadi_Psi(l, p1, omega_xT, sc=sc))
            elif l == m + 1:
                Btemp = ca.horzcat(e3 / self.mass, ca.DM.zeros(3, 3))
            elif l > m + 1 and l <= 2 * m:
                Btemp = ca.horzcat(ca.mpower(omega_xT, l - m - 1) @ e3 / self.mass,
                                   self.casadi_Psi(l - m, y1, omega_xT, sc=sc))
            elif l > 2 * m + 1 and l <= 3 * m:
                Btemp = ca.horzcat(ca.DM.zeros(3, 1), self.casadi_Psi(l - 2 * m, h1, omega_xT, sc=sc))
            elif l == 3 * m + 1:
                Btemp = ca.DM.zeros(9, 4)
            else:
                Btemp = ca.horzcat(ca.DM.zeros(9, 1), ca.kron(ca.DM.eye(3), R) @ self.casadi_G(l - 3 * m, omega_xT, sc=sc))
    
            if l <= 3 * m:
                Bp_temp.append(Btemp)
            else:
                Ba_temp.append(Btemp)
    
        Bp = ca.vertcat(*Bp_temp)
        Ba = ca.vertcat(*Ba_temp)
    
        calB = ca.vertcat(Bp, Ba)
    
        return calB

    def casadi_Psi(self, l, a, omega_xT, sc=1):
        """
        Helper function to calculate Psi matrix for CasADi.
    
        Parameters
        ----------
        l : int
            Index of the term in the series.
        a : casadi.DM or casadi.MX
            Input vector.
        omega_xT : casadi.DM or casadi.MX
            Transpose of the skew-symmetric omega matrix.
    
        Returns
        -------
        Psi_l : casadi.MX or casadi.DM
        """



        Psi_l = ca.DM.zeros(3, 3)
        J_i = ca.inv(self.J)
        for i in range(1, l):
            Psi_l += ca.mpower(omega_xT, i - 1) @ self.utilsf.casadi_skew(ca.mpower(omega_xT, l - 1 - i) @ a) @ J_i
        return Psi_l

    def casadi_G(self, l, omega_xT, sc=1):
        """
        Helper function to calculate G matrix for CasADi.
    
        Parameters
        ----------
        l : int
            Index of the term in the series.
        omega_xT : casadi.DM or casadi.MX
            Transpose of the skew-symmetric omega matrix.
    
        Returns
        -------
        G_l : casadi.MX or casadi.DM
        """

        g1 = ca.DM.zeros(3, 3)
        g2 = ca.DM.zeros(3, 3)
        g3 = ca.DM.zeros(3, 3)
        J_i = ca.inv(self.J)
        for i in range(1, l):
            g1 += ca.mpower(omega_xT, i - 1) @ self.utilsf.casadi_skew(J_i[:, 0]) @ ca.mpower(omega_xT, l - 1 - i)
            g2 += ca.mpower(omega_xT, i - 1) @ self.utilsf.casadi_skew(J_i[:, 1]) @ ca.mpower(omega_xT, l - 1 - i)
            g3 += ca.mpower(omega_xT, i - 1) @ self.utilsf.casadi_skew(J_i[:, 2]) @ ca.mpower(omega_xT, l - 1 - i)
        G_l = ca.horzcat(g1.reshape((9, 1)), g2.reshape((9, 1)), g3.reshape((9, 1)))
        return G_l

    
    
    # function to convert lifted control to actual control
    def fcn_U2u(self, x, U):
        
        """
        x is the 18x1 actual states
        U is the lifted control
        """
        calB = self.fcn_calB(x)
        ubar =  np.linalg.pinv(calB) @ self.fcn_Bbar() @ U
        u = np.vstack([ubar[0].reshape(-1,1), ubar[1:].reshape(-1,1)+ 
                   self.utilsf.skew(x[15:18]) @ self.J @ x[15:18]])
        
        return u
    
    # function to convert actual control to lifted control
    def fcn_u2U(self,x,u, sc=1):


        ubar = np.vstack([u[0].reshape(-1,1), (1/sc)*u[1:].reshape(-1,1) 
                    -sc*self.utilsf.skew(x[15:18]) @ self.J @ x[15:18]])
        U = np.linalg.pinv(self.fcn_Bbar()) @ self.fcn_calB(x) @ ubar
        return U
    
    def fcn_u2u_tilde(self, x, u, sc = 1):

        u_tilde = np.vstack([u[0].reshape(-1,1), (1/sc)*u[1:].reshape(-1,1) \
                    -sc*self.utilsf.skew(x[15:18]) @ self.J @ x[15:18].reshape(-1,1)])
        return u_tilde
    
    def fcn_u_tilde2u(self,x,u_tilde, sc=1):


        u = np.vstack([u_tilde[0].reshape(-1,1), sc*u_tilde[1:].reshape(-1,1) 
                    + (sc**2)*self.utilsf.skew(x[15:18]) @ self.J @ x[15:18].reshape(-1,1)])
        return u

    def casadi_U2u(self, x, U):
        """
        Convert lifted control to actual control using CasADi.

        Parameters
        ----------
        x : casadi.DM or casadi.MX
            18x1 actual state vector.
        U : casadi.DM or casadi.MX
            Lifted control vector.

        Returns
        -------
        u : casadi.MX or casadi.DM
            Actual control vector.
        """
        calB = self.casadi_calB(x)
        ubar = ca.pinv(calB) @ self.fcn_Bbar() @ U
        u = ca.vertcat(
            ubar[0],
            ubar[1:] + self.utilsf.casadi_skew(x[15:18]) @ self.J @ x[15:18]
        )
        return u

    def casadi_u2U(self, x, u):
        """
        Convert actual control to lifted control using CasADi.

        Parameters
        ----------
        x : casadi.DM or casadi.MX
            18x1 actual state vector.
        u : casadi.DM or casadi.MX
            Actual control vector.

        Returns
        -------
        U : casadi.MX or casadi.DM
            Lifted control vector.
        """
        ubar = ca.vertcat(
            u[0],
            u[1:] - self.utilsf.casadi_skew(x[15:18]) @ self.J @ x[15:18]
        )
        U = ca.pinv(self.fcn_Bbar()) @ self.casadi_calB(x) @ ubar
        return U

    def casadi_u2u_tilde(self, x, u, sc=1):
        """
        Convert actual control to u_tilde using CasADi.

        Parameters
        ----------
        x : casadi.DM or casadi.MX
            18x1 actual state vector.
        u : casadi.DM or casadi.MX
            Actual control vector.

        Returns
        -------
        u_tilde : casadi.MX or casadi.DM
            Modified control vector (u_tilde).
        """

        
        u_tilde = ca.vertcat(
            u[0],
            (1/sc)*u[1:] - sc*self.utilsf.casadi_skew(x[15:18]) @ self.J @ x[15:18]
        )
        return u_tilde

    def casadi_u_tilde2u(self, x, u_tilde, sc=1):
        """
        Convert u_tilde to actual control using CasADi.

        Parameters
        ----------
        x : casadi.DM or casadi.MX
            18x1 actual state vector.
        u_tilde : casadi.DM or casadi.MX
            Modified control vector (u_tilde).

        Returns
        -------
        u : casadi.MX or casadi.DM
            Actual control vector.
        """
        
        u = ca.vertcat(
            u_tilde[0],
            sc*u_tilde[1:] + (sc**2)*self.utilsf.casadi_skew(x[15:18]) @ self.J @ x[15:18]
        )
        return u
    
    # def fcn_U2fw(self,xdot,x,u_mpc):
    #     M = u_mpc[1:].reshape(-1,1)
    #     J= self.J
    #     wdot = xdot[15:18].reshape(-1,1)
    #     w = x[15:18].reshape()
        


"""
Begins the Utilities Class
"""

class UtilsFunctions:
    
    def __init__(self, M=None, N=None, params=None):
        
        # Inertial parameters
        Ixx = 0.002354405654  # kg-m^2
        Iyy = 0.002629495468
        Izz = 0.003186782158
        x_rot = 0.175 / 2
        y_rot = 0.131 / 2

        if M is None and N is None and params is None:
            print("No input parameters provided ::: using default values..")
            print("\t M = 3, N = 3")
            self.M = 3
            self.N = 3
            # Define and return the parameters as a dictionary
            self.params = {
                'mass': 0.904,  # kg
                "J": np.diag([Ixx, Iyy, Izz]),
                "d": np.sqrt(x_rot**2 + y_rot**2),
                "k_eta": 0.0000007949116377,
                "c_tau": 1.969658853113417e-8,
                "wr_max": 2500,
                "g": 9.81,  # gravity
                "e3": np.array([0, 0, 1]).reshape(-1,1)
            }
        elif M is not None and N is not None and params is not None:
            self.M = M
            self.N = N
            self.params = params
        elif M is not None and N is not None:
            self.M = M
            self.N = N
            self.params = {
                'mass': 0.904,  # kg
                "J": np.diag([Ixx, Iyy, Izz]),
                "d": np.sqrt(x_rot**2 + y_rot**2),
                "k_eta": 0.0000007949116377,
                "c_tau": 1.969658853113417e-8,
                "wr_max": 3100,
                "g": 9.81,  # gravity
                "e3": np.array([0, 0, 1]).reshape(-1,1)
            }
        else:
            raise ValueError("Invalid input parameters ::: at least provide M,N")
 
    def get_quad_params(self):
        return self.params, self.M, self.N


    @staticmethod
    def skew(v):
        """
        Convert a 3x1 vector to a skew-symmetric matrix.

        Parameters:
        v (numpy array): Input 3x1 vector.

        Returns:
        numpy array: Skew-symmetric matrix.
        """
        v=np.squeeze(v)
        
        return np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
  
            #raise ValueError("Input must be a 3x1 vector.")

    @staticmethod
    def fcn_vee(ss):
        """
        

        Parameters
        ----------
        ss : 3x3 skew symmetric matrix

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        vec : 3x1 vector, corresponding to the input skew-symmetric matrix

        """
        # Check if the matrix is square
        if ss.shape[0] != ss.shape[1]:
            raise ValueError("Input must be a square matrix.")
        
        vec = np.array([-ss[1,2], ss[0,2], -ss[0,1]]).reshape(-1,1)
        
        return vec
    
    @staticmethod
    def casadi_skew(omega):
        """
        Convert a 3x1 vector to a skew-symmetric matrix.
        ** Intended to be used for casadi only (symbolic)

        Parameters:
         omega (numpy array): Input 3x1 vector.

        Returns:
        omega_x: Skew-symmetric matrix.
        """
        omega_x = ca.vertcat(
                ca.horzcat(0, -omega[2], omega[1]),
                ca.horzcat(omega[2], 0, -omega[0]),
                ca.horzcat(-omega[1], omega[0], 0)
                )
        return omega_x
    
    @staticmethod
    def casadi_vee(omega_x):
        omega = ca.vertcat(-omega_x[1,2], omega_x[0,2],-omega_x[0,1])
        return omega
    
    @staticmethod
    def fcn_meas_noise(X, noise_power=0.01):
        std= noise_power
        std_s, std_v, std_rot, std_w = std, std/3, std, std/10
        X[0:3] += np.random.uniform(-std_s,std_s)
        X[3:6] += np.random.uniform(-std_v,std_v)
        R = X[6:15].reshape((3,3), order='F')
        # extract Euler angles
        eul = rot.from_matrix(R).as_euler('xyz', degrees=True)
        # Adding noise to each Euler angle
        eul[0] += np.random.uniform(-std_rot, std_rot)
        eul[1] += np.random.uniform(-std_rot, std_rot)
        eul[2] += np.random.uniform(-std_rot, std_rot)
        # Converting back to a noisy rotation matrix
        R_noisy = rot.from_euler('xyz', eul, degrees=True).as_matrix()
        X[6:15] = R_noisy.flatten(order='F').reshape(-1,1)
        X[15:18] += np.random.uniform(-std_w, std_w)
        return X
    

    @staticmethod
    def fcn_generate_errors(Xd, X, time_sim):
        # Ensure X and Xd are column vectors if they have more than 18 columns
        if X.shape[1] > 18:
            X = X.T
        
        if Xd.shape[1] > 18:
            Xd = Xd.T
    
        # Initialize respective errors
        Psi = np.zeros(len(time_sim))
        ex = np.zeros((len(time_sim), 3))
        ev = np.zeros((len(time_sim), 3))
        ew = np.zeros((len(time_sim), 3))
    
        for jj in range(len(time_sim)):
            # Extract current measurements using column-major order ('F')
            R = X[jj, 6:15].reshape((3, 3), order='F')
            x = X[jj, 0:3]
            v = X[jj, 3:6]
            w = X[jj, 15:18]
    
            # Extract desired/reference values using column-major order ('F')
            Rd = Xd[jj, 6:15].reshape((3, 3), order='F')
            xd = Xd[jj, 0:3]
            vd = Xd[jj, 3:6]
            wd = Xd[jj, 15:18]
    
            # Calculate errors
            Psi[jj] = np.trace(np.eye(3) - Rd.T @ R)/2  # Equivalent to trace(eye(3)-Rd'*R)
            ew[jj, :] = np.abs(wd - w)
            ex[jj, :] = np.abs(xd - x)
            ev[jj, :] = np.abs(vd - v)
    
        # Normalize errors
        ex_norm = ex / np.linalg.norm(ex, axis=0, keepdims=True)
        ev_norm = ev / np.linalg.norm(ev, axis=0, keepdims=True)
        ew_norm = ew / np.linalg.norm(ew, axis=0, keepdims=True)
    
        return ex_norm, ev_norm, Psi, ew_norm, ex, ev, ew
    
    @staticmethod
    def rotation_matrix_to_rpy(rotation_matrix,deg=True):
        """
        Convert a rotation matrix to roll, pitch, and yaw angles using scipy.

        Parameters:
            rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
            deg: boolean, True => degrees true, else radians

        Returns:
            rpy: np array,  roll, pitch, yaw.
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")

        # Create a Rotation object from the matrix
        rotation = rot.from_matrix(rotation_matrix)

        # Extract roll, pitch, and yaw in radians (ZYX order)
        if deg:
            roll, pitch, yaw = rotation.as_euler('ZYX', degrees=True)
        else:
            roll, pitch, yaw = rotation.as_euler('ZYX', degrees=False)
        rpy = np.array([roll, pitch, yaw])
        
        return rpy


    def get_max_tilt_angles_degrees(self, X):
        """
        Calculates the maximum absolute roll and pitch angles (in degrees) from a quadrotor trajectory.
        Assumes the rotation matrix vec(R) is stored in COLUMN-MAJOR order.
        Uses the existing rotation_matrix_to_rpy function.

        Args:
            X (np.ndarray): 18xN state matrix. States are [x, y, z, vx, vy, vz, R11, R21, R31, R12, R22, R32, R13, R23, R33, wx, wy, wz]^T.

        Returns:
            tuple: (max_roll_deg, max_pitch_deg) The maximum absolute roll and pitch angles in degrees.
        """
        
        num_states, N = X.shape
        # Extract the rotation matrix part (elements 6 to 14 inclusive)
        R_vecs = X[6:15, :].T  # Shape is now (N, 9)

        # Initialize arrays to store the ABSOLUTE roll and pitch angles for each timestep
        all_roll_deg = np.zeros(N)
        all_pitch_deg = np.zeros(N)

        for i in range(N):
            # Get the 9-element column-major vector for this timestep
            r_vec = R_vecs[i, :]
            # Reshape it into a 3x3 matrix using COLUMN-MAJOR ('F'ortran) order
            R_matrix = r_vec.reshape(3, 3, order='F')
            
            # Use your existing function to get RPY angles in degrees
            roll, pitch, yaw = self.rotation_matrix_to_rpy(R_matrix, deg=True)
            
            # Store the absolute value of the angles (magnitude of tilt)
            all_roll_deg[i] = np.abs(roll)
            all_pitch_deg[i] = np.abs(pitch)

        # Find the maximum absolute angles across the entire trajectory
        max_roll_deg = np.max(all_roll_deg)
        max_pitch_deg = np.max(all_pitch_deg)

        return max_roll_deg, max_pitch_deg
    
    def fcn_x2x_scl(self, x, frak_w):
        
        x[15:18] = (1/frak_w)*x[15:18]
        return x.reshape(-1,1)
    
    def fcn_x_scl2x(self, x_scl, frak_w):
        x_scl[15:18] = frak_w*x_scl[15:18]
        return x_scl.reshape(-1,1)
    
    def ca_x2x_scl(self, x, frak_w):
        x[15:18] = (1/frak_w)*x[15:18]
        return x
    
    def ca_x_scl2x(self, x_scl, frak_w):
        x_scl[15:18] = frak_w*x_scl[15:18]
        return x_scl
