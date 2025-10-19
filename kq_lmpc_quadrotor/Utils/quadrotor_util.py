"""
This module is from ACRL, UIUC

"""

import os
import errno
import shutil
import numpy as np
import casadi as cs
import pyquaternion     
import matplotlib.pyplot as plt
# from utils.george_params import quad_params
import copy

"""
Some useful util functions used for MPC and online learning

"""

# f_to_TM: convert individual thrusts to thrust-moment outputs, not changed by the number of rotors
def f_to_TM(f, quad_params, ifcs=False):
    # load the quadrotor parameters
    rotor_dir       = quad_params['rotor_directions']
    num_rotors      = quad_params['num_rotors']
    rotor_pos       = quad_params['rotor_pos']
    k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
    k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
    k = k_m/k_eta

    f_to_TM = np.vstack((np.ones((1,num_rotors)),
                         np.hstack([np.cross(rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in rotor_pos]), 
                         k*rotor_dir))
    if ifcs:
        TM = cs.mtimes(f_to_TM, f)
    else:
        TM = f_to_TM @ f
    return TM 

# TM_to_f4: convert thrust-moment outputs to individual thrusts, 4 rotors functioning 
def TM_to_f4(TM, quad_params):
    # load the quadrotor parameters
    rotor_dir       = quad_params['rotor_directions']
    num_rotors      = quad_params['num_rotors']
    rotor_pos       = quad_params['rotor_pos']
    k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
    k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
    k = k_m/k_eta

    f_to_TM = np.vstack((np.ones((1,num_rotors)),
                         np.hstack([np.cross(rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in rotor_pos]), 
                         k*rotor_dir))
    TM_to_f4 = np.linalg.inv(f_to_TM)
    f4 = TM_to_f4 @ TM
    return f4 

def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory: {0}'.format(directory))

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)

def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.

    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return cs.vertcat(t0, t1, t2, t3)
    

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q


def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f, ind):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), velocity (3D), angle (4D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :param ind: Only used for trajectory tracking. Index of cost function to use.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and
    """

    if isinstance(cost_f, list):
        # Select the list of cost functions
        cost_f = cost_f[ind * m_steps_per_point:(ind + 1) * m_steps_per_point]
    else:
        cost_f = [cost_f]

    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    x0 = x
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(x=x, u=u)['x_dot']
        k2 = dynamics_f(x=x + dt / 2 * k1, u=u)['x_dot']
        k3 = dynamics_f(x=x + dt / 2 * k2, u=u)['x_dot']
        k4 = dynamics_f(x=x + dt * k3, u=u)['x_dot']
        x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x_out

        if cost_f and cost_f[j] is not None:
            q = q + cost_f[j](x=x, u=u)['q']

    return cs.Function('F', [x0, u], [x, q], ['x0', 'p'], ['xf', 'qf'])

def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires knowledge of the previous quaternion
    in the series, q_past
    :param q_past: 4-dimensional vector representing a quaternion in wxyz form.
    :param q_current: 4-dimensional vector representing a quaternion in wxyz form. Will be corrected if it generates
    a flip wrt q_past.
    :return: q_current with the flip removed if necessary
    """

    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(np.sum((q_past + q_current) ** 2)):
        return -q_current
    return q_current

def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.

    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])

def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref, quad_params, 
                                        map_limits=None, plot=False, to_list=False, return_TM=True):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    quad_mass = quad_params['mass']
    # Ixx = quad_params['Ixx']
    # Iyy = quad_params['Iyy']
    # Izz = quad_params['Izz']
    # Ixy = quad_params['Ixy']
    # Ixz = quad_params['Ixz']
    # Iyz = quad_params['Iyz']
    # quad_J = np.array([[Ixx, Ixy, Ixz],
    #                        [Ixy, Iyy, Iyz],
    #                        [Ixz, Iyz, Izz]])
    # Note: only use diagonal entries 
    J=quad_params['J']

    quad_J = np.array([J[0,0],J[1,1],J[2,2]])
    # quad_inv_J = np.linalg.inv(quad_J)

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = traj_derivatives[2, :, :].T + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust ** 2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))
    for i in range(len_traj):
        f_t[i, 0] = quad_mass * z_b[i].dot(thrust[i, :].T)

    if yawing:
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.concatenate((np.cos(yaw_derivatives[0, :])[:, np.newaxis],
                              np.sin(yaw_derivatives[0, :])[:, np.newaxis],
                              np.zeros(len_traj)[:, np.newaxis]), 1)
        y_b = np.cross(z_b, x_c)
        y_b = y_b / np.sqrt(np.sum(y_b ** 2, axis=1))[:, np.newaxis]
        x_b = np.cross(y_b, z_b)

        # Rotation matrix (from body to world)
        b_r_w = np.concatenate((x_b[:, :, np.newaxis], y_b[:, :, np.newaxis], z_b[:, :, np.newaxis]), -1)
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(rotation_matrix_to_quat(b_r_w[i]))
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the quadrotor acceleration into the Z body axis
        a_proj = np.zeros((len_traj, 1))

        for i in range(len_traj):
            a_proj[i, 0] = z_b[i].dot(traj_derivatives[3, :, i])

        h_omega = quad_mass / f_t * (traj_derivatives[3, :, :].T - a_proj * z_b)
        for i in range(len_traj):
            rate[i, 0] = -h_omega[i].dot(y_b[i])
            rate[i, 1] = h_omega[i].dot(x_b[i])
            rate[i, 2] = -yaw_derivatives[1, i] * np.array([0, 0, 1]).dot(z_b[i])

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
        q = q / np.sqrt(np.sum(q ** 2, 1))[:, np.newaxis]

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=0) / discretization_dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]

        go_crazy_about_yaw = True
        if go_crazy_about_yaw:
            #print("Maximum yawrate before adaption: %.3f" % np.max(np.abs(rate[:, 2])))
            q_new = q
            yaw_corr_acc = 0.0
            for i in range(1, len_traj):
                yaw_corr = -rate[i, 2] * discretization_dt
                yaw_corr_acc += yaw_corr
                q_corr = np.array([np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)])
                q_new[i, :] = q_dot_q(q[i, :], q_corr)
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]

            q_new_dot = np.gradient(q_new, axis=0) / discretization_dt
            for i in range(1, len_traj):
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]

            q = q_new
            rate[:, 0] = w_int[:, 0]
            rate[:, 1] = w_int[:, 1]
            rate[:, 2] = w_int[:, 2]
            #print("Maximum yawrate after adaption: %.3f" % np.max(np.abs(rate[:, 2])))

    # Compute inputs
    rate_dot = np.gradient(rate, axis=0) / discretization_dt
   
    rate_x_Jrate = np.array([(quad_J[2] - quad_J[1]) * rate[:, 2] * rate[:, 1],
                             (quad_J[0] - quad_J[2]) * rate[:, 0] * rate[:, 2],
                             (quad_J[1] - quad_J[0]) * rate[:, 1] * rate[:, 0]]).T
    


    tau = rate_dot * quad_J[np.newaxis, :] + rate_x_Jrate
    b = np.concatenate((f_t, tau), axis=-1)
    # a_mat = np.concatenate((quad.y_f[np.newaxis, :], -quad.x_f[np.newaxis, :],
    #                         quad.z_l_tau[np.newaxis, :], np.ones_like(quad.z_l_tau)[np.newaxis, :]), 0)

    if return_TM:
        reference_u = b
    else:
        reference_u = np.zeros((len_traj, 4))
        for i in range(len_traj):
            reference_u[i, :] = TM_to_f4(b[i, :], quad_params=quad_params)

        # reference_u[i, :] = np.linalg.solve(a_mat, b[i, :])
    # reference_u = b

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    reference_traj = np.concatenate((full_pos, full_vel, q, rate), 1)

    if map_limits is None:
        # Locate starting point right at x=0 and y=0.
        reference_traj[:, 0] -= reference_traj[0, 0]
        reference_traj[:, 1] -= reference_traj[0, 1]

    else:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]
        z_max_range = map_limits["z"][1] - map_limits["z"][0]

        x_center = x_max_range / 2 + map_limits["x"][0]
        y_center = y_max_range / 2 + map_limits["y"][0]
        z_center = z_max_range / 2 + map_limits["z"][0]

        # Center circle to center of map XY plane
        reference_traj[:, :3] += np.array([x_center, y_center, 0])
        reference_traj[:, 2] = z_center
   
    if plot:
        draw_poly(reference_traj, reference_u, t_ref)


    if to_list:
        reference_traj = [full_pos, full_vel, q, rate]

    return reference_traj, t_ref, reference_u

def draw_poly(traj, u_traj, t, target_points=None, target_t=None):
    """
    Plots the generated trajectory of length n with the used keypoints.
    :param traj: Full generated reference trajectory. Numpy array of shape nx13
    :param u_traj: Generated reference inputs. Numpy array of shape nx4
    :param t: Timestamps of the references. Numpy array of length n
    :param target_points: m position keypoints used for trajectory generation. Numpy array of shape 3 x m.
    :param target_t: Timestamps of the reference position keypoints. If not passed, then they are extracted from the
    t vector, assuming constant time separation.
    """

    ders = 2
    dims = 3

    y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$']
    dim_legends = ['x', 'y', 'z']

    if target_t is None and target_points is not None:
        target_t = np.linspace(0, t[-1], target_points.shape[1])

    p_traj = traj[:, :3] # pos
    a_traj = traj[:, 6:10]
    v_traj = traj[:, 3:6]
    r_traj = traj[:, 10:] # rate

    plt_traj = [p_traj, v_traj]

    fig = plt.figure()
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1)

        for dim in range(dims):

            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[dim])

            if d_ord == 0 and target_points is not None:
                plt.plot(target_t, target_points[dim, :], 'bo')

        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])

    dim_legends = [['w', 'x', 'y', 'z'], ['x', 'y', 'z']]
    y_labels = [r'att $[quat]$', r'rate $[rad/s]$']
    plt_traj = [a_traj, r_traj]
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1 + ders * 2)
        for dim in range(plt_traj[d_ord].shape[1]):
            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[d_ord][dim])

        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])
        if d_ord == ders - 1:
            plt.xlabel(r'time $[s]$')
        else:
            plt.gca().set_xticklabels([])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plt.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2])
    if target_points is not None:
        plt.plot(target_points[0, :], target_points[1, :], target_points[2, :], 'bo')
    plt.title('Target position trajectory')
    ax.set_xlabel(r'$p_x [m]$')
    ax.set_ylabel(r'$p_y [m]$')
    ax.set_zlabel(r'$p_z [m]$')

    plt.subplot(ders + 1, 2, (ders + 1) * 2)
    for i in range(u_traj.shape[1]):
        plt.plot(t, u_traj[:, i], label=r'$u_{}$'.format(i))
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel(r'time $[s]$')
    plt.ylabel(r'single thrusts $[N]$')
    plt.title('Control inputs')

    plt.suptitle('Generated polynomial trajectory')

    plt.show()

def activation(x, act):
    if act == 'relu':
        if isinstance(x, np.ndarray):
            return np.maximum(0,x)
        return cs.if_else(x < 0, 0.* x, x)
    if act == 'linear':
        return x
    if act =='elu':
        if isinstance(x, np.ndarray):
            return np.maximum(0,x)
        return cs.if_else(x < 0, -0.2 * x, x)