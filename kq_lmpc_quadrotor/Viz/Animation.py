import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R  # Using SciPy for Euler conversion
import matplotlib
#%matplotlib qt
# Set the ffmpeg path if needed (adjust for your system)
#matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' # for mac

matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # for ubuntu
# ---------------- Dummy Data Generation ----------------
dt = 0.01
num_frames = 200
t = np.linspace(0, 2*np.pi, num_frames)

# Generate dummy rotation matrices (for the quadrotor) and angular rates
rotation_matrices = np.array([
    np.linalg.qr(np.eye(3) + 0.1 * np.random.randn(3, 3))[0]
    for _ in range(num_frames)
])
angular_rates = 0.1 * np.random.randn(num_frames, 3)

# Generate position (sinusoidal with noise) and its derivative (velocity)
position = np.vstack([
    3*np.sin(t) + 0.1 * np.random.randn(num_frames),
    3*np.cos(t) + 0.1 * np.random.randn(num_frames),
    1.5*np.sin(2*t) + 0.05 * np.random.randn(num_frames)
]).T
velocity = np.gradient(position, axis=0)

# Generate a reference position (for position tracking)
ref_position = np.vstack([
    3*np.sin(t) + 0.1 * np.random.randn(num_frames),
    3*np.cos(t) + 0.1 * np.random.randn(num_frames),
    1.5*np.sin(2*t) + 0.05 * np.random.randn(num_frames)
]).T

# Generate dummy reference rotation matrices and flatten them.
rotation_matrices_ref = np.array([
    np.linalg.qr(np.eye(3) + 0.05 * np.random.randn(3, 3))[0]
    for _ in range(num_frames)
])
rotation_ref_flat = rotation_matrices_ref.reshape(num_frames, 9)

# Build reference_trajectory: concatenate reference position (columns 0–2)
# with the flattened reference rotation matrix (columns 3–11)
reference_trajectory = np.hstack([ref_position, rotation_ref_flat])

# Build trajectory_data: [position | velocity | flattened rotation | angular rates]
trajectory_data = np.hstack([
    position,
    velocity,
    rotation_matrices.reshape(num_frames, 9),
    angular_rates
])

# Dummy control inputs (for control mode)
control_inputs = np.vstack([
    10 + 2*np.sin(t) + 0.1 * np.random.randn(num_frames),
    0.5*np.sin(t) + 0.1 * np.random.randn(num_frames),
    0.5*np.cos(t) + 0.1 * np.random.randn(num_frames),
    0.3*np.sin(2*t) + 0.1 * np.random.randn(num_frames)
]).T
control_inputs_ref = np.vstack([
    10 + 2*np.sin(t),
    0.5*np.sin(t),
    0.5*np.cos(t),
    0.3*np.sin(2*t)
]).T

# ---------------- Animation Function Definition ----------------
def animate_quadrotor(trajectory_data, reference_trajectory, control_inputs, control_inputs_ref,
                      dt=0.01, save_video=False, save_gif=False, show_tracking=True,
                      output_filename="quadrotor_animation.mp4",
                      gif_filename=None, control_umin=None, control_umax=None,
                      task_id=0, three_d_only=False, display_plot=True):
    """
    Animate the quadrotor with various layouts.
    
    Layouts:
      - If three_d_only is True: Only the 3D quadrotor animation is shown.
      - Otherwise:
          * Tracking Mode (show_tracking=True):  
              Left: 3D quadrotor animation.
              Middle: Position tracking (s_x, s_y, s_z; actual vs. reference).
              Right: Euler angle tracking (roll, pitch, yaw in deg; actual vs. reference),
                computed from the flattened rotation data in reference_trajectory (columns 3–11).
          * Control Mode (show_tracking=False):  
              Left: 3D quadrotor animation.
              Right: 2×2 grid for control signals with dashed gray lines showing input bounds.
    
    Additional Parameters:
      control_umin : array-like of shape (4,), lower bounds for control inputs.
      control_umax : array-like of shape (4,), upper bounds for control inputs.
                     If not provided, dummy bounds are used.
      save_gif    : Boolean flag. If True, export the animation as a GIF (using the same fps as video).
      gif_filename: Filename for saving the GIF. If None, it is derived from output_filename.
      task_id     : If set to 1 or 2, a red cross marker (target) is added to the 3D plot.
                    (The target is the last point of the reference position, i.e. reference_trajectory[:,:3].)
      three_d_only: If True, only the 3D plot is displayed.
    """
    num_frames = trajectory_data.shape[0]
    
    # Use dummy control bounds if not provided.
    if control_umin is None or control_umax is None:
        control_umin = np.array([8, -0.6, -0.6, -0.3])
        control_umax = np.array([12,  0.6,  0.6,  0.3])
    control_bounds = np.column_stack((control_umin, control_umax))
    
    # Helper function to draw a rotor circle.
    def draw_rotor_circle(center, R_mat, radius=0.1, num_points=20):
        theta = np.linspace(0, 2*np.pi, num_points)
        circle = np.array([radius*np.cos(theta),
                           radius*np.sin(theta),
                           np.zeros_like(theta)])
        return R_mat @ circle + center.reshape(3,1)
    
    # Compute overall position limits from both actual and reference data.
    # For the 3D plot and for position tracking, we use the first 3 columns of reference_trajectory.
    pos_all = np.concatenate([trajectory_data[:, :3], reference_trajectory[:, :3]], axis=0)
    x_max = np.max(np.abs(pos_all[:, 0]))
    y_max = np.max(np.abs(pos_all[:, 1]))
    z_max = np.max(pos_all[:, 2])
    
    # --------------------- 3D-Only Layout ---------------------
    if three_d_only:
        fig = plt.figure(figsize=(12, 9))
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.set_xlabel("X-axis", fontsize=16)
        ax3d.set_ylabel("Y-axis", fontsize=16)
        ax3d.set_zlabel("Z-axis", fontsize=16)
        ax3d.set_title("Quadrotor 3D Animation")
        ax3d.tick_params(axis='both', labelsize=14)
        # Set axis limits: x and y symmetric, z from 0 to max+1.
        ax3d.set_xlim([-x_max - 1, x_max + 1])
        ax3d.set_ylim([-y_max - 1, y_max + 1])
        if task_id > 2:
            ax3d.set_zlim([-z_max - 1, z_max + 1])
        else:
            ax3d.set_zlim([0, z_max + 1])
        
        # Add target marker if task_id is 1 or 2.
        if task_id in [1, 2]:
            target = reference_trajectory[-1, :3]
            ax3d.plot([target[0]], [target[1]], [target[2]],
                     'rx', markersize=15, markeredgewidth=3, label="Target")
        
        # Create 3D quadrotor graphics objects.
        start_marker, = ax3d.plot([trajectory_data[0,0]], [trajectory_data[0,1]], [trajectory_data[0,2]],
                                   'go', markersize=10, label="Start")
        arm_len = 0.3
        rotor_rad = 0.1
        quad_arms, = ax3d.plot([], [], [], 'k-', lw=3)
        quad_arms2, = ax3d.plot([], [], [], 'k-', lw=3)
        rotor_patches = [ax3d.plot([], [], [], 'r-', lw=2)[0] for _ in range(4)]
        actual_traj, = ax3d.plot([], [], [], 'b-', alpha=0.8, lw=2, label="Actual Trajectory")
        reference_traj, = ax3d.plot([], [], [], 'r--', alpha=0.8, lw=2, label="Reference Trajectory")
        ax3d.legend(loc="upper left")
        time_text = fig.text(0.5, 0.05, "", ha="center", fontsize=16)
        
        def init_3d():
            quad_arms.set_data([], []);       quad_arms.set_3d_properties([])
            quad_arms2.set_data([], []);      quad_arms2.set_3d_properties([])
            actual_traj.set_data([], []);     actual_traj.set_3d_properties([])
            reference_traj.set_data([], []);  reference_traj.set_3d_properties([])
            for rotor in rotor_patches:
                rotor.set_data([], []); rotor.set_3d_properties([])
            time_text.set_text("")
            return (quad_arms, quad_arms2, actual_traj, reference_traj,
                    *rotor_patches, time_text)
        
        def update_3d(frame):
            pos = trajectory_data[frame, :3]
            R_mat = trajectory_data[frame, 6:15].reshape(3, 3)
            arm1 = np.array([[arm_len, 0, 0], [-arm_len, 0, 0]]).T
            arm2 = np.array([[0, arm_len, 0], [0, -arm_len, 0]]).T
            arm1_rot = R_mat @ arm1 + pos.reshape(3,1)
            arm2_rot = R_mat @ arm2 + pos.reshape(3,1)
            quad_arms.set_data(arm1_rot[0,:], arm1_rot[1,:])
            quad_arms.set_3d_properties(arm1_rot[2,:])
            quad_arms2.set_data(arm2_rot[0,:], arm2_rot[1,:])
            quad_arms2.set_3d_properties(arm2_rot[2,:])
            rotor_positions = [arm1_rot[:,0], arm1_rot[:,1], arm2_rot[:,0], arm2_rot[:,1]]
            for i, rotor in enumerate(rotor_patches):
                circle = draw_rotor_circle(rotor_positions[i], R_mat, radius=rotor_rad)
                rotor.set_data(circle[0,:], circle[1,:])
                rotor.set_3d_properties(circle[2,:])
            actual_traj.set_data(trajectory_data[:frame, 0], trajectory_data[:frame, 1])
            actual_traj.set_3d_properties(trajectory_data[:frame, 2])
            reference_traj.set_data(reference_trajectory[:frame, 0],
                                     reference_trajectory[:frame, 1])
            reference_traj.set_3d_properties(reference_trajectory[:frame, 2])
            time_text.set_text(f"Time: {frame*dt:.2f} s")
            return (quad_arms, quad_arms2, actual_traj, reference_traj,
                    *rotor_patches, time_text)
        
        fps = int(round(1/dt))
        ani = animation.FuncAnimation(fig, update_3d, frames=num_frames,
                                      init_func=init_3d, interval=dt*1000, blit=False)
    
    # --------------------- Multi-Panel Layout ---------------------
    else:
        if show_tracking:
            # Precompute Euler angles for tracking mode.
            euler_actual_all = np.zeros((num_frames, 3))
            euler_ref_all = np.zeros((num_frames, 3))
            for f in range(num_frames):
                R_actual = trajectory_data[f, 6:15].reshape(3,3)
                angles = R.from_matrix(R_actual).as_euler('zyx', degrees=True)
                euler_actual_all[f, :] = [angles[2], angles[1], angles[0]]
                # For the reference, extract columns 3 to 12 and reshape.
                R_ref_mat = reference_trajectory[f, 3:12].reshape(3,3)
                angles_ref = R.from_matrix(R_ref_mat).as_euler('zyx', degrees=True)
                euler_ref_all[f, :] = [angles_ref[2], angles_ref[1], angles_ref[0]]
            
            fig = plt.figure(figsize=(20, 12))
            gs_main = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
            gs_main.update(wspace=0.3)
            
            # Left: 3D Quadrotor Plot
            ax3d = fig.add_subplot(gs_main[0,0], projection='3d')
            ax3d.set_xlabel("X-axis", fontsize=16)
            ax3d.set_ylabel("Y-axis", fontsize=16)
            ax3d.set_zlabel("Z-axis", fontsize=16)
            ax3d.set_title("Quadrotor 3D Animation")
            ax3d.tick_params(axis='both', labelsize=14)
            # Set 3D axis limits based on computed data.
            ax3d.set_xlim([-x_max - 1, x_max + 1])
            ax3d.set_ylim([-y_max - 1, y_max + 1])
            ax3d.set_zlim([0, z_max + 1])
            ax3d.view_init(elev=30)
            if task_id in [1, 2]:
                target = reference_trajectory[-1, :3]
                ax3d.plot([target[0]], [target[1]], [target[2]],
                         'rx', markersize=15, markeredgewidth=3, label="Target")
            
            # Middle: 3x1 Grid for Position Tracking
            gs_xyz = gs_main[0,1].subgridspec(3, 1)
            ax_x = fig.add_subplot(gs_xyz[0,0])
            ax_y = fig.add_subplot(gs_xyz[1,0])
            ax_z = fig.add_subplot(gs_xyz[2,0])
            ax_x.set_title("X Tracking")
            ax_y.set_title("Y Tracking")
            ax_z.set_title("Z Tracking")
            ax_x.set_ylabel(r"$s_x$ (m)")
            ax_y.set_ylabel(r"$s_y$ (m)")
            ax_z.set_ylabel(r"$s_z$ (m)")
            ax_x.set_xlabel("")
            ax_x.set_xticklabels([])
            ax_y.set_xlabel("")
            ax_y.set_xticklabels([])
            ax_z.set_xlabel("Time (s)")
            # For x and y, use symmetric limits.
            x_all = np.concatenate([trajectory_data[:, 0], reference_trajectory[:, 0]])
            y_all = np.concatenate([trajectory_data[:, 1], reference_trajectory[:, 1]])
            x_lim = np.max(np.abs(x_all)) + 1
            y_lim = np.max(np.abs(y_all)) + 1
            ax_x.set_ylim([-x_lim, x_lim])
            ax_y.set_ylim([-y_lim, y_lim])
            # For s_z, lower limit is 0.
            z_all = np.concatenate([trajectory_data[:, 2], reference_trajectory[:, 2]])
            z_lim = np.max(z_all) + 1
            ax_z.set_ylim([0, z_lim])
            
            # Right: 3x1 Grid for Euler Angle Tracking
            gs_euler = gs_main[0,2].subgridspec(3, 1)
            ax_roll = fig.add_subplot(gs_euler[0,0])
            ax_pitch = fig.add_subplot(gs_euler[1,0])
            ax_yaw = fig.add_subplot(gs_euler[2,0])
            ax_roll.set_title("Roll Tracking (deg)")
            ax_pitch.set_title("Pitch Tracking (deg)")
            ax_yaw.set_title("Yaw Tracking (deg)")
            ax_roll.set_ylabel(r"$\phi$ (deg)")
            ax_pitch.set_ylabel(r"$\theta$ (deg)")
            ax_yaw.set_ylabel(r"$\psi$ (deg)")
            ax_roll.set_xlabel("")
            ax_roll.set_xticklabels([])
            ax_pitch.set_xlabel("")
            ax_pitch.set_xticklabels([])
            ax_yaw.set_xlabel("Time (s)")
            for ax in [ax_roll, ax_pitch, ax_yaw]:
                ax.set_xlim(0, num_frames*dt)
                ax.set_ylim(-180, 180)
            
            line_x_ref, = ax_x.plot([], [], '--', label=r"$s_x^{ref}$")
            line_x_act, = ax_x.plot([], [], '-', label=r"$s_x$")
            line_y_ref, = ax_y.plot([], [], '--', label=r"$s_y^{ref}$")
            line_y_act, = ax_y.plot([], [], '-', label=r"$s_y$")
            line_z_ref, = ax_z.plot([], [], '--', label=r"$s_z^{ref}$")
            line_z_act, = ax_z.plot([], [], '-', label=r"$s_z$")
            ax_x.legend(loc="upper right")
            ax_y.legend(loc="upper right")
            ax_z.legend(loc="upper right")
            
            line_roll_ref, = ax_roll.plot([], [], '--', label=r"$\phi^{ref}$")
            line_roll_act, = ax_roll.plot([], [], '-', label=r"$\phi$")
            line_pitch_ref, = ax_pitch.plot([], [], '--', label=r"$\theta^{ref}$")
            line_pitch_act, = ax_pitch.plot([], [], '-', label=r"$\theta$")
            line_yaw_ref, = ax_yaw.plot([], [], '--', label=r"$\psi^{ref}$")
            line_yaw_act, = ax_yaw.plot([], [], '-', label=r"$\psi$")
            ax_roll.legend(loc="upper right")
            ax_pitch.legend(loc="upper right")
            ax_yaw.legend(loc="upper right")
            
            time_text = fig.text(0.1, 0.03, "", ha="center", fontsize=16)
            
            start_marker, = ax3d.plot([trajectory_data[0,0]], [trajectory_data[0,1]], [trajectory_data[0,2]],
                                       'go', markersize=10, label="Start")
            arm_len = 0.3
            rotor_rad = 0.1
            quad_arms, = ax3d.plot([], [], [], 'k-', lw=3)
            quad_arms2, = ax3d.plot([], [], [], 'k-', lw=3)
            rotor_patches = [ax3d.plot([], [], [], 'r-', lw=2)[0] for _ in range(4)]
            actual_traj, = ax3d.plot([], [], [], 'b-', alpha=0.8, lw=2, label="Actual Trajectory")
            reference_traj, = ax3d.plot([], [], [], 'r--', alpha=0.8, lw=2, label="Reference Trajectory")
            ax3d.legend(loc="upper left")
            
            def init_tracking():
                quad_arms.set_data([], []);       quad_arms.set_3d_properties([])
                quad_arms2.set_data([], []);      quad_arms2.set_3d_properties([])
                actual_traj.set_data([], []);     actual_traj.set_3d_properties([])
                reference_traj.set_data([], []);  reference_traj.set_3d_properties([])
                for rotor in rotor_patches:
                    rotor.set_data([], []); rotor.set_3d_properties([])
                for line in [line_x_ref, line_x_act, line_y_ref, line_y_act, line_z_ref, line_z_act,
                             line_roll_ref, line_roll_act, line_pitch_ref, line_pitch_act, line_yaw_ref, line_yaw_act]:
                    line.set_data([], [])
                time_text.set_text("")
                return (quad_arms, quad_arms2, actual_traj, reference_traj,
                        *rotor_patches,
                        line_x_ref, line_x_act, line_y_ref, line_y_act, line_z_ref, line_z_act,
                        line_roll_ref, line_roll_act, line_pitch_ref, line_pitch_act, line_yaw_ref, line_yaw_act,
                        time_text)
            
            def update_tracking(frame):
                pos = trajectory_data[frame, :3]
                R_mat = trajectory_data[frame, 6:15].reshape(3, 3)
                arm1 = np.array([[arm_len, 0, 0], [-arm_len, 0, 0]]).T
                arm2 = np.array([[0, arm_len, 0], [0, -arm_len, 0]]).T
                arm1_rot = R_mat @ arm1 + pos.reshape(3,1)
                arm2_rot = R_mat @ arm2 + pos.reshape(3,1)
                quad_arms.set_data(arm1_rot[0, :], arm1_rot[1, :])
                quad_arms.set_3d_properties(arm1_rot[2, :])
                quad_arms2.set_data(arm2_rot[0, :], arm2_rot[1, :])
                quad_arms2.set_3d_properties(arm2_rot[2, :])
                rotor_positions = [arm1_rot[:,0], arm1_rot[:,1], arm2_rot[:,0], arm2_rot[:,1]]
                for i, rotor in enumerate(rotor_patches):
                    circle = draw_rotor_circle(rotor_positions[i], R_mat, radius=rotor_rad)
                    rotor.set_data(circle[0, :], circle[1, :])
                    rotor.set_3d_properties(circle[2, :])
                actual_traj.set_data(trajectory_data[:frame, 0], trajectory_data[:frame, 1])
                actual_traj.set_3d_properties(trajectory_data[:frame, 2])
                reference_traj.set_data(reference_trajectory[:frame, 0], reference_trajectory[:frame, 1])
                reference_traj.set_3d_properties(reference_trajectory[:frame, 2])
                
                time_arr = np.arange(frame) * dt
                line_x_ref.set_data(time_arr, reference_trajectory[:frame, 0])
                line_x_act.set_data(time_arr, trajectory_data[:frame, 0])
                line_y_ref.set_data(time_arr, reference_trajectory[:frame, 1])
                line_y_act.set_data(time_arr, trajectory_data[:frame, 1])
                line_z_ref.set_data(time_arr, reference_trajectory[:frame, 2])
                line_z_act.set_data(time_arr, trajectory_data[:frame, 2])
                
                line_roll_ref.set_data(time_arr, euler_ref_all[:frame, 0])
                line_roll_act.set_data(time_arr, euler_actual_all[:frame, 0])
                line_pitch_ref.set_data(time_arr, euler_ref_all[:frame, 1])
                line_pitch_act.set_data(time_arr, euler_actual_all[:frame, 1])
                line_yaw_ref.set_data(time_arr, euler_ref_all[:frame, 2])
                line_yaw_act.set_data(time_arr, euler_actual_all[:frame, 2])
                
                time_text.set_text(f"Time: {frame*dt:.2f} s")
                return (quad_arms, quad_arms2, actual_traj, reference_traj,
                        *rotor_patches,
                        line_x_ref, line_x_act, line_y_ref, line_y_act, line_z_ref, line_z_act,
                        line_roll_ref, line_roll_act, line_pitch_ref, line_pitch_act, line_yaw_ref, line_yaw_act,
                        time_text)
            
            ani = animation.FuncAnimation(fig, update_tracking, frames=num_frames,
                                          init_func=init_tracking, interval=dt*1000, blit=False)
        
        else:
            # ----------------- Control Mode Layout -----------------
            fig = plt.figure(figsize=(20, 12))
            gs_main = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
            gs_main.update(wspace=0.3)
            
            # Left: 3D Quadrotor Plot
            ax3d = fig.add_subplot(gs_main[0,0], projection='3d')
            ax3d.set_xlabel("X-axis", fontsize=16)
            ax3d.set_ylabel("Y-axis", fontsize=16)
            ax3d.set_zlabel("Z-axis", fontsize=16)
            ax3d.set_title("Quadrotor 3D Animation")
            ax3d.tick_params(axis='both', labelsize=14)
            ax3d.set_xlim([-x_max - 1, x_max + 1])
            ax3d.set_ylim([-y_max - 1, y_max + 1])
            ax3d.set_zlim([0, z_max + 1])
            if task_id in [1, 2]:
                target = reference_trajectory[-1, :3]
                ax3d.plot([target[0]], [target[1]], [target[2]],
                         'rx', markersize=15, markeredgewidth=3, label="Target")
            
            # Right: 2x2 Grid for Control Signals
            gs_ctrl = gs_main[0,1].subgridspec(2, 2)
            ax_ctrl = [fig.add_subplot(gs_ctrl[i, j]) for i in range(2) for j in range(2)]
            ctrl_titles = [r"$f$", r"$M_1$", r"$M_2$", r"$M_3$"]
            for i, ax in enumerate(ax_ctrl):
                ax.set_title(ctrl_titles[i])
                ax.set_xlabel("Time (s)")
                # For control signals, use their own min/max values.
                dmin = min(control_inputs[:, i].min(), control_inputs_ref[:, i].min())
                dmax = max(control_inputs[:, i].max(), control_inputs_ref[:, i].max())
                ax.set_xlim(0, num_frames * dt)
                ax.set_ylim(dmin - 1, dmax + 1)
                ax.axhline(dmin, color='gray', linestyle='--', linewidth=1)
                ax.axhline(dmax, color='gray', linestyle='--', linewidth=1)
                ax.legend(loc="upper right")
            
            ctrl_lines = []
            for i, ax in enumerate(ax_ctrl):
                ref_line, = ax.plot([], [], '--', label=f"${ctrl_titles[i]}^{{ref}}$")
                act_line, = ax.plot([], [], '-', label=f"${ctrl_titles[i]}$")
                ctrl_lines.append((ref_line, act_line))
            
            time_text = fig.text(0.1, 0.03, "", ha="center", fontsize=16)
            
            start_marker, = ax3d.plot([trajectory_data[0, 0]], [trajectory_data[0, 1]], [trajectory_data[0, 2]],
                                       'go', markersize=10, label="Start")
            arm_len = 0.3
            rotor_rad = 0.1
            quad_arms, = ax3d.plot([], [], [], 'k-', lw=3)
            quad_arms2, = ax3d.plot([], [], [], 'k-', lw=3)
            rotor_patches = [ax3d.plot([], [], [], 'r-', lw=2)[0] for _ in range(4)]
            actual_traj, = ax3d.plot([], [], [], 'b-', alpha=0.8, lw=2, label="Actual Trajectory")
            reference_traj, = ax3d.plot([], [], [], 'r--', alpha=0.8, lw=2, label="Reference Trajectory")
            ax3d.legend(loc="upper left")
            
            def init_control():
                quad_arms.set_data([], []);       quad_arms.set_3d_properties([])
                quad_arms2.set_data([], []);      quad_arms2.set_3d_properties([])
                actual_traj.set_data([], []);     actual_traj.set_3d_properties([])
                reference_traj.set_data([], []);  reference_traj.set_3d_properties([])
                for rotor in rotor_patches:
                    rotor.set_data([], []); rotor.set_3d_properties([])
                for pair in ctrl_lines:
                    pair[0].set_data([], [])
                    pair[1].set_data([], [])
                time_text.set_text("")
                return (quad_arms, quad_arms2, actual_traj, reference_traj,
                        *[line for pair in ctrl_lines for line in pair],
                        time_text)
            
            def update_control(frame):
                pos = trajectory_data[frame, :3]
                R_mat = trajectory_data[frame, 6:15].reshape(3,3)
                arm1 = np.array([[arm_len, 0, 0], [-arm_len, 0, 0]]).T
                arm2 = np.array([[0, arm_len, 0], [0, -arm_len, 0]]).T
                arm1_rot = R_mat @ arm1 + pos.reshape(3,1)
                arm2_rot = R_mat @ arm2 + pos.reshape(3,1)
                quad_arms.set_data(arm1_rot[0, :], arm1_rot[1, :])
                quad_arms.set_3d_properties(arm1_rot[2, :])
                quad_arms2.set_data(arm2_rot[0, :], arm2_rot[1, :])
                quad_arms2.set_3d_properties(arm2_rot[2, :])
                rotor_positions = [arm1_rot[:,0], arm1_rot[:,1], arm2_rot[:,0], arm2_rot[:,1]]
                for i, rotor in enumerate(rotor_patches):
                    circle = draw_rotor_circle(rotor_positions[i], R_mat, radius=rotor_rad)
                    rotor.set_data(circle[0, :], circle[1, :])
                    rotor.set_3d_properties(circle[2, :])
                actual_traj.set_data(trajectory_data[:frame, 0], trajectory_data[:frame, 1])
                actual_traj.set_3d_properties(trajectory_data[:frame, 2])
                reference_traj.set_data(reference_trajectory[:frame, 0], reference_trajectory[:frame, 1])
                reference_traj.set_3d_properties(reference_trajectory[:frame, 2])
                time_arr = np.arange(frame) * dt
                for i, (ref_line, act_line) in enumerate(ctrl_lines):
                    ref_line.set_data(time_arr, control_inputs_ref[:frame, i])
                    act_line.set_data(time_arr, control_inputs[:frame, i])
                time_text.set_text(f"Time: {frame*dt:.2f} s")
                return (quad_arms, quad_arms2, actual_traj, reference_traj,
                        *[line for pair in ctrl_lines for line in pair],
                        time_text)
            
            ani = animation.FuncAnimation(fig, update_control, frames=num_frames,
                                          init_func=init_control, interval=dt*1000, blit=False)
        
        fps = int(round(1/dt))
    
    # --------------------- Export Options ---------------------
    if save_video:
        writer = animation.FFMpegWriter(
            fps=fps, codec='libx264', bitrate=16000,
            extra_args=["-vf", "scale=1920:1080", "-crf", "18", "-pix_fmt", "yuv420p"]
        )
        print(f"Saving video to {output_filename}... Please wait.")
        ani.save(output_filename, writer=writer)
        print(f"Video saved as {output_filename}!")
    
    if save_gif:
        if gif_filename is None:
            gif_filename = output_filename.rsplit('.', 1)[0] + ".gif"
        gif_writer = animation.PillowWriter(fps=fps)
        print(f"Saving GIF to {gif_filename}... Please wait.")
        ani.save(gif_filename, writer=gif_writer)
        print(f"GIF saved as {gif_filename}!")
    
    if display_plot:
        plt.show()
    return ani

# ---------------- Run the Animation ----------------
# Example: Run in control mode with three_d_only enabled.
# The target marker will display if task_id is 1 or 2.
# anim = animate_quadrotor(
#     trajectory_data, reference_trajectory, control_inputs, control_inputs_ref,
#     dt=dt, save_video=False, save_gif=False, show_tracking=False,
#     output_filename="quadrotor_animation.mp4",
#     control_umin=np.array([8, -0.6, -0.6, -0.3]),
#     control_umax=np.array([12,  0.6,  0.6,  0.3]),
#     task_id=1,
#     three_d_only=False  # Set to True to display only the 3D plot.
# )
