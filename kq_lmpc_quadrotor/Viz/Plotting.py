import numpy as np
import matplotlib.pyplot as plt



class Plot_Results:
        
    @staticmethod
    def fcn_plot_traj_hover(sd,X,Xd,X0, task_id):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the drone's trajectory using the first three rows of X
        ax.plot(X[0, :], X[1, :], X[2, :], color='blue', label='Trajectory', linewidth=2)

        # Mark the start and end points
        ax.scatter(X0[0],X0[1],X0[2], color='green', s=50, label='Start', marker='o')
        ax.scatter(Xd[0,-1], Xd[1,-1], Xd[2,-1], color='red', s=50, label='Ref', marker='x')
        ax.view_init(elev=20, azim=-40)
        # Labels and title
        ax.set_xlabel('$s_x$ (m)')
        ax.set_ylabel('$s_y$ (m)')
        ax.set_zlabel('$s_z$ (m)')
        ax.set_title('Hover Task')
        if task_id == 1:
            ax.set_xlim([-sd[0]-1, sd[0]+1])  # x-axis limits
            ax.set_ylim([-sd[1]-1, sd[1]+1])  # y-axis limits
            ax.set_zlim([0, sd[2]+0.5])  # z-axis limits
        else:
            ax.set_xlim([-3, 3])  # x-axis limits
            ax.set_ylim([-3, 3])  # y-axis limits
            ax.set_zlim([0, sd[2]+0.5])  # z-axis limits

        # Adjust the layout to make space for the legend
        #plt.subplots_adjust(bottom=0.25)  # Adjust bottom space to fit the legend

        # Add a legend below the plot
        fig.legend()
        #plt.tight_layout()
        # Show the plot
        plt.show()
        return fig
    
    @staticmethod
    def fcn_plot_traj_track(X,Xd,X0, task_id=None):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Mark the start and end points
        ax.plot(Xd[0, :], Xd[1, :], Xd[2, :], color='blue', label='Ref',linewidth=2)

        # Plot the drone's trajectory using the first three rows of X
        ax.plot(X[0, :], X[1, :], X[2, :], color='red',linestyle='--', label='Trajectory', linewidth=2)


        ax.scatter(X0[0],X0[1],X0[2], color='green', s=50, label='Start', marker='o')
        
        ax.view_init(elev=30, azim=-160)
        # Labels and title
        ax.set_xlabel('$s_x$ (m)')
        ax.set_ylabel('$s_y$ (m)')
        ax.set_zlabel('$s_z$ (m)')
        if task_id is None:
            ax.set_title('Trajectory Tracking')
        elif task_id == 2:
            ax.set_title('Follow a Line and Hover')
        elif task_id == 3:
            ax.set_title('Lemniscate')
        elif task_id == 4 or task_id ==5:
            ax.set_title('Helics')
        elif task_id ==6:
            ax.set_title('Knot')
        ax.set_ylim([-np.abs(np.max(X[1, :])) -0.3, 
                     np.abs(np.max(X[1, :])) + 0.3])
        ax.set_xlim([-np.abs(np.max(X[0, :])) -0.3
                           , np.abs(np.max(X[0, :])) + 0.3])
        ax.set_zlim([np.min(X[2, :])-0.5, np.abs(np.max(X[2, :])) + 0.5])
        fig.legend(loc='upper left')
        plt.tight_layout()
        # Show the plot
        plt.show()
        return fig
    
    
    @staticmethod
    def plot_tracking_errors_normalized(time_sim, ex_norm, ev_norm, Psi, ew_norm):
        # Create a figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot Attitude Tracking Error
        axs[0, 0].plot(time_sim, Psi, linewidth=3)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Attitude Tracking Error', fontsize=16)
        axs[0, 0].set_xlabel('Time (s)', fontsize=14)
        axs[0, 0].set_ylabel(r'$\Psi = \mathrm{tr}(\mathbf{I} - R_d^T R)$', fontsize=14)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Angular Velocity Error (Normalized)
        axs[0, 1].plot(time_sim, ew_norm, linewidth=3)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Angular Velocity Error (Normalized)', fontsize=16)
        axs[0, 1].set_xlabel('Time (s)', fontsize=14)
        axs[0, 1].set_ylabel('Error', fontsize=14)
        axs[0, 1].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'], fontsize=12)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Position Error (Normalized)
        axs[1, 0].plot(time_sim, ex_norm, linewidth=3)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Position Error (Normalized)', fontsize=16)
        axs[1, 0].set_xlabel('Time (s)', fontsize=14)
        axs[1, 0].set_ylabel('Error', fontsize=14)
        axs[1, 0].legend(['$s_x$', '$s_y$', '$s_z$'], fontsize=12)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Velocity Error (Normalized)
        axs[1, 1].plot(time_sim, ev_norm, linewidth=3)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Velocity Error (Normalized)', fontsize=16)
        axs[1, 1].set_xlabel('Time (s)', fontsize=14)
        axs[1, 1].set_ylabel('Error', fontsize=14)
        axs[1, 1].legend(['$v_x$', '$v_y$', '$v_z$'], fontsize=12)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
    
        # Adjust layout and position
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
        # Show the plot
        plt.show()
        return fig
    
    @staticmethod
    def plot_tracking_errors(time_sim, ex_norm, ev_norm, Psi, ew_norm):
        # Create a figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot Attitude Tracking Error
        axs[0, 0].plot(time_sim, Psi, linewidth=3)
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Attitude Tracking Error', fontsize=16)
        axs[0, 0].set_xlabel('Time (s)', fontsize=14)
        axs[0, 0].set_ylabel(r'$\Psi = \mathrm{tr}(\mathbf{I} - R_d^T R)$', fontsize=14)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Angular Velocity Error (Normalized)
        axs[0, 1].plot(time_sim, ew_norm, linewidth=3)
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Angular Velocity Error', fontsize=16)
        axs[0, 1].set_xlabel('Time (s)', fontsize=14)
        axs[0, 1].set_ylabel('Error', fontsize=14)
        axs[0, 1].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'], fontsize=12)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Position Error (Normalized)
        axs[1, 0].plot(time_sim, ex_norm, linewidth=3)
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Position Error', fontsize=16)
        axs[1, 0].set_xlabel('Time (s)', fontsize=14)
        axs[1, 0].set_ylabel('Error', fontsize=14)
        axs[1, 0].legend(['$s_x$', '$s_y$', '$s_z$'], fontsize=12)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        
        # Plot Velocity Error (Normalized)
        axs[1, 1].plot(time_sim, ev_norm, linewidth=3)
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Velocity Error', fontsize=16)
        axs[1, 1].set_xlabel('Time (s)', fontsize=14)
        axs[1, 1].set_ylabel('Error', fontsize=14)
        axs[1, 1].legend(['$v_x$', '$v_y$', '$v_z$'], fontsize=12)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
    
        # Adjust layout and position
        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
        # Show the plot
        plt.show()
        return fig

    @staticmethod
    def plot_trajectory_and_error(Xd, X, time_sim):
        """
        Plots trajectory tracking and attitude tracking error.
        
        Parameters:
            Xd (numpy.ndarray): Reference trajectory of shape (18, N).
            X (numpy.ndarray): Actual trajectory of shape (18, N).
            time_sim (array-like): Array of N timesteps.
        """
        # Calculate attitude tracking error
        attitude_errors = []
        for i in range(X.shape[1]):  # Loop over timesteps
            # Reference rotation matrix
            Rd = Xd[6:15, i].reshape(3, 3, order='F')
            # Actual rotation matrix
            R_act = X[6:15, i].reshape(3, 3, order='F')
            error = 0.5 * np.trace(np.eye(3) - np.dot(Rd.T, R_act))
            attitude_errors.append(error)

        attitude_errors = np.array(attitude_errors)

        # Create figure
        fig, axs = plt.subplots(
            4, 2, figsize=(12, 16))  # 4 rows, 2 columns
        fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

        # Plot positions s_x, s_y, s_z
        axs[0, 0].plot(time_sim, Xd[0, :], label='$s_{x,d}$', color='blue')
        axs[0, 0].plot(time_sim, X[0, :], label='$s_x$',
                       color='red', linestyle='--')
        axs[0, 0].set_title('$s_x$ vs $s_{x,d}$')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Position (m)')
        axs[0, 0].legend()
        axs[0, 0].set_ylim([-np.abs(np.max(X[0, :])) -
                           1, np.abs(np.max(X[0, :])) + 1])

        axs[0, 1].plot(time_sim, Xd[1, :], label='$s_{y,d}$', color='blue')
        axs[0, 1].plot(time_sim, X[1, :], label='$s_y$',
                       color='red', linestyle='--')
        axs[0, 1].set_title('$s_y$ vs $s_{y,d}$')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Position (m)')
        axs[0, 1].legend()
        axs[0, 1].set_ylim([-np.abs(np.max(X[1, :])) -
                           1, np.abs(np.max(X[1, :])) + 1])

        axs[1, 0].plot(time_sim, Xd[2, :], label='$s_{z,d}$', color='blue')
        axs[1, 0].plot(time_sim, X[2, :], label='$s_z$',
                       color='red', linestyle='--')
        axs[1, 0].set_title('$s_z$ vs $s_{z,d}$')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Position (m)')
        axs[1, 0].legend()
        axs[1, 0].set_ylim([-np.abs(np.max(X[2, :])) -
                           1, np.abs(np.max(X[2, :])) + 1])

        # Plot velocities v_x, v_y, v_z
        axs[1, 1].plot(time_sim, Xd[3, :], label='$v_{x,d}$', color='blue')
        axs[1, 1].plot(time_sim, X[3, :], label='$v_x$',
                       color='red', linestyle='--')
        axs[1, 1].set_title('$v_x$ vs $v_{x,d}$')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (m/s)')
        axs[1, 1].legend()
        axs[1, 1].set_ylim([-np.abs(np.max(X[3, :])) -
                           1, np.abs(np.max(X[3, :])) + 1])

        axs[2, 0].plot(time_sim, Xd[4, :], label='$v_{y,d}$', color='blue')
        axs[2, 0].plot(time_sim, X[4, :], label='$v_y$',
                       color='red', linestyle='--')
        axs[2, 0].set_title('$v_y$ vs $v_{y,d}$')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Velocity (m/s)')
        axs[2, 0].legend()
        axs[2, 0].set_ylim([-np.abs(np.max(X[4, :])) -
                           1, np.abs(np.max(X[4, :])) + 1])

        axs[2, 1].plot(time_sim, Xd[5, :], label='$v_{z,d}$', color='blue')
        axs[2, 1].plot(time_sim, X[5, :], label='$v_z$',
                       color='red', linestyle='--')
        axs[2, 1].set_title('$v_z$ vs $v_{z,d}$')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Velocity (m/s)')
        axs[2, 1].legend()
        axs[2, 1].set_ylim([-np.abs(np.max(X[5, :])) -
                           1, np.abs(np.max(X[5, :])) + 1])

        # Plot attitude tracking error
        ax_att = fig.add_subplot(4, 1, 4)
        ax_att.plot(time_sim, attitude_errors,
                    label='Attitude Tracking Error', color='purple')
        ax_att.set_title('Attitude Tracking Error')
        ax_att.set_xlabel('Time (s)')
        ax_att.set_ylabel('Error')
        ax_att.legend()
        ax_att.set_ylim([0, np.max(attitude_errors)+0.1])

        plt.tight_layout()
        # Display the plot
        plt.show()

        return fig

    @staticmethod
    def plot_control_inputs_comparison(reference_u, u_act, time):
        """
        Plot a comparison between reference control inputs and actual control inputs in a 2x2 grid.
        
        Parameters:
        - reference_u: Nx4 matrix, reference control inputs [f, M1, M2, M3].
        - u_act: Nx4 matrix, actual control inputs [f, M1, M2, M3].
        - time: List of time values of size N for the x-axis.
        """
        # Ensure the inputs are numpy arrays
        reference_u = np.array(reference_u)
        u_act = np.array(u_act)
        time = np.array(time)  # Convert time to numpy array

        # Check that the dimensions match
        assert reference_u.shape == u_act.shape, "Dimensions of reference_u and u_act must match."
        assert len(
            time) == reference_u.shape[0], "Time vector length must match the number of rows in reference_u and u_act."

        # Labels for the control inputs
        labels = ['f (Thrust)', 'M1 (Moment X)',
                  'M2 (Moment Y)', 'M3 (Moment Z)']

        # Create 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            'Comparison of Min-Snap Reference and Actual Control Inputs', fontsize=16)

        # Flatten the axes array for easy iteration
        axs = axs.flatten()

        for i in range(4):
            axs[i].plot(time, reference_u[:, i],
                        label=f'Reference {labels[i]}', linestyle='--', linewidth=2)
            axs[i].plot(time, u_act[:, i],
                        label=f'Actual {labels[i]}', linewidth=1.5)
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(labels[i])
            axs[i].legend()
            axs[i].grid()

        # Adjust layout
        # Leave space for the main title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig