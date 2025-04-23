import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams, plot_state_trajectory

rod = {
    'm': 1.0,  # Mass of rod
    'l': 1.0,  # Length of rod
    # 'I': 1.0,  # Moment of inertia of rod
}

D_PARAMS = {
    'rod': rod,
    'g': 9.81, # Acceleration due to gravity
    'mu': 0.1, # Coefficient of friction
}

def single_pend_dynamics_w_contact(x, u, params, t=None):
    ''' Dynamics function for a single pendulum with contact dynamics.
    Args:
        x (np.ndarray): State vector (theta, theta_dot).
        u (np.ndarray): Control and constraint vector.
    '''
    # Get current torque values
    t = u[0]
    # Unpack state variables
    theta, theta_dot = x[0], x[1]
    # Unpack parameters
    m = D_PARAMS['rod']['m']  # Mass of rod
    l = D_PARAMS['rod']['l']  # Length of rod
    g = D_PARAMS['g']         # Gravitational acceleration
    # Pre-calculate trigonometric terms for efficiency
    s = ca.sin(theta)
    c = ca.cos(theta)
    # --- Assemble the Matrix Equation M * theta_ddot = N ---
    # Mass Matrix M(theta)
    M = 1/3 * m * l**2
    # Coriolis and centrifugal forces
    C = 0
    # Gravity forces
    G = m * g * l/2 * s
    # Control forces
    T = t
    # --- Compute the acceleration ---
    # Angular acceleration
    theta_ddot = (T + C + G) / M
    # --- Return the state derivative ---
    x_dot = ca.vertcat(theta_dot, theta_ddot)
    return x_dot

def single_pend_cost_fn(opti, X, U, params):
    # Define a cost function (e.g., minimize control effort)
    t = U[0, :]  # Torque applied to the pendulum
    cost = ca.sumsqr(t) * params.T # Minimize control effort 
    # cost = params.T # Minimize time
    return cost

def single_pend_constraints(opti, X, U, params):
    # Normal and Tangent force constraints
    # Unpack state variables
    theta = X[0, :-1]
    theta_dot = X[1, :-1]
    # Unpack control variables
    t = U[0, :]  # Torque applied to the pendulum
    theta_ddot = [single_pend_dynamics_w_contact(X[:, i], U[:, i], params)[1] for i in range(U.shape[1])]
    theta_ddot = ca.horzcat(*theta_ddot)  # Convert list to CasADi matrix
    # Unpack parameters
    m = D_PARAMS['rod']['m']  # Mass of rod
    l = D_PARAMS['rod']['l']  # Length of rod
    g = D_PARAMS['g']         # Gravitational acceleration
    # Calculate forces
    # T = -m * l/2 * (theta_ddot*cos(theta) - theta_dot**2*sin(theta))
    # N = m * g - m * l/2 * (theta_ddot*sin(theta) + theta_dot**2*cos(theta))
    T = -m * l/2 * (theta_ddot * ca.cos(theta) - theta_dot**2 * ca.sin(theta))
    N = m * g - m * l/2 * (theta_ddot * ca.sin(theta) + theta_dot**2 * ca.cos(theta))
    # Add constraints to the optimization problem
    opti.subject_to(N >= 0)  # Normal force must be non-negative
    opti.subject_to(ca.fabs(T) <= N * D_PARAMS['mu'])  # Friction force must be less than or equal to the normal force times the friction coefficient
    # Add any additional constraints here
    # opti.subject_to(opti.bounded(0.1, params.T, 5))  # Example limit on total time

def animate_single_pendulum(x, params):
    """
    Animates the single pendulum system based on the state trajectory.

    Args:
        x (np.ndarray): State trajectory array (n_x, N+1).
                        Assumes x = [theta, px, py, theta_dot, px_dot, py_dot].
        params (dict): Dictionary containing system parameters like 'L', 'T'.
                       It also reads 'dt_x' or 'dt' for timing.
    """
    theta_opt = x[0, :]
    l = D_PARAMS['rod']['l']  # Length of rod
    T = params.T  # Total time
    dt = params.dt_x  # Time step for state
    N = int(T / dt)  # Number of control intervals
    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
    ax_anim.set_aspect('equal') # Ensure correct aspect ratio
    ax_anim.grid(True)
    ax_anim.set_xlabel("X Position (m)")
    ax_anim.set_ylabel("Y Position (m)")
    ax_anim.set_title("Single Pendulum Animation")
    # Determine plot limits dynamically
    max_x_traj = l
    plot_margin = max_x_traj * 0.2 # Add margin around trajectory
    ax_anim.set_xlim(-max_x_traj - plot_margin, max_x_traj + plot_margin)
    ax_anim.set_ylim(-max_x_traj - plot_margin, max_x_traj + plot_margin)
    # Define plot elements (use lists for easy access in nested functions if needed)
    rod_line, = ax_anim.plot([], [], 'o-', lw=3, color='blue', markersize=6)
    time_template = 'Time = %.2fs'
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)
    def init_anim_single_pend():
        """Initializes the animation plot."""
        rod_line.set_data([], [])
        time_text.set_text('')
        return rod_line, time_text
    def update_anim_single_pend(i):
        """Updates the animation plot for frame i."""
        theta_i = theta_opt[i]
        # Calculate the position of the pendulum joint
        x1 = -l * np.sin(theta_i)
        y1 = l * np.cos(theta_i)
        # Update the lines and patches
        rod_line.set_data([0, x1], [0, y1])
        time_text.set_text(time_template % (i * dt))
        return rod_line, time_text
    # Create and display the animation
    FPS = 30
    # Calculate the frame step to achieve the desired FPS
    frame_step = int(max(1, N // (T * FPS)))  # Aim for ~FPS frames per second
    # Adjust the interval to match the frame step and dt
    ani_interval = int(dt * frame_step * 1000)  # Convert to milliseconds
    ani = animation.FuncAnimation(fig_anim, update_anim_single_pend,
                                  frames=range(0, N + 1, frame_step),
                                  interval=ani_interval,
                                  blit=True, init_func=init_anim_single_pend,
                                  repeat=True, repeat_delay=1000)
    return ani


def main():
    # Define the parameters
    params = {
        'T': 5.0,  # Total time
        'dt_x': 0.01,  # Time step for state
        'dt_u': 0.02,   # Time step for control
        'n_x': 2,      # State dimension
        'n_u': 1,      # Control dimension
        # Bounds
        # 'u_lb': (-10,),  # Lower bound for control
        # 'u_ub': (10,),   # Upper bound for control
        # Initial and final values
        'x0': (np.pi, 0),  # Initial state
        'xf': (0, 0),  # Final state
    }
    params = TrajectoryOptimizerParams(**params)
    # Create an instance of the optimizer
    optimizer = TrajectoryOptimizer(
        dynamics_fn=single_pend_dynamics_w_contact,
        cost_fn=single_pend_cost_fn,
        constraints_fn=single_pend_constraints,
        # boundary_conditions_fn=cartpole_boundary_conditions,
        # initial_guess_fn=cartpole_initial_guess,
        integration_method='trapezoidal',
        params=params,
        # is_t_variable=True,
    )
    # Setup the optimization problem
    optimizer.setup()
    # Solve the optimization problem
    plugin_opts = {
        "expand": True
    }
    solver_opts = {
        "max_iter": 10000,
        "print_level": 1,
    }
    try:
        x_opt, u_opt = optimizer.solve(p_opts=plugin_opts, s_opts=solver_opts)
        print("Optimal solution found!")
        # Plot the results
        _, ax_plot = plot_state_trajectory(x_opt, u_opt, params)
        # Animate
        ani = animate_single_pendulum(x_opt, params)
        plt.show()
    except Exception as e:
        raise e
        # print("An error occurred during optimization:", e)
        # Handle the error (e.g., log it, retry, etc.)

if __name__ == "__main__":
    main()
