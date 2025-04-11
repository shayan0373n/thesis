import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams, plot_state_trajectory

rod1 = {
    'm': 1.0,  # Mass of rod 1
    'l': 1.0,  # Length of rod 1
    # 'I': 1.0,  # Moment of inertia of rod 1
}
rod2 = {
    'm': 1.0,  # Mass of rod 2
    'l': 1.0,  # Length of rod 2
    # 'I': 1.0,  # Moment of inertia of rod 2
}
D_PARAMS = {
    'rod1': rod1,
    'rod2': rod2,
    'g': 9.81, # Acceleration due to gravity
}

def double_pend_dynamics(x, u, params, t=None):
    # Get current torque values
    t1, t2 = u[0], u[1]
    t2 = 0.0  # Turn off the second motor
    # Unpack state variables
    theta1, theta2, theta1_dot, theta2_dot = x[0], x[1], x[2], x[3]

    # Unpack parameters
    m1 = D_PARAMS['rod1']['m']  # Mass of rod 1
    l1 = D_PARAMS['rod1']['l']  # Length of rod 1
    m2 = D_PARAMS['rod2']['m']  # Mass of rod 2
    l2 = D_PARAMS['rod2']['l']  # Length of rod 2
    g = D_PARAMS['g']           # Gravitational acceleration

    # Pre-calculate trigonometric terms for efficiency
    c12 = ca.cos(theta1 - theta2)
    s12 = ca.sin(theta1 - theta2)
    s1 = ca.sin(theta1)
    s2 = ca.sin(theta2)

    # --- Assemble the Matrix Equation M * theta_ddot = N ---

    # Mass Matrix M(theta)
    M11 = (m1/3 + m2) * l1**2
    M12 = (1/2) * m2 * l1 * l2 * c12
    M21 = M12 # Symmetric
    M22 = (1/3) * m2 * l2**2

    M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M21, M22))

    # Right-hand side vector N(theta, theta_dot, torques)
    # From Eq1: (m1/3 + m2) l1^2 th1_dd + (1/2)m2 l1 l2 c12 th2_dd = t1 - t2 - (1/2)m2 l1 l2 s12 th2_d^2 + (m1/2 + m2) l1 g s1
    N1 = (t1 - t2) - (1/2)*m2*l1*l2*s12*theta2_dot**2 + (m1/2 + m2)*l1*g*s1

    # From Eq2: (1/2) m2 l1 l2 c12 th1_dd + (1/3) m2 l2^2 th2_dd = t2 - (- (1/2) m2 l1 l2 s12 th1_d^2) + (1/2) m2 g l2 s2
    N2 = t2 + (1/2)*m2*l1*l2*s12*theta1_dot**2 + (1/2)*m2*g*l2*s2

    N = ca.vertcat(N1, N2)

    # Solve for angular accelerations symbolically: theta_ddot = M^-1 * N
    theta_ddot  = ca.solve(M, N, 'symbolicqr')
    
    # Extract angular accelerations
    theta1_ddot = theta_ddot[0]
    theta2_ddot = theta_ddot[1]

    # Return the derivative of the state vector
    dxdt = ca.vertcat(theta1_dot, theta2_dot, theta1_ddot, theta2_ddot)
    return dxdt

def double_pend_cost_fn(opti, X, U, params):
    # Define a cost function (e.g., minimize control effort)
    cost = ca.sumsqr(U)  # Minimize control effort
    return cost

def double_pend_constraints(opti, X, U, params):
    # Add any additional constraints here
    # Keep the end effector's x equal to zero
    l1 = D_PARAMS['rod1']['l']
    l2 = D_PARAMS['rod2']['l']
    for i in range(X.shape[1]):
        # End effector position
        ee_x = -l1 * ca.sin(X[0, i]) - l2 * ca.sin(X[1, i])
        opti.subject_to(opti.bounded(-1, ee_x, 1))

    # Turn off the second motor
    # opti.subject_to(opti.bounded(-0.001, U[1, :], 0.001))

def animate_double_pendulum(x, params):
    """
    Animates the double pendulum system based on the state trajectory.

    Args:
        x (np.ndarray): State trajectory array (n_x, N+1).
                        Assumes x = [theta1, theta2, theta1_dot, theta2_dot].
        params (dict): Dictionary containing system parameters like 'L', 'T'.
                       It also reads 'dt_x' or 'dt' for timing.
    """
    theta1_opt = x[0, :]
    theta2_opt = x[1, :]
    l1 = D_PARAMS['rod1']['l']  # Length of rod 1
    l2 = D_PARAMS['rod2']['l']  # Length of rod 2
    T = params.T  # Total time
    dt = params.dt_x  # Time step for state
    N = int(T / dt)  # Number of control intervals
    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
    ax_anim.set_aspect('equal') # Ensure correct aspect ratio
    ax_anim.grid(True)
    ax_anim.set_xlabel("X Position (m)")
    ax_anim.set_ylabel("Y Position (m)")
    ax_anim.set_title("Double Pendulum Animation")
    # Determine plot limits dynamically
    max_x_traj = l1 + l2
    plot_margin = max_x_traj * 0.2 # Add margin around trajectory
    ax_anim.set_xlim(-max_x_traj - plot_margin, max_x_traj + plot_margin)
    ax_anim.set_ylim(-max_x_traj - plot_margin, max_x_traj + plot_margin)
    # Set fixed Y limits (e.g., ground to above the max pole height)
    # ax_anim.set_ylim(-l1 - l2 - plot_margin, l1 + l2 + plot_margin)
    # Define plot elements (use lists for easy access in nested functions if needed)
    rod1_line, = ax_anim.plot([], [], 'o-', lw=3, color='blue', markersize=6)
    rod2_line, = ax_anim.plot([], [], 'o-', lw=3, color='red', markersize=6)
    time_template = 'Time = %.2fs'
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)
    def init_anim_double_pend():
        """Initializes the animation plot."""
        rod1_line.set_data([], [])
        rod2_line.set_data([], [])
        time_text.set_text('')
        return rod1_line, rod2_line, time_text
    def update_anim_double_pend(i):
        """Updates the animation plot for frame i."""
        theta1_i = theta1_opt[i]
        theta2_i = theta2_opt[i]
        # Calculate the positions of the pendulum joints
        x1 = -l1 * np.sin(theta1_i)
        y1 = l1 * np.cos(theta1_i)
        x2 = x1 - l2 * np.sin(theta2_i)
        y2 = y1 + l2 * np.cos(theta2_i)
        # Update the lines and patches
        rod1_line.set_data([0, x1], [0, y1])
        rod2_line.set_data([x1, x2], [y1, y2])
        time_text.set_text(time_template % (i * dt))
        return rod1_line, rod2_line, time_text
    # Create and display the animation
    FPS = 30
    # Calculate the frame step to achieve the desired FPS
    frame_step = int(max(1, N // (T * FPS)))  # Aim for ~FPS frames per second
    # Adjust the interval to match the frame step and dt
    ani_interval = int(dt * frame_step * 1000)  # Convert to milliseconds
    ani = animation.FuncAnimation(fig_anim, update_anim_double_pend,
                                  frames=range(0, N + 1, frame_step),
                                  interval=ani_interval,
                                  blit=True, init_func=init_anim_double_pend,
                                  repeat=True, repeat_delay=1000)
    return ani

def main():
    # Define the parameters
    params = {
        'T': 5.0,  # Total time
        'dt_x': 0.01,  # Time step for state
        'dt_u': 0.02,   # Time step for control
        'n_x': 4,      # State dimension
        'n_u': 2,      # Control dimension
        # Bounds
        # 'u_lb': (-15.0, ),  # Lower bound for control
        # 'u_ub': (15.0, ),   # Upper bound for control
        # 'x_lb': (-0.5, None, None, None),  # Lower bound for state
        # 'x_up': (0.5, None, None, None),   # Upper bound for state
        # Initial and final values
        'x0': (np.pi, np.pi, 0, 0),  # Initial state
        'xf': (0, 0, 0, 0),      # Final state
    }
    params = TrajectoryOptimizerParams(**params)
    # Create an instance of the optimizer
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_pend_dynamics,
        cost_fn=double_pend_cost_fn,
        constraints_fn=double_pend_constraints,
        # boundary_conditions_fn=cartpole_boundary_conditions,
        # initial_guess_fn=cartpole_initial_guess,
        integration_method='trapezoidal',
        params=params,
    )
    # Setup the optimization problem
    optimizer.setup()
    # Solve the optimization problem
    plugin_opts = {
        "expand": True
    }
    solver_opts = {
        "max_iter": 3000,
        "print_level": 1,
    }
    try:
        x_opt, u_opt = optimizer.solve(p_opts=plugin_opts, s_opts=solver_opts)
        print("Optimal solution found!")
        # Plot the results
        _, ax_plot = plot_state_trajectory(x_opt, u_opt, params)
        # Animate
        ani = animate_double_pendulum(x_opt, params)
        plt.show()
    except Exception as e:
        raise e
        # print("An error occurred during optimization:", e)
        # Handle the error (e.g., log it, retry, etc.)

if __name__ == "__main__":
    main()
