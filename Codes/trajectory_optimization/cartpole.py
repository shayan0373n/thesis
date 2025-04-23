import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams, plot_state_trajectory

# Cartpole parameters
D_PARAMS = {
    'm': 0.1,  # Mass of the pole (kg)
    'M': 1,  # Mass of the cart (kg)
    'L': 0.5,  # Length of the pole (m)
    'g': 9.81, # Gravitational acceleration (m/s^2)
}

def smooth_relu_quad(x, delta=1e-3):
    """ Smooth approximation of max(x, 0) using sqrt """
    # delta controls the smoothness (smaller delta = sharper corner)
    return 0.5 * (ca.sqrt(x**2 + delta**2) + x)

def cartpole_dynamics(x, u, params, t=None):
    m, M, L, g = D_PARAMS['m'], D_PARAMS['M'], D_PARAMS['L'], D_PARAMS['g']
    x, theta, x_dot, theta_dot = x[0], x[1], x[2], x[3]
    f = u[0] # Force applied to the cart
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    denom = M + m * sin_theta**2
    x_ddot = (f - m * L * theta_dot**2 * sin_theta + m * g * sin_theta * cos_theta) / denom
    theta_ddot = (f * cos_theta - m * L * theta_dot**2 * sin_theta * cos_theta + (M + m) * g * sin_theta) / (L * denom)
    x_dot = ca.vertcat(x_dot, theta_dot, x_ddot, theta_ddot)
    return x_dot

def cartpole_constraints(opti, X, U, params):
    # Add any additional constraints here
    opti.subject_to(opti.bounded(-0.5, X[0, :], 0.5))  # Example limit on cart position
    # opti.subject_to(opti.bounded(-7.0, U[0, :], 7.0))  # Example limit on control force

def cartpole_boundary_conditions(opti, X, U, params):
    # Initial conditions
    opti.subject_to(X[0, 0] == 0)        # Initial cart position = 0
    opti.subject_to(X[1, 0] == np.pi)    # Initial pole angle = pi (hanging down)
    opti.subject_to(X[2, 0] == 0)        # Initial cart velocity = 0
    opti.subject_to(X[3, 0] == 0)        # Initial pole angular velocity = 0

    # Final conditions
    opti.subject_to(X[0, -1] == 0)       # Final cart position = 0
    opti.subject_to(X[1, -1] == 0)       # Final pole angle = 0 (upright)
    opti.subject_to(X[2, -1] == 0)       # Final cart velocity = 0
    opti.subject_to(X[3, -1] == 0)       # Final pole angular velocity = 0

def cartpole_cost_fn(opti, X, U, params):
    # Define a cost function (e.g., minimize control effort)
    cost = ca.sumsqr(U)  # Minimize control effort
    return cost

def cartpole_cost_time(opti, X, U, params):
    # Minimize time spent in the system
    T = params.T  # Total time
    return T

def cartpole_cost_energy(opti, X, U, params):
    # Minimize energy consumption
    P = X[2, :-1] * U[0, :]  # Power = force * velocity
    positive_P = smooth_relu_quad(P, delta=1E-3)  # Smooth ReLU to avoid negative power
    cost = ca.sum2(positive_P) * params.dt_x # Total energy consumed
    return cost

def cartpole_initial_guess(opti, X, U, params):
    angle = X[1, :]
    X_size = X.shape[1]
    # Simple linear interpolation often works reasonably well
    opti.set_initial(angle, np.linspace(np.pi, 0, X_size))

def animate_cartpole(x, params):
    """
    Animates the cartpole system based on the state trajectory.

    Args:
        x (np.ndarray): State trajectory array (n_x, N+1).
                        Assumes x = [pos, angle, vel, ang_vel].
        params (dict): Dictionary containing system parameters like 'L', 'T'.
                       It also reads 'dt_x' or 'dt' for timing.
    """
    pos_opt = x[0, :]
    angle_opt = x[1, :]
    L = D_PARAMS['L']  # Length of the pole
    T = params.T  # Total time
    dt = params.dt_x  # Time step for state
    N = int(T / dt)  # Number of control intervals

    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
    ax_anim.set_aspect('equal') # Ensure correct aspect ratio
    ax_anim.grid(True)
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_ylabel("Vertical Position (m)")
    ax_anim.set_title("Cartpole Animation")
    # Determine plot limits dynamically
    min_x_traj = np.min(pos_opt)
    max_x_traj = np.max(pos_opt)
    plot_margin = L * 1.5 # Add margin around trajectory
    ax_anim.set_xlim(min_x_traj - plot_margin, max_x_traj + plot_margin)
    # Set fixed Y limits (e.g., ground to above the max pole height)
    ax_anim.set_ylim(-L * 1.4, L * 1.6)

    # Define plot elements (use lists for easy access in nested functions if needed)
    cart_width = L / 2.0
    cart_height = cart_width / 2.0
    pole_line, = ax_anim.plot([], [], 'o-', lw=3, color='blue', markersize=6)
    cart_patch = Rectangle((np.nan, np.nan), cart_width, cart_height, fc='gray')
    ax_anim.add_patch(cart_patch)
    time_template = 'Time = %.2fs'
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)

    def init_anim_cart():
        """Initializes the animation plot."""
        pole_line.set_data([], [])
        cart_patch.set_xy((-cart_width / 2, -cart_height / 2)) # Initial position
        time_text.set_text('')
        return pole_line, cart_patch, time_text

    def update_anim_cart(i):
        """Updates the animation plot for frame i."""
        cart_x = pos_opt[i]
        cart_y = 0 # Cart vertical position
        angle_i = angle_opt[i] # Pole angle (0 is up)

        # Pole pivot is assumed at the center of the cart base
        pivot_x = cart_x
        pivot_y = cart_y

        # Pole endpoint calculation (using theta=0 as UP)
        # Tip coordinates relative to pivot: (-L*sin(theta), L*cos(theta))
        pole_end_x = pivot_x - L * np.sin(angle_i)
        pole_end_y = pivot_y + L * np.cos(angle_i)

        pole_line.set_data([pivot_x, pole_end_x], [pivot_y, pole_end_y])
        cart_patch.set_xy((cart_x - cart_width / 2, cart_y - cart_height / 2))
        time_text.set_text(time_template % (i * dt))
        return pole_line, cart_patch, time_text

    # Create and display the animation
    FPS = 30
    # Calculate the frame step to achieve the desired FPS
    frame_step = int(max(1, N // (T * FPS)))  # Aim for ~FPS frames per second
    # Adjust the interval to match the frame step and dt
    ani_interval = int(dt * frame_step * 1000)  # Convert to milliseconds
    ani = animation.FuncAnimation(fig_anim, update_anim_cart,
                                  frames=range(0, N + 1, frame_step),
                                  interval=ani_interval,
                                  blit=True, init_func=init_anim_cart,
                                  repeat=True, repeat_delay=1000)
    return ani


def main():
    params = {
        'T': 2.0,  # Total time
        'dt_x': 0.1,  # Time step for state
        'dt_u': 0.2,   # Time step for control
        'n_x': 4,      # State dimension
        'n_u': 1,      # Control dimension
        # Bounds
        'u_lb': (-10.0, ),  # Lower bound for control
        'u_ub': (10.0, ),   # Upper bound for control
        # 'x_lb': (-0.5, None, None, None),  # Lower bound for state
        # 'x_ub': (0.5, None, None, None),   # Upper bound for state
        # Initial and final values
        'x0': (0, np.pi, 0, 0),  # Initial state
        'xf': (0, 0, 0, 0),      # Final state
    }
    params = TrajectoryOptimizerParams(**params)
    # Create an instance of the optimizer
    optimizer = TrajectoryOptimizer(
        dynamics_fn=cartpole_dynamics,
        # cost_fn=cartpole_cost_fn,
        # cost_fn=cartpole_cost_time,
        cost_fn=cartpole_cost_energy,
        # constraints_fn=cartpole_constraints,
        # boundary_conditions_fn=cartpole_boundary_conditions,
        # initial_guess_fn=cartpole_initial_guess,
        integration_method='trapezoidal',
        is_t_variable=True,
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
        "print_level": 5,
    }
    try:
        x_opt, u_opt = optimizer.solve(p_opts=plugin_opts, s_opts=solver_opts)
        print("Optimal solution found!")
        # Plot the results
        _, ax_plot = plot_state_trajectory(x_opt, u_opt, params)
        # Animate the cartpole
        ani = animate_cartpole(x_opt, params)
        plt.show()
    except Exception as e:
        raise e
        # print("An error occurred during optimization:", e)
        # Handle the error (e.g., log it, retry, etc.)

if __name__ == "__main__":
    main()
