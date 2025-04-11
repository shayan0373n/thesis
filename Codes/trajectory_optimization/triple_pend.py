import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams, plot_state_trajectory # Assuming these are available

class TriplePendulum:
    """
    Represents a triple pendulum system, handling dynamics derivation,
    optimization setup, and animation.
    """
    def __init__(self, params):
        """
        Initializes the TriplePendulum instance.

        Args:
            params (dict): Dictionary containing system parameters.
                           Expected keys: 'rod1', 'rod2', 'rod3' (each with 'm', 'l'),
                           'g', 'T', 'dt_x', 'dt_u', 'n_x', 'n_u', 'x0', 'xf'.
        """
        self.params = params
        # Unpack frequently used parameters for convenience
        self.m1 = params['rod1']['m']
        self.l1 = params['rod1']['l']
        self.m2 = params['rod2']['m']
        self.l2 = params['rod2']['l']
        self.m3 = params['rod3']['m']
        self.l3 = params['rod3']['l']
        self.g = params['g']
        # Moments of inertia about CoM
        self.I1 = (1/12) * self.m1 * self.l1**2
        self.I2 = (1/12) * self.m2 * self.l2**2
        self.I3 = (1/12) * self.m3 * self.l3**2

        # Derive and store the symbolic dynamics function
        self._dynamics_fn_sym = self._derive_symbolic_dynamics()
        # Create a callable lambda matching the expected signature for the optimizer
        self.dynamics_fn = lambda x, u, params, t=None: self._dynamics_fn_sym(x, u)
        self.cost_fn = lambda opti, X, U, params: self._cost_fn(opti, X, U)
        self.constraints_fn = lambda opti, X, U, params: self._constraints_fn(opti, X, U)
        print("TriplePendulum initialized.")

    def _derive_symbolic_dynamics(self):
        """
        Derives the symbolic dynamics function using CasADi and Lagrangian mechanics.
        Internal helper method called by __init__.

        Returns:
            ca.Function: A CasADi function f(x, u) -> dxdt.
        """
        print("Deriving symbolic dynamics...")
        th1, th2, th3 = ca.SX.sym('th1'), ca.SX.sym('th2'), ca.SX.sym('th3')
        q = ca.vertcat(th1, th2, th3)
        om1, om2, om3 = ca.SX.sym('om1'), ca.SX.sym('om2'), ca.SX.sym('om3')
        q_dot = ca.vertcat(om1, om2, om3)
        tau1, tau2, tau3 = ca.SX.sym('tau1'), ca.SX.sym('tau2'), ca.SX.sym('tau3')
        x = ca.vertcat(q, q_dot)
        u_input = ca.vertcat(tau1, tau2, tau3)
        u = ca.vertcat(u_input[0], 0, 0)  # Only use tau1 for now (i.e., turn off tau2 and tau3)
        # Use parameters stored in the instance
        m1, l1, m2, l2, m3, l3 = self.m1, self.l1, self.m2, self.l2, self.m3, self.l3
        g = self.g
        I1, I2, I3 = self.I1, self.I2, self.I3

        # Kinematics (0=up, positive=CCW)
        p_j0 = ca.vertcat(0, 0)
        p_c1 = p_j0 + ca.vertcat(-(l1/2) * ca.sin(th1), (l1/2) * ca.cos(th1))
        p_j1 = p_j0 + ca.vertcat(-l1 * ca.sin(th1), l1 * ca.cos(th1))
        p_c2 = p_j1 + ca.vertcat(-(l2/2) * ca.sin(th2), (l2/2) * ca.cos(th2))
        p_j2 = p_j1 + ca.vertcat(-l2 * ca.sin(th2), l2 * ca.cos(th2))
        p_c3 = p_j2 + ca.vertcat(-(l3/2) * ca.sin(th3), (l3/2) * ca.cos(th3))

        # Velocities
        v_c1 = ca.jacobian(p_c1, q) @ q_dot
        v_c2 = ca.jacobian(p_c2, q) @ q_dot
        v_c3 = ca.jacobian(p_c3, q) @ q_dot

        # Lagrangian (L = KE - PE)
        KE1 = 0.5 * m1 * ca.dot(v_c1, v_c1) + 0.5 * I1 * om1**2
        KE2 = 0.5 * m2 * ca.dot(v_c2, v_c2) + 0.5 * I2 * om2**2
        KE3 = 0.5 * m3 * ca.dot(v_c3, v_c3) + 0.5 * I3 * om3**2
        KE = KE1 + KE2 + KE3

        PE1 = m1 * g * p_c1[1]
        PE2 = m2 * g * p_c2[1]
        PE3 = m3 * g * p_c3[1]
        PE = PE1 + PE2 + PE3
        L = KE - PE

        # Euler-Lagrange Equations
        dL_dq = ca.gradient(L, q)
        dL_dqdot = ca.gradient(L, q_dot)
        M = ca.jacobian(dL_dqdot, q_dot)
        # M = ca.simplify(M) # Simplification can be slow, often optional

        C_G = ca.jtimes(dL_dqdot, q, q_dot) - dL_dq # Coriolis, Centrifugal, Gravity terms
        # C_G = ca.simplify(C_G) # Optional simplification

        N = u - C_G # RHS for M*q_ddot = N

        # Solve for angular accelerations
        q_ddot = ca.solve(M, N)
        # q_ddot = ca.simplify(q_ddot) # Optional simplification

        # State derivative
        dxdt = ca.vertcat(q_dot, q_ddot)
        dynamics_function = ca.Function('triple_pend_dynamics', [x, u_input], [dxdt],
                                        ['x', 'u'], ['dxdt'])
        print("Symbolic dynamics derived.")
        return dynamics_function

    def _cost_fn(self, opti, X, U):
        """
        Default cost function: minimize sum squared control effort.
        Can be overridden or replaced in TrajectoryOptimizer setup.
        """
        return ca.sumsqr(U)

    def _constraints_fn(self, opti, X, U):
        """
        Default constraints function. Example: constrain end-effector x-pos.
        Can be overridden or replaced in TrajectoryOptimizer setup.
        """
        print("Applying constraints...")
        # Example constraint: Keep the end effector's x within bounds [-0.1, 0.1]
        l1, l2, l3 = self.l1, self.l2, self.l3
        n_steps = X.shape[1] # Number of trajectory points (N+1)

        for i in range(n_steps):
            th1_i = X[0, i]
            th2_i = X[1, i]
            th3_i = X[2, i]
            # End effector X position (using 0=up, positive=CCW convention)
            ee_x = -l1 * ca.sin(th1_i) - l2 * ca.sin(th2_i) - l3 * ca.sin(th3_i)
            # Apply constraint (e.g., keep it near the vertical line)
            opti.subject_to(opti.bounded(-0.1, ee_x, 0.1))

        # Add other constraints if needed, e.g., on torques U:
        # max_torque = 10.0
        # opti.subject_to(opti.bounded(-max_torque, U, max_torque))
        print("Constraints applied.")


    def setup_optimizer(self, params, integration_method='rk4'):
        """
        Creates and configures the TrajectoryOptimizer instance.
        """
        print(f"Setting up TrajectoryOptimizer with integration: {integration_method}")
        optimizer = TrajectoryOptimizer(
            dynamics_fn=self.dynamics_fn,
            cost_fn=self.cost_fn,
            constraints_fn=self.constraints_fn,
            integration_method=integration_method,
            params=params # Pass the full params dict
        )
        optimizer.setup()
        print("Optimizer setup complete.")
        return optimizer

    def solve_optimization(self, optimizer, plugin_opts=None, solver_opts=None):
        """
        Solves the trajectory optimization problem.
        """
        default_plugin_opts = {"expand": True} # Helps with CasADi symbolic -> numeric
        default_solver_opts = {"max_iter": 3000, "print_level": 5} # Default IPOPT verbosity

        p_opts = plugin_opts if plugin_opts is not None else default_plugin_opts
        s_opts = solver_opts if solver_opts is not None else default_solver_opts

        print("Solving optimization problem...")
        try:
            x_opt, u_opt = optimizer.solve(p_opts=p_opts, s_opts=s_opts)
            print("Optimal solution found!")
            return x_opt, u_opt
        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            # Optionally re-raise or return None
            raise e
            # return None, None


    def animate(self, x_opt, params):
        """
        Animates the triple pendulum system based on the state trajectory.

        Args:
            x_opt (np.ndarray): Optimal state trajectory array (n_x, N+1).
        """
        print("Starting animation setup...")
        theta1_opt = x_opt[0, :]
        theta2_opt = x_opt[1, :]
        theta3_opt = x_opt[2, :] # Get theta3
        l1, l2, l3 = self.l1, self.l2, self.l3
        T = params.T
        dt = params.dt_x # State integration time step
        N = int(T / dt)

        fig_anim, ax_anim = plt.subplots(figsize=(8, 8)) # Adjusted size
        ax_anim.set_aspect('equal')
        ax_anim.grid(True)
        ax_anim.set_xlabel("X Position (m)")
        ax_anim.set_ylabel("Y Position (m)")
        ax_anim.set_title("Triple Pendulum Animation")

        # Determine plot limits dynamically
        max_reach = l1 + l2 + l3
        plot_margin = max_reach * 0.15 # Add margin
        ax_anim.set_xlim(-max_reach - plot_margin, max_reach + plot_margin)
        ax_anim.set_ylim(-max_reach - plot_margin, max_reach + plot_margin)

        # Define plot elements
        rod1_line, = ax_anim.plot([], [], 'o-', lw=3, color='blue', markersize=6, label='Rod 1')
        rod2_line, = ax_anim.plot([], [], 'o-', lw=3, color='red', markersize=6, label='Rod 2')
        rod3_line, = ax_anim.plot([], [], 'o-', lw=3, color='green', markersize=6, label='Rod 3') # Added rod 3
        time_template = 'Time = %.2fs'
        time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes)
        ax_anim.legend(loc='upper right')

        def init_anim():
            """Initializes the animation plot."""
            rod1_line.set_data([], [])
            rod2_line.set_data([], [])
            rod3_line.set_data([], []) # Init rod 3
            time_text.set_text('')
            return rod1_line, rod2_line, rod3_line, time_text # Return all elements

        def update_anim(i):
            """Updates the animation plot for frame i."""
            th1, th2, th3 = theta1_opt[i], theta2_opt[i], theta3_opt[i]

            # Calculate joint positions using 0=up, positive=CCW convention
            x0, y0 = 0, 0
            x1 = -l1 * np.sin(th1)
            y1 =  l1 * np.cos(th1)
            x2 = x1 - l2 * np.sin(th2)
            y2 = y1 + l2 * np.cos(th2)
            x3 = x2 - l3 * np.sin(th3) # Calculate x3
            y3 = y2 + l3 * np.cos(th3) # Calculate y3

            # Update plot data
            rod1_line.set_data([x0, x1], [y0, y1])
            rod2_line.set_data([x1, x2], [y1, y2])
            rod3_line.set_data([x2, x3], [y2, y3]) # Update rod 3
            time_text.set_text(time_template % (i * dt))
            return rod1_line, rod2_line, rod3_line, time_text # Return all elements

        # Animation parameters
        FPS = 30
        frame_step = int(max(1, N // (T * FPS))) # Ensure at least 1 step
        ani_interval = int(dt * frame_step * 1000) # Interval in milliseconds

        print("Creating FuncAnimation...")
        # Create and return the animation object
        ani = animation.FuncAnimation(fig_anim, update_anim,
                                      frames=range(0, N + 1, frame_step),
                                      interval=ani_interval,
                                      blit=True, # Use blitting for efficiency
                                      init_func=init_anim,
                                      repeat=True, repeat_delay=1000)
        print("Animation object created.")
        return ani

# ==============================================================================
# Main execution block
# ==============================================================================
def main():
    SAVE_ANIM = False # Set to True to save the animation
    print("Setting up Triple Pendulum simulation...")
    # Define the parameters for the triple pendulum
    # Ensure consistency in parameter names if TrajectoryOptimizer expects specific ones
    rod1_p = {'m': 1.0, 'l': 1.0}
    rod2_p = {'m': 1.0, 'l': 0.8}
    rod3_p = {'m': 0.5, 'l': 0.5}
    d_params = {
        'rod1': rod1_p,
        'rod2': rod2_p,
        'rod3': rod3_p,
        'g': 9.81, # Gravity (m/s^2)
    }
    params = {
        # Timing
        'T': 5.0,       # Total time horizon (s)
        'dt_x': 0.02,   # State integration time step (s)
        'dt_u': 0.04,   # Control interval time step (dt_u >= dt_x)
        # Dimensions
        'n_x': 6,       # State dimension [th1, th2, th3, om1, om2, om3]
        'n_u': 3,       # Control dimension [tau1, tau2, tau3]
        # Initial and final states (ensure length matches n_x)
        # Example: Start hanging down, finish upright
        'x0': np.array([np.pi, np.pi, np.pi, 0.0, 0.0, 0.0]), # Hanging down
        'xf': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),     # Straight up
        # Bounds (Optional - add if needed by optimizer/constraints)
        # 'u_lb': np.array([-10.0, -10.0, -10.0]),
        # 'u_ub': np.array([ 10.0,  10.0,  10.0]),
        # 'x_lb': np.array([-2*np.pi]*3 + [-10.0]*3), # Example state bounds
        # 'x_ub': np.array([ 2*np.pi]*3 + [ 10.0]*3),
    }
    params = TrajectoryOptimizerParams(**params) # Convert to TrajectoryOptimizerParams
    # Create the pendulum instance (derives dynamics)
    pendulum = TriplePendulum(d_params)

    # Setup the optimizer using the pendulum's methods
    optimizer = pendulum.setup_optimizer(params, integration_method='rk4') # Or 'euler', 'trapezoidal'

    # Define solver options if defaults are not desired
    # plugin_opts = {"expand": True}
    # solver_opts = {"max_iter": 3000, "print_level": 5, "tol": 1e-4}

    # Solve the optimization problem
    x_opt, u_opt = pendulum.solve_optimization(optimizer) #, solver_opts=solver_opts)

    if x_opt is not None and u_opt is not None:
        print("Optimization successful. Plotting and animating...")
        # Plot the results (assuming plot_state_trajectory handles 6 states)
        try:
            plot_state_trajectory(x_opt, u_opt, params)
        except Exception as plot_err:
            print(f"Could not plot state trajectory: {plot_err}") # Handle if plotter fails

        # Animate the result
        ani = pendulum.animate(x_opt, params)
        if SAVE_ANIM:
            ani.save('triple_pendulum_animation.mp4', writer='ffmpeg', fps=30, dpi=200)
            print("Animation saved as 'triple_pendulum_animation.mp4'.")

        # Keep plots displayed
        plt.show()
    else:
        print("Optimization failed or did not run.")

if __name__ == "__main__":
    main()