import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams, plot_state_trajectory # Assuming these are available

def smooth_relu_quad(x, delta=1e-3):
    """ Smooth approximation of max(x, 0) using sqrt """
    # delta controls the smoothness (smaller delta = sharper corner)
    return 0.5 * (ca.sqrt(x**2 + delta**2) + x)

class TriplePendulum:
    """
    Represents a triple pendulum system, handling dynamics derivation,
    optimization setup, and animation.
    """
    def __init__(self, params, **kwargs):
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
        self.mu = params.get('mu', None)  # Default friction coefficient
        # Moments of inertia about CoM
        self.I1 = (1/12) * self.m1 * self.l1**2
        self.I2 = (1/12) * self.m2 * self.l2**2
        self.I3 = (1/12) * self.m3 * self.l3**2

        # Derive and store the symbolic dynamics function
        dynamics_fn_sym = self._derive_symbolic_dynamics()
        self.dynamics_fn = lambda x, u, params: dynamics_fn_sym(x, u)  # Wrap in a lambda to match expected signature
        # Derive and store the symbolic center of mass dynamics function
        self.com_dynamics_fn = self._derive_symbolic_forward_dynamics_com()

        self.R = kwargs.get('R', np.diag([1, 1, 1]))  # Default cost matrix
        print("TriplePendulum initialized.")

    def _derive_symbolic_dynamics(self):
        """
        Derives the symbolic dynamics function using CasADi and Lagrangian mechanics.
        Internal helper method called by __init__.

        Returns:
            ca.Function: A CasADi function f(x, u) -> dxdt.
        """
        print("Deriving symbolic dynamics...")
        th1, th2, th3 = ca.SX.sym('th1'), ca.SX.sym('th2'), ca.SX.sym('th3') # type: ignore
        q = ca.vertcat(th1, th2, th3)
        om1, om2, om3 = ca.SX.sym('om1'), ca.SX.sym('om2'), ca.SX.sym('om3') # type: ignore
        q_dot = ca.vertcat(om1, om2, om3)
        tau1, tau2, tau3 = ca.SX.sym('tau1'), ca.SX.sym('tau2'), ca.SX.sym('tau3') # type: ignore
        x = ca.vertcat(q, q_dot)
        u_input = ca.vertcat(tau1, tau2, tau3)
        # u = ca.vertcat(u_input[0], 0, 0)  # Only use tau1 for now (i.e., turn off tau2 and tau3)
        u = ca.vertcat(u_input[0], u_input[1], u_input[2])  # Use all three torques
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
    
    def _derive_symbolic_forward_dynamics_com(self):
        """
        Derives the x_com_ddot and y_com_ddot for the center of mass of the pendulum.
        """
        # +x is right, +y is up
        th1, th2, th3 = ca.SX.sym('th1'), ca.SX.sym('th2'), ca.SX.sym('th3') # type: ignore
        th1_dot, th2_dot, th3_dot = ca.SX.sym('th1_dot'), ca.SX.sym('th2_dot'), ca.SX.sym('th3_dot') # type: ignore
        th1_ddot, th2_ddot, th3_ddot = ca.SX.sym('th1_ddot'), ca.SX.sym('th2_ddot'), ca.SX.sym('th3_ddot') # type: ignore
        th = ca.vertcat(th1, th2, th3)
        th_dot = ca.vertcat(th1_dot, th2_dot, th3_dot)
        th_ddot = ca.vertcat(th1_ddot, th2_ddot, th3_ddot)
        # Unpack parameters
        l1, l2, l3 = self.l1, self.l2, self.l3
        m1, m2, m3 = self.m1, self.m2, self.m3
        M = m1 + m2 + m3
        x_com1 = -l1 * ca.sin(th1) / 2
        y_com1 = l1 * ca.cos(th1) / 2
        x_com2 = -l1 * ca.sin(th1) - l2 * ca.sin(th2) / 2
        y_com2 = l1 * ca.cos(th1) + l2 * ca.cos(th2) / 2
        x_com3 = -l1 * ca.sin(th1) - l2 * ca.sin(th2) - l3 * ca.sin(th3) / 2
        y_com3 = l1 * ca.cos(th1) + l2 * ca.cos(th2) + l3 * ca.cos(th3) / 2
        x_com = (m1 * x_com1 + m2 * x_com2 + m3 * x_com3) / M
        y_com = (m1 * y_com1 + m2 * y_com2 + m3 * y_com3) / M
        # Calculate the derivatives
        H_x_com, grad_x_com = ca.hessian(x_com, th)
        H_y_com, grad_y_com = ca.hessian(y_com, th)
        x_com_ddot = th_dot.T @ H_x_com @ th_dot + grad_x_com.T @ th_ddot
        y_com_ddot = th_dot.T @ H_y_com @ th_dot + grad_y_com.T @ th_ddot
        # Create a CasADi function for the center of mass dynamics
        com_dynamics = ca.Function('com_dynamics', [th, th_dot, th_ddot], [x_com_ddot, y_com_ddot],
                                  ['th', 'th_dot', 'th_ddot'], ['x_com_ddot', 'y_com_ddot'])
        print("Symbolic center of mass dynamics derived.")
        return com_dynamics

    def base_dynamics(self, x, u, params):
        """
        Returns (T, N) for the base of the pendulum.
        Args:
            x (np.ndarray): State vector (theta, theta_dot).
            u (np.ndarray): Control vector.
        """
        x_ddot = self.dynamics_fn(x, u, params)
        x_com_ddot, y_com_ddot = self.com_dynamics_fn(x[0:3], x[3:6], x_ddot[3:6])
        M = self.m1 + self.m2 + self.m3
        g = self.g
        T = M * x_com_ddot
        N = M * y_com_ddot + M * g
        return ca.vertcat(T, N)
    
    def _cost_fn(self, opti, X, U, params):
        """
        Cost function
        """
        cost =  self._cost_control_effort(opti, X, U, params)
        # self._cost_energy(opti, X, U, params)
        return cost

    def _cost_control_effort(self, opti, X, U, params):
        sum = 0
        for i in range(U.shape[1]):
            u = U[:, i]
            sum += u.T @ self.R @ u * params.dt_x
        return sum
    
    def _cost_energy(self, opti, X, U, params):
        """
        Cost function to minimize energy consumption.
        """
        P = X[3:, :-1] * U  # Power = torque * angular velocity
        cost = ca.sumsqr(P) * params.dt_x  # Total energy consumed
        return cost

    def _constraints_fn(self, opti, X, U, params):
        """
        Default constraints function. Example: constrain end-effector x-pos.
        Can be overridden or replaced in TrajectoryOptimizer setup.
        """
        print("Applying constraints...")
        # Keep the hips' y-position above the lower leg
        hip_y = self.l1 * ca.cos(X[0, :]) + self.l2 * ca.cos(X[1, :])
        opti.subject_to(hip_y >= self.l1) # Avoid penetration
        # Keep the knee angle positive
        knee_angle = X[1, :] - X[0, :]
        opti.subject_to(knee_angle >= 0)
        self._base_force_constraints(opti, X, U, params)

    def _base_force_constraints(self, opti, X, U, params):
        """
        Constraints for the base of the pendulum.
        """
        for i in range(U.shape[1]):
            u = U[:, i]
            x = X[:, i]
            base_force = self.base_dynamics(x, u, params)
            T = base_force[0]
            N = base_force[1]
            # Constraints on the base forces
            opti.subject_to(N >= 0)  # Normal force must be positive
            opti.subject_to(ca.fabs(T) <= N * self.mu)  # Friction constraint


    def setup_optimizer(self, params, is_variable_time=False, integration_method='rk4'):
        """
        Creates and configures the TrajectoryOptimizer instance.
        """
        print(f"Setting up TrajectoryOptimizer with integration: {integration_method}")
        optimizer = TrajectoryOptimizer(
            dynamics_fn=self.dynamics_fn,
            cost_fn=self._cost_fn,
            constraints_fn=self._constraints_fn,
            integration_method=integration_method,
            is_t_variable=is_variable_time,
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
    M = 74 # Mass of the pendulum (kg)
    L = 1.74 # Length of the pendulum (m)
    rod1_p = {'m': 0.186 * M / 2, 'l': 0.478 * L / 2}
    rod2_p = {'m': 0.4 * M / 2, 'l': 0.489 * L / 2}
    rod3_p = {'m': 1.356 * M / 2, 'l': 0.932 * L / 2}
    d_params = {
        'rod1': rod1_p,
        'rod2': rod2_p,
        'rod3': rod3_p,
        'g': 9.81, # Gravity (m/s^2)
        'mu': 0.6, # Friction coefficient (example)
    }
    params = {
        # Timing
        # 'N': 200,
        'T': 1.5,       # Total time horizon (s)
        'dt_x': 0.01,   # State integration time step (s)
        'dt_u': 0.01,   # Control interval time step (dt_u >= dt_x)
        # Dimensions
        'n_x': 6,       # State dimension [th1, th2, th3, om1, om2, om3]
        'n_u': 3,       # Control dimension [tau1, tau2, tau3]
        # Initial and final states (ensure length matches n_x)
        # Example: Start hanging down, finish upright
        'x0': np.array([0, np.pi/2, 0, 0.0, 0.0, 0.0]), # Sitting
        'xf': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),     # Standing
        # Bounds (Optional - add if needed by optimizer/constraints)
        # 'u_lb': np.array([-10.0, -10.0, -10.0]),
        # 'u_ub': np.array([ 10.0,  10.0,  10.0]),
        # 'x_lb': np.array([-2*np.pi]*3 + [-10.0]*3), # Example state bounds
        # 'x_ub': np.array([ 2*np.pi]*3 + [ 10.0]*3),
    }
    R = np.diag([1, 1, 10])
    params = TrajectoryOptimizerParams(**params) # Convert to TrajectoryOptimizerParams
    # Create the pendulum instance (derives dynamics)
    pendulum = TriplePendulum(d_params, R=R)

    # Setup the optimizer using the pendulum's methods
    optimizer = pendulum.setup_optimizer(params,
                                         integration_method='trapezoidal',
                                         is_variable_time=False)

    # Define solver options if defaults are not desired
    # plugin_opts = {"expand": True}
    # solver_opts = {"max_iter": 3000, "print_level": 5, "tol": 1e-4}

    # Solve the optimization problem
    x_opt, u_opt = pendulum.solve_optimization(optimizer) #, solver_opts=solver_opts)

    # Reconstruct base forces
    base_forces = np.zeros((2, u_opt.shape[1]))
    for i in range(u_opt.shape[1]):
        u = u_opt[:, i]
        x = x_opt[:, i]
        base_forces[:, i] = np.array(pendulum.base_dynamics(x, u, params)).flatten()
    # Reconstruct x_com
    x_com = np.zeros((1, u_opt.shape[1]))
    for i in range(u_opt.shape[1]):
        x = x_opt[:3, i]
        th1, th2, th3 = x[0], x[1], x[2]
        l1, l2, l3 = pendulum.l1, pendulum.l2, pendulum.l3
        m1, m2, m3 = pendulum.m1, pendulum.m2, pendulum.m3
        M = m1 + m2 + m3
        x_com1 = -l1 * np.sin(th1) / 2
        x_com2 = -l1 * np.sin(th1) - l2 * np.sin(th2) / 2
        x_com3 = -l1 * np.sin(th1) - l2 * np.sin(th2) - l3 * np.sin(th3) / 2
        x_com[:, i] = (m1 * x_com1 + m2 * x_com2 + m3 * x_com3) / M
        

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



        t = np.linspace(0, params.T, u_opt.shape[1])
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        # Plot base forces
        ax = axs[0]
        ax.plot(t, base_forces[0, :], label='Base Force (T)')
        ax.plot(t, base_forces[1, :], label='Base Force (N)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Base Forces (N)')
        ax.set_title('Base Forces over Time')
        ax.legend()
        ax.grid()

        # Plot center of mass trajectory
        ax = axs[1]
        ax.plot(t, x_com[0, :], label='x_com')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('x_com (m)')
        ax.set_title('Center of Mass Trajectory over Time')
        ax.legend()
        ax.grid()
        
        plt.show()
        

    else:
        print("Optimization failed or did not run.")

if __name__ == "__main__":
    main()