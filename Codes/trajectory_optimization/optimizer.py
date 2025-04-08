import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TrajectoryOptimizer:
    def __init__(self, dynamics_fn, cost_fn,
                 constraints_fn=None,
                 boundary_conditions_fn=None,
                 initial_guess_fn=None,
                 integration_method='trapezoidal',
                 params={}):
        # Store functions and parameters
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.constraints_fn = constraints_fn
        self.initial_guess_fn = initial_guess_fn
        self.boundary_conditions_fn = boundary_conditions_fn
        self.integration_method = integration_method
        self.params = params

        # Extract parameters
        self.T = params['T']  # Total time
        self.dt_x = params.get('dt_x', params.get('dt', 0.01))  # Time step for state
        self.dt_u = params.get('dt_u', params.get('dt', 0.01))  # Time step for control
        self.N = int(self.T / self.dt_x)  # Number of control intervals
        self.n_x = params['n_x'] # State dimension
        self.n_u = params['n_u'] # Control dimension
    
    def setup(self):
        """Sets up the optimization problem."""
        # Setup optimization variables
        self.opti = ca.Opti()  # Create an optimization problem
        self.X = self.opti.variable(self.n_x, self.N + 1)
        self.U = self.opti.variable(self.n_u, self.N)
        # --- Set Objective ---
        cost = self.cost_fn(self.opti, self.X, self.U, self.params)
        self.opti.minimize(cost)
        # --- Apply Constraints ---
        self._apply_dynamics_constraints()
        self._apply_control_constraints()
        self._apply_custom_constraints()
        self._apply_boundary_conditions()
        print("Problem setup complete: Objective and constraints applied.")
        
    def _apply_dynamics_constraints(self):
        """Applies the dynamics constraints using Trapezoidal integration."""
        for i in range(self.N):
            x_i = self.X[:, i]
            x_i_plus_1 = self.X[:, i + 1]
            u_i = self.U[:, i]
            # It's common practice to assume u stays constant over the state interval dt_x
            # If u can change faster, the discretization needs more thought.
            # Let's assume u_i applies for the dynamics calculation at both ends.
            if self.integration_method == 'trapezoidal':
                # Trapezoidal integration: x_{i+1} = x_i + (dt_x/2) * (f_i + f_{i+1})
                # where f_i = f(x_i, u_i) and f_{i+1} = f(x_{i+1}, u_i)
                # Here we assume u is constant over dt_x
                f_i = self.dynamics_fn(x_i, u_i, self.params)
                f_i_plus_1 = self.dynamics_fn(x_i_plus_1, u_i, self.params)
                self.opti.subject_to(x_i_plus_1 == x_i + (self.dt_x / 2) * (f_i + f_i_plus_1))
            elif self.integration_method == 'euler':
                # Euler integration: x_{i+1} = x_i + dt_x * f_i
                f_i = self.dynamics_fn(x_i, u_i, self.params)
                self.opti.subject_to(x_i_plus_1 == x_i + self.dt_x * f_i)
            elif self.integration_method == 'rk4':
                # Runge-Kutta 4th order integration: x_{i+1} = x_i + (dt_x/6) * (k1 + 2*k2 + 2*k3 + k4)
                k1 = self.dynamics_fn(x_i, u_i, self.params)
                k2 = self.dynamics_fn(x_i + (self.dt_x / 2) * k1, u_i, self.params)
                k3 = self.dynamics_fn(x_i + (self.dt_x / 2) * k2, u_i, self.params)
                k4 = self.dynamics_fn(x_i + self.dt_x * k3, u_i, self.params)
                self.opti.subject_to(x_i_plus_1 == x_i + (self.dt_x / 6) * (k1 + 2*k2 + 2*k3 + k4))
            else:
                raise NotImplementedError(f"Unknown integration method: {self.integration_method}")

    def _apply_control_constraints(self):
        """Applies Zero-Order Hold (ZOH) constraints to the control inputs."""
        # Check if dt_u is actually smaller than dt_x, otherwise ZOH doesn't make sense here.
        if self.dt_u <= self.dt_x:
             # If control updates as fast or faster than state, no ZOH needed between state steps
             print("Warning: dt_u <= dt_x, ZOH constraint between state steps is trivial. Each U[:,i] can be independent.")
             return
        # Calculate how many state steps correspond to one control step
        steps_per_control = round(self.dt_u / self.dt_x)
        if not np.isclose(steps_per_control * self.dt_x, self.dt_u):
            print(f"Warning: dt_u ({self.dt_u}) is not an integer multiple of dt_x ({self.dt_x}). ZOH implementation might be approximate.")
        for i in range(1, self.N):
            # If the current state step 'i' does NOT start a new control interval
            # (i.e., it's not 0, steps_per_control, 2*steps_per_control, ...)
            # then the control U[:, i] must be the same as the previous one U[:, i-1].
            if i % steps_per_control != 0:
                self.opti.subject_to(self.U[:, i] == self.U[:, i-1])

    def _apply_custom_constraints(self):
        """
        Applies constraints. Prioritizes the user-defined constraints_fn.
        If constraints_fn is None, applies simple box constraints if defined
        in params (x_lower_bound, x_upper_bound, u_lower_bound, u_upper_bound).
        """
        if self.constraints_fn is not None:
            # Apply user-defined constraints
            print("Applying user-defined constraints.")
            self.constraints_fn(self.opti, self.X, self.U, self.params)
        else:
            # Apply default box constraints if defined in params
            print("Applying default box constraints.")
            x_lower_bound = self.params.get('x_lower_bound', (-ca.inf,) * self.n_x)
            x_upper_bound = self.params.get('x_upper_bound', (ca.inf,) * self.n_x)
            u_lower_bound = self.params.get('u_lower_bound', (-ca.inf,) * self.n_u)
            u_upper_bound = self.params.get('u_upper_bound', (ca.inf,) * self.n_u)
            # Apply box constraints for state and control
            for i in range(self.n_x):
                lb = x_lower_bound[i] if x_lower_bound[i] is not None else -ca.inf
                ub = x_upper_bound[i] if x_upper_bound[i] is not None else ca.inf
                self.opti.subject_to(self.opti.bounded(lb, self.X[i, :], ub))
            for i in range(self.n_u):
                lb = u_lower_bound[i] if u_lower_bound[i] is not None else -ca.inf
                ub = u_upper_bound[i] if u_upper_bound[i] is not None else ca.inf
                self.opti.subject_to(self.opti.bounded(lb, self.U[i, :], ub))

    def _apply_boundary_conditions(self):
        """Applies user-defined boundary conditions."""
        if self.boundary_conditions_fn is not None:
            self.boundary_conditions_fn(self.opti, self.X, self.U, self.params)

    def solve(self, solver_name='ipopt', p_opts=None, s_opts=None):
        """Solves the optimization problem."""
        # Set default options if not provided     
        if self.initial_guess_fn is not None:
            # Set initial guess for optimization variables
            self.initial_guess_fn(self.opti, self.X, self.U, self.params)
        # Prepare options for the solver
        default_p_opts = {"expand": True}
        default_s_opts = {
            "max_iter": 3000,       # Increase max iterations if needed
            "print_level": 5,       # IPOPT verbosity (0=silent)
        }
        if p_opts is not None:
            default_p_opts.update(p_opts)
        if s_opts is not None:
            default_s_opts.update(s_opts)
        # Set options for the solver
        self.opti.solver(solver_name, default_p_opts, default_s_opts)
        # Solve the optimization problem
        print("Solving the optimization problem.")
        try:
            sol = self.opti.solve()
            print("Optimization problem solved successfully.")
            # Extract the optimized trajectory
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            return x_opt, u_opt
        except RuntimeError as e:
            print("Error during optimization:", e)
            raise e # Re-raise the error for further handling
    


def cartpole_dynamics(x, u, params, t=None):
    m, M, L, g = params['m'], params['M'], params['L'], params['g']
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

def cartpole_initial_guess(opti, X, U, params):
    angle = X[1, :]
    X_size = X.shape[1]
    # Simple linear interpolation often works reasonably well
    opti.set_initial(angle, np.linspace(np.pi, 0, X_size))

def cartpole_plot(x, u, params):
    pos_opt = x[0, :]
    angle_opt = x[1, :]
    vel_opt = x[2, :]
    ang_vel_opt = x[3, :]
    u_opt = u[:]
    t = np.linspace(0, params['T'], x.shape[1], endpoint=True)

    plt.figure("Cartpole Trajectory Optimization Results", figsize=(10, 10))
    plt.suptitle(f'Cartpole Swing-up Trajectory (T={params['T']})', fontsize=14)
    plt.subplot(5, 1, 1)
    plt.plot(t, pos_opt, label='x (m)')
    plt.ylabel('Position'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 2)
    plt.plot(t, angle_opt, label='theta (rad)')
    plt.plot(t, np.zeros_like(t), 'k--', label='target')
    plt.ylabel('Angle'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(t, vel_opt, label='x_dot (m/s)')
    plt.ylabel('Velocity'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(t, ang_vel_opt, label='theta_dot (rad/s)')
    plt.ylabel('Ang Vel'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 5)
    plt.stairs(u_opt, t, baseline=None, color='r', label='u (Force N)')
    plt.ylabel('Control'); plt.xlabel('Time (s)'); plt.grid(True); plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


def main():
    params = {
        'T': 2.0,  # Total time
        'dt_x': 0.001,  # Time step for state
        'dt_u': 0.1,   # Time step for control
        'n_x': 4,      # State dimension
        'n_u': 1,      # Control dimension
        # Cartpole parameters
        'M': 1.0,  # Mass of the cart
        'm': 0.1,  # Mass of the pole
        'L': 0.5,  # Length of the pole
        'g': 9.81, # Acceleration due to gravity
        # Bounds
        'u_lower_bound': (-15.0, ),  # Lower bound for control
        'u_upper_bound': (15.0, ),   # Upper bound for control
        'x_lower_bound': (-0.5, None, None, None),  # Lower bound for state
        'x_upper_bound': (0.5, None, None, None),   # Upper bound for state
    }

    # Create an instance of the optimizer
    optimizer = TrajectoryOptimizer(
        dynamics_fn=cartpole_dynamics,
        cost_fn=cartpole_cost_fn,
        # constraints_fn=cartpole_constraints,
        boundary_conditions_fn=cartpole_boundary_conditions,
        initial_guess_fn=cartpole_initial_guess,
        integration_method='trapezoidal',
        params=params
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
        cartpole_plot(x_opt, u_opt, params)
    except Exception as e:
        print("An error occurred during optimization:")
        # Handle the error (e.g., log it, retry, etc.)

if __name__ == "__main__":
    main()
