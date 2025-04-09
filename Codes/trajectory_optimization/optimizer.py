import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_state_trajectory(x, u, params,
                          state_labels=None,
                          control_labels=None,
                          ylabels_state=None,
                          ylabels_control=None,
                          suptitle="State and Control Trajectory"):
    """
    Plots the results of the trajectory optimization.
    """
    t = np.linspace(0, params['T'], x.shape[1], endpoint=True)
    nx = params['n_x']
    nu = params['n_u']
    xf = params.get('xf', None)
    x_lb = params.get('x_lb', None)
    x_ub = params.get('x_ub', None)
    u_lb = params.get('u_lb', None)
    u_ub = params.get('u_ub', None)

    if state_labels is None:
        state_labels = [f'x[{i}]' for i in range(nx)]
        if ylabels_state is None:
            ylabels_state = state_labels
    if control_labels is None:
        control_labels = [f'u[{i}]' for i in range(nu)]
        if ylabels_control is None:
            ylabels_control = control_labels

    fig, ax = plt.subplots(nx + nu, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(suptitle, fontsize=14)
    
    for i in range(nx):
        ax[i].plot(t, x[i, :], label=state_labels[i])
        ax[i].set_ylabel(ylabels_state[i])
        if xf is not None and xf[i] is not None:
            ax[i].plot(t, xf[i] * np.ones_like(t), 'k--', label='target')
        if x_lb is not None and x_lb[i] is not None:
            ax[i].plot(t, x_lb[i] * np.ones_like(t), 'g:', label='x_lb')
        if x_ub is not None and x_ub[i] is not None:
            ax[i].plot(t, x_ub[i] * np.ones_like(t), 'g:', label='x_ub')
    for i in range(nu):
        ax[nx + i].stairs(u[i, :], t, baseline=None, label=control_labels[i], color='orange')
        ax[nx + i].set_ylabel(ylabels_control[i])
        if u_lb is not None and u_lb[i] is not None:
            ax[nx + i].plot(t[:-1], u_lb[i] * np.ones_like(t[:-1]), 'g:', label='u_lb')
        if u_ub is not None and u_ub[i] is not None:
            ax[nx + i].plot(t[:-1], u_ub[i] * np.ones_like(t[:-1]), 'g:', label='u_ub')
    ax[nx + nu - 1].set_xlabel('Time (s)')

    for a in ax:
        a.grid(True)
        a.legend()
        a.set_xlim(t[0], t[-1])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    return fig, ax


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
        in params (x_lb, x_ub, u_lb, u_ub).
        """
        if self.constraints_fn is not None:
            # Apply user-defined constraints
            print("Applying user-defined constraints.")
            self.constraints_fn(self.opti, self.X, self.U, self.params)
        else:
            # Apply default box constraints if defined in params
            print("Applying default box constraints.")
            x_lower_bound = self.params.get('x_lb', (-ca.inf,) * self.n_x)
            x_upper_bound = self.params.get('x_ub', (ca.inf,) * self.n_x)
            u_lower_bound = self.params.get('u_lb', (-ca.inf,) * self.n_u)
            u_upper_bound = self.params.get('u_ub', (ca.inf,) * self.n_u)
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
        """
        Applies boundary conditions. Prioritizes boundary_conditions_fn.
        If None, applies simple state equality constraints if 'x0' or 'xf'
        are defined in params.
        """
        if self.boundary_conditions_fn is not None:
            # Apply user-defined boundary conditions
            print("Applying user-defined boundary conditions.")
            self.boundary_conditions_fn(self.opti, self.X, self.U, self.params)
        else:
            # Apply default boundary conditions if defined in params
            print("Applying default boundary conditions.")
            x0 = self.params.get('x0', None)
            xf = self.params.get('xf', None)
            if x0 is not None:
                self.opti.subject_to(self.X[:, 0] == x0)
            if xf is not None:
                self.opti.subject_to(self.X[:, -1] == xf)

    def solve(self, solver_name='ipopt', p_opts=None, s_opts=None):
        """Solves the optimization problem."""
        # Set default options if not provided     
        if self.initial_guess_fn is not None:
            # Set initial guess for optimization variables
            print("Setting initial guess using user-defined function.")
            self.initial_guess_fn(self.opti, self.X, self.U, self.params)
        elif "x0" in self.params and "xf" in self.params:
                # If initial and final states are provided, use them to set initial guess
                print("Setting initial guess using provided x0 and xf.")
                x0 = np.asarray(self.params['x0']).flatten()
                xf = np.asarray(self.params['xf']).flatten()
                if x0.shape != (self.n_x,) or xf.shape != (self.n_x,):
                    raise ValueError("x0 and xf must have the same shape as the state dimension.")
                x_init = np.linspace(x0, xf, self.N + 1, axis=1)
                self.opti.set_initial(self.X, x_init)
        else:
            print("Warning: No initial guess provided. Using solver's default initialization (likely zeros).")
        default_p_opts = {"expand": True}
        default_s_opts = {
            "max_iter": 3000,      # Increase max iterations if needed
            "print_level": 5,      # IPOPT verbosity (0=silent)
            "tol": 1e-6,           # Solver tolerance
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
            x_opt = sol.value(self.X).reshape(self.n_x, self.N + 1)
            u_opt = sol.value(self.U).reshape(self.n_u, self.N)
            return x_opt, u_opt
        except RuntimeError as e:
            print("Error during optimization:", e)
            raise e # Re-raise the error for further handling
    
