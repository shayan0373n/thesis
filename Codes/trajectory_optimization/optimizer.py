from dataclasses import dataclass
from typing import Callable, Optional, Union
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
    t = np.linspace(0, params.T, x.shape[1], endpoint=True)
    nx = params.n_x
    nu = params.n_u
    xf = params.xf
    x_lb = params.x_lb
    x_ub = params.x_ub
    u_lb = params.u_lb
    u_ub = params.u_ub

    if state_labels is None:
        state_labels = [f'x[{i}]' for i in range(nx)]
    # Assign default ylabels if None, using the potentially generated/provided state_labels
    if ylabels_state is None:
        ylabels_state = state_labels

    if control_labels is None:
        control_labels = [f'u[{i}]' for i in range(nu)]
    # Assign default ylabels if None, using the potentially generated/provided control_labels
    if ylabels_control is None:
        ylabels_control = control_labels

    fig, axs = plt.subplots(nx + nu, 1, figsize=(10, 10), sharex=True)
    axs = np.atleast_1d(axs) # Ensure axs is always array-like # type: ignore 
    fig.suptitle(suptitle, fontsize=14)
    
    for i in range(nx):
        axs[i].plot(t, x[i, :], label=state_labels[i])
        axs[i].set_ylabel(ylabels_state[i])
        if xf is not None and xf[i] is not None:
            axs[i].plot(t, xf[i] * np.ones_like(t), 'k--', label='target')
        if x_lb is not None and x_lb[i] is not None:
            axs[i].plot(t, x_lb[i] * np.ones_like(t), 'g:', label='x_lb')
        if x_ub is not None and x_ub[i] is not None:
            axs[i].plot(t, x_ub[i] * np.ones_like(t), 'g:', label='x_ub')
    for i in range(nu):
        axs[nx + i].stairs(u[i, :], t, baseline=None, label=control_labels[i], color='orange')
        axs[nx + i].set_ylabel(ylabels_control[i])
        if u_lb is not None and u_lb[i] is not None:
            axs[nx + i].plot(t[:-1], u_lb[i] * np.ones_like(t[:-1]), 'g:', label='u_lb')
        if u_ub is not None and u_ub[i] is not None:
            axs[nx + i].plot(t[:-1], u_ub[i] * np.ones_like(t[:-1]), 'g:', label='u_ub')
    axs[nx + nu - 1].set_xlabel('Time (s)')

    for a in axs:
        a.grid(True)
        a.legend()
        a.set_xlim(t[0], t[-1])
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout for suptitle
    return fig, axs

@dataclass
class TrajectoryOptimizerParams:
    n_x: int = 0
    n_u: int = 0
    N: int = 0
    dt_x: float = 0.0
    dt_u: float = 0.0
    T: float = 0.0
    x0: tuple = None
    xf: tuple = None
    x_lb: tuple = None
    x_ub: tuple = None
    u_lb: tuple = None
    u_ub: tuple = None
    T_lb: float = None
    T_ub: float = None
    T_guess: float = 1.0
    other: dict = None

class TrajectoryOptimizer:
    def __init__(self, dynamics_fn, cost_fn,
                 constraints_fn=None,
                 boundary_conditions_fn=None,
                 initial_guess_fn=None,
                 control_input_method ='ZOH',
                 integration_method='trapezoidal',
                 is_t_variable=False,
                 params=None):
        # Store functions and parameters
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.constraints_fn = constraints_fn
        self.initial_guess_fn = initial_guess_fn
        self.boundary_conditions_fn = boundary_conditions_fn
        self.control_input_method = control_input_method
        self.integration_method = integration_method
        self.is_t_variable = is_t_variable
        self.params = params

        # Evaluate parameters
        if self.params.n_x == 0:
            raise ValueError("n_x must be defined in params.")
        if self.params.n_u == 0:
            raise ValueError("n_u must be defined in params.")
        if is_t_variable:
            if self.params.N == 0:
                print("Warning: N is not defined, setting N to 100 for T variable optimization.")
                self.params.N = 100
        else:
            if self.params.N == 0:
                self.params.N = int(self.params.T / self.params.dt_x)
            if self.params.dt_x == 0:
                self.params.dt_x = self.params.T / self.params.N
            if self.params.dt_u == 0:
                self.params.dt_u = self.params.dt_x
            if self.params.T == 0:
                self.params.T = self.params.N * self.params.dt_x
        if self.params.x0 is None:
            self.params.x0 = (0,) * self.params.n_x
        if self.params.xf is None:
            self.params.xf = (0,) * self.params.n_x
        if self.params.x_lb is None:
            self.params.x_lb = (None,) * self.params.n_x
        if self.params.x_ub is None:
            self.params.x_ub = (None,) * self.params.n_x
        if self.params.u_lb is None:
            self.params.u_lb = (None,) * self.params.n_u
        if self.params.u_ub is None:
            self.params.u_ub = (None,) * self.params.n_u
    
    def setup(self):
        """Sets up the optimization problem."""
        # Setup optimization variables
        self.opti = ca.Opti()  # Create an optimization problem
        self.X = self.opti.variable(self.params.n_x, self.params.N + 1)
        self.U = self.opti.variable(self.params.n_u, self.params.N)
        if self.is_t_variable:
            self.params.T = self.opti.variable(1)
            self.params.dt_x = self.params.T / self.params.N
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
        for i in range(self.params.N):
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
                self.opti.subject_to(x_i_plus_1 == x_i + (self.params.dt_x / 2) * (f_i + f_i_plus_1))
            elif self.integration_method == 'euler':
                # Euler integration: x_{i+1} = x_i + dt_x * f_i
                f_i = self.dynamics_fn(x_i, u_i, self.params)
                self.opti.subject_to(x_i_plus_1 == x_i + self.params.dt_x * f_i)
            elif self.integration_method == 'rk4':
                # Runge-Kutta 4th order integration: x_{i+1} = x_i + (dt_x/6) * (k1 + 2*k2 + 2*k3 + k4)
                k1 = self.dynamics_fn(x_i, u_i, self.params)
                k2 = self.dynamics_fn(x_i + (self.params.dt_x / 2) * k1, u_i, self.params)
                k3 = self.dynamics_fn(x_i + (self.params.dt_x / 2) * k2, u_i, self.params)
                k4 = self.dynamics_fn(x_i + self.params.dt_x * k3, u_i, self.params)
                self.opti.subject_to(x_i_plus_1 == x_i + (self.params.dt_x / 6) * (k1 + 2*k2 + 2*k3 + k4))
            else:
                raise NotImplementedError(f"Unknown integration method: {self.integration_method}")

    def _apply_control_constraints(self):
        """Applies Zero-Order Hold (ZOH) constraints to the control inputs."""
        if self.control_input_method == 'ZOH':
            # Check if dt_u is actually smaller than dt_x, otherwise ZOH doesn't make sense here.
            if self.is_t_variable:
                print("ZOH is not applicable when T is a variable. Using dt_x for control intervals.")
                return
            # Calculate how many state steps correspond to one control step
            steps_per_control = round(self.params.dt_u / self.params.dt_x)
            if not np.isclose(steps_per_control * self.params.dt_x, self.params.dt_u):
                print(f"Warning: dt_u ({self.params.dt_u}) is not an integer multiple of dt_x ({self.params.dt_x}). ZOH implementation might be approximate.")
            for i in range(1, self.params.N):
                # If the current state step 'i' does NOT start a new control interval
                # (i.e., it's not 0, steps_per_control, 2*steps_per_control, ...)
                # then the control U[:, i] must be the same as the previous one U[:, i-1].
                if i % steps_per_control != 0:
                    self.opti.subject_to(self.U[:, i] == self.U[:, i-1])
        else:
            raise NotImplementedError(f"Unknown control input method: {self.control_input_method}")

    def _apply_custom_constraints(self):
        """
        Applies constraints. Prioritizes the user-defined constraints_fn.
        If constraints_fn is None, applies simple box constraints if defined
        in params (x_lb, x_ub, u_lb, u_ub).
        """
        if self.is_t_variable:
            # Make sure T is positive
            self.opti.subject_to(self.params.T >= 1E-6)
        if self.constraints_fn is not None:
            # Apply user-defined constraints
            print("Applying user-defined constraints.")
            self.constraints_fn(self.opti, self.X, self.U, self.params)
        else:
            # Apply default box constraints if defined in params
            print("Applying default box constraints.")
            # Apply box constraints for state and control
            for i in range(self.params.n_x):
                lb = self.params.x_lb[i] if self.params.x_lb[i] is not None else -ca.inf
                ub = self.params.x_ub[i] if self.params.x_ub[i] is not None else ca.inf
                self.opti.subject_to(self.opti.bounded(lb, self.X[i, :], ub))
            for i in range(self.params.n_u):
                lb = self.params.u_lb[i] if self.params.u_lb[i] is not None else -ca.inf
                ub = self.params.u_ub[i] if self.params.u_ub[i] is not None else ca.inf
                self.opti.subject_to(self.opti.bounded(lb, self.U[i, :], ub))
            if self.is_t_variable:
                # Apply upper bound for T if defined in params
                if self.params.T_ub is not None:
                    self.opti.subject_to(self.params.T <= self.params.T_ub)
                # Apply lower bound for T if defined in params
                if self.params.T_lb is not None:
                    self.opti.subject_to(self.params.T >= self.params.T_lb)

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
            self.opti.subject_to(self.X[:, 0] == self.params.x0)
            self.opti.subject_to(self.X[:, -1] == self.params.xf)

    def solve(self, solver_name='ipopt', p_opts=None, s_opts=None):
        """Solves the optimization problem."""
        # Set default options if not provided     
        if self.initial_guess_fn is not None:
            # Set initial guess for optimization variables
            print("Setting initial guess using user-defined function.")
            self.initial_guess_fn(self.opti, self.X, self.U, self.params)
        else:
            # Set default initial guess using linear interpolation
            print("Setting initial guess using default linear interpolation.")
            x_init = np.linspace(self.params.x0, self.params.xf, self.params.N + 1, endpoint=True, axis=1)
            u_init = np.zeros((self.params.n_u, self.params.N))
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
        if self.is_t_variable:
            # Set initial guess for time variable T
            self.opti.set_initial(self.params.T, self.params.T_guess)

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
            print("Final cost:", sol.value(self.cost_fn(self.opti, self.X, self.U, self.params)))
            # Extract the optimized trajectory
            x_opt = sol.value(self.X).reshape(self.params.n_x, self.params.N + 1)
            u_opt = sol.value(self.U).reshape(self.params.n_u, self.params.N)
            if self.is_t_variable:
                T_opt = sol.value(self.params.T)
                print(f"Optimized time variable T: {T_opt}")
                self.params.T = T_opt
                self.params.dt_x = T_opt / self.params.N
                self.params.dt_u = self.params.dt_x
            return x_opt, u_opt
        except RuntimeError as e:
            print("Error during optimization:", e)
            raise e # Re-raise the error for further handling
    
