import casadi as ca
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Sequence, Any

# --- Global Constants ---
INITIAL_H_GUESS_FACTOR = 0.95 # Factor to scale initial guess for time steps when T is an optimization variable
MIN_TIME_STEP = 1e-6 # Minimum time step for numerical stability

# --- Utility Functions ---
def stretch_array(arr: np.ndarray, new_col_size: int) -> np.ndarray:
    """Stretches a 2D array to a new size."""
    old_col_size = arr.shape[1]
    old_index = np.linspace(0, old_col_size - 1, old_col_size)
    new_index = np.linspace(0, old_col_size - 1, new_col_size)
    stretched_arr = np.zeros((arr.shape[0], new_col_size))
    for i in range(arr.shape[0]):
        stretched_arr[i, :] = np.interp(new_index, old_index, arr[i, :])
    return stretched_arr

# --- Data Structures for Hybrid System Definition ---

@dataclass
class Mode:
    """Defines a single operational mode of the hybrid system."""
    name: str
    n_x: int # State dimension
    n_u: int # Control dimension
    # Dynamics function: f(x, u, params) -> x_dot
    dynamics_fn: Callable[[ca.MX, ca.MX, dict[str, Any]], ca.MX]
    # Optional mode-specific path constraints: g(opti, x, u, params)
    constraints_fn: Callable[[ca.Opti, ca.MX, ca.MX, dict[str, Any]], None] | None = None
    # Mode specific bounds
    x_lb: Sequence | None = None # Lower bound on state
    x_ub: Sequence | None = None # Upper bound on state
    u_lb: Sequence | None = None # Lower bound on control
    u_ub: Sequence | None = None # Upper bound on control
    # Mode specific initial guess
    initial_x_guess: npt.ArrayLike | None = None # Initial guess for state
    initial_u_guess: npt.ArrayLike | None = None # Initial guess for control
    # Mode specific other parameters
    other: dict[str, Any] = field(default_factory=dict) # For dynamics, costs etc.

    def __post_init__(self):
        if self.x_lb is None:
            self.x_lb = [None] * self.n_x
        if self.x_ub is None:
            self.x_ub = [None] * self.n_x
        if self.u_lb is None:
            self.u_lb = [None] * self.n_u
        if self.u_ub is None:
            self.u_ub = [None] * self.n_u
        if self.initial_x_guess is None:
            self.initial_x_guess = np.zeros((self.n_x, 1))
        else:
            self.initial_x_guess = np.asarray(self.initial_x_guess)
        if self.initial_u_guess is None:
            self.initial_u_guess = np.zeros((self.n_u, 1))
        else:
            self.initial_u_guess = np.asarray(self.initial_u_guess)
        self._validate()

    def _validate(self):
        """Basic validation of the mode parameters."""
        assert self.n_x > 0, "n_x must be positive."
        assert self.n_u >= 0, "n_u cannot be negative."
        assert len(self.x_lb) == self.n_x, "x_lb must match n_x dimension."
        assert len(self.x_ub) == self.n_x, "x_ub must match n_x dimension."
        assert len(self.u_lb) == self.n_u, "u_lb must match n_u dimension."
        assert len(self.u_ub) == self.n_u, "u_ub must match n_u dimension."
        assert self.initial_x_guess.shape[0] == self.n_x, "initial_x_guess must match n_x dimension."
        assert self.initial_u_guess.shape[0] == self.n_u, "initial_u_guess must match n_u dimension."

    def copy(self):
        """Creates a deep copy of the mode."""
        return Mode(
            name=self.name,
            n_x=self.n_x,
            n_u=self.n_u,
            dynamics_fn=self.dynamics_fn,
            constraints_fn=self.constraints_fn,
            x_lb=self.x_lb.copy(),
            x_ub=self.x_ub.copy(),
            u_lb=self.u_lb.copy(),
            u_ub=self.u_ub.copy(),
            initial_x_guess=self.initial_x_guess.copy(),
            initial_u_guess=self.initial_u_guess.copy(),
            other=self.other.copy()
        )


@dataclass
class Transition:
    """Defines a transition between two modes."""
    # Guard function: g(x, u, params) -> scalar_value (transition occurs when value == 0)
    guard_fn: Callable[[ca.MX, ca.MX, dict[str, Any]], ca.MX]
    # Reset map: r(x, u, params) -> x_plus (state immediately after transition)
    reset_map_fn: Callable[[ca.MX, ca.MX, dict[str, Any]], ca.MX]


@dataclass
class HybridSystemParams:
    """Parameters for the hybrid trajectory optimization problem."""
    mode_sequence: Sequence[Mode] # Sequence of modes
    transitions: Sequence[Transition] # List of transitions
    # --- Global parameters ---
    num_knot_points_per_segment: int # N: Number of intervals (N+1 points) per segment
    # --- Initial/Final conditions ---
    x0: Sequence # Initial state constraint
    xf: Sequence | None = None # Final state constraint (optional)
    # --- Time parameters ---
    T: float | None = None # Fixed total time T (if None, T is optimized)
    # --- Time Optimization (if T is None) ---
    T_guess: float = 1.0
    T_lb: float | None = 1e-4
    T_ub: float | None = None
    # --- Other user params ---
    other: dict[str, Any] = field(default_factory=dict) # For dynamics, costs etc.

    def __post_init__(self):
        if self.xf is None:
            self.xf = [None] * self.mode_sequence[-1].n_x
        self._validate()

    def _validate(self):
        """Basic validation of the parameters."""
        assert len(self.mode_sequence) > 0, "At least one mode must be in the sequence."
        assert len(self.transitions) == len(self.mode_sequence) - 1, "Number of transitions must be one less than number of modes."
        assert self.num_knot_points_per_segment > 0, "num_knot_points_per_segment must be positive."
        assert len(self.x0) == self.mode_sequence[0].n_x, "x0 must match n_x dimension of the first mode."
        # Add more checks as needed (e.g., bounds dimensions)


# --- The Optimizer Class ---

class HybridTrajectoryOptimizer:
    def __init__(self,
                 params: HybridSystemParams,
                 cost_fn: Callable[[ca.Opti, list[ca.MX], list[ca.MX], list[ca.MX], HybridSystemParams], ca.MX],
                 global_constraints_fn: Callable[[ca.Opti, list[ca.MX], list[ca.MX], list[ca.MX], HybridSystemParams], None] | None = None,
                 solver_name: str = 'ipopt',
                 solver_options: dict | None = None):
        """
        Initializes the hybrid trajectory optimizer.

        Args:
            params: An instance of HybridSystemParams defining the problem.
            cost_fn: Function to define the optimization objective.
                     Signature: cost_fn(opti, X_vars, U_vars, H_vars, params) -> cost_value
            global_constraints_fn: Optional function for global constraints.
                     Signature: global_constraints_fn(opti, X_vars, U_vars, H_vars, params)
            solver_name: Name of the NLP solver (default: 'ipopt').
            solver_options: Dictionary of options for the NLP solver.
        """
        self.params = params
        self.cost_fn = cost_fn
        self.global_constraints_fn = global_constraints_fn
        self.solver_name = solver_name
        self.solver_options = solver_options if solver_options is not None else {}

        self.opti = ca.Opti()
        self.num_modes = len(params.mode_sequence)
        self.N = params.num_knot_points_per_segment # Knot points per segment (N intervals)

        # --- Decision Variables ---
        self.X_vars: list[ca.MX] = [] # State variables [n_x, N+1] for each segment k
        self.U_vars: list[ca.MX] = [] # Control variables [n_u, N] for each segment k
        self.H_vars: list[ca.MX] = [] # Time steps [1, N] for each segment k
        self.T_var: ca.MX | None = None # Total time (if optimized)

        self._setup_variables()
        self._setup_constraints()
        self._setup_objective()


    def _setup_variables(self):
        """Define CasADi decision variables for the optimizer."""
        N = self.N

        # Time variable if T is not fixed
        if self.params.T is None:
            self.T_var = self.opti.variable(1)
            self.opti.set_initial(self.T_var, self.params.T_guess)
            if self.params.T_lb is not None:
                self.opti.subject_to(self.T_var >= self.params.T_lb) # type: ignore
            if self.params.T_ub is not None:
                self.opti.subject_to(self.T_var <= self.params.T_ub) # type: ignore
            # Guess segment durations equally, slightly less than total guess
            total_h_guess = self.params.T_guess * INITIAL_H_GUESS_FACTOR
            h_guess_val = total_h_guess / (self.num_modes * N)
        else:
            h_guess_val = self.params.T / (self.num_modes * N)

        # Variables for each segment
        for k in range(self.num_modes):
            n_x = self.params.mode_sequence[k].n_x
            n_u = self.params.mode_sequence[k].n_u
            # --- Decision Variables ---
            # States
            Xk = self.opti.variable(n_x, N + 1)
            self.X_vars.append(Xk)
            # Controls
            Uk = self.opti.variable(n_u, N)
            self.U_vars.append(Uk)
            # Time steps
            Hk = self.opti.variable(1, N)
            self.H_vars.append(Hk)
            self.opti.subject_to(Hk >= MIN_TIME_STEP) # Time steps must be positive
            self.opti.set_initial(Hk, h_guess_val)

            # --- Set initial guesses if provided ---
            init_x_guess = self.params.mode_sequence[k].initial_x_guess
            init_x_guess_stretched = stretch_array(init_x_guess, N + 1)
            self.opti.set_initial(Xk, init_x_guess_stretched)
            # Set initial guess for control
            init_u_guess = self.params.mode_sequence[k].initial_u_guess
            init_u_guess_stretched = stretch_array(init_u_guess, N)
            self.opti.set_initial(Uk, init_u_guess_stretched)


    def _setup_constraints(self):
        """Apply all constraints: dynamics, transitions, bounds, global, boundary."""
        N = self.N

        # --- Total Time Constraint ---
        total_h_sum = ca.sum2(ca.horzcat(*self.H_vars)) # Sum of all time steps
        if self.T_var is not None: # Variable time
            self.opti.subject_to(total_h_sum == self.T_var)
        else: # Fixed time
            self.opti.subject_to(total_h_sum == self.params.T)

        # --- Boundary Conditions ---
        # Initial state
        self.opti.subject_to(self.X_vars[0][:, 0] == self.params.x0)
        # Final state (optional)
        for i, xf in enumerate(self.params.xf):
            if xf is not None:
                self.opti.subject_to(self.X_vars[-1][i, -1] == xf)

        # --- Constraints per Segment ---
        for k in range(self.num_modes):
            mode = self.params.mode_sequence[k]
            Xk = self.X_vars[k]
            Uk = self.U_vars[k]
            Hk = self.H_vars[k]

            # --- 4a. Collocation Constraints (Dynamics within segment) ---
            # Midpoint collocation: x_{n+1} - x_n = h_n * f( (x_n+x_{n+1})/2, (u_n+u_{n+1})/2 )
            # NOTE: Assuming ZOH control within the interval for simplicity here.
            #       Midpoint collocation on control might be better: U_mid = 0.5*(Uk[:,n] + Uk[:,n+1])?
            #       Let's stick to ZOH U_k,n for now, as in the bouncing ball.
            for n in range(N):
                x_n = Xk[:, n]
                x_n_plus_1 = Xk[:, n+1]
                u_n = Uk[:, n] # Control assumed constant over interval n
                h_n = Hk[0, n]
                x_mid = 0.5 * (x_n + x_n_plus_1)
                f_mid = mode.dynamics_fn(x_mid, u_n, self.params.other)
                self.opti.subject_to(x_n_plus_1 - x_n == h_n * f_mid)

            # --- 4b. Mode-Specific Constraints ---
            if mode.constraints_fn:
                 # Apply to all points in the segment? Or just intermediate? Assume all for now.
                mode.constraints_fn(self.opti, Xk, Uk, self.params.other)

            # --- 4c. Transition Constraints (Guard and Reset) ---
            if k < self.num_modes - 1:
                transition = self.params.transitions[k]

                x_end_k = Xk[:, -1]
                u_end_k = Uk[:, -1] # Control at the end of the segment
                x_start_k_plus_1 = self.X_vars[k+1][:, 0]

                # Guard Constraint: guard_fn(x_end, u_end) == 0
                guard_value = transition.guard_fn(x_end_k, u_end_k, self.params.other)
                self.opti.subject_to(guard_value == 0)

                # Guard constraints: guard_fn must be positive for all other points
                for n in range(1, N):
                    x_n = Xk[:, n]
                    u_n = Uk[:, n] 
                    guard_value = transition.guard_fn(x_n, u_n, self.params.other)
                    self.opti.subject_to(guard_value > 0)

                # Reset Map: x_start_{k+1} = reset_map(x_end_k, u_end_k)
                reset_value = transition.reset_map_fn(x_end_k, u_end_k, self.params.other)
                self.opti.subject_to(x_start_k_plus_1 == reset_value)

            # --- Apply Box Constraints ---
            # State bounds
            for i in range(mode.n_x):
                if mode.x_lb[i] is not None:
                    self.opti.subject_to(Xk[i, :] >= mode.x_lb[i])
                if mode.x_ub[i] is not None:
                    self.opti.subject_to(Xk[i, :] <= mode.x_ub[i])
            # Control bounds
            for i in range(mode.n_u):
                if mode.u_lb[i] is not None:
                    self.opti.subject_to(Uk[i, :] >= mode.u_lb[i])
                if mode.u_ub[i] is not None:
                    self.opti.subject_to(Uk[i, :] <= mode.u_ub[i])

        # --- Apply Optional Global Constraints ---
        if self.global_constraints_fn:
             self.global_constraints_fn(self.opti, self.X_vars, self.U_vars, self.H_vars, self.params)


    def _setup_objective(self):
        """Define the objective function."""
        cost = self.cost_fn(self.opti, self.X_vars, self.U_vars, self.H_vars, self.params)
        self.opti.minimize(cost)


    def solve(self):
        """Solve the optimization problem."""
        print(f"Solving hybrid trajectory optimization with {self.num_modes} segments...")
        # Set solver options
        p_opts = {"expand": True} # Needed for some CasADi functions inside constraints
        s_opts = {"max_iter": 3000, "print_level": 5} # Default IPOPT options
        s_opts.update(self.solver_options) # Overwrite with user options

        self.opti.solver(self.solver_name, p_opts, s_opts)

        try:
            sol = self.opti.solve()
            print("Optimization SUCCESSFUL!")

        except Exception as e:
            print(f"Optimization FAILED. Error: {e}")
            # You might be able to get partial results from opti.debug.value(var)
            # Or check opti.debug.stats()
            return {
                "success": False,
                "times": None,
                "states": None,
                "controls": None,
                "timesteps": None,
                "T": None,
                "optimizer": self.opti
            }
        # --- Extract Results ---
        res_x = []
        res_u = []
        res_h = []
        for k in range(self.num_modes):
            n_x_k = self.params.mode_sequence[k].n_x
            n_u_k = self.params.mode_sequence[k].n_u
            res_x.append(sol.value(self.X_vars[k]).reshape((n_x_k, -1)))
            res_u.append(sol.value(self.U_vars[k]).reshape((n_u_k, -1)))
            res_h.append(sol.value(self.H_vars[k]).reshape((1, -1)))
        res_t = self.params.T # Fixed time
        if self.T_var is not None:
            res_t = sol.value(self.T_var) # Optimized time

        # Reconstruct continuous time and state/control trajectories
        times_list = []
        current_time = 0.0

        for k in range(self.num_modes):
            h_k = res_h[k]
            times_k = np.cumsum(h_k) + current_time
            times_k = np.insert(times_k, 0, current_time) # Add start time
            times_list.append(times_k)
            current_time = times_k[-1] # Update current time for next segment

        print(f"  Final Time: {times_list[-1][-1]:.4f} (Target: {res_t:.4f})")
        print(f"  Final State: {res_x[-1][:, -1].flatten()}")
        if self.T_var is not None:
                print(f"  Optimized Total Time T: {res_t:.4f}")

        return {
            "success": True,
            "times": times_list,
            "states": res_x,
            "controls": res_u,
            "timesteps": res_h,
            "T": res_t,
            "optimizer": self.opti,
        }



# --- Simple Example: Switching Linear System ---
if __name__ == "__main__":

    # --- Define the Hybrid System Components ---

    # Mode 1: Simple integrator, drifts right
    def dynamics_mode1(x, u, params):
        # x = [p], u = [v]
        return u[0] + 0.1 # Drift term

    # Mode 2: Simple integrator, drifts left
    def dynamics_mode2(x, u, params):
        # x = [p], u = [v]
        return u[0] - 0.1 # Drift term

    # Constraint for Mode 2: Keep position positive
    def constraints_mode2(opti, Xk, Uk, params):
        opti.subject_to(Xk[0, :] >= -0.1) # Allow slight negative for numerics

    # Transition from Mode 1 to Mode 2 when x > 1.0
    def guard_1_to_2(x, u, params):
        return 1.0 - x[0] # Trigger when x[0] == 1.0

    # Transition from Mode 2 to Mode 1 when x < 0.5
    def guard_2_to_1(x, u, params):
        return x[0] - 0.5 # Trigger when x[0] == 0.5

    # Reset map (identity map in this case)
    def reset_identity(x, u, params):
        return x

    # --- Define Modes and Transitions ---
    n_x = 1 # State dimension
    n_u = 1 # Control dimension
    u_lb = -1.0
    u_ub = 1.0
    mode1 = Mode(name="DriftRight", n_x=n_x, n_u=n_u,
                 dynamics_fn=dynamics_mode1,
                 u_lb=[u_lb], u_ub=[u_ub],)
    mode2 = Mode(name="DriftLeft", n_x=n_x, n_u=n_u,
                 dynamics_fn=dynamics_mode2,
                 constraints_fn=constraints_mode2,
                 u_lb=[u_lb], u_ub=[u_ub],)
    trans12 = Transition(guard_fn=guard_1_to_2, reset_map_fn=reset_identity)
    trans21 = Transition(guard_fn=guard_2_to_1, reset_map_fn=reset_identity)

    # --- Setup Parameters ---
    params = HybridSystemParams(
        mode_sequence=[mode1, mode2, mode1],
        transitions=[trans12, trans21],
        num_knot_points_per_segment=20, # N
        T=5.0, # Fixed total time
        x0=[0.0],      # Start at position 0
        xf=[1.5],      # End at position 1.5
    )

    # --- Cost Function: Minimize control effort ---
    def cost_function(opti, X_vars, U_vars, H_vars, params):
        total_cost = 0
        for k in range(len(U_vars)):
            Uk = U_vars[k]
            # Integrate squared controls over the segment
            # Using sum (Uk^2 * Hk) as approximation
            # total_cost += ca.sumsqr(Uk) * ca.sum(Hk) / params.num_knot_points_per_segment # Approximate integral
            # Or just sumsqr(Uk) without time scaling? Let's do simple sumsqr
            total_cost += ca.sumsqr(Uk)

        return total_cost
    
    def cost_function_minimum_time(opti, X_vars, U_vars, H_vars, params):
        total_time = 0
        for k in range(len(H_vars)):
            Hk = H_vars[k]
            # Integrate time over the segment
            total_time += ca.sum(Hk)
        return total_time

    # --- Create and Solve ---
    optimizer = HybridTrajectoryOptimizer(params, cost_fn=cost_function)
    results = optimizer.solve()

    # --- Plotting ---
    if results["success"]:
        times = results["times"]
        states = results["states"]
        controls = results["controls"]
        num_modes = len(params.mode_sequence)
        N = params.num_knot_points_per_segment

        plt.figure(figsize=(12, 8))

        # Plot state
        plt.subplot(2, 1, 1)
        for k in range(num_modes):
            plt.plot(times[k], states[k][0, :], 'b.-', label=f'Mode {k+1} x(t)')
        # Mark segment boundaries and modes
        for k in range(num_modes):
            t_start = times[k][0]
            # Plot vertical line at transition (except start)
            if k > 0:
                 plt.axvline(t_start, color='r', linestyle='--', linewidth=0.8)


        # Plot target state
        if params.xf[0] is not None:
            plt.plot(times[-1][-1], params.xf[0], 'go', markersize=10, label='Target State')
        # Plot initial state
        plt.plot(times[0][0], params.x0[0], 'ro', markersize=10, label='Start State')

        plt.title("Hybrid System Trajectory")
        plt.ylabel("State x")
        plt.legend()
        plt.grid(True)

        # Plot control
        plt.subplot(2, 1, 2)
        # Use stairs for ZOH control - need time points matching control intervals
        for k in range(num_modes):
            plt.stairs(controls[k][0, :], times[k], baseline=None, color='orange', label=f'Mode {k+1} u(t)', lw=2)

        # Mark segment boundaries
        for k in range(num_modes):
            t_start = times[k][0]
            # Plot vertical line at transition (except start)
            if k > 0:
                 plt.axvline(t_start, color='r', linestyle='--', linewidth=0.8)

        # Plot control bounds
        plt.axhline(u_lb, color='g', linestyle=':', label='u_lb')
        plt.axhline(u_ub, color='g', linestyle=':', label='u_ub')

        plt.ylabel("Control u")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.xlim(times[0][0], times[-1][-1])

        plt.tight_layout()
        plt.show()