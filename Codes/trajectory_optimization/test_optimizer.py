# test_optimizer.py

import pytest
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt # Optional: for debugging plots during test development
# Assuming your optimizer class is in optimizer.py
from optimizer import TrajectoryOptimizer, TrajectoryOptimizerParams

# ---- System Definition: ----
def double_integrator_dynamics(x, u, params):
    """ x = [pos, vel], u = [accel] -> dx/dt = [vel, accel] """
    return ca.vertcat(x[1], u[0])

def dummy_dynamics(x, u, params):
    """ Dummy dynamics for testing purposes """
    return ca.vertcat(x[0], u[0])

# ---- Cost Functions ----
def cost_min_u_squared(opti, X, U, params):
    """Minimize integral(u^2 dt) approx sum(u^2 * dt)"""
    T = params.T # Total time
    N = X.shape[1] - 1 # Number of intervals from state array
    dt = T / N
    # Integrate U^2 over the N intervals
    return ca.sumsqr(U) * dt

# ---- Custom BC Function ----
def custom_bc_func(opti, X, U, params):
    """ Example custom BC: start=[0,0], end=[2,0] """
    opti.subject_to(X[0, 0] == 0.0) # Initial position
    opti.subject_to(X[1, 0] == 0.0) # Initial velocity
    opti.subject_to(X[0, -1] == 2.0) # Final position
    opti.subject_to(X[1, -1] == 0.0) # Final velocity (at rest)

# ---- Pytest Fixture for Default Params ----
@pytest.fixture
def default_params():
    """Provides default parameters for tests, easily modifiable."""
    N_test = 20 # Number of intervals for tests (keep reasonably small)
    T_test = 2.0
    params = {
        'T': T_test,
        # 'N': N_test, # N is derived from T/dt in fixed time version
        'dt_x': 0.1, # Calculate dt based on desired N for test
        'dt_u': 0.1, # Assume control updates each step for simplicity here
        'n_x': 2,
        'n_u': 1,
        # Add other base params if needed by system/cost funcs
    }
    return TrajectoryOptimizerParams(**params)

# ---- Test Suite ----
def test_basic_one_input_two_states(default_params):
    """Tests basic object creation and attribute setup."""
    print("\nTesting Basic Object Creation")
    params = default_params
    # Create a TrajectoryOptimizer object with default params
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        # No custom BCs or constraints for this basic test
    )
    optimizer.setup()  # Setup the optimizer
    # Check if the optimizer is set up correctly
    assert optimizer.opti is not None, "Optimizer not set up correctly"
    assert optimizer.dynamics_fn is not None, "Dynamics function not set correctly"
    assert optimizer.cost_fn is not None, "Cost function not set correctly"
    assert optimizer.params is not None, "Parameters not set correctly"

    x_opt, u_opt = optimizer.solve()  # Solve the optimization problem
    # Check the solution dimensions
    assert x_opt.shape == (params.n_x, params.N + 1), "x_opt shape mismatch"
    assert u_opt.shape == (params.n_u, params.N), "u_opt shape mismatch"

def test_double_integrator_different_rates(default_params):
    """Test double integrator with different state/control update rates."""
    params = default_params
    params.x0 = [0.0, 0.0]
    params.xf = [1.0, 0.0]
    params.T = 2.0 # Time horizon
    params.dt_x = 0.01 # State update interval
    params.dt_u = 0.05 # Control update interval

    print("\nTesting Double Integrator (Double Rate)")
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        # No custom BCs or constraints for this test
    )
    optimizer.setup()
    solver_opts = {"print_level": 0, "tol": 1e-6} # Suppress IPOPT output, set tolerance
    x_opt, u_opt = optimizer.solve(s_opts=solver_opts)

    # Assertions
    assert np.allclose(x_opt[:, 0], params.x0, atol=1e-5), "Initial condition mismatch"
    assert np.allclose(x_opt[:, -1], params.xf, atol=1e-5), "Final condition mismatch"
    # Check if control input remains constant due to the ZOH
    assert np.allclose(u_opt[:, 0], u_opt[:, 1], atol=1e-5), "Control input not constant"


def test_goto_defaults_min_effort(default_params):
    """Test reaching xf=[1,0] from x0=[0,0] using fallbacks and min U^2."""
    params = default_params
    params.x0 = [0.0, 0.0]
    params.xf = [1.0, 0.0]
    # Use default 'trapezoidal' integration

    print("\nTesting Go-to-Point (Defaults, Min Effort)")
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        # Using default fallback for BCs (x0, xf in params)
        # Using default fallback for constraints (none defined)
        # Using default fallback for initial guess (x0, xf -> linspace)
    )
    optimizer.setup()
    # Add reasonable solver options for tests
    solver_opts = {"print_level": 0, "tol": 1e-6} # Suppress IPOPT output, set tolerance
    x_opt, u_opt = optimizer.solve(s_opts=solver_opts)

    # Assertions
    print(f"  Initial state: {x_opt[:, 0]}")
    print(f"  Final state: {x_opt[:, -1]}")
    assert np.allclose(x_opt[:, 0], params.x0, atol=1e-5), "Initial condition mismatch"
    assert np.allclose(x_opt[:, -1], params.xf, atol=1e-5), "Final condition mismatch"
    # Optional: Check if cost is reasonable (e.g., non-negative)
    final_cost = cost_min_u_squared(optimizer.opti, x_opt, u_opt, params)
    print(f"  Final cost (U^2): {final_cost}")
    assert final_cost >= 0

def test_goto_control_bounds(default_params):
    """Test reaching xf with control bounds using fallback constraints."""
    params = default_params
    params.x0 = [0.0, 0.0]
    params.xf = [1.0, 0.0]
    params.u_lb = [-0.8,] # Control bounds (as tuple/list length nu)
    params.u_ub = [0.8,]
    params.T = 3.0 # Time horizon

    print("\nTesting Go-to-Point (Control Bounds)")
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        constraints_fn=None, # Explicitly use box constraint fallback
    )
    optimizer.setup()
    # Assertions related to optimizer setup
    assert optimizer.params.T == 3.0, "Time horizon mismatch"
    assert optimizer.params.N == 30, "Number of intervals mismatch" # Based on dt_u and T
    assert params.N == 30, "Number of intervals mismatch" # Based on dt_u and T
    solver_opts = {"print_level": 0, "tol": 1e-6}
    x_opt, u_opt = optimizer.solve(s_opts=solver_opts)

    # Assertions
    print(f"  Final state: {x_opt[:, -1]}")
    print(f"  Min/Max control: {np.min(u_opt):.3f} / {np.max(u_opt):.3f}")
    assert np.allclose(x_opt[:, 0], params.x0, atol=1e-5)
    assert np.allclose(x_opt[:, -1], params.xf, atol=1e-5)
    # Check bounds (allow small tolerance for solver precision)
    assert np.all(u_opt[0, :] >= params.u_lb[0] - 1e-6), "Control violated lower bound"
    assert np.all(u_opt[0, :] <= params.u_ub[0] + 1e-6), "Control violated upper bound"
    # Check if bounds seem active (optional, might not always happen)
    # assert np.isclose(np.max(np.abs(u_opt[0, :])), params['u_ub'][0], atol=1e-3), "Control bounds may not be active"

def test_goto_state_bounds(default_params):
    """Test reaching xf with state bounds using fallback constraints."""
    params = default_params
    params.x0 = [0.0, 0.0]
    params.xf = [2.0, 0.0]
    params.x_ub = [None, 1.0] # State bounds (as tuple/list length nx)
    params.T = 3.0 # Time horizon

    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        constraints_fn=None, # Explicitly use box constraint fallback
    )
    optimizer.setup()
    solver_opts = {"print_level": 0, "tol": 1e-6}
    x_opt, u_opt = optimizer.solve(s_opts=solver_opts)

    # Assertions
    print(f"  Final state: {x_opt[:, -1]}")
    print(f"  Min/Max state: {np.min(x_opt[1, :]):.3f} / {np.max(x_opt[1, :]):.3f}")
    assert np.allclose(x_opt[:, 0], params.x0, atol=1e-5)
    assert np.allclose(x_opt[:, -1], params.xf, atol=1e-5)
    # Check bounds (allow small tolerance for solver precision)
    assert np.all(x_opt[1, :] <= params.x_ub[1] + 1e-6), "State violated upper bound"
    # Check if bounds seem active (optional, might not always happen)
    assert np.isclose(np.max(np.abs(x_opt[1, :])), params.x_ub[1], atol=1e-3), "State bounds may not be active"

def test_custom_bc_function(default_params):
    """Test custom boundary conditions."""
    params = default_params
    params.x0 = [0.0, 0.0] # This will be overridden by custom BCs
    params.xf = [1.0, 0.0] # This will be overridden by custom BCs
    params.T = 3.0 # Time horizon

    print("\nTesting Custom Boundary Conditions")
    optimizer = TrajectoryOptimizer(
        dynamics_fn=double_integrator_dynamics,
        cost_fn=cost_min_u_squared,
        params=params,
        boundary_conditions_fn=custom_bc_func, # Use custom BC function
    )
    optimizer.setup()
    solver_opts = {"print_level": 0, "tol": 1e-6}
    x_opt, u_opt = optimizer.solve(s_opts=solver_opts)

    # Assertions
    print(f"  Initial state: {x_opt[:, 0]}")
    print(f"  Final state: {x_opt[:, -1]}")
    assert np.allclose(x_opt[:, 0], 0, atol=1e-5), "Initial condition mismatch"
    assert np.isclose(x_opt[0, -1], 2.0, atol=1e-5), "Final condition mismatch" # Custom BCs set to 2.0
    assert np.isclose(x_opt[1, -1], 0, atol=1e-5), "Final condition mismatch" # Custom BCs set to 2.0

