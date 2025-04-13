import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def optimize_bouncing_ball_casadi(
    num_bounces: int,
    total_time: float,
    target_pos: float,
    target_vel: float,
    initial_pos_guess: float = 1.0,
    initial_vel_guess: float = 0.0,
    gravity: float = 9.81,
    restitution_coeff: float = 0.8,
    num_knot_points_per_segment: int = 10, # N: knots excluding start
):
    """
    Optimizes the initial state (y0, y_dot0) of a bouncing ball using CasADi
    to reach a target state (yf, y_dotf) in a fixed total time T,
    with a specified number of bounces.

    Args:
        num_bounces: The exact number of bounces (K).
        total_time: The total duration of the trajectory (T).
        target_pos: The final desired position (yf).
        target_vel: The final desired velocity (y_dotf).
        initial_pos_guess: Initial guess for y0.
        initial_vel_guess: Initial guess for y_dot0.
        gravity: Acceleration due to gravity (g).
        restitution_coeff: Coefficient of restitution (e).
        num_knot_points_per_segment: Number of knot points N (incl start/end).
                                     N-1 intervals per segment.

    Returns:
        A tuple: (success, optimized_x0, times, states) or (False, None, None, None)
    """
    opti = ca.Opti()

    # --- System Dynamics ---
    # State x = [y, y_dot]
    def continuous_dynamics(x):
        # Need to ensure x is a CasADi type if called within opti context
        return ca.vertcat(x[1], -gravity)

    num_segments = num_bounces + 1 # K+1 flight segments
    N = num_knot_points_per_segment # Short name for clarity

    # --- Decision Variables ---
    # Initial state
    x0_var = opti.variable(2, 1)
    opti.set_initial(x0_var, [initial_pos_guess, initial_vel_guess])

    # Time steps for each interval within each segment
    h_vars = [] # List of CasADi vars for dt in each segment
    for k in range(num_segments):
        # N knot points => N-1 intervals
        h_k = opti.variable(1, N)
        opti.set_initial(h_k, total_time / (num_segments * N))
        opti.subject_to(h_k > 0) # Time steps must be positive
        h_vars.append(h_k)

    # State variables at each knot point for all segments
    x_vars = [] # List of CasADi vars [y, y_dot] for each segment
    for k in range(num_segments):
        x_k = opti.variable(2, N + 1)
        # Simple linear guess for states
        y_guess = np.linspace(initial_pos_guess if k==0 else 0,
                              target_pos if k==num_segments-1 else 0,
                              N + 1)
        ydot_guess = np.zeros(N + 1) # Could use initial_vel_guess etc.
        opti.set_initial(x_k, np.vstack([y_guess, ydot_guess]))
        x_vars.append(x_k)

    # --- Constraints ---
    # 1. Total Time Constraint
    total_h = ca.vertcat(*h_vars)
    opti.subject_to(ca.sum(total_h) == total_time)

    # 2. Link Initial State Decision Variable
    opti.subject_to(x_vars[0][:, 0] == x0_var)

    # 3. Final State Constraint
    if target_pos is not None:
        opti.subject_to(x_vars[-1][0, -1] == target_pos)
    if target_vel is not None:
        opti.subject_to(x_vars[-1][1, -1] == target_vel)

    # 4. Dynamics, Non-Penetration, Guard, Reset Constraints (Segment by Segment)
    for k in range(num_segments):
        h_k = h_vars[k] # Time steps for segment k (N-1 of them)
        x_k = x_vars[k] # States for segment k (N knot points)

        # 4a. Collocation Constraints (Dynamics within segment)
        # Midpoint collocation: enforces x_n+1 - x_n = h_n * f( (x_n+x_n+1)/2 )
        for n in range(N):
            x_n = x_k[:, n]
            x_n_plus_1 = x_k[:, n+1]
            h_n = h_k[n]
            x_mid = 0.5 * (x_n + x_n_plus_1)
            f_mid = continuous_dynamics(x_mid)
            opti.subject_to(x_n_plus_1 - x_n == h_n * f_mid)

        # 4b. Guard Constraint (y > 0 for intermediate points)
        # Apply to all points except the very last point of segments that hit the ground (k < K)
        for n in range(N):
            opti.subject_to(x_k[0, n] > 0)

        # 4c. Guard and Reset Constraints (Linking segments k and k+1)
        if k < num_bounces: # If not the last segment, it must end in a bounce
            # Guard: Position y=0 at the end of segment k
            opti.subject_to(x_k[0, -1] == 0) # y_end = 0

            # Reset: Link state after impact (start of k+1) to state before (end of k)
            x_next_start = x_vars[k+1][:, 0]
            x_curr_end = x_k[:, -1]
            # y_start(k+1) = y_end(k) (=0 implied by guard)
            opti.subject_to(x_next_start[0] == x_curr_end[0])
            # y_dot_start(k+1) = -e * y_dot_end(k)
            opti.subject_to(x_next_start[1] == -restitution_coeff * x_curr_end[1])

    # --- Objective Function ---
    # Feasibility problem (no objective), or minimize initial speed etc.
    # opti.minimize(0)
    opti.minimize(ca.sumsqr(x0_var[1])) # Minimize initial velocity squared

    # --- Solve ---
    print(f"Solving for K={num_bounces} bounces using CasADi...")
    # Set solver options (optional)
    p_opts = {"expand": True} # Expand symbolic expressions if needed
    s_opts = {"max_iter": 3000, "print_level": 0} # Suppress IPOPT output (set >0 for details)
                                                 # Increase max_iter if needed
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        print("Optimization SUCCESSFUL!")
    except Exception as e:
        print(f"Optimization FAILED for K={num_bounces}. Error: {e}")
        # You might be able to get partial results from opti.debug.value(var)
        # Or check opti.debug.stats()
        return False, None, None, None


    # --- Process Results ---
    optimized_x0 = sol.value(x0_var)
    print(f"  Optimized Initial State x0 = {optimized_x0.flatten()}")

    # Reconstruct time and state arrays
    times_list = [0.0]
    states_list = [sol.value(x_vars[0][:, 0])] # Initial state
    current_time = 0.0

    for k in range(num_segments):
        h_sol_k = sol.value(h_vars[k])
        x_sol_k = sol.value(x_vars[k])
        for n in range(N):
            current_time += h_sol_k[n]
            times_list.append(current_time)
            states_list.append(x_sol_k[:, n+1]) # State at end of interval n

    times = np.array(times_list)
    states = np.array(states_list).T # Shape (2, num_total_knots)

    print(f"  Achieved Total Time: {times[-1]:.4f} (Target: {total_time})")
    print(f"  Final State: {states[:,-1]}")

    return True, optimized_x0.flatten(), times, states


# --- Example Usage ---
if __name__ == "__main__":
    # Problem Parameters (same as Drake example)
    K = 2 # Number of bounces to try
    T = 2.0 # Total time
    Y_TARGET = 0.5 # Target final height
    Y_DOT_TARGET = None # Target final velocity
    G = 9.81
    E = 0.8 # Coefficient of restitution
    N_KNOTS_PER_SEG = 30 # Resolution between bounces (including start/end points)
    Y0_GUESS = 2.0 # Initial guess for starting height
    YDOT0_GUESS = 0.0 # Initial guess for starting velocity

    success, opt_x0, times_sol, states_sol = optimize_bouncing_ball_casadi(
        num_bounces=K,
        total_time=T,
        target_pos=Y_TARGET,
        target_vel=Y_DOT_TARGET,
        initial_pos_guess=Y0_GUESS,
        initial_vel_guess=YDOT0_GUESS,
        gravity=G,
        restitution_coeff=E,
        num_knot_points_per_segment=N_KNOTS_PER_SEG
    )

    # --- Plotting ---
    if success:
        plt.figure(figsize=(10, 6))

        # Plot trajectory using knot points
        plt.plot(times_sol, states_sol[0, :], '.-', label='Position y(t)')
        plt.plot(times_sol, states_sol[1, :], '.-', label='Velocity y_dot(t)')

        # Mark target
        if Y_TARGET is not None:
            plt.plot(T, Y_TARGET, 'ro', markersize=10, label='Target State')
        if Y_DOT_TARGET is not None:
            plt.plot(T, Y_DOT_TARGET,'r*', markersize=8) # Velocity target marker

        # Mark start
        plt.plot(0, opt_x0[0], 'go', markersize=10, label='Optimized Start State')
        plt.plot(0, opt_x0[1],'g*', markersize=8) # Velocity start marker

        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.title(f"Bouncing Ball Trajectory Optimization (CasADi, K={K} bounces)")
        plt.legend()
        plt.grid(True)
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8) # Ground
        plt.ylim(bottom=min(np.min(states_sol[0,:])-0.5, np.min(states_sol[1,:])*1.1),
                 top=max(np.max(states_sol[0,:])+0.5, np.max(states_sol[1,:])*1.1)) # Adjust ylim
        plt.show()