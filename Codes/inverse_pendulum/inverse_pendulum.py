import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import collections

# --- 1. System Parameters ---
# Assuming thin rods
RodParams = collections.namedtuple('RodParams', ['m', 'l'])
Params = collections.namedtuple('Params', ['rod1', 'rod2', 'g'])
State = collections.namedtuple('State', ['theta1', 'theta2', 'theta1_dot', 'theta2_dot'])
rod1 = RodParams(m=1.0, l=1.0)  # Mass (kg) and length (m) of rod 1
rod2 = RodParams(m=1.0, l=1.0)  # Mass (kg) and length (m) of rod 2
params = Params(rod1=rod1, rod2=rod2, g=9.81)  # Gravitational acceleration (m/s^2)

# --- 2. Torque Input Functions ---
# Define how torques t1 and t2 behave.
# They are functions of time 't' and state 'x'
# Example: Constant torques (replace with controllers or other functions as needed)
def get_torque1(t, x):
    """External torque on rod 1."""
    return 0.0 # N*m

def get_torque2(t, x):
    """Internal motor torque between rod 1 and rod 2."""
    # Example: No motor torque
    # return 0.0 # N*m
    # --- Example: Simple stabilization torque trying to keep rod 2 vertical ---
    theta1, theta2, theta1_dot, theta2_dot = x
    kp = 150.0 # Proportional gain
    kd = 20.0  # Derivative gain
    target_theta2 = 0.0 # Target angle (vertical upward)
    error = target_theta2 - theta2
    # Torque applied *by motor onto rod 2* to correct error
    motor_torque = kp * error - kd * theta2_dot
    return motor_torque
    # ---

def get_torque(t, x):
    """Returns the torques for both rods as a column vector."""
    t1 = get_torque1(t, x)
    t2 = get_torque2(t, x)
    return t1, t2

# --- 3. ODE Function ---
def pendulum_ode(t, x, params, torque_func):
    """
    Calculates the derivative of the state vector for the double pendulum.
    State x = [theta1, theta1_dot
    Returns dx/dt = [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]
    """

    # Get current torque values
    t1, t2 = torque_func(t, x)

    # Unpack state variables
    theta1, theta2, theta1_dot, theta2_dot = x

    m1 = params.rod1.m  # Mass of rod 1
    l1 = params.rod1.l  # Length of rod 1
    m2 = params.rod2.m  # Mass of rod 2
    l2 = params.rod2.l  # Length of rod 2
    g = params.g        # Gravitational acceleration

    # Pre-calculate trigonometric terms for efficiency
    c12 = np.cos(theta1 - theta2)
    s12 = np.sin(theta1 - theta2)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)

    # --- Assemble the Matrix Equation M * theta_ddot = N ---

    # Mass Matrix M(theta)
    M11 = (m1/3 + m2) * l1**2
    M12 = (1/2) * m2 * l1 * l2 * c12
    M21 = M12 # Symmetric
    M22 = (1/3) * m2 * l2**2
    M = np.array([[M11, M12], [M21, M22]])

    # Right-hand side vector N(theta, theta_dot, torques)
    # From Eq1: (m1/3 + m2) l1^2 th1_dd + (1/2)m2 l1 l2 c12 th2_dd = t1 - t2 - (1/2)m2 l1 l2 s12 th2_d^2 + (m1/2 + m2) l1 g s1
    N1 = (t1 - t2) - (1/2)*m2*l1*l2*s12*theta2_dot**2 + (m1/2 + m2)*l1*g*s1

    # From Eq2: (1/2) m2 l1 l2 c12 th1_dd + (1/3) m2 l2^2 th2_dd = t2 - (- (1/2) m2 l1 l2 s12 th1_d^2) + (1/2) m2 g l2 s2
    N2 = t2 + (1/2)*m2*l1*l2*s12*theta1_dot**2 + (1/2)*m2*g*l2*s2

    N = np.array([N1, N2])

    # Solve for angular accelerations: theta_ddot = M^-1 * N
    try:
        # Using solve is generally better than calculating inverse explicitly
        theta_ddot = np.linalg.solve(M, N)
    except np.linalg.LinAlgError:
        # Handle cases where M might become singular (shouldn't happen with l > 0)
        print(f"Warning: Singular matrix encountered at t={t:.2f}. Setting accelerations to zero.")
        theta_ddot = np.zeros(2)

    theta1_ddot = theta_ddot[0]
    theta2_ddot = theta_ddot[1]

    # Return the derivative of the state vector
    dxdt = [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]
    return dxdt

# --- 4. Simulation Setup ---
# Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
# Start slightly perturbed from the unstable upright equilibrium
theta1_0 = 0  # radians (~5.7 degrees)
theta1_dot_0 = 0.0 # rad/s
theta2_0 = 0.1  # radians (~5.7 degrees)
theta2_dot_0 = 0.0 # rad/s
x0 = [theta1_0, theta2_0, theta1_dot_0, theta2_dot_0]  # Initial state vector

# Time span for the simulation
t_start = 0.0
t_end = 10.0 # seconds
t_span = (t_start, t_end)

# Points in time where the solution is saved (for plotting)
t_eval = np.linspace(t_start, t_end, 500)

# --- 5. Run Simulation ---
print("Running simulation...")
sol = solve_ivp(
    pendulum_ode,
    t_span,
    x0,
    method='RK45',  # Standard Runge-Kutta method, good general choice
    t_eval=t_eval,
    args=(params, get_torque),  # Pass parameters and torque function
    dense_output=True # Allows interpolation if needed
)
print("Simulation complete.")

# --- 6. Plot Results ---
if sol.success:
    t = sol.t
    x_sim = sol.y # Shape (4, num_time_points)

    theta1_sim = x_sim[0, :]
    theta2_sim = x_sim[1, :]
    theta1_dot_sim = x_sim[2, :]
    theta2_dot_sim = x_sim[3, :]

    plt.figure(figsize=(12, 9))

    # Plot angles
    plt.subplot(2, 1, 1)
    plt.plot(t, np.rad2deg(theta1_sim), label=r'$\theta_1$ (Rod 1)')
    plt.plot(t, np.rad2deg(theta2_sim), label=r'$\theta_2$ (Rod 2)', linestyle='--')
    plt.title('Inverted Double Pendulum Simulation (Thin Rods)')
    plt.ylabel('Angle (degrees from vertical)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot angular velocities
    plt.subplot(2, 1, 2)
    plt.plot(t, theta1_dot_sim, label=r'$\dot{\theta}_1$')
    plt.plot(t, theta2_dot_sim, label=r'$\dot{\theta}_2$', linestyle='--')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust spacing between plots
    plt.show()

    # Optional: Plot torques if they are dynamic
    t1_sim = [get_torque1(ti, x_sim[:, i]) for i, ti in enumerate(t)]
    t2_sim = [get_torque2(ti, x_sim[:, i]) for i, ti in enumerate(t)]
    plt.figure()
    plt.plot(t, t1_sim, label='Torque t1 (External)')
    plt.plot(t, t2_sim, label='Torque t2 (Internal Motor)', linestyle='--')
    plt.title('Input Torques Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N*m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# (Your existing simulation code from before should be here...)
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# import collections
# ... (parameters, torque functions, ODE function) ...
# ... (simulation setup and run) ...
# Ensure 'sol', 'params' are available from the simulation run

# --- 7. Animation ---
if sol.success:
    print("Preparing animation...")
    import matplotlib.animation as animation

    t = sol.t
    x_sim = sol.y # Shape (4, num_time_points) as [theta1, theta2, theta1_dot, theta2_dot]

    # Extract angles and parameters needed for plotting
    theta1_sim = x_sim[0, :]
    theta2_sim = x_sim[1, :]
    l1 = params.rod1.l
    l2 = params.rod2.l

    # --- Animation Setup ---
    fig = plt.figure(figsize=(7, 7)) # Adjust figure size for better aspect ratio
    ax = fig.add_subplot(111, autoscale_on=False)

    # Determine plot limits: needs to accommodate full swing + pivot
    max_reach = l1 + l2
    ax.set_xlim(-max_reach * 1.1, max_reach * 1.1)
    # Make sure y-limits include downward swing if it happens, plus upward reach
    ax.set_ylim(-max_reach * 1.1, max_reach * 1.1)
    ax.set_aspect('equal') # Crucial for correct proportions
    ax.grid(True)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Double Pendulum Animation')

    # Define plot elements to be updated: line segments for rods
    # Using 'o-' style to show joints/ends
    line1, = ax.plot([], [], 'o-', lw=2, color='blue', markersize=6, label='Rod 1') # Pivot P1 to P2
    line2, = ax.plot([], [], 'o-', lw=2, color='red', markersize=6, label='Rod 2')  # Pivot P2 to end
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes) # Position text in axes coordinates
    ax.legend(loc='lower left')

    # Initialization function (draws a blank frame)
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line1, line2, time_text

    # Update function (called for each animation frame)
    def update(i):
        # Calculate coordinates for the pendulum at simulation time step 'i'
        theta1 = theta1_sim[i]
        theta2 = theta2_sim[i]

        # Pivot P1 (origin)
        x_p1, y_p1 = 0, 0

        # Pivot P2 (end of rod 1)
        # Remember: theta measured from UPWARD vertical
        x_p2 = l1 * np.sin(theta1)
        y_p2 = l1 * np.cos(theta1)

        # End of rod 2
        x_e2 = x_p2 + l2 * np.sin(theta2)
        y_e2 = y_p2 + l2 * np.cos(theta2)

        # Update the line data for the rods
        line1.set_data([x_p1, x_p2], [y_p1, y_p2]) # Rod 1 from P1 to P2
        line2.set_data([x_p2, x_e2], [y_p2, y_e2]) # Rod 2 from P2 to End

        # Update the time display
        time_text.set_text(time_template % t[i])

        # Return the plot elements that have changed
        return line1, line2, time_text

    # --- Create and Display Animation ---
    # Decide which simulation frames to use for animation
    # Using a step > 1 makes the animation faster but less smooth
    frame_step = 2 # Use every 2nd simulation frame
    frame_indices = range(0, len(t), frame_step)

    # Calculate interval between frames in milliseconds
    # Try to match real time: interval = time_step_between_frames * 1000
    interval_ms = (t[frame_step] - t[0]) * 1000 if len(t) > frame_step else 50

    ani = animation.FuncAnimation(fig, update, frames=frame_indices,
                                  init_func=init, blit=True, interval=interval_ms,
                                  repeat=False) # Don't repeat animation

    # Show the animation
    # Depending on your environment (Jupyter, script, etc.), this might
    # open a new window or display inline.
    plt.show()

    # --- Optional: Save Animation ---
    # Uncomment the following lines to save as MP4 (requires ffmpeg)
    # print("Saving animation (may take a while)...")
    # try:
    #     ani.save('double_pendulum_animation.mp4', fps=int(1000 / interval_ms),
    #              writer='ffmpeg', dpi=150)
    #     print("Animation saved successfully.")
    # except Exception as e:
    #     print(f"Error saving animation: {e}")
    #     print("Ensure ffmpeg is installed and accessible in your system PATH.")

else:
    print(f"Simulation failed: {sol.message}")