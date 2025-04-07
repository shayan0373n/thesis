import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.linalg import expm # Import for matrix exponential (discretization)

class SEA_Device:
    def __init__(self, J_m, J_l, K_s, B_m, B_l, gear_ratio=1.0):
        """
        Initializes the SEA device parameters.

        J_m: Motor inertia (kg*m^2)
        J_l: Load inertia (kg*m^2)
        K_s: Spring stiffness (Nm/rad)
        B_m: Motor damping (Nms/rad)
        B_l: Load damping (Nms/rad)
        gear_ratio: Gear ratio between motor and spring (motor angle to spring angle)
        """
        self.J_m = J_m
        self.J_l = J_l
        self.K_s = K_s
        self.B_m = B_m
        self.B_l = B_l
        self.gear_ratio = gear_ratio
        self.fixed_motor_angle_passive = 0.0 # Define a fixed motor angle for passive mode

    def equations_of_motion_active(self, t, state, control_torque_func):
        """
        Equations of motion for active mode.

        state: [theta_m, omega_m, theta_l, omega_l] - motor angle, motor angular velocity, load angle, load angular velocity
        control_torque_func: Function that returns the desired motor torque at time t and current state.
        """
        theta_m, omega_m, theta_l, omega_l = state

        # Spring deflection
        spring_deflection = (theta_m / self.gear_ratio) - theta_l

        # Spring torque and damping torque
        spring_torque = self.K_s * spring_deflection
        damping_torque_spring = 0 # Simplified - could add damping in spring if needed

        # Motor torque from control
        motor_torque = control_torque_func(t, state)

        # Equations of motion
        d_omega_m_dt = (motor_torque - spring_torque - self.B_m * omega_m) / self.J_m
        d_omega_l_dt = (spring_torque - self.B_l * omega_l + damping_torque_spring) / self.J_l

        return [omega_m, d_omega_m_dt, omega_l, d_omega_l_dt]


    def equations_of_motion_passive(self, t, state):
        """
        Equations of motion for passive mode (motor angle is fixed).

        state: [theta_m, omega_m, theta_l, omega_l] -  Although we use 4-state, theta_m and omega_m are fixed in passive mode.
        """
        theta_m, omega_m, theta_l, omega_l = state

        # In passive mode, enforce motor angle and velocity to be constant
        theta_m_passive = self.fixed_motor_angle_passive # Fixed motor angle
        omega_m_passive = 0.0  # Fixed motor angular velocity

        # Spring deflection - use the *fixed* motor angle
        spring_deflection = (theta_m_passive / self.gear_ratio) - theta_l

        # Spring torque and damping torque
        spring_torque = self.K_s * spring_deflection
        damping_torque_spring = 0 # Simplified - could add damping in spring if needed

        # Motor torque - not relevant as motor angle is fixed, but conceptually, it would be the torque required to keep motor fixed.
        motor_torque = 0.0 # Not used in load equation but for completeness set to 0.

        # Equations of motion
        d_omega_m_dt = 0.0 # Motor angle is fixed, so no acceleration
        d_omega_l_dt = (spring_torque - self.B_l * omega_l + damping_torque_spring) / self.J_l

        return [omega_m_passive, d_omega_m_dt, omega_l, d_omega_l_dt] # Return fixed motor velocity and zero acceleration


def simulate_SEA(sea_device, initial_state, time_span, mode='active', control_torque_func=None):
    """
    Simulates the SEA device in active or passive mode.

    sea_device: SEA_Device object
    initial_state: Initial state [theta_m, omega_m, theta_l, omega_l]
    time_span: Time duration for simulation [t_start, t_end]
    mode: 'active' or 'passive'
    control_torque_func: Function for control torque (only for active mode)
    """

    if mode == 'active':
        if control_torque_func is None:
            raise ValueError("Control torque function must be provided for active mode.")
        sol = solve_ivp(sea_device.equations_of_motion_active, time_span, initial_state,
                        args=(control_torque_func,), dense_output=True, max_step=0.001) # Reduced max_step for smoother results
    elif mode == 'passive':
        sol = solve_ivp(sea_device.equations_of_motion_passive, time_span, initial_state,
                        dense_output=True, max_step=0.001) # Reduced max_step for smoother results
    else:
        raise ValueError("Mode must be 'active' or 'passive'.")

    return sol


def pid_controller(t, state, target_load_angle, kp, ki, kd, integral_error, last_error, dt):
    """
    Simple PID controller for load angle.

    t: Current time
    state: [theta_m, omega_m, theta_l, omega_l]
    target_load_angle: Desired load angle (rad)
    kp, ki, kd: PID gains
    integral_error: Accumulated integral error from previous steps (mutable list to persist value)
    last_error: Error from previous step (mutable list to persist value)
    dt: Time step since last control calculation
    """
    theta_l = state[2]
    error = target_load_angle - theta_l

    integral_error[0] += error * dt
    derivative_error = (error - last_error[0]) / dt if dt > 0 else 0 # avoid division by zero in initial step
    last_error[0] = error

    control_torque = kp * error + ki * integral_error[0] + kd * derivative_error
    return control_torque

def lqr_controller(t, desired_state, current_state, K):
    """
    LQR controller for load angle with reference tracking.

    t: Current time (not used in this simple state feedback LQR)
    desired_state: Desired state [theta_m_d, omega_m_d, theta_l_d, omega_l_d] (reference state)
    current_state: Current state [theta_m, omega_m, theta_l, omega_l]
    K: LQR gain matrix (state feedback gain)
    """
    state_error = np.array(current_state) - np.array(desired_state) # Error = current - desired
    control_torque = -K @ state_error # Control law u = -K * e
    return control_torque


# --- Simulation Parameters ---
# SEA Device Parameters (Example values - adjust as needed)
J_m = 0.001  # Motor inertia
J_l = 1.0   # Load inertia
K_s = 1.0    # Spring stiffness
B_m = 0.001  # Motor damping
B_l = 0.1   # Load damping
gear_ratio = 1.0
dt = 0.01

# Simulation Time
time_span = [0, 5] # seconds
time_points = np.linspace(time_span[0], time_span[1], 500) # More points for smoother plots

# Initial State [theta_m, omega_m, theta_l, omega_l]
initial_state_active = [0.0, 0.0, 0.0, 0.0] # Initial state for ACTIVE mode (start from rest)
initial_state_passive = [0.0, 0.0, 0.5, 0.0] # Initial state for PASSIVE mode (initial load angle)

# --- Create SEA Device Instance ---
sea_device = SEA_Device(J_m, J_l, K_s, B_m, B_l, gear_ratio)


# --- LQR Controller Design ---
# Define state-space matrices A, B, C for LQR design (linearized around equilibrium)
A_cont = np.array([[0,          1,          0,           0],
              [-sea_device.K_s/(sea_device.J_m*sea_device.gear_ratio), -sea_device.B_m/sea_device.J_m,     sea_device.K_s/sea_device.J_m,       0],
              [0,          0,          0,           1],
              [sea_device.K_s/(sea_device.J_l*sea_device.gear_ratio),  0,          -sea_device.K_s/sea_device.J_l,     -sea_device.B_l/sea_device.J_l]])

B_cont = np.array([[0],
              [1/sea_device.J_m],
              [0],
              [0]])

C = np.array([[0, 0, 1, 0]]) # Output is theta_l (load angle)


# Discretize the continuous-time system - Zero Order Hold (ZOH) method
n = A_cont.shape[0] # System order (4)
M = np.block([[A_cont, B_cont],
              [np.zeros((1, n)), np.zeros((1, 1))]]) # Augmented matrix for discretization

Md = expm(M * dt) # Matrix exponential for discretization

Ad = Md[:n, :n] # Discrete-time A matrix
Bd = Md[:n, n:] # Discrete-time B matrix
Cd = C # Output matrix remains the same for ZOH


print("Ad Matrix:\n", Ad)
print("Bd Matrix:\n", Bd)
print("C Matrix:\n", C)


# Define Q and R matrices for LQR cost function (tuning parameters)
Q_lqr = np.diag([0, 0, 1000, 0.0]) # Penalize theta_m, omega_m, theta_l, omega_l  (Load angle error is weighted higher)
R_lqr = np.array([[0.1]])           # Penalize control effort (motor torque)


# Solve Discrete Algebraic Riccati Equation to find LQR gain K
# For discrete-time LQR, we need to solve the Discrete ARE (DARE), not CARE
def solve_discrete_are(Ad, Bd, Q, R):
    """Solves the Discrete Algebraic Riccati Equation (DARE) using iterative method."""
    X = Q # Initialize X (e.g., with Q)
    for _ in range(1000): # Iterate until convergence (adjust max iterations if needed)
        X_next = Q + Ad.T @ X @ Ad - Ad.T @ X @ Bd @ np.linalg.inv(R + Bd.T @ X @ Bd) @ Bd.T @ X @ Ad
        if np.linalg.norm(X_next - X) < 1e-6: # Convergence check
            print("DARE Converged!")
            break
        X = X_next
    return X_next

X_lqr_discrete = solve_discrete_are(Ad, Bd, Q_lqr, R_lqr) # Solve DARE
K_lqr = np.linalg.solve(R_lqr + Bd.T @ X_lqr_discrete @ Bd, (Bd.T @ X_lqr_discrete @ Ad)).reshape((1, 4)) # Discrete LQR gain


print("LQR Gain Matrix K_lqr:\n", K_lqr)


# --- Active Mode Simulation with LQR ---
# Target Trajectory (Example: Sinusoidal for load angle) - same as before
amplitude = np.pi/4  # Amplitude of sinusoidal trajectory (45 degrees)
frequency = 0.5       # Frequency of sinusoidal trajectory (0.5 Hz)
target_load_angle_osc = lambda t: amplitude * np.sin(2 * np.pi * frequency * t)
target_load_velocity_osc = lambda t: 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t) # Derivative of target angle
target_load_angle_constant = lambda t: amplitude * np.ones_like(t) # Constant target angle
target_load_velocity_constant = lambda t: np.zeros_like(t) # Constant target velocity
target_load_angle = target_load_angle_osc # Choose target trajectory function
target_load_velocity = target_load_velocity_osc # Choose target velocity function

def active_control_torque_lqr(t, state):
    desired_state = [0.0, 0.0, target_load_angle(t), target_load_velocity(t)] # Desired state with target load angle
    control_torque = lqr_controller(t, desired_state, state, K_lqr)
    return control_torque


active_sol_lqr = simulate_SEA(sea_device, initial_state_active, time_span, mode='active', control_torque_func=active_control_torque_lqr)
active_state_interp_lqr = active_sol_lqr.sol(time_points)
active_theta_m_lqr = active_state_interp_lqr[0, :]
active_omega_m_lqr = active_state_interp_lqr[1, :]
active_theta_l_lqr = active_state_interp_lqr[2, :]
active_omega_l_lqr = active_state_interp_lqr[3, :]

# Calculate active mode spring torque for plotting (LQR)
active_spring_deflection_lqr = (active_theta_m_lqr / sea_device.gear_ratio) - active_theta_l_lqr
active_spring_torque_lqr = sea_device.K_s * active_spring_deflection_lqr

# Calculate active mode motor torque (approximated from equation of motion) (LQR)
active_motor_torque_lqr = sea_device.J_m * np.gradient(active_omega_m_lqr, time_points) + active_spring_torque_lqr + sea_device.B_m * active_omega_m_lqr


# --- Passive Mode Simulation --- (Same as before)
passive_sol = simulate_SEA(sea_device, initial_state_passive, time_span, mode='passive') # Use different initial state
passive_state_interp = passive_sol.sol(time_points)
passive_theta_m = passive_state_interp[0, :]
passive_omega_m = passive_state_interp[0, :] # Corrected to use theta_m[0] as it's fixed
passive_theta_l = passive_state_interp[2, :]
passive_omega_l = passive_state_interp[3, :]

# Calculate passive mode spring torque for plotting
passive_spring_deflection = (sea_device.fixed_motor_angle_passive / sea_device.gear_ratio) - passive_theta_l
passive_spring_torque = sea_device.K_s * passive_spring_deflection


# --- Plotting ---
plt.figure(figsize=(16, 10)) # Increased figure size for more subplots

# Active Mode Plots (LQR)
plt.subplot(2, 3, 1) # 2 rows, 3 columns, 1st subplot
plt.plot(time_points, active_theta_l_lqr, label='Active (LQR) Load Angle (theta_l)')
plt.plot(time_points, target_load_angle(time_points), '--', label='Active Target Angle')
plt.title('Active Mode (LQR): Load Angle and Target')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2) # 2 rows, 3 columns, 2nd subplot
plt.plot(time_points, active_spring_torque_lqr, label='Active (LQR) Spring Torque')
plt.plot(time_points, active_motor_torque_lqr, label='Active (LQR) Motor Torque (approx.)')
plt.title('Active Mode (LQR): Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3) # 2 rows, 3 columns, 3rd subplot
plt.plot(active_theta_l_lqr, active_spring_torque_lqr, label='Active (LQR) Spring Torque vs. Load Angle')
plt.title('Active Mode (LQR): Torque vs. Angle')
plt.xlabel('Load Angle (rad)')
plt.ylabel('Spring Torque (Nm)')
plt.grid(True)
plt.legend()


# Passive Mode Plots
plt.subplot(2, 3, 4) # 2 rows, 3 columns, 4th subplot
plt.plot(time_points, passive_theta_l, label='Passive Load Angle (theta_l)')
plt.title('Passive Mode: Load Angle (Initial Displacement, Fixed Motor)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5) # 2 rows, 3 columns, 5th subplot
plt.plot(time_points, passive_spring_torque, label='Passive Spring Torque')
plt.title('Passive Mode: Spring Torque (Initial Displacement, Fixed Motor)')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6) # 2 rows, 3 columns, 6th subplot
plt.plot(passive_theta_l, passive_spring_torque, label='Passive Spring Torque vs. Load Angle')
plt.title('Passive Mode: Torque vs. Angle (Fixed Motor)')
plt.xlabel('Load Angle (rad)')
plt.ylabel('Spring Torque (Nm)')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()