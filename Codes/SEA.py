import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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



# --- Simulation Parameters ---
# SEA Device Parameters (Example values - adjust as needed)
J_m = 0.001  # Motor inertia
J_l = 1.0    # Load inertia
K_s = 8.0    # Spring stiffness
B_m = 0.001  # Motor damping
B_l = 0.01   # Load damping
gear_ratio = 1.0

# Simulation Time
time_span = [0, 5] # seconds
time_points = np.linspace(time_span[0], time_span[1], 500) # More points for smoother plots

# Initial State [theta_m, omega_m, theta_l, omega_l]
initial_state_active = [0.0, 0.0, 0.0, 0.0] # Initial state for ACTIVE mode (start from rest)
initial_state_passive = [0.0, 0.0, 0.5, 0.0] # Initial state for PASSIVE mode (initial load angle)

# --- Create SEA Device Instance ---
sea_device = SEA_Device(J_m, J_l, K_s, B_m, B_l, gear_ratio)


# --- Active Mode Simulation ---
# Target Trajectory (Example: Sinusoidal for load angle)
amplitude = np.pi/4  # Amplitude of sinusoidal trajectory (45 degrees)
frequency = 0.5       # Frequency of sinusoidal trajectory (0.5 Hz)
target_load_trajectory = lambda t: amplitude * np.sin(2 * np.pi * frequency * t)

# PID Controller Gains (Tune these for desired performance)
kp = 2.0  # Example - You'll need to tune these
ki = 0.0   # Example
kd = 0.0   # Example
integral_error_active = [0.0] # Mutable list to hold integral error for active control
last_error_active = [0.0] # Mutable list to hold last error for derivative term
control_dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.0 # Approximate dt

def active_control_torque(t, state):
    target_angle = target_load_trajectory(t)
    return pid_controller(t, state, target_angle, kp, ki, kd, integral_error_active, last_error_active, control_dt)


active_sol = simulate_SEA(sea_device, initial_state_active, time_span, mode='active', control_torque_func=active_control_torque)
active_state_interp = active_sol.sol(time_points)
active_theta_m = active_state_interp[0, :]
active_omega_m = active_state_interp[1, :]
active_theta_l = active_state_interp[2, :]
active_omega_l = active_state_interp[3, :]

# Calculate active mode spring torque for plotting
active_spring_deflection = (active_theta_m / sea_device.gear_ratio) - active_theta_l
active_spring_torque = sea_device.K_s * active_spring_deflection

# Calculate active mode motor torque (approximated from equation of motion)
active_motor_torque = sea_device.J_m * np.gradient(active_omega_m, time_points) + active_spring_torque + sea_device.B_m * active_omega_m


# --- Passive Mode Simulation ---
passive_sol = simulate_SEA(sea_device, initial_state_passive, time_span, mode='passive') # Use different initial state
passive_state_interp = passive_sol.sol(time_points)
passive_theta_m = passive_state_interp[0, :]
passive_omega_m = passive_state_interp[0, :] # Corrected to use theta_m[0] as it's fixed
passive_theta_l = passive_state_interp[2, :]
passive_omega_l = passive_state_interp[3, :]

# Calculate passive mode spring torque for plotting
passive_spring_deflection = (sea_device.fixed_motor_angle_passive / sea_device.gear_ratio) - passive_theta_l # Use fixed motor angle
passive_spring_torque = sea_device.K_s * passive_spring_deflection


# --- Plotting ---
plt.figure(figsize=(16, 10)) # Increased figure size for more subplots

# Active Mode Plots
plt.subplot(2, 3, 1) # 2 rows, 3 columns, 1st subplot
plt.plot(time_points, active_theta_l, label='Active Load Angle (theta_l)')
plt.plot(time_points, target_load_trajectory(time_points), '--', label='Active Target Angle')
plt.title('Active Mode: Load Angle and Target')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2) # 2 rows, 3 columns, 2nd subplot
plt.plot(time_points, active_spring_torque, label='Active Spring Torque')
plt.plot(time_points, active_motor_torque, label='Active Motor Torque (approx.)')
plt.title('Active Mode: Torques')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3) # 2 rows, 3 columns, 3rd subplot
plt.plot(active_theta_l, active_spring_torque, label='Active Spring Torque vs. Load Angle')
plt.title('Active Mode: Torque vs. Angle')
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