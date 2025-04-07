import numpy as np
import matplotlib.pyplot as plt

def derivative(arr, time_step):
    """
    Calculate the derivative of an array using central difference method.
    
    Args:
        arr: A numpy array of values.
        time_step: The time step between data points.
        
    Returns:
        A numpy array of derivative values.
    """
    derivative = np.zeros_like(arr)
    for i in range(1, len(arr) - 1):
        derivative[i] = (arr[i+1] - arr[i-1]) / (2 * time_step)
    derivative[0] = (arr[1] - arr[0]) / time_step  # Forward difference
    derivative[-1] = (arr[-1] - arr[-2]) / time_step  # Backward difference
    return derivative

def calculate_optimal_stiffness(joint_angles, joint_torques, joint_velocities, time_step, joint_type, plot=True):
    """
    Calculates the optimal series elastic actuator (SEA) stiffness for a given joint.

    Args:
        joint_angles: A numpy array of joint angles (in radians) over time.
        joint_torques: A numpy array of joint torques (normalized to body mass) over time.
        joint_velocities: A numpy array of joint angular velocities (in radians/s) over time.
        time_step: The time step between data points (in seconds).
        joint_type: A string indicating the joint type ('hip', 'knee', or 'ankle').
        plot: show plot if it is true.

    Returns:
        A tuple containing:
            - optimal_stiffness: The optimal stiffness value (normalized).
            - min_criterion: The minimum value of the optimization criterion (energy or peak power).
            - all_stiffnesses: A numpy array of all tested stiffness values.
            - all_criteria: A numpy array of the optimization criterion values for all tested stiffnesses.
    """
    M_ex_dot = derivative(joint_torques, time_step)
    if joint_velocities is None:
        joint_velocities = derivative(joint_angles, time_step)

    # --- 1. Define Stiffness Range ---
    stiffnesses = np.linspace(0.01, 1, 10_000, endpoint=True)  # Normalized stiffness values [Nm/(rad*kg)]

    # --- 2. Optimization Criterion ---
    if joint_type == 'ankle':
        optimization_criterion = 'peak_power'
    else:  # 'hip' or 'knee'
        optimization_criterion = 'energy'

    # --- 3. Initialize ---
    all_criteria = []
    min_criterion = float('inf')
    optimal_stiffness = None

    # --- 4. Brute-Force Search ---
    for Ks in stiffnesses:
        motor_powers = []
        
        # Assume spring is at rest at the begining of the gait cycle
        theta_0 = 0

        for i in range(len(joint_angles)):            
            # --- Calculate Motor Angular Velocity (theta_m_dot) ---
            theta_m_dot = joint_velocities[i] + M_ex_dot[i] / Ks  # Equation (7)

            # --- Calculate Motor Power (Pm) ---
            Pm = joint_torques[i] * theta_m_dot   # Equation (8) -  Mm = Mex
            motor_powers.append(Pm)

        # --- Calculate Optimization Criterion Value ---
        if optimization_criterion == 'energy':
            # Calculate energy requirement
            total_energy = np.sum(np.array(motor_powers) * time_step) 
            criterion_value = total_energy
        else:  # peak_power
            # Calculate peak power (absolute value)
            criterion_value = np.max(np.abs(motor_powers))
            
        all_criteria.append(criterion_value)

        # --- Update Optimal Stiffness ---
        if criterion_value < min_criterion:
            min_criterion = criterion_value
            optimal_stiffness = Ks

    if plot:
        # --- Plot Results ---
        plt.figure(figsize=(8, 6))
        plt.plot(stiffnesses, all_criteria)
        plt.xlabel('Stiffness (Nm/(rad*kg))')
        if optimization_criterion == 'energy':
            plt.ylabel('Energy Requirement (J/(kg*m))')
        else:
            plt.ylabel('Peak Power (W/kg)')

        plt.title(f'Optimal Stiffness Search for {joint_type.capitalize()} ({optimization_criterion.replace("_", " ").title()})')
        plt.scatter(optimal_stiffness, min_criterion, color='red', marker='o', label=f'Optimal Ks = {optimal_stiffness:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_stiffness, min_criterion, stiffnesses, all_criteria



# --- Example Usage with Mock Data ---

# Create time vector
num_points = 101
time = np.linspace(0, 2, num_points, endpoint=True)  # 1 second gait cycle
time_step = time[1] - time[0]

# load data
data = np.loadtxt("my_data.csv", delimiter=',')
joint_angles = data[:, 1]
joint_torques = data[:, 0]
joint_velocities = derivative(joint_angles, time_step)

# --- Run Optimization ---
# optimal_stiffness_hip, min_energy_hip, _, _ = calculate_optimal_stiffness(joint_angles, joint_torques, joint_velocities, time_step, 'hip')
# optimal_stiffness_knee, min_energy_knee, _, _ = calculate_optimal_stiffness(joint_angles, joint_torques, joint_velocities, time_step, 'knee')
optimal_stiffness_ankle, min_peak_power_ankle, _, _ = calculate_optimal_stiffness(joint_angles, joint_torques, joint_velocities, time_step, 'ankle')

# print(f"Optimal Stiffness (Hip): {optimal_stiffness_hip:.2f} Nm/(rad*kg), Min Energy: {min_energy_hip:.4f} J/(kg*m)")
# print(f"Optimal Stiffness (Knee): {optimal_stiffness_knee:.2f} Nm/(rad*kg), Min Energy: {min_energy_knee:.4f} J/(kg*m)")
print(f"Optimal Stiffness (Ankle): {optimal_stiffness_ankle:.4f} Nm/(rad*kg), Min Peak Power: {min_peak_power_ankle:.4f} W/kg")