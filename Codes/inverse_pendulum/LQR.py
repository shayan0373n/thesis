import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

class LQRController:
    """
    A class to design and implement a discrete-time LQR controller
    for a continuous-time system.
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
        """
        Initializes the LQR controller.

        Args:
            A (np.ndarray): Continuous-time state matrix.
            B (np.ndarray): Continuous-time input matrix.
            Q (np.ndarray): State weighting matrix for the cost function.
            R (np.ndarray): Input weighting matrix for the cost function.
            dt (float): Sampling time for discretization.
        """
        if dt <= 0:
            raise ValueError("Sampling time dt must be positive.")

        self.A_cont = A
        self.B_cont = B
        self.Q = Q
        self.R = R
        self.dt = dt

        # Discretize the system
        self.Ad, self.Bd, _, _, _ = scipy.signal.cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))), dt)
        print(f"Discrete A (Ad):\n{self.Ad}")
        print(f"Discrete B (Bd):\n{self.Bd}")


        # Solve the Discrete Algebraic Riccati Equation (DARE)
        self.P = self._solve_dare(self.Ad, self.Bd, self.Q, self.R)

        # Compute the optimal feedback gain K
        self.K = self._compute_gain(self.Ad, self.Bd, self.Q, self.R, self.P)
        print(f"Optimal Gain K:\n{self.K}")

        self.last_u = None # Last control input
        self.last_t = None # Last time step

        self.t_hist = []
        self.u_hist = []


    def _solve_dare(self, Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Solves the Discrete Algebraic Riccati Equation (DARE).
        P = Ad.T P Ad - (Ad.T P Bd) inv(R + Bd.T P Bd) (Bd.T P Ad) + Q
        """
        try:
            P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
            return P
        except np.linalg.LinAlgError as e:
            print(f"Error solving DARE: {e}")
            print("Check if (Ad, Bd) is stabilizable and (Ad, sqrt(Q)) is detectable.")
            print("Also ensure R is positive definite and Q is positive semi-definite.")
            raise

    def _compute_gain(self, Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Computes the optimal LQR gain K.
        K = inv(R + Bd.T P Bd) (Bd.T P Ad)
        """
        # K = inv(R + B'.P.B) * B'.P.A
        R_term = R + Bd.T @ P @ Bd
        Ad_term = Bd.T @ P @ Ad
        try:
            K = np.linalg.solve(R_term, Ad_term) # More stable than inv()
            # K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) # Alternative
            return K
        except np.linalg.LinAlgError as e:
            print(f"Error computing gain K (matrix inversion/solve failed): {e}")
            print("Check if (R + Bd.T P Bd) is invertible.")
            raise

    def compute_control(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Computes the control input u = -Kx.

        Args:
            x (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Control input vector u.
        """
        if self.last_t is not None and (t - self.last_t) < self.dt:
            # Avoid too frequent calls to compute_control
            return self.last_u
        x = x.reshape(-1, 1) # Ensure x is a column vector
        if x.shape[0] != self.K.shape[1]:
             raise ValueError(f"State vector shape {x.shape} incompatible with gain matrix K shape {self.K.shape}")

        u = (-self.K @ x).flatten() # Compute control input
        self.last_u = u # Store last control input
        self.last_t = t # Store last time step
        self.t_hist.append(t) # Store time step
        self.u_hist.append(u) # Store control input history
        return u # Return as a 1D array typically
    
    def reset(self):
        """
        Resets the controller state (if needed).
        """
        self.last_u = None
        self.last_t = None
        self.t_hist = []
        self.u_hist = []

    def get_u_hist(self):
        """
        Returns the history of control inputs computed by the controller as a 2xN array.
        Row 0: time steps, Row 1: control inputs.
        """
        t_hist = np.array(self.t_hist)[:, np.newaxis] # Convert to column vector
        u_hist = np.array(self.u_hist)
        return np.hstack((t_hist, u_hist))

# --- Example & Simulation ---
if __name__ == '__main__':
    # Example: Simple inverted pendulum on a cart (linearized around upright)
    g = 9.81
    m_c = 1.0  # Cart mass
    m_p = 0.1  # Pendulum mass
    L = 0.5   # Pendulum length

    A_cont = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, m_p * g / m_c, 0, 0],
        [0, (m_c + m_p) * g / (m_c * L), 0, 0]
    ])

    B_cont = np.array([
        [0],
        [0],
        [1 / m_c],
        [1 / (m_c * L)]
    ])

    # State x = [position, angle, velocity, angular_velocity]
    # Input u = [force]

    # LQR weighting matrices
    Q = np.diag([1.0, 10.0, 0.1, 0.1]) # Penalize angle deviation most
    R = np.array([[0.1]])            # Low penalty on control effort

    # Sampling time
    dt_controller = 0.1 # seconds
    dt_simulation = 0.001 # seconds (for simulation, not controller)

    print("Initializing LQR Controller...")

    lqr = LQRController(A_cont, B_cont, Q, R, dt_controller)

    # --- Simulation Setup ---
    T_sim = 5.0  # Simulation time in seconds
    # Initial state: pendulum displaced by 0.3 rad (~17 deg), cart at 0
    x0 = np.array([0.0, 0.3, 0.0, 0.0])

    print(f"\nSimulating system for {T_sim} seconds...")
    print(f"Initial State (x0): {x0}")

    num_steps = int(T_sim / dt_simulation)
    t_hist = np.linspace(0, T_sim, num_steps + 1)
    x_hist = np.zeros((lqr.Ad.shape[0], num_steps + 1)) # State history (pos, angle, vel, ang_vel)
    u_hist = np.zeros((lqr.Bd.shape[1], num_steps))     # Control input history (force)

    x_hist[:, 0] = x0 # Set initial state

    # --- Simulation Loop ---
    for k in range(num_steps):
        # Get current state
        x_k = x_hist[:, k] # Ensure x_k is a column vector

        # Compute control input using the controller
        u_k = lqr.compute_control(t_hist[k], x_k)
        u_hist[:, k] = u_k # Store control input

        # Simulate *continuous* system
        x_k_plus_1 = x_k + dt_simulation * (lqr.A_cont @ x_k + lqr.B_cont @ u_k)
        # Note: This is a simple Euler integration step. For better accuracy,
        # consider using a more sophisticated method like RK4.
        # Alternatively, use scipy's solve_ivp for more complex dynamics.

        # Store next state
        x_hist[:, k+1] = x_k_plus_1.flatten()

    print("Simulation complete.")

    # --- Plotting ---
    print("Generating plots...")

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('LQR Control of Inverted Pendulum', fontsize=16)

    # Plot states: Position and Angle
    axs[0].plot(t_hist, x_hist[0, :], label='Cart Position (m)', color='blue')
    axs[0].plot(t_hist, x_hist[1, :], label='Pendulum Angle (rad)', color='red')
    axs[0].set_ylabel('Position / Angle')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title('State Trajectories')

    # Plot states: Velocities
    axs[1].plot(t_hist, x_hist[2, :], label='Cart Velocity (m/s)', color='blue', linestyle='--')
    axs[1].plot(t_hist, x_hist[3, :], label='Pendulum Ang. Vel. (rad/s)', color='red', linestyle='--')
    axs[1].set_ylabel('Velocities')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Plot control input
    # Use steps-post to show discrete nature of control signal changes
    u = lqr.get_u_hist()
    axs[2].plot(u[:, 0], u[:, 1:], label='Control Force (N)', drawstyle='steps-post', color='green')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Control Input')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    axs[2].set_title('Control Input')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    print("Plots displayed.")