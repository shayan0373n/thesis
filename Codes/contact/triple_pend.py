from typing import Callable
from dataclasses import dataclass
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation

@dataclass
class Link:
    m: float  # Mass of the link (kg)
    l: float  # Length of the link (m)
    I: float | None = None  # Moment of inertia (kg*m^2) around the CoM, optional

    def __post_init__(self):
        if self.I is None:
            # If I is not provided, calculate it for a uniform rod
            self.I = (1/12) * self.m * self.l**2

class TriplePendulum:
    """
    Represents a triple pendulum system, handling dynamics derivation,
    optimization setup, and animation.
    """
    def __init__(self, link1: Link, link2: Link, link3: Link, **kwargs):
        """
        Initializes the TriplePendulum instance.

        Args:
            params (dict): Dictionary containing system parameters.
                           Expected keys: 'rod1', 'rod2', 'rod3' (each with 'm', 'l'), 'g'
        """
        self.link1 = link1
        self.link2 = link2
        self.link3 = link3
        self.g = kwargs.get('g', 9.81)  # Default gravity (m/s^2)
        self.kwargs = kwargs  # Store additional parameters

        # Derive and store the symbolic dynamics function
        self.manipulator_coefficients_fn: Callable[[ca.DM], tuple[ca.DM, ca.DM, ca.DM]] = self._derive_symbolic_manipulator_coefficients_fn()
        # Derive and store the center of mass dynamics function
        self.com_dynamics_fn: Callable[[ca.DM, ca.DM], ca.DM] = self._derive_symbolic_com_dynamics_fn()

    def dynamics_fn(self, x: ca.DM, u: ca.DM) -> ca.DM:
        """
        Computes the dynamics of the triple pendulum system.

        Args:
            x: State vector (theta, theta_dot).
            u: Control vector (torques).

        Returns:
            Derivative of the state vector (x_ddot).
        """
        B = np.array([
            [1, -1, 0],
            [0, 1, -1],
            [0, 0, 1],
        ])
        M, C, G = self.manipulator_coefficients_fn(x)
        # Unpack state vector
        q_dot = x[3:6]
        # M(q) * q_ddot + C(q, q_dot) + G(q) = u
        # q_ddot = M(q)^(-1) * (u - C(q, q_dot) - G(q))
        q_dotdot = ca.solve(M, B @ u - C - G, 'symbolicqr')
        # Return the full state derivative
        return ca.vertcat(q_dot, q_dotdot)
     
    def base_force_fn(self, x: ca.DM, u: ca.DM) -> ca.DM:
        """
        Computes the base force of the triple pendulum system.

        Args:
            x: State vector (theta, theta_dot).
            u: Control vector (torques).

        Returns:
            Base force vector.
        """
        com_ddot = self.com_dynamics_fn(x, u)
        # M * com_ddot = F + W
        M = self.link1.m + self.link2.m + self.link3.m
        W = ca.vertcat(0, -M * self.g)  # Weight vector
        F = M * com_ddot - W  # Base force
        return F
    
    def com_pos(self, x: ca.DM) -> ca.DM:
        """
        Computes the center of mass position of the triple pendulum system.
        Args:
            x: State vector (theta, theta_dot).
        Returns:
            Center of mass position vector.
        """
        # Unpack state vector
        q = x[0:3]
        l1, l2, l3 = self.link1.l, self.link2.l, self.link3.l
        m1, m2, m3 = self.link1.m, self.link2.m, self.link3.m
        M = m1 + m2 + m3
        x_com1 = -l1 * ca.sin(q[0]) / 2
        y_com1 = l1 * ca.cos(q[0]) / 2
        x_com2 = -l1 * ca.sin(q[0]) - l2 * ca.sin(q[1]) / 2
        y_com2 = l1 * ca.cos(q[0]) + l2 * ca.cos(q[1]) / 2
        x_com3 = -l1 * ca.sin(q[0]) - l2 * ca.sin(q[1]) - l3 * ca.sin(q[2]) / 2
        y_com3 = l1 * ca.cos(q[0]) + l2 * ca.cos(q[1]) + l3 * ca.cos(q[2]) / 2
        x_com = (m1 * x_com1 + m2 * x_com2 + m3 * x_com3) / M
        y_com = (m1 * y_com1 + m2 * y_com2 + m3 * y_com3) / M
        return ca.vertcat(ca.reshape(x_com, 1, -1), ca.reshape(y_com, 1, -1))
    
    def cop_x(self, x: ca.DM, u: ca.DM) -> ca.DM:
        """
        Computes the center of pressure (CoP) x-coordinate of the triple pendulum system.
        Args:
            x: State vector (theta, theta_dot).
            u: Control vector (torques).
        Returns:
            Center of pressure x-coordinate.
        """
        com = self.com_pos(x)
        com_ddot = self.com_dynamics_fn(x, u)
        cop_x = com[0] - (com_ddot[0] * com[1]) / (self.g + com_ddot[1])
        return cop_x 
        
    def _derive_symbolic_manipulator_coefficients_fn(self) -> Callable[[ca.DM], tuple[ca.DM, ca.DM, ca.DM]]:
        """
        Returns a function that computes [M(q), C(q, q_dot), G(q)] for the manipulator dynamics.
        Such that M(q) * q_ddot + C(q, q_dot) + G(q) = u
        """
        # Define symbolic variables
        th1, th2, th3 = ca.MX.sym('th1'), ca.MX.sym('th2'), ca.MX.sym('th3')
        om1, om2, om3 = ca.MX.sym('om1'), ca.MX.sym('om2'), ca.MX.sym('om3')
        q = ca.vertcat(th1, th2, th3)
        q_dot = ca.vertcat(om1, om2, om3)
        x = ca.vertcat(q, q_dot)
        m1 = self.link1.m
        m2 = self.link2.m
        m3 = self.link3.m
        l1 = self.link1.l
        l2 = self.link2.l
        l3 = self.link3.l
        # Moments of inertia around CoM
        I1 = self.link1.I
        I2 = self.link2.I
        I3 = self.link3.I

        g = self.g

        # Kinematics (0=up, positive=CCW)
        p_j0 = ca.vertcat(0, 0)
        p_c1 = p_j0 + ca.vertcat(-(l1/2) * ca.sin(th1), (l1/2) * ca.cos(th1))
        p_j1 = p_j0 + ca.vertcat(-l1 * ca.sin(th1), l1 * ca.cos(th1))
        p_c2 = p_j1 + ca.vertcat(-(l2/2) * ca.sin(th2), (l2/2) * ca.cos(th2))
        p_j2 = p_j1 + ca.vertcat(-l2 * ca.sin(th2), l2 * ca.cos(th2))
        p_c3 = p_j2 + ca.vertcat(-(l3/2) * ca.sin(th3), (l3/2) * ca.cos(th3))

        # Velocities
        v_c1 = ca.jacobian(p_c1, q) @ q_dot
        v_c2 = ca.jacobian(p_c2, q) @ q_dot
        v_c3 = ca.jacobian(p_c3, q) @ q_dot

        # Lagrangian (L = KE - PE)
        KE1 = 0.5 * m1 * ca.dot(v_c1, v_c1) + 0.5 * I1 * om1**2
        KE2 = 0.5 * m2 * ca.dot(v_c2, v_c2) + 0.5 * I2 * om2**2
        KE3 = 0.5 * m3 * ca.dot(v_c3, v_c3) + 0.5 * I3 * om3**2
        T = KE1 + KE2 + KE3

        PE1 = m1 * g * p_c1[1]
        PE2 = m2 * g * p_c2[1]
        PE3 = m3 * g * p_c3[1]
        V = PE1 + PE2 + PE3
        # L = T - V

        # Euler-Lagrange Equations
        dV_dq = ca.gradient(V, q)
        dT_dq = ca.gradient(T, q)
        M, dT_dqdot = ca.hessian(T, q_dot) # Only the KE term can be a function of q_dot
        # M = ca.simplify(M) # Simplification can be slow, often optional

        C = ca.jtimes(dT_dqdot, q, q_dot) - dT_dq # Coriolis, Centrifugal terms
        G = dV_dq # Gravity terms


        manipulator_coeffs_fn = ca.Function('manipulator_coefficients', [x], [M, C, G],
                                        ['x'], ['M', 'C', 'G'])
        return manipulator_coeffs_fn

    def _derive_symbolic_com_dynamics_fn(self) -> Callable[[ca.DM, ca.DM], ca.DM]:
        """
        Derives the x_com_ddot and y_com_ddot for the center of mass of the pendulum.
        """
        # +x is right, +y is up
        q = ca.MX.sym('th', 3, 1) # type: ignore
        q_dot = ca.MX.sym('th_dot', 3, 1) # type: ignore
        u = ca.MX.sym('u', 3, 1) # type: ignore
        x = ca.vertcat(q, q_dot)
        q_ddot = self.dynamics_fn(x, u)[3:6] # Get q_ddot
        com_pos = self.com_pos(x) # Get the center of mass position
        x_com = com_pos[0]
        y_com = com_pos[1]
        # Calculate the derivatives
        H_x_com, grad_x_com = ca.hessian(x_com, q)
        H_y_com, grad_y_com = ca.hessian(y_com, q)
        x_com_ddot = q_dot.T @ H_x_com @ q_dot + grad_x_com.T @ q_ddot
        y_com_ddot = q_dot.T @ H_y_com @ q_dot + grad_y_com.T @ q_ddot
        com_ddot = ca.vertcat(x_com_ddot, y_com_ddot)
        # Create a CasADi function for the center of mass dynamics
        com_dynamics_fn = ca.Function('com_dynamics', [x, u], [com_ddot],
                                    ['x', 'u'], ['com_ddot'])
        print("Symbolic center of mass dynamics derived.")
        return com_dynamics_fn

    def animate(self, x: np.ndarray, dt: float) -> animation.FuncAnimation:
        """
        Animates the triple pendulum system based on the state trajectory.

        Args:
            x: State trajectory (shape: [n_x, N+1]).
            dt: Time step for the animation.
        """
        print("Starting animation setup...")
        theta1_opt = x[0, :]
        theta2_opt = x[1, :]
        theta3_opt = x[2, :] # Get theta3
        l1, l2, l3 = self.link1.l, self.link2.l, self.link3.l # Link lengths
        N = x.shape[1] - 1 # Number of time steps
        T = dt * N # Total time

        fig_anim, ax_anim = plt.subplots(figsize=(8, 8)) # Adjusted size
        ax_anim.set_aspect('equal')
        ax_anim.grid(True)
        ax_anim.set_xlabel("X Position (m)")
        ax_anim.set_ylabel("Y Position (m)")
        ax_anim.set_title("Triple Pendulum Animation")

        # Determine plot limits dynamically
        max_reach = l1 + l2 + l3
        plot_margin = max_reach * 0.15 # Add margin
        ax_anim.set_xlim(-max_reach - plot_margin, max_reach + plot_margin)
        ax_anim.set_ylim(-max_reach - plot_margin, max_reach + plot_margin)

        # Define plot elements
        rod1_line, = ax_anim.plot([], [], 'o-', lw=3, color='blue', markersize=6, label='Rod 1')
        rod2_line, = ax_anim.plot([], [], 'o-', lw=3, color='red', markersize=6, label='Rod 2')
        rod3_line, = ax_anim.plot([], [], 'o-', lw=3, color='green', markersize=6, label='Rod 3') # Added rod 3
        time_template = 'Time = %.2fs'
        time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes)
        ax_anim.legend(loc='upper right')

        def init_anim():
            """Initializes the animation plot."""
            rod1_line.set_data([], [])
            rod2_line.set_data([], [])
            rod3_line.set_data([], []) # Init rod 3
            time_text.set_text('')
            return rod1_line, rod2_line, rod3_line, time_text # Return all elements

        def update_anim(i):
            """Updates the animation plot for frame i."""
            th1, th2, th3 = theta1_opt[i], theta2_opt[i], theta3_opt[i]

            # Calculate joint positions using 0=up, positive=CCW convention
            x0, y0 = 0, 0
            x1 = -l1 * np.sin(th1)
            y1 =  l1 * np.cos(th1)
            x2 = x1 - l2 * np.sin(th2)
            y2 = y1 + l2 * np.cos(th2)
            x3 = x2 - l3 * np.sin(th3) # Calculate x3
            y3 = y2 + l3 * np.cos(th3) # Calculate y3

            # Update plot data
            rod1_line.set_data([x0, x1], [y0, y1])
            rod2_line.set_data([x1, x2], [y1, y2])
            rod3_line.set_data([x2, x3], [y2, y3]) # Update rod 3
            time_text.set_text(time_template % (i * dt))
            return rod1_line, rod2_line, rod3_line, time_text # Return all elements

        # Animation parameters
        FPS = 30
        frame_step = int(max(1, N // (T * FPS))) # Ensure at least 1 step
        ani_interval = int(dt * frame_step * 1000) # Interval in milliseconds

        print("Creating FuncAnimation...")
        # Create and return the animation object
        ani = animation.FuncAnimation(fig_anim, update_anim,
                                      frames=range(0, N + 1, frame_step),
                                      interval=ani_interval,
                                      blit=True, # Use blitting for efficiency
                                      init_func=init_anim,
                                      repeat=True, repeat_delay=1000)
        print("Animation object created.")
        return ani

# ==============================================================================
# Main execution block
# ==============================================================================
def main():
    # Example parameters for the triple pendulum
    link1 = Link(m=1.0, l=1.0)  # Link 1 parameters
    link2 = Link(m=1.0, l=1.0)  # Link 2 parameters
    link3 = Link(m=1.0, l=1.0)  # Link 3 parameters

    # Create the triple pendulum instance
    pendulum = TriplePendulum(link1, link2, link3, g=9.81)

    # Example state trajectory (for testing purposes)
    dt = 0.01  # Time step size
    T = 5.0  # Total time
    N = int(T/dt)  # Number of time steps
    t = np.linspace(0, T, N + 1)  # Time vector
    x = np.zeros((6, N + 1))  # State trajectory (angles and angular velocities)
    u = np.zeros((3, N))  # Control inputs (torques)
    F = np.zeros((2, N))  # Contact forces (x and y components)
    com_pos = np.zeros((2, N))  # Center of mass position
    poc_x = np.zeros((1, N))  # Center of pressure x-coordinate
    x0 = np.array([0.0, 0, 0, 0, 0, 0])  # Initial state (angles and angular velocities)
    x[:, 0] = x0  # Set initial state
    for i in range(N):
        x_dot = np.array(pendulum.dynamics_fn(x[:, i], u[:, i])).flatten()
        x[:, i + 1] = x[:, i] + x_dot * dt
        F[:, i] = np.array(pendulum.base_force_fn(x[:, i], u[:, i])).flatten()
        com_pos[:, i] = np.array(pendulum.com_pos(x[:, i])).flatten()
        poc_x[:, i] = np.array(pendulum.cop_x(x[:, i], u[:, i])).flatten()

    # Animate the triple pendulum
    ani = pendulum.animate(x, dt)

    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Contact Forces
    axes[0].plot(t[:-1], F[0, :], label='Contact X')
    axes[0].plot(t[:-1], F[1, :], label='Contact Y')
    axes[0].set_ylabel('Contact force (N)')
    axes[0].legend()
    axes[0].grid()

    # CoM Position
    axes[1].plot(t[:-1], com_pos[0, :], label='CoM X')
    axes[1].plot(t[:-1], com_pos[1, :], label='CoM Y')
    axes[1].set_ylabel('CoM position (m)')
    axes[1].legend()
    axes[1].grid()

    # CoP Position
    axes[2].plot(t[:-1], poc_x[0, :], label='CoP X')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('CoP position (m)')
    axes[2].legend()
    axes[2].grid()

    plt.xlim(0, T)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()