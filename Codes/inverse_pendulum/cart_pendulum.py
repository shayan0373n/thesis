# nonlinear_sim_scipy_rk45_direct_nodefense_v2.py

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate
# Assuming lqr_zoh_controller.py exists and LQRController is correct
from LQR import LQRController

# ==============================================================
#  Nonlinear Dynamics Function (Removed checks)
# ==============================================================
def inverted_pendulum_nonlinear_dynamics(t, x, u, params):
    """ Calculates dxdt assuming inputs are perfect """
    g, M, m, L = params
    pos, angle, vel, ang_vel = x
    u = np.asarray(u)  # Ensure u is an array
    force = u[0]
    #force = 0
    sa = math.sin(angle); ca = math.cos(angle)
    # <<< Assume denominator is never zero
    denom = M + m * sa**2
    cart_accel = (force + m * sa * (-L * ang_vel**2 + g * ca)) / denom
    ang_accel = (force * ca - m * L * ang_vel**2 * sa * ca + (M + m) * g * sa) / (L * denom)
    dxdt = np.array([vel, ang_vel, cart_accel, ang_accel])
    return dxdt

# ==============================================================
#  ODE Function for solve_ivp (Calls controller)
# ==============================================================
def system_ode_for_solver(t, x, controller, params):
    """ ODE function passed to solver. Assumes controller works. """
    # Assume compute_control returns a suitable format
    u = controller.compute_control(t, x)
    # Assume dynamics function can handle the format of u directly
    # (If u is scalar, u[0] in dynamics will raise IndexError)
    # Let's ensure u passed to dynamics is always an array format
    dxdt = inverted_pendulum_nonlinear_dynamics(t, x, u, params)
    return dxdt

# ==============================================================
#  Simulation Script (Most defensive elements removed)
# ==============================================================
if __name__ == '__main__':
    # --- Parameters & Setup ---
    g = 9.81; M = 1.0; m = 0.1; L = 0.5
    params = [g, M, m, L]
    A_lin = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, m * g / M, 0, 0],
                      [0, (M + m) * g / (M * L), 0, 0]]) # Linearized around upright position
    B_lin = np.array([[0],
                      [0],
                      [1 / M],
                      [1 / (M * L)]]) # Linearized input matrix
    Q = np.diag([1.0, 10.0, 0.1, 1.0])
    R = np.array([[0.1]])
    dt_controller = 0.02
    dt_output = 0.001
    T_sim = 5.0
    x0 = np.array([0.0, 0.0, 4.0, 0.0])

    # --- Initialize Controller ---
    # Assume LQRController init works
    lqr = LQRController(A_lin, B_lin, Q, R, dt_controller)
    lqr.reset()

    # --- Run Simulation ---
    # Assume parameters result in valid t_eval
    t_eval = np.linspace(0, T_sim, int(T_sim / dt_output) + 1)
    # Assume solve_ivp works and converges
    sol = scipy.integrate.solve_ivp(
        fun=system_ode_for_solver,
        t_span=(0, T_sim),
        y0=x0,
        method='LSODA',
        t_eval=t_eval,
        args=(lqr, params),
    )

    print("Nonlinear simulation complete.")

    # --- Extract Results ---
    # Assume sol contains valid .t and .y
    t_hist = sol.t
    x_hist = sol.y

    # --- Control Input History ---
    u_hist = lqr.get_u_hist()

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 9))
    fig.suptitle('LQR Control (Internal ZOH) on NONLINEAR Pendulum', fontsize=16)

    # Define grid shape and create axes
    grid_shape = (3, 2)
    ax_x     = plt.subplot2grid(grid_shape, (0, 0))
    ax_theta = plt.subplot2grid(grid_shape, (0, 1))
    ax_v     = plt.subplot2grid(grid_shape, (1, 0))
    ax_omega = plt.subplot2grid(grid_shape, (1, 1))
    ax_u     = plt.subplot2grid(grid_shape, (2, 0), colspan=2) # Spans bottom row

    # --- Plot data on respective axes ---
    ax_x.plot(t_hist, x_hist[0, :], label='Position (m)', color='blue')
    ax_theta.plot(t_hist, x_hist[1, :], label='Angle (rad)', color='orange')
    ax_v.plot(t_hist, x_hist[2, :], label='Velocity (m/s)', color='green')
    ax_omega.plot(t_hist, x_hist[3, :], label='Angular Velocity (rad/s)', color='red')
    ax_u.step(u_hist[:, 0], u_hist[:, 1:], where='post', label='Control Input (N)', color='purple')

    # --- Set unique labels ---
    ax_x.set_ylabel('Position (m)')
    ax_theta.set_ylabel('Angle (rad)')
    ax_v.set_ylabel('Velocity (m/s)')
    ax_omega.set_ylabel('Angular Velocity (rad/s)')
    ax_u.set_ylabel('Control Input (N)')
    ax_u.set_xlabel('Time (s)') # Only the bottom-most axis needs x-label usually

    # --- Apply common settings to all axes ---
    # Iterate through all axes belonging to the figure
    for ax in fig.axes:
        ax.grid(True)
        ax.set_xlim([0, T_sim])
        # Optional: add legend if you used 'label' in plot commands
        # ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Plots displayed.")

    # ==============================================================
    #  Animation Setup (Add this section)
    # ==============================================================
    import matplotlib.animation as animation

    # --- Get simulation results ---
    t_anim = t_hist # Use the times from the simulation
    x_pos = x_hist[0, :]
    theta = x_hist[1, :]

    # --- Set up the figure for animation ---
    fig_anim = plt.figure(figsize=(8, 5))
    ax_anim = fig_anim.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-0.5, 1.5))
    ax_anim.set_aspect('equal')
    ax_anim.grid()
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_title("Cartpole Simulation Animation")

    # --- Define elements to animate ---
    # Cart (represented as a rectangle)
    cart_width = 0.4
    cart_height = 0.2
    # Use global variables or a class to store line objects for updating
    # Use lists to make them accessible inside update function easily
    line, = ax_anim.plot([], [], 'o-', lw=2, color='blue', label='Pole') # Pole line
    cart = ax_anim.add_patch(plt.Rectangle((0, 0), cart_width, cart_height, fc='gray', label='Cart')) # Cart patch
    time_template = 'time = %.1fs'
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)

    # --- Initialization function for animation ---
    def init_anim():
        line.set_data([], [])
        cart.set_xy((-cart_width / 2, -cart_height / 2)) # Initial cart centered at 0
        time_text.set_text('')
        # Important: return the iterable of artists that need updating
        return line, cart, time_text

    # --- Animation update function ---
    def update_anim(i):
        # Cart position
        cart_x = x_pos[i]
        cart_y = 0 # Cart stays on y=0 axis

        # Pole endpoints
        pole_x1 = cart_x
        pole_y1 = cart_y + cart_height / 2 # Pivot point slightly above cart base
        # We derived theta=0 as UP, positive CCW. x_m = x - L*sin(theta), y_m = L*cos(theta) + pole_y1
        # But matplotlib plots angle=0 as horizontal right, positive CCW.
        # Let's stick to physics coords: Pole tip relative to pivot point (pole_x1, pole_y1)
        # Use the derived coords: displacement is (-L*sin(theta), L*cos(theta)) relative to pivot
        pole_x2 = pole_x1 - L * math.sin(theta[i])
        pole_y2 = pole_y1 + L * math.cos(theta[i])

        # Update plot elements
        line.set_data([pole_x1, pole_x2], [pole_y1, pole_y2])
        cart.set_xy((cart_x - cart_width / 2, cart_y - cart_height / 2))
        time_text.set_text(time_template % t_anim[i])

        # Important: return the iterable of artists that were updated
        return line, cart, time_text

    # --- Create the animation ---
    # Decide on animation speed / frame skipping
    # If t_anim has many points, animation might be too slow. Skip frames.
    frame_step = max(1, int(len(t_anim) / (T_sim * 30))) # Aim for approx 30 fps visual rate

    ani = animation.FuncAnimation(fig_anim, update_anim, frames=range(0, len(t_anim), frame_step),
                                interval=dt_output * frame_step * 1000, # Interval in ms
                                blit=True, init_func=init_anim, repeat=False)

    # Keep the animation object alive (important in some environments)
    # If running as a script, plt.show() below handles this.
    # If in Jupyter, you might need `%matplotlib notebook` or `HTML(ani.to_jshtml())`

    # Display the animation figure (this will run AFTER the time history plots)
    plt.show() # The existing plt.show() will display all figures including this one.

    # If you want to save the animation instead of showing it:
    # Requires ffmpeg or other writer installed
    # ani.save('cartpole_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # print("Animation saved to cartpole_animation.mp4")

    # ==============================================================