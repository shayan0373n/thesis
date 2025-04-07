# cartpole_swingup_trajopt.py

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Define Parameters
# ---------------------
T = 5.0     # Total time (s) - May need adjustment for cartpole
dt = 0.01   # Time step (s) - Adjust for smoother control
N = int(T / dt) # Number of control intervals
g = 9.81
# Cartpole specific params from your previous simulation
M = 1.0     # Cart mass (kg)
m = 0.1     # Pole mass (kg)
L = 0.5     # Pole length (m) - Corresponds to 'l' in derivations

R_cost = 0.01 # Weight for control cost term

# 2. Set up CasADi Optimization Problem
# -------------------------------------
opti = ca.Opti()

# Define Optimization Variables (State + Control)
X = opti.variable(4, N + 1) # State: [x, theta, x_dot, theta_dot]
pos     = X[0, :] # Cart position
angle   = X[1, :] # Pole angle (0=upright, pi=down)
vel     = X[2, :] # Cart velocity
ang_vel = X[3, :] # Pole angular velocity
U = opti.variable(1, N)     # Control: u (Force F)

# 3. Define Cost Function (Minimize Control Effort)
# -------------------------------------------------
# Minimize integral of R*u^2 dt
cost = ca.sumsqr(U) * R_cost * dt
opti.minimize(cost)

# 4. Define System Dynamics Constraints (Trapezoid Method)
# -------------------------------------------------------
# Helper function for cartpole dynamics: dX/dt = f(X, u)
def cartpole_dynamics(x_state, u, M, m, L, g):
    x, th, x_d, th_d = x_state[0], x_state[1], x_state[2], x_state[3]
    force = u

    # Use CasADi's trig functions for symbolic differentiation
    sina = ca.sin(th)
    cosa = ca.cos(th)
    # Denominator term (ensure it doesn't become zero)
    denom = M + m * sina**2
    # Avoid division by zero (maybe add small epsilon if needed, but unlikely here)
    # denom = M + m * sa**2 + 1e-9

    # Calculate accelerations using equations derived earlier
    # x_ddot = ( F - m*l*theta_dot^2*sin(theta) + m*g*sin(theta)cos(theta) ) / ( M + m*sin^2(theta) )
    x_dd = (force - m * L * th_d**2 * sina + m * g * sina * cosa) / denom

    # theta_ddot = ( F*cos(theta) - m*l*theta_dot^2*sin(theta)cos(theta) + (M + m)*g*sin(theta) ) / ( l * (M + m*sin^2(theta)) )
    th_dd = (force * cosa - m * L * th_d**2 * sina * cosa + (M + m) * g * sina) / (L * denom)

    # Return state derivatives vector
    return ca.vertcat(x_d, th_d, x_dd, th_dd)

# Loop through intervals and apply trapezoid integration constraint
for k in range(N):
    x_k = X[:, k]
    x_k_plus_1 = X[:, k+1]
    u_k = U[:, k] # Control applied over the interval k

    # Calculate state derivatives at the start and end of the interval
    f_k = cartpole_dynamics(x_k, u_k, M, m, L, g)
    f_k_plus_1 = cartpole_dynamics(x_k_plus_1, u_k, M, m, L, g) # Use u_k at k+1 for trapezoid

    # Trapezoid rule constraint: enforce system dynamics across interval
    opti.subject_to( x_k_plus_1 == x_k + (dt / 2.0) * (f_k + f_k_plus_1) )

    # Optional: Add control limits (if needed)
    # opti.subject_to(opti.bounded(-2.0, u_k, 2.0)) # Example limit on control force

    # Optional: Add state limits (if needed)
    opti.subject_to(opti.bounded(-0.5, x_k[0], 0.5)) # Example limit on cart position

# 5. Define Boundary Conditions
# -----------------------------
opti.subject_to(pos[0] == 0)        # Initial cart position = 0
opti.subject_to(angle[0] == np.pi)  # Initial pole angle = pi (hanging down)
opti.subject_to(vel[0] == 0)        # Initial cart velocity = 0
opti.subject_to(ang_vel[0] == 0)    # Initial pole angular velocity = 0

opti.subject_to(pos[N] == 0)        # Final cart position = 0 (Try this first)
opti.subject_to(angle[N] == 0)      # Final pole angle = 0 (upright)
opti.subject_to(vel[N] == 0)        # Final cart velocity = 0
opti.subject_to(ang_vel[N] == 0)    # Final pole angular velocity = 0

# 6. Optional: Add Control/State Limits (Uncomment if needed)
# ----------------------------------------------------------
# u_max = 25.0  # Max force - adjust as needed
# opti.subject_to(opti.bounded(-u_max, U, u_max))
# pos_limit = 2.4 # Limit cart travel +/- from origin
# opti.subject_to(opti.bounded(-pos_limit, pos, pos_limit))

# 7. Provide Initial Guess
# ------------------------
# Simple linear interpolation often works reasonably well
opti.set_initial(pos, np.linspace(0, 0, N + 1))            # Guess cart stays at 0
opti.set_initial(angle, np.linspace(np.pi, 0, N + 1))      # Guess linear angle swing
opti.set_initial(vel, np.zeros(N + 1))                   # Guess zero velocity
opti.set_initial(ang_vel, np.zeros(N + 1))               # Guess zero angular velocity
opti.set_initial(U, np.zeros(N))                         # Guess zero control effort

# 8. Configure Solver (Using IPOPT)
# ---------------------------------
p_opts = {"expand": True} # CasADi specific plugin option
s_opts = {
    "max_iter": 3000,       # Increase max iterations if needed
    "print_level": 5,       # IPOPT verbosity (0=silent)
    #"tol": 1e-6,           # Solver tolerance
    #"acceptable_tol": 1e-4 # Accept slightly less optimal if main tolerance hard to meet
}
opti.solver('ipopt', p_opts, s_opts)

# 9. Solve the NLP
# ----------------
print(f"Solving the cartpole swing-up optimal control problem (T={T}s, N={N})...")
try:
    sol = opti.solve()
    print("Optimal solution found!")

    # 10. Extract Results
    # -------------------
    pos_opt = sol.value(pos)
    angle_opt = sol.value(angle)
    vel_opt = sol.value(vel)
    ang_vel_opt = sol.value(ang_vel)
    u_opt = sol.value(U)
    t = np.linspace(0, T, N + 1)
    t_u = np.linspace(0, T - dt, N)

    # 11. Plot Results
    # ----------------
    plt.figure("Cartpole Trajectory Optimization Results", figsize=(10, 10))
    plt.suptitle(f'Cartpole Swing-up Trajectory (T={T}s, N={N}, R={R_cost})', fontsize=14)
    plt.subplot(5, 1, 1)
    plt.plot(t, pos_opt, label='x (m)')
    plt.ylabel('Position'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 2)
    plt.plot(t, angle_opt, label='theta (rad)')
    plt.plot(t, np.zeros_like(t), 'k--', label='target')
    plt.ylabel('Angle'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(t, vel_opt, label='x_dot (m/s)')
    plt.ylabel('Velocity'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(t, ang_vel_opt, label='theta_dot (rad/s)')
    plt.ylabel('Ang Vel'); plt.grid(True); plt.legend()
    plt.subplot(5, 1, 5)
    plt.step(t_u, u_opt, 'r-', where='post', label='u (Force N)')
    plt.ylabel('Control'); plt.xlabel('Time (s)'); plt.grid(True); plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # 12. Animation
    # -------------
    fig_anim = plt.figure("Cartpole Animation", figsize=(8, 5))
    ax_anim = fig_anim.add_subplot(111, autoscale_on=False,
                                   xlim=(-max(1.5, np.max(np.abs(pos_opt))*1.2)),
                                   ylim=(-L*1.5, L*1.5))
    ax_anim.set_aspect('equal')
    ax_anim.grid()
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_ylabel("Height (m)")
    ax_anim.set_title("Cartpole Swing-up Animation")

    cart_width = 0.4
    cart_height = 0.2
    # Define plot elements (use lists for easy access in nested functions if needed)
    pole_line, = ax_anim.plot([], [], 'o-', lw=2, color='blue', markersize=6)
    cart_patch = plt.Rectangle((np.nan, np.nan), cart_width, cart_height, fc='gray')
    ax_anim.add_patch(cart_patch)
    time_template = 'time = %.2fs'
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)

    def init_anim_cart():
        """Initializes the animation plot."""
        pole_line.set_data([], [])
        cart_patch.set_xy((-cart_width / 2, -cart_height / 2)) # Initial position
        time_text.set_text('')
        return pole_line, cart_patch, time_text

    def update_anim_cart(i):
        """Updates the animation plot for frame i."""
        cart_x = pos_opt[i]
        cart_y = 0 # Cart vertical position
        angle_i = angle_opt[i] # Pole angle (0 is up)

        # Pole pivot is assumed at the center of the cart base
        pivot_x = cart_x
        pivot_y = cart_y

        # Pole endpoint calculation (using theta=0 as UP)
        # Tip coordinates relative to pivot: (-L*sin(theta), L*cos(theta))
        pole_end_x = pivot_x - L * np.sin(angle_i)
        pole_end_y = pivot_y + L * np.cos(angle_i)

        pole_line.set_data([pivot_x, pole_end_x], [pivot_y, pole_end_y])
        cart_patch.set_xy((cart_x - cart_width / 2, cart_y - cart_height / 2))
        time_text.set_text(time_template % (i * dt))
        return pole_line, cart_patch, time_text

    # Create and display the animation
    frame_step = max(1, N // int(T * 30)) # Aim for ~30fps visual rate
    ani = animation.FuncAnimation(fig_anim, update_anim_cart,
                                  frames=range(0, N + 1, frame_step),
                                  interval=dt * frame_step * 1000,
                                  blit=True, init_func=init_anim_cart,
                                  repeat=False)

    plt.show()

except RuntimeError as e:
    print(f"\nSolver failed: {e}\n")
    print("Cartpole swing-up is more challenging. Suggestions:")
    print(" - Increase maneuver time T (e.g., T=4 or T=5)")
    print(" - Increase number of intervals N (e.g., N=150 or N=200)")
    print(" - Relax final cart position constraint: Comment out 'opti.subject_to(pos[N] == 0)'")
    print(" - Add/adjust control limits: Uncomment and tune 'opti.subject_to(opti.bounded(...))'")
    print(" - Adjust control cost weight R_cost")
    print(" - Check IPOPT solver options or try providing a better initial guess")