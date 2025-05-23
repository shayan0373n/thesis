import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from triple_pend import TriplePendulum, Link
from hybrid_trajectory_optimizer import HybridTrajectoryOptimizer, HybridSystemParams, Mode, Transition

class STS(TriplePendulum):
    def __init__(self, link1: Link, link2: Link, link3: Link,
                 seat_pos: tuple | None = None,
                 foot_x_lims: tuple | None = None,
                 **kwargs):
        super().__init__(link1, link2, link3, **kwargs)
        if seat_pos is None:
            self.seat_pos = (-link2.l, link1.l)
        else:
            self.seat_pos = seat_pos

        self.foot_x_lims = foot_x_lims

    def sit_dynamics_fn(self, x, u):
        """
        Calculate the dynamics of the sit system.
        x: state vector (6D)
        u: control input (3D)
        """
        hip_torque = u[2]
        torso_angle = x[2]
        torso_angle_dot = x[5]
        # Parameters
        g = self.g
        torso_l = self.link3.l  # Length of torso link
        torso_m = self.link3.m
        # We have torso I around the CoM, but we need it around the pivot point
        # Using parallel axis theorem: I = I_com + m * d^2
        # where d is the distance from the CoM to the pivot point
        torso_I = self.link3.I + torso_m * (torso_l / 2)**2 # Assuming pivot is at the top of the torso
        # Sit model is a simple inverted pendulum
        # Dynamics equations
        torso_angle_ddot = (torso_m * g * torso_l / 2 * ca.sin(torso_angle) + hip_torque) / torso_I
        return ca.vertcat(0, 0, torso_angle_dot, 0, 0, torso_angle_ddot)
    
    def stand_dynamics_fn(self, x, u):
        """
        Calculate the dynamics of the stand system.
        x: state vector (6D)
        u: control input (3D)
        """
        return self.dynamics_fn(x, u)
    
    def sit_contact_force_fn(self, x, u):
        """
        Calculate the contact force during the sit phase.
        x: state vector (6D)
        u: control input (3D)
        """
        # Constraints: pos(joint_2) = C
        l1 = self.link1.l
        l2 = self.link2.l
        q = x[0:3]
        q_dot = x[3:]
        q1 = q[0]
        q2 = q[1]
        q1_dot = q_dot[0]
        q2_dot = q_dot[1]
        # Jacobian of the constraint w.r.t. the state
        Jh = ca.vertcat(
            ca.horzcat(-l1 * ca.cos(q1), -l2 * ca.cos(q2), 0),
            ca.horzcat(-l1 * ca.sin(q1), -l2 * ca.sin(q2), 0),
        )
        Jh_dot = ca.vertcat(
            ca.horzcat(l1 * ca.sin(q1) * q1_dot, l2 * ca.sin(q2) * q2_dot, 0),
            ca.horzcat(-l1 * ca.cos(q1) * q1_dot, -l2 * ca.cos(q2) * q2_dot, 0),
        )
        # Contact force
        M, C, G = self.manipulator_coefficients_fn(x)
        # CF = (Jh @ M^-1 @ Jh.T) @ (-Jh_dot @ q_dot - Jh @ M^-1 @ (u - C - G))
        Minv = ca.inv(M, 'symbolicqr')
        CF = ca.solve(Jh @ Minv @ Jh.T, -Jh_dot @ q_dot - Jh @ ca.solve(M, u - C - G, 'symbolicqr'), 'symbolicqr')
        return CF
    
    def j2_pos(self, x):
        """
        Calculate the position of joint 2.
        x: state vector (6D)
        """
        q1 = x[0]
        q2 = x[1]
        # x position of joint 2
        x_j2 = -self.link1.l * ca.sin(q1) - self.link2.l * ca.sin(q2)
        # y position of joint 2
        y_j2 = self.link1.l * ca.cos(q1) + self.link2.l * ca.cos(q2)
        return ca.vertcat(x_j2, y_j2)
    

class STSTrajectoryOptimizer:
    def __init__(self, sts: STS,
                 state_weights: tuple = (1, 1, 1, 1, 1, 1),
                 control_weights: tuple = (1, 1, 1)):
        self.sts = sts
        self.Q = ca.diag(state_weights)
        self.R = ca.diag(control_weights)

    def sit_stand_map_fn(self, x, u):
        """
        Map function to transform the state from sit to stand phase.
        x: state vector (6D)
        u: control input (3D)
        """
        # In this case, we don't need to change the state
        return x
    
    def cost_function(self, X_vars: list, U_vars: list, H_vars: list):
        total_cost = 0
        # total_cost += 1E7 * self.cost_function_cop(X_vars, U_vars, H_vars)
        total_cost += self.cost_function_control_effort(X_vars, U_vars, H_vars)
        # total_cost += 1E5 * self.cost_function_time(X_vars, U_vars, H_vars)
        # total_cost += 1E5 * self.cost_function_positive_energy(X_vars, U_vars, H_vars)
        # total_cost += 1E3 * self.cost_function_max_torque(X_vars, U_vars, H_vars)
        return total_cost
    
    def cost_function_max_torque(self, X_vars: list, U_vars: list, H_vars: list):
        """
        Cost function to minimize the maximum torque.
        X_vars: state variables
        U_vars: control variables
        H_vars: hybrid variables
        """
        U = ca.hcat(U_vars)
        # Calculate the maximum torque
        max_torque = ca.norm_inf(U)
        # Cost function: minimize the maximum torque
        return max_torque
    
    def cost_function_cop(self, X_vars: list, U_vars: list, H_vars: list):
        """
        Cost function to minimize the center of pressure in the standing phase.
        X_vars: state variables
        U_vars: control variables
        H_vars: hybrid variables
        """
        total_cost = 0
        X = X_vars[1] # Use the state variables from the stand phase
        U = U_vars[1]
        H = H_vars[1]
        for n in range(H.shape[1]):
            x = X[:, n]
            u = U[:, n]
            h = H[0, n]
            # Calculate the CoP
            cop_x = self.sts.cop_x(x, u)
            # Cost function: minimize the CoP
            total_cost += cop_x**2 * h
        return total_cost

    def cost_function_control_effort(self, X_vars: list, U_vars: list, H_vars: list):
        """
        Cost function to minimize the control effort.
        X_vars: state variables
        U_vars: control variables
        H_vars: hybrid variables
        """
        total_cost = 0
        for u, h in zip(U_vars, H_vars):
            for n in range(h.shape[1]):
                # Cost function: minimize the control effort
                uk = u[:, n]
                uk_next = u[:, n + 1]
                hk = h[0, n]
                total_cost += hk * (uk.T @ self.R @ uk + uk_next.T @ self.R @ uk_next)
        return total_cost
    
    def cost_function_positive_energy(self, X_vars: list, U_vars: list, H_vars: list):
        """
        Cost function to minimize the positive energy.
        X_vars: state variables
        U_vars: control variables
        H_vars: hybrid variables
        """
        total_cost = 0
        for X, U, H in zip(X_vars, U_vars, H_vars):
            for n in range(H.shape[1]):
                # Cost function: minimize the positive energy
                q_dot = X[3:, n]
                v = ca.vertcat(q_dot[0],
                               q_dot[1] - q_dot[0],
                               q_dot[2] - q_dot[1])
                u = U[:, n]
                h = H[0, n]
                # Calculate the kinetic energy
                T = u.T @ v
                # Calculate the positive energy
                T_pos = ca.fmax(0, T)
                total_cost += h * T_pos
        return total_cost
    
    def cost_function_time(self, X_vars: list, U_vars: list, H_vars: list):
        """
        Cost function to minimize the time.
        X_vars: state variables
        U_vars: control variables
        H_vars: hybrid variables
        """
        total_cost = 0
        for h in H_vars:
            total_cost += ca.sum(h)
        return total_cost

    def j2_y_pos_higher_than_seat_constraint_fn(self, opti, Xk, Uk, Hk):
        """
        Constraint function to ensure that the y position of joint 2 is higher than the seat in the stand phase.
        """
        for n in range(Xk.shape[1]):
            x = Xk[:, n]
            y_j2 = self.sts.j2_pos(x)[1]
            opti.subject_to(y_j2 >= self.sts.seat_pos[1])

    def base_contact_no_slip_global_constraint_fn(self, opti, X_vars, U_vars, H_vars):
        """
        Constraint function to ensure that the base does not slip.
        Applies to all modes.
        """
        for Xk, Uk, in zip(X_vars, U_vars):
            for n in range(Xk.shape[1]):
                x = Xk[:, n]
                u = Uk[:, n]
                # Base contact force
                base_force = self.sts.base_force_fn(x, u)
                # Base contact force in the y direction
                base_force_y = base_force[1]
                base_force_x = base_force[0]
                # Constraint: base contact force in the y direction should be greater than 0
                opti.subject_to(base_force_y >= 0)
                if 'mu' in self.sts.kwargs:
                    # Constraint: base contact force in the x direction should be less than the friction force
                    mu = self.sts.kwargs['mu']
                    opti.subject_to(ca.fabs(base_force_x) <= mu * base_force_y)
                else:
                    print("Warning: Friction coefficient not provided. No friction constraint applied.")

    def seat_contact_no_slip_constraint_fn(self, opti, Xk, Uk, Hk):
        """
        Constraint function to ensure that the seat does not slip.
        """
        for n in range(Xk.shape[1] - 1): # Exclude the last knot point
            x = Xk[:, n]
            u = Uk[:, n]
            # Seat contact force
            seat_force = self.sts.sit_contact_force_fn(x, u)
            # Seat contact force in the y direction
            seat_force_y = seat_force[1]
            seat_force_x = seat_force[0]
            # Constraint: seat contact force in the y direction should be greater than 0
            opti.subject_to(seat_force_y >= 0)
            if 'mu' in self.sts.kwargs:
                # Constraint: seat contact force in the x direction should be less than the friction force
                mu = self.sts.kwargs['mu']
                opti.subject_to(ca.fabs(seat_force_x) <= mu * seat_force_y)
            else:
                print("Warning: Friction coefficient not provided. No friction constraint applied.")

    def equal_time_segment_global_constraint_fn(self, opti, X_vars, U_vars, H_vars):
        """
        Constraint function to ensure that the time segments are equal.
        """
        for Hk in H_vars:
            for n in range(Hk.shape[1] - 1):
                hk = Hk[0, n]
                hk_next = Hk[0, n + 1]
                opti.subject_to(hk == hk_next)

    def knee_joint_constraint_fn(self, opti, Xk, Uk, Hk):
        """
        Constraint function to ensure that the knee joint cannot bend backwards.
        """
        for n in range(Xk.shape[1]):
            x = Xk[:, n]
            q1 = x[0]
            q2 = x[1]
            knee_joint_angle = q2 - q1
            # Constraint: knee joint angle should be greater than 0
            opti.subject_to(knee_joint_angle >= 0)

    def initial_x_seat_pos_constraint_fn(self, opti, Xk, Uk, Hk):
        """
        Constraint function to ensure that the initial state is such that the j2 positions equals the seat position.
        """
        j2_pos_initial = self.sts.j2_pos(Xk[:, 0])
        # Constraint: j2 position should be equal to the seat position
        if self.sts.seat_pos[0] is not None:
            opti.subject_to(j2_pos_initial[0] == self.sts.seat_pos[0])
        if self.sts.seat_pos[1] is not None:
            opti.subject_to(j2_pos_initial[1] == self.sts.seat_pos[1])

    def COP_constraint_fn(self, opti, Xk, Uk, Hk):
        """
        Constraint function to ensure that the COP remains inside the support polygon.
        """
        if sts.foot_x_lims is None:
            print("Warning: Foot x limits not provided. No COP constraint applied.")
            return
        for n in range(Xk.shape[1]):
            x = Xk[:, n]
            u = Uk[:, n]
            # Calculate the CoP
            cop_x = self.sts.cop_x(x, u)
            # Constraint: CoP should be inside the foot limits
            opti.subject_to(cop_x >= self.sts.foot_x_lims[0])
            opti.subject_to(cop_x <= self.sts.foot_x_lims[1])

    def control_effort_vanishes_global_constraint_fn(self, opti, X_vars, U_vars, H_vars):
        """
        Constraint function to ensure that the control effort stops out.
        """
        U_last = U_vars[-1]
        u_last = U_last[:, -1]
        # Constraint: last control input should be equal to 0
        opti.subject_to(u_last == 0)
            

###
MU = 0.5 # Coefficient of friction
T_UB = 10.0 # Upper bound on total time
N_knots = 25 # Number of knot points per segment
M = 74 # Mass of the pendulum (kg)
L = 1.74 # Length of the pendulum (m)
lleg = Link(m=0.186 / 2 * M, l=0.478 / 2 * L)  # Lower leg link
uleg = Link(m=0.4 / 2 * M, l=0.489 / 2 * L)  # Upper leg link
torso = Link(m=1.356 / 2 * M, l=0.932 / 2 * L)  # Torso link
# lleg = Link(m=1, l=1)  # Link 1
# uleg = Link(m=1, l=1)  # Link 2
# torso = Link(m=1, l=1)  # Link 3
SEAT_POS = (-0.2, 0.4) # Seat position
FOOT_X_LIMS = (-0.05, 0.2) # Foot x limits
R_weights = (1, 3, 1) # Control weights

U_LB = (None,) * 3 # Lower bound on control inputs
U_UB = (None,) * 3 # Upper bound on control inputs

# Define the system
sts = STS(link1=lleg, link2=uleg, link3=torso,
          mu=MU, seat_pos=SEAT_POS, foot_x_lims=FOOT_X_LIMS)
sts_optimizer = STSTrajectoryOptimizer(sts,
                                       control_weights=R_weights,
)

x0 = (None, None, 0, 0, 0, 0) # Initial state
xf = (0, 0, 0, 0, 0, 0)

sit_mode_initial_guess = np.array([0, np.pi / 2, 0, 0, 0, 0]).reshape(-1, 1) 

mode_sit = Mode(
    name='sit',
    dynamics_fn=sts.sit_dynamics_fn,
    n_x=6,
    n_u=3,
    constraints_fns=[
                sts_optimizer.seat_contact_no_slip_constraint_fn,
                sts_optimizer.initial_x_seat_pos_constraint_fn,
                     ],
    initial_x_guess=sit_mode_initial_guess,
    u_lb=U_LB,
    u_ub=U_UB,
)

mode_stand = Mode(
    name='stand',
    dynamics_fn=sts.stand_dynamics_fn,
    n_x=6,
    n_u=3,
    constraints_fns=[   
                sts_optimizer.COP_constraint_fn,
                sts_optimizer.knee_joint_constraint_fn,
                    ],
    u_lb=U_LB,
    u_ub=U_UB,
)

transition_sit_stand = Transition(
    guard_fn=lambda x, u: sts.sit_contact_force_fn(x, u)[1], # y component of the contact force
    reset_map_fn=sts_optimizer.sit_stand_map_fn,
)

hybrid_params = HybridSystemParams(
    mode_sequence=[mode_sit, mode_stand],
    transitions=[transition_sit_stand],
    x0=x0, # Initial state
    xf=xf, # Final state
    # T = 1.5, # Total time
    T_ub = T_UB, # Upper bound on total time
    min_time_step = 1E-6,
    num_knot_points_per_segment = N_knots, # Number of knot points per segment
)

hybrid_optimizer = HybridTrajectoryOptimizer(
    params=hybrid_params,
    cost_fn=sts_optimizer.cost_function,
    global_constraints_fns=[sts_optimizer.base_contact_no_slip_global_constraint_fn,
                            sts_optimizer.equal_time_segment_global_constraint_fn,
                            sts_optimizer.control_effort_vanishes_global_constraint_fn,
                            ],
)

# Solve the optimization problem
solver_name = 'ipopt'
solver_options = {
    'max_iter': 10000,
    'fixed_variable_treatment': 'make_constraint',
    'tol': 1E-6,
}
solution = hybrid_optimizer.solve(solver_name=solver_name, solver_options=solver_options)

times = solution['times']
states = solution['states']
controls = solution['controls']
t_all = np.hstack(times)
x_all = np.hstack(states)
u_all = np.hstack(controls)
t_linear = np.linspace(0, t_all[-1], 100)
dt = t_linear[1] - t_linear[0]
x_sampled = np.zeros((x_all.shape[0], len(t_linear)))
u_sampled = np.zeros((u_all.shape[0], len(t_linear)))
for i in range(x_all.shape[0]):
    # Interpolate the state variables
    x_sampled[i, :] = np.interp(t_linear, t_all, x_all[i, :])
for i in range(u_all.shape[0]):
    # Interpolate the control variables
    u_sampled[i, :] = np.interp(t_linear, t_all, u_all[i, :])
com_sampled = sts.com_pos(x_sampled)
cop_sampled = np.zeros((1, len(t_linear)))
for i in range(len(t_linear)):
    # Calculate the CoP
    cop_sampled[:, i] = sts.cop_x(x_sampled[:, i], u_sampled[:, i])
ani = sts.animate(x_sampled, dt=dt)

fig, ax = plt.subplots(5, 1, figsize=(10, 8))
ax[0].plot(t_linear, x_sampled[:3, :].T, label=['q1', 'q2', 'q3'])
ax[0].set_title('Joint Angles')
ax[0].set_ylabel('Angle (rad)')
ax[1].plot(t_linear, x_sampled[3:, :].T, label=['q1_dot', 'q2_dot', 'q3_dot'])
ax[1].set_title('Joint Velocities')
ax[1].set_ylabel('Velocity (rad/s)')
ax[2].plot(t_linear, u_sampled.T, label=['u1', 'u2', 'u3'])
ax[2].set_title('Control Inputs')
ax[2].set_ylabel('Torque (Nm)')
ax[3].plot(t_linear, com_sampled.T, label=['CoM x', 'CoM y'])
ax[3].set_title('Center of Mass')
ax[3].set_ylabel('CoM (m)')
ax[4].plot(t_linear, cop_sampled.T, label='CoP')
ax[4].set_title('Center of Pressure')
ax[4].set_ylabel('CoP (m)')
ax[-1].set_xlabel('Time (s)')
for a in ax:
    a.legend()
    a.grid()
    a.set_xlim([0, t_linear[-1]])
plt.show()
