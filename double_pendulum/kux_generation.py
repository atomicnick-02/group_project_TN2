import numpy as np
import matplotlib.pyplot as plt
import jax
import cyipopt
import jax.numpy as jnp
import os
from collections import namedtuple

jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────
# Model parameters  (cloudpendulum dp_fwd_inv_dynamics, system-identified)
# ─────────────────────────────────────────────
ModelParams = namedtuple(
    "ModelParams",
    "m1 m2 l1 l2 lc1 lc2 I1 I2 b1 b2 mu1 mu2 g gr1 gr2 Ir torque_limit",
)

P = ModelParams(
    m1=0.10548177618443695,
    m2=0.07619744360415454,
    l1=0.05,
    l2=0.05,
    lc1=0.05,
    lc2=0.03670036749567022,
    I1=0.00046166221821039165,
    I2=0.00023702395072092597,
    b1=7.634058385430087e-12,
    b2=0.0005106535523065844,
    mu1=0.00305,
    mu2=0.0007777,
    g=9.81,
    gr1=403.0,
    gr2=379.0,
    Ir=0.0,                 # rotor inertia: not provided by the notebook
    torque_limit=0.07,      # hard actuator limit [Nm] (old_pendulum/simul.ipynb)
)

# arctan(FRICTION_SCALE · q̇) is the smooth sign(q̇) used for Coulomb friction.
FRICTION_SCALE = 100.0


# ─────────────────────────────────────────────
# Dynamics (distributed-inertia model from the notebook)
# ─────────────────────────────────────────────
def M(x):
    q2 = x[1]
    c2 = jnp.cos(q2)
    rotor = P.gr1 ** 2 * P.Ir + P.Ir
    m00 = P.I1 + P.I2 + P.l1 ** 2 * P.m2 + 2 * P.l1 * P.m2 * P.lc2 * c2 + rotor
    m01 = P.I2 + P.l1 * P.m2 * P.lc2 * c2
    m11 = P.I2
    return jnp.array([[m00, m01], [m01, m11]])


def C(x):
    q2, q1_dot, q2_dot = x[1], x[2], x[3]
    s2 = jnp.sin(q2)
    k = P.l1 * P.m2 * P.lc2
    c00 = -2.0 * q2_dot * k * s2
    c01 = -q2_dot * k * s2
    c10 = q1_dot * k * s2
    return jnp.array([[c00, c01], [c10, 0.0]])


def G(x):
    q1, q2 = x[0], x[1]
    g0 = -P.g * P.m1 * P.lc1 * jnp.sin(q1) \
         - P.g * P.m2 * (P.l1 * jnp.sin(q1) + P.lc2 * jnp.sin(q1 + q2))
    g1 = -P.g * P.m2 * P.lc2 * jnp.sin(q1 + q2)
    return jnp.array([g0, g1])


def friction(dq):
    # Viscous + arctan-smoothed Coulomb friction (cloudpendulum sys-id).
    f0 = P.b1 * dq[0] + P.mu1 * jnp.arctan(FRICTION_SCALE * dq[0])
    f1 = P.b2 * dq[1] + P.mu2 * jnp.arctan(FRICTION_SCALE * dq[1])
    return jnp.array([f0, f1])


def dynamics(x, u):
    u = u.reshape(2)
    dq = x[2:].reshape(2)
    rhs = u + G(x) - C(x) @ dq - friction(dq)
    ddq = jnp.linalg.solve(M(x), rhs)
    return jnp.concatenate([dq, ddq]).flatten()


@jax.jit
def rk4_step(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def state_dist(x1, x2):
	"""xi = [q1, q2, q1_dot, q2_dot]
	Compute the distance between two states,wrapping angles properly 
	"""
	diff = x1 - x2
	diff = diff.at[0].set((diff[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
	diff = diff.at[1].set((diff[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
	return diff


# ─────────────────────────────────────────────
# QP problem Equations
# ─────────────────────────────────────────────
def l(xk, uk, x_goal):
    se = xk - x_goal
    return se.T @ Q @ se + uk.T @ R @ uk

def total_objective(z, steps, nx, nu, x_goal, dt=0.05):
    # z is the flattened decision vector [x0, x1, ..., xN, u0, u1, ..., uN-1]
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps, nu))
    
    point_costs = jax.vmap(lambda x, u: l(x, u, x_goal)*0.5)(X, U)
    # 1. Final State Cost
    final_diff = X[-1] - x_goal
    point_costs = point_costs.at[-1].set(final_diff.T @ Qfin @ final_diff)
    # 2. Apply Weights 
    cost = jnp.sum(point_costs) 
    
    return cost

def l_deriv(xk, uk, x_goal):
    se = xk - x_goal
    grad_x = 2 * Q @ se
    grad_u = 2 * R @ uk
    return grad_x, grad_u

def total_objective_deriv(z, steps, nx, nu, x_goal, dt=0.05):
    # 1. Reshape z back into state and input trajectories
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps , nu))
    
    # 2. Compute raw gradients at each time step using l_deriv
    point_grads_X, point_grads_U = jax.vmap(lambda x, u: l_deriv(x, u, x_goal))(X, U)
    
    # 3. Apply the weighting 
    weights = jnp.ones(steps)* 0.5

    # Broadcast weights to match the shape of gradients
    grad_X_weighted = point_grads_X * weights[:, None]
    grad_U_weighted = point_grads_U * weights[:, None]
    
    # 4. Add Terminal Cost Gradient
    # The terminal cost is applied only to the last state X[-1]
    terminal_grad_X = 2 * Qfin @ (X[-1] - x_goal)
    grad_X_weighted = grad_X_weighted.at[-1].set(terminal_grad_X)
    
    # 5. Flatten and concatenate to form the full gradient vector
    # Expected size: steps*nx + steps*nu
    grad = jnp.concatenate([grad_X_weighted.flatten(), grad_U_weighted.flatten()])
    
    return grad

def trapezoidal_collocation(x_k, x_kp1, u_k, u_kp1, dynamics, dt):
    """
    Enforces a linear consistency between two nodes.
    x_k, x_kp1: states at node k and k+1
    u_k, u_kp1: controls at node k and k+1
    returns: Scalar constraint value showing the collocation error
    """
    # 1. Compute dynamics at the nodes
    f_k = dynamics(x_k, u_k)
    f_kp1 = dynamics(x_kp1, u_kp1)
    
    # 2. Trapezoidal Rule constraint
    collocation_constraint = x_kp1 - x_k - (dt / 2.0) * (f_k + f_kp1)
    
    return collocation_constraint.flatten()

def constraints(z, steps, nx, nu, x0, x_goal, dt=0.05):
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps, nu)) 
    
    # Initial state constraint
    c_init = X[0] - x0
    # We have 'steps - 1' intervals to constrain
    x_curr = X[:-1] # size: steps - 1
    x_next = X[1:]  # size: steps - 1
    
    u_curr = U[:-1] # size: steps - 1 
    u_next = U[1:]  # size: steps - 1 
    
    hs_vmap = jax.vmap(lambda xk, xkp1, uk, ukp1: 
                       trapezoidal_collocation(xk, xkp1, uk, ukp1, dynamics, dt))
    
    c_dyn = hs_vmap(x_curr, x_next, u_curr, u_next).flatten()
    
    # Final state constraint
    c_final = X[-1] - x_goal
    return jnp.concatenate([c_init, c_dyn, c_final])



class Problem:
	"""
 	Defines a minimization problem with the following form:
		min f(z) (z = [x0, x1, ..., xN, u0, u1, ..., uN-1])
	s.t. h(z) = 0
		lb <= z <= ub
	f(z) = objective function
	"""
	def __init__(self, steps, nx, nu, x0, xgoal, dt):
		self.steps = steps
		self.nx = nx # Number of states
		self.nu = nu # Number of inputs
		print(f"Problem initialized with {steps} steps, {nx} states, {nu} inputs.")
		# JIT-compiled functions with fixed problem dimensions and xgoal/x0
		self._obj_jit = jax.jit(lambda z: total_objective(z, self.steps, self.nx, self.nu, xgoal)) # f(z)
		# self._grad_jit = jax.jit(jax.grad(lambda z: total_objective(z, self.steps, self.nx, self.nu, xgoal))) #	nablaf(z)
		self._grad_jit = lambda z: total_objective_deriv(z, self.steps, self.nx, self.nu, xgoal)
		
  		# pass x0 and xgoal into constraints/jacobian to match signature
		self._cons_jit = jax.jit(lambda z: constraints(z, self.steps, self.nx, self.nu, x0, xgoal, dt)) # h(z)
		self._jac_jit = jax.jit(jax.jacobian(lambda z: constraints(z, self.steps, self.nx, self.nu, x0, xgoal, dt))) # nablag(z)

	def objective(self, z):
		return float(self._obj_jit(z))

	def gradient(self, z):
		return np.array(self._grad_jit(z)).ravel().astype(np.float64)

	def constraints(self, z):
		return np.array(self._cons_jit(z))

	def jacobian(self, z):
		jac = self._jac_jit(z)
		# return flattened Jacobian (rows*cols) - matches earlier usage
		return jac.ravel()

Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin =  Q
R = jnp.diag(jnp.array([1, 1])) * 0.01


# Simulation parameters
T = 2 # Total simulation time in seconds
dt = 0.05
steps = int(T / dt) + 1
time_span = jnp.linspace(0, T, steps)

# Initial state
x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
xgoal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
# Storage for history
x_history = [x0]
u_history = []
force_history = []

# Initialize x_traj with random positions, but keep start and end fixed
key = jax.random.PRNGKey(0)
x_traj = jax.random.uniform(key, shape=(steps, 4), minval=-jnp.pi, maxval=jnp.pi)
u_traj = jax.random.uniform(key, shape=(steps, 2), minval=P.torque_limit, maxval=-P.torque_limit)

x_traj = x_traj.at[0].set(x0)
x_traj = x_traj.at[-1].set(xgoal)

grad_func = jax.jit(jax.grad(total_objective), static_argnums=(1, 2, 3))


nx = 4
nu = 2
num_vars = steps * nx + steps * nu
num_constraints = (steps - 1) * nx + nx + nx 
 

x_init_guess = jnp.zeros((steps, 4))
x_init_guess = x_init_guess.at[:, 0].set(jnp.linspace(0, jnp.pi, steps))
z0 = jnp.concatenate([x_init_guess.flatten(), u_traj.flatten()])

# Initial guess
z0 = jnp.concatenate([x_traj.flatten(), u_traj.flatten()])

# Bounds for variables
# State bounds unbounded
x_lb = jnp.full((steps, nx), -jnp.inf)
x_ub = jnp.full((steps, nx), jnp.inf)
# Input bounds
u_lb = jnp.full((steps, nu), -P.torque_limit)
u_ub = jnp.full((steps, nu), P.torque_limit)

lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])
# set the second motor torque limits tighter

# Constraint bounds: equality constraints (dynamics + initial state) => zeros
cl = jnp.zeros(num_constraints)
cu = jnp.zeros(num_constraints)


problem = cyipopt.Problem(
	n=num_vars,
	m=num_constraints,
	problem_obj=Problem(steps, nx, nu, x0, xgoal, dt),
	lb=lb,
	ub=ub,
	cl=cl,
	cu=cu
)


# Set options
problem.add_option('max_iter', 100)
problem.add_option('tol', 1e-3)
problem.add_option('print_level', 0) # 5 is default, 0 is silent

# Solve
z_opt, info = problem.solve(z0)

# Unpack results
x_traj = z_opt[:steps * nx].reshape((steps, nx))
u_traj = z_opt[steps * nx:].reshape((steps, nu))
force_history = u_traj

import matplotlib.pyplot as plt

# Plots
x_history = jnp.array(x_history)
ref = jnp.ones(steps) * jnp.pi
font = 14
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
print("last second", time_span[-1], x_traj[-1])

# Plot Joint Angles
ax1.plot(time_span, x_traj[:, 0], label=r'$q_1$ (rad)')
ax1.plot(time_span, x_traj[:, 1], label=r'$q_2$ (rad)')
ax1.set_title('Joint States over Time')
ax1.set_ylabel('Angle (rad)', fontsize=font)
ax1.set_xlabel('Time (s)', fontsize=font)
ax1.legend()
ax1.grid(True)

# Plot Joint Velocities
ax2.plot(time_span, x_traj[:, 2], label=r'$\dot{q}_1$ (rad/s)', linestyle='--')
ax2.plot(time_span, x_traj[:, 3], label=r'$\dot{q}_2$ (rad/s)', linestyle='--')
ax2.plot(time_span, ref, 'k:')
ax2.set_title('Joint Vel over Time')
ax2.set_ylabel('Value (rad/s)', fontsize=font)
ax2.set_xlabel('Time (s)', fontsize=font)
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# plot Forces
plt.figure(figsize=(7, 4))
plt.plot(time_span[:], force_history[:, 0], label=r'$u_1$ (Nm)')
plt.plot(time_span[:], force_history[:, 1], label=r'$u_2$ (Nm)')
plt.title('Control Inputs (Torques) over Time')
plt.xlabel('Time (s)',fontsize=font)
plt.ylabel('Torque (Nm)',fontsize=font)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Construct filename suffix from Q and R matrices
q_diag = jnp.diag(Q)
r_diag = jnp.diag(R)
# filename_suffix = f"R_{int(r_diag[0])}_{int(r_diag[1])}_Q_{int(q_diag[0])}_{int(q_diag[1])}_{int(q_diag[2])}_{int(q_diag[3])}"

# # Save figures
# fig_joint_states = plt.figure(1)
# fig_torques = plt.figure(2)

# fig_joint_states.savefig(f"Joint_states_{filename_suffix}.png")
# fig_torques.savefig(f"Torques_{filename_suffix}.png")

plt.show()
