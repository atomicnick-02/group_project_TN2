import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cyipopt

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────
#  Cost weights
# ─────────────────────────────────────────────
Q    = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin = Q
R    = jnp.diag(jnp.array([1.0, 1.0])) * 0.01


# ─────────────────────────────────────────────
#  Dynamics
# ─────────────────────────────────────────────
class DoublePendulum:
	"""
	Equations of motion for a 2-DOF double-pendulum arm.

	State  : x = [q1, q2, q1_dot, q2_dot]
	Control: u = [tau1, tau2]
	"""
	def __init__(self, mass=(1.0, 1.0), length=(0.05, 0.049), gravity=9.81):
		self.m1, self.m2 = mass
		self.l1, self.l2 = length
		self.g = gravity

	# ── Matrices ──────────────────────────────

	def M(self, x):
		"""Mass (inertia) matrix."""
		_, q2, _, _ = x
		c2 = jnp.cos(q2)
		m00 = self.l1**2 * self.m1 + (self.l1**2 + self.l2**2 + 2*self.l1*self.l2*c2) * self.m2
		m01 = self.m2 * (self.l2**2 + self.l1*self.l2*c2)
		m11 = self.m2 * self.l2**2
		return jnp.array([[m00, m01],
						  [m01, m11]])

	def C(self, x):
		"""Coriolis / centripetal matrix."""
		_, q2, q1d, q2d = x
		s2 = jnp.sin(q2)
		h  = self.l1 * self.m2 * self.l2 * s2
		return jnp.array([[-2*q2d*h, -q2d*h],
						  [ q1d*h,    0.0  ]])

	def G(self, x):
		"""Gravity vector."""
		q1, q2, _, _ = x
		s1  = jnp.sin(q1)
		s12 = jnp.sin(q1 + q2)
		g0  = -self.g * (self.m1*self.l1*s1 + self.m2*(self.l1*s1 + self.l2*s12))
		g1  = -self.g * self.m2 * self.l2 * s12
		return jnp.array([g0, g1])

	# ── Equations of motion ───────────────────

	def continuous(self, x, u):
		"""Continuous-time dynamics: xdot = f(x, u)."""
		q   = x[:2]
		dq  = x[2:]
		ddq = jnp.linalg.solve(self.M(x), u - self.C(x) @ dq + self.G(x))
		return jnp.concatenate([dq, ddq])

	def rk4(self, x, u, dt):
		"""4th-order Runge-Kutta integration step."""
		f  = self.continuous
		k1 = f(x,                   u)
		k2 = f(x + 0.5*dt*k1,       u)
		k3 = f(x + 0.5*dt*k2,       u)
		k4 = f(x +     dt*k3,       u)
		return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

	def trapezoidal_collocation(self, xk, xkp1, uk, ukp1, dt):
		"""
		Trapezoidal collocation defect:
			x_{k+1} - x_k - dt/2 * (f_k + f_{k+1}) = 0
		"""
		fk    = self.continuous(xk,   uk)
		fkp1  = self.continuous(xkp1, ukp1)
		return (xkp1 - xk - (dt/2.0)*(fk + fkp1)).flatten()

	@staticmethod
	def angle_diff(x1, x2):
		"""State difference with angle wrapping for q1, q2."""
		d = x1 - x2
		d = d.at[0].set((d[0] + jnp.pi) % (2*jnp.pi) - jnp.pi)
		d = d.at[1].set((d[1] + jnp.pi) % (2*jnp.pi) - jnp.pi)
		return d

# ─────────────────────────────────────────────
#  Objective & constraints
# ─────────────────────────────────────────────
def stage_cost(x, u, x_goal):
	e = x - x_goal
	return e @ Q @ e + u @ R @ u

def total_objective(z, steps, nx, nu, x_goal):
	X = z[:steps*nx].reshape(steps, nx)
	U = z[steps*nx:].reshape(steps, nu)

	costs = jax.vmap(lambda x, u: stage_cost(x, u, x_goal))(X, U) * 0.5
	terminal = (X[-1] - x_goal) @ Qfin @ (X[-1] - x_goal)
	return jnp.sum(costs.at[-1].set(terminal))

def total_objective_grad(z, steps, nx, nu, x_goal):
	X = z[:steps*nx].reshape(steps, nx)
	U = z[steps*nx:].reshape(steps, nu)

	gX, gU = jax.vmap(lambda x, u: (2*Q@(x-x_goal), 2*R@u))(X, U)
	gX = (gX * 0.5).at[-1].set(2 * Qfin @ (X[-1] - x_goal))
	gU = gU * 0.5
	return jnp.concatenate([gX.flatten(), gU.flatten()])

def build_constraints(dyn: DoublePendulum, dt):
	"""Return a constraint function closed over *dyn* and *dt*."""

	def constraints(z, steps, nx, nu, x0, x_goal):
		X = z[:steps*nx].reshape(steps, nx)
		U = z[steps*nx:].reshape(steps, nu)

		c_init  = X[0]  - x0
		c_final = X[-1] - x_goal

		trap = jax.vmap(
			lambda xk, xkp1, uk, ukp1:
				dyn.trapezoidal_collocation(xk, xkp1, uk, ukp1, dt)
		)
		c_dyn = trap(X[:-1], X[1:], U[:-1], U[1:]).flatten()

		return jnp.concatenate([c_init, c_dyn, c_final])

	return constraints


# ─────────────────────────────────────────────
#  IPOPT problem wrapper
# ─────────────────────────────────────────────
class OCProblem:
	"""
	Optimal control problem passed to cyipopt.

	Decision vector: z = [x_0, ..., x_N, u_0, ..., u_N]
	"""

	def __init__(self, steps, nx, nu, x0, x_goal, dt, dyn: DoublePendulum):
		self.steps   = steps
		self.nx, self.nu = nx, nu
		print(f"OCProblem: {steps} steps | nx={nx} | nu={nu}")

		cons_fn = build_constraints(dyn, dt)

		self._obj  = jax.jit(lambda z: total_objective(z, steps, nx, nu, x_goal))
		self._grad = lambda z: total_objective_grad(z, steps, nx, nu, x_goal)
		self._cons = jax.jit(lambda z: cons_fn(z, steps, nx, nu, x0, x_goal))
		self._jac  = jax.jit(jax.jacobian(lambda z: cons_fn(z, steps, nx, nu, x0, x_goal)))

	def objective(self, z):
		return float(self._obj(z))

	def gradient(self, z):
		return np.asarray(self._grad(z), dtype=np.float64).ravel()

	def constraints(self, z):
		return np.asarray(self._cons(z), dtype=np.float64)

	def jacobian(self, z):
		return np.asarray(self._jac(z), dtype=np.float64).ravel()

def plot_results(time_span, x_traj, u_traj, output_dir="graphs"):
	"""Save joint-state and torque plots to *output_dir*."""
	os.makedirs(output_dir, exist_ok=True)
	font = 14

	# — Joint angles & velocities —
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

	ax1.plot(time_span, x_traj[:, 0], label=r"$q_1$ (rad)")
	ax1.plot(time_span, x_traj[:, 1], label=r"$q_2$ (rad)")
	ax1.set(title="Joint Angles over Time", ylabel="Angle (rad)", xlabel="Time (s)")
	ax1.legend(); ax1.grid(True)

	ax2.plot(time_span, x_traj[:, 2], label=r"$\dot{q}_1$ (rad/s)", linestyle="--")
	ax2.plot(time_span, x_traj[:, 3], label=r"$\dot{q}_2$ (rad/s)", linestyle="--")
	ax2.axhline(jnp.pi, color="k", linestyle=":")
	ax2.set(title="Joint Velocities over Time", ylabel="Velocity (rad/s)", xlabel="Time (s)")
	ax2.legend(); ax2.grid(True)

	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "joint_states.png"), dpi=150)

	# — Torques —
	fig2, ax = plt.subplots(figsize=(7, 4))
	ax.plot(time_span, u_traj[:, 0], label=r"$u_1$ (Nm)")
	ax.plot(time_span, u_traj[:, 1], label=r"$u_2$ (Nm)")
	ax.set(title="Control Inputs (Torques) over Time",
		   xlabel="Time (s)", ylabel="Torque (Nm)")
	ax.legend(); ax.grid(True)
	plt.tight_layout()
	fig2.savefig(os.path.join(output_dir, "torques.png"), dpi=150)



def main():
	# ── Problem parameters ──────────────────
	T    = 2.0
	dt   = 0.05
	steps = int(T / dt) + 1
	time_span = jnp.linspace(0, T, steps)

	x0    = jnp.array([0.0,    0.0, 0.0, 0.0])
	xgoal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

	peak_torque   = 1.0
	torque_limits = (-peak_torque, peak_torque)

	nx, nu = 4, 2

	# ── Dynamics ────────────────────────────
	dyn = DoublePendulum(mass=(1, 1), length=(0.05, 0.049))

	# ── Initial guess ───────────────────────
	key    = jax.random.PRNGKey(0)
	x_init = jnp.zeros((steps, nx)).at[:, 0].set(jnp.linspace(0, jnp.pi, steps))
	u_init = jax.random.uniform(key, (steps, nu),
								minval=torque_limits[0], maxval=torque_limits[1])
	z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

	# ── Variable bounds ──────────────────────
	x_lb = jnp.full((steps, nx), -jnp.inf)
	x_ub = jnp.full((steps, nx),  jnp.inf)
	u_lb = jnp.full((steps, nu),  torque_limits[0])
	u_ub = jnp.full((steps, nu),  torque_limits[1])
	lb   = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
	ub   = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

	# ── Constraint bounds (all equalities) ──
	num_constraints = nx + (steps - 1)*nx + nx      # init + dynamics + final
	cl = cu = jnp.zeros(num_constraints)

	# ── Solve ───────────────────────────────
	num_vars = steps * nx + steps * nu
	problem  = cyipopt.Problem(
		n=num_vars,
		m=num_constraints,
		problem_obj=OCProblem(steps, nx, nu, x0, xgoal, dt, dyn),
		lb=lb, ub=ub, cl=cl, cu=cu,
	)
	problem.add_option("max_iter",    100)
	problem.add_option("tol",         1e-3)
	problem.add_option("print_level", 0)

	z_opt, info = problem.solve(z0)
	print("Solver status:", info["status_msg"])

	# ── Unpack & report ──────────────────────
	x_traj = z_opt[:steps*nx].reshape(steps, nx)
	u_traj = z_opt[steps*nx:].reshape(steps, nu)
	print("Final state:", x_traj[-1])

	# ── Save results ─────────────────────────
	output_dir = os.path.join(os.path.dirname(__file__), "results")
	os.makedirs(output_dir, exist_ok=True)
	np.savetxt(os.path.join(output_dir, "trajectory.csv"), x_traj,
			   delimiter=",", header="q1,q2,q1_dot,q2_dot", comments="")
	np.savetxt(os.path.join(output_dir, "inputs.csv"), u_traj,
			   delimiter=",", header="u1,u2", comments="")

	graph_dir = os.path.join(os.path.dirname(__file__), "graphs")
	os.makedirs(graph_dir, exist_ok=True)
	# ── Plot ─────────────────────────────────
	plot_results(time_span, x_traj, u_traj, graph_dir)


if __name__ == "__main__":
	main()