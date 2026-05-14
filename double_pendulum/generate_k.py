import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cyipopt

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────
# Fixed parameters
# ─────────────────────────────────────────────
mass = [0.10548177618443695, 0.07619744360415454]
length = [0.05, 0.05]
gravity = 9.81

g = gravity
l1, l2 = length
m1, m2 = mass

peak_torque = 0.04
torque_limits = jnp.array([-peak_torque, peak_torque])

Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin = Q
R = jnp.diag(jnp.array([1.0, 1.0])) * 0.01


# ─────────────────────────────────────────────
# Dynamics (point-mass model)
# ─────────────────────────────────────────────
def M(q):
    _, q2, _, _ = q
    m00 = l1**2 * m1 + l2**2 * m2 + l1**2 * m2 + 2 * l1 * m2 * l2 * jnp.cos(q2)
    m01 = l2**2 * m2 + l1 * m2 * l2 * jnp.cos(q2)
    m11 = l2**2 * m2
    return jnp.array([[m00, m01], [m01, m11]])


def C(q):
    _, q2, q1_dot, q2_dot = q
    c00 = -2 * q2_dot * l1 * m2 * l2 * jnp.sin(q2)
    c01 = -q2_dot * l1 * m2 * l2 * jnp.sin(q2)
    c10 = q1_dot * l1 * m2 * l2 * jnp.sin(q2)
    return jnp.array([[c00, c01], [c10, 0.0]])


def G(q):
    q1, q2, _, _ = q
    g00 = -g * m1 * l1 * jnp.sin(q1) - g * m2 * (l1 * jnp.sin(q1) + l2 * jnp.sin(q1 + q2))
    g10 = -g * m2 * l2 * jnp.sin(q1 + q2)
    return jnp.array([[g00], [g10]])


def dynamics(x, u):
    u = u.reshape(2, 1)
    dq = x[2:].reshape(2, 1)
    ddq = jnp.linalg.solve(M(x), u - C(x) @ dq + G(x))
    return jnp.concatenate([dq, ddq]).flatten()


# ─────────────────────────────────────────────
# Objective and gradient
# ─────────────────────────────────────────────
def stage_cost(x, u, x_goal):
    e = x - x_goal
    return e @ Q @ e + u @ R @ u


def total_objective(z, steps, nx, nu, x_goal):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    costs = jax.vmap(lambda x, u: stage_cost(x, u, x_goal))(X, U) * 0.5
    terminal = (X[-1] - x_goal) @ Qfin @ (X[-1] - x_goal)
    return jnp.sum(costs.at[-1].set(terminal))


def total_objective_grad(z, steps, nx, nu, x_goal):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    gX, gU = jax.vmap(lambda x, u: (2 * Q @ (x - x_goal), 2 * R @ u))(X, U)
    gX = (gX * 0.5).at[-1].set(2 * Qfin @ (X[-1] - x_goal))
    gU = gU * 0.5
    return jnp.concatenate([gX.flatten(), gU.flatten()])


# ─────────────────────────────────────────────
# Trapezoidal collocation constraints
# ─────────────────────────────────────────────
def trapezoidal_collocation(xk, xkp1, uk, ukp1, dt):
    fk = dynamics(xk, uk)
    fkp1 = dynamics(xkp1, ukp1)
    return (xkp1 - xk - (dt / 2.0) * (fk + fkp1)).flatten()


def constraints(z, steps, nx, nu, x0, x_goal, dt):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    c_init = X[0] - x0
    c_final = X[-1] - x_goal
    trap = jax.vmap(
        lambda xk, xkp1, uk, ukp1: trapezoidal_collocation(xk, xkp1, uk, ukp1, dt)
    )
    c_dyn = trap(X[:-1], X[1:], U[:-1], U[1:]).flatten()
    return jnp.concatenate([c_init, c_dyn, c_final])


# ─────────────────────────────────────────────
# IPOPT problem wrapper
# ─────────────────────────────────────────────
class Problem:
    """
    min  f(z)   with z = [x_0, ..., x_N, u_0, ..., u_N]
    s.t. h(z) = 0,  lb <= z <= ub
    """
    def __init__(self, steps, nx, nu, x0, x_goal, dt):
        print(f"Problem: {steps} steps | nx={nx} | nu={nu}")
        self._obj = jax.jit(lambda z: total_objective(z, steps, nx, nu, x_goal))
        self._grad = lambda z: total_objective_grad(z, steps, nx, nu, x_goal)
        self._cons = jax.jit(lambda z: constraints(z, steps, nx, nu, x0, x_goal, dt))
        self._jac = jax.jit(jax.jacobian(
            lambda z: constraints(z, steps, nx, nu, x0, x_goal, dt)
        ))

    def objective(self, z):
        return float(self._obj(z))

    def gradient(self, z):
        return np.asarray(self._grad(z), dtype=np.float64).ravel()

    def constraints(self, z):
        return np.asarray(self._cons(z), dtype=np.float64)

    def jacobian(self, z):
        return np.asarray(self._jac(z), dtype=np.float64).ravel()


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_results(time_span, x_traj, u_traj, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(time_span, x_traj[:, 0], label=r'$q_1$ (rad)')
    ax1.plot(time_span, x_traj[:, 1], label=r'$q_2$ (rad)')
    ax1.set(title='Joint Angles over Time', xlabel='Time (s)', ylabel='Angle (rad)')
    ax1.legend(); ax1.grid(True)

    ax2.plot(time_span, x_traj[:, 2], '--', label=r'$\dot{q}_1$ (rad/s)')
    ax2.plot(time_span, x_traj[:, 3], '--', label=r'$\dot{q}_2$ (rad/s)')
    ax2.set(title='Joint Velocities over Time', xlabel='Time (s)', ylabel='Velocity (rad/s)')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "joint_states.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_span, u_traj[:, 0], label=r'$u_1$ (Nm)')
    ax.plot(time_span, u_traj[:, 1], label=r'$u_2$ (Nm)')
    ax.set(title='Control Inputs (Torques) over Time',
           xlabel='Time (s)', ylabel='Torque (Nm)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "torques.png"), dpi=150)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    T = 2.0
    dt = 0.05
    steps = int(T / dt) + 1
    time_span = jnp.linspace(0, T, steps)
    nx, nu = 4, 2

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    x_goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    # Initial guess: linear angle ramp + random torques
    key = jax.random.PRNGKey(0)
    x_init = jnp.zeros((steps, nx)).at[:, 0].set(jnp.linspace(0, jnp.pi, steps))
    u_init = jax.random.uniform(key, (steps, nu),
                                minval=torque_limits[0], maxval=torque_limits[1])
    z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

    # Variable bounds
    x_lb = jnp.full((steps, nx), -jnp.inf)
    x_ub = jnp.full((steps, nx),  jnp.inf)
    u_lb = jnp.full((steps, nu), torque_limits[0])
    u_ub = jnp.full((steps, nu), torque_limits[1])
    lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
    ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

    # Equality constraints: init + dynamics + final
    num_constraints = nx + (steps - 1) * nx + nx
    cl = cu = jnp.zeros(num_constraints)
    num_vars = steps * nx + steps * nu

    problem = cyipopt.Problem(
        n=num_vars, m=num_constraints,
        problem_obj=Problem(steps, nx, nu, x0, x_goal, dt),
        lb=lb, ub=ub, cl=cl, cu=cu,
    )
    problem.add_option('max_iter', 100)
    problem.add_option('tol', 1e-3)
    problem.add_option('print_level', 0)

    z_opt, info = problem.solve(z0)
    print("Solver status:", info['status_msg'])

    x_traj = z_opt[:steps * nx].reshape(steps, nx)
    u_traj = z_opt[steps * nx:].reshape(steps, nu)
    print("Final state:", x_traj[-1])

    # Save results
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(os.path.join(results_dir, "trajectory.csv"), x_traj,
               delimiter=",", header="q1,q2,q1_dot,q2_dot", comments="")
    np.savetxt(os.path.join(results_dir, "inputs.csv"), u_traj,
               delimiter=",", header="u1,u2", comments="")

    plot_results(time_span, x_traj, u_traj, os.path.join(base, "graphs/reference"))


if __name__ == "__main__":
    main()