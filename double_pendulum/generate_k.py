import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cyipopt
import mujoco
from pathlib import Path
from collections import namedtuple

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────
# Model parameters: READ DIRECTLY FROM dp.xml
# ─────────────────────────────────────────────
# The whole point: the trajectory optimizer below and the MuJoCo simulator used
# for deployment must describe the SAME physical system. Previously this file
# hard-coded a frictionless point-mass model whose inertia, gravity arms, damping
# and friction did NOT match dp.xml -- so the "expert" actions never actually
# swung up the simulated pendulum. We now pull every parameter from the XML, so
# editing dp.xml automatically updates the planner; they can't silently drift.
#
# Derivation note (validated against MuJoCo's mj_fullM at q2=0):
#   a1 = I1 + m1*lc1^2 + m2*l1^2     a2 = I2 + m2*lc2^2     a3 = m2*l1*lc2
#   M = [[a1 + a2 + 2*a3*cos q2, a2 + a3*cos q2],
#        [a2 + a3*cos q2,        a2           ]]
#   -> M[0,0]|_{q2=0} = 1.535e-3, matching MuJoCo's 0.00153516.

XML_PATH = Path(__file__).resolve().parent / "dp.xml"

ModelParams = namedtuple(
    "ModelParams",
    "m1 m2 lc1 lc2 l1 I1 I2 b1 b2 f1 f2 g tau1 tau2",
)


def load_params(xml_path=XML_PATH):
    """Extract the planar 2-link dynamics parameters straight from the MJCF."""
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    b1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link1")
    b2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link2")

    # Hinge axis is y (axis="0 1 0") and body iquat is identity, so the inertia
    # about the rotation axis is component [1] of diaginertia.
    return ModelParams(
        m1=float(m.body_mass[b1_id]),
        m2=float(m.body_mass[b2_id]),
        lc1=float(-m.body_ipos[b1_id][2]),       # COM distance from joint 1
        lc2=float(-m.body_ipos[b2_id][2]),       # COM distance from joint 2
        l1=float(-m.body_pos[b2_id][2]),         # joint1 -> joint2 distance
        I1=float(m.body_inertia[b1_id][1]),      # about hinge (y) axis, at COM
        I2=float(m.body_inertia[b2_id][1]),
        b1=float(m.dof_damping[0]),              # viscous damping, joint 1
        b2=float(m.dof_damping[1]),
        f1=float(m.dof_frictionloss[0]),         # Coulomb friction, joint 1
        f2=float(m.dof_frictionloss[1]),
        g=float(-m.opt.gravity[2]),
        tau1=float(m.actuator_ctrlrange[0, 1] * m.actuator_gear[0, 0]),
        tau2=float(m.actuator_ctrlrange[1, 1] * m.actuator_gear[1, 0]),
    )


P = load_params()

# Coulomb friction is non-smooth (sign of velocity), which breaks gradient-based
# collocation. Approximate sign(v) with tanh(v/eps); smaller eps -> closer to
# MuJoCo's dry friction but stiffer for the optimizer. 0.05 rad/s is a good
# compromise for this slow swing-up.
FRICTION_EPS = 0.05


# Cost weights for the standalone single-solve in main(). The dataset generator
# (generate_dataset.py) passes its own Q/R grid and does not use these.
Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin = Q
R = jnp.diag(jnp.array([1.0, 1.0])) * 0.01


# ─────────────────────────────────────────────
# Dynamics (distributed-inertia model, parameterized from dp.xml)
# ─────────────────────────────────────────────
def M(q):
    q2 = q[1]
    a1 = P.I1 + P.m1 * P.lc1 ** 2 + P.m2 * P.l1 ** 2
    a2 = P.I2 + P.m2 * P.lc2 ** 2
    a3 = P.m2 * P.l1 * P.lc2
    c2 = jnp.cos(q2)
    return jnp.array([[a1 + a2 + 2 * a3 * c2, a2 + a3 * c2],
                      [a2 + a3 * c2,          a2]])


def C(q):
    q2, q1_dot, q2_dot = q[1], q[2], q[3]
    a3 = P.m2 * P.l1 * P.lc2
    s2 = jnp.sin(q2)
    # Canonical Coriolis matrix (Spong); C @ dq reproduces the centrifugal/
    # Coriolis generalized forces.
    return jnp.array([[-a3 * s2 * q2_dot, -a3 * s2 * (q1_dot + q2_dot)],
                      [ a3 * s2 * q1_dot,  0.0]])


def G(q):
    # Sign convention matches the EOM below (M ddq = u - C dq + G ...):
    # q1 = 0 is hanging down (stable), q1 = pi is upright (goal).
    q1, q2 = q[0], q[1]
    g1 = -P.g * (P.m1 * P.lc1 + P.m2 * P.l1) * jnp.sin(q1) \
         - P.g * P.m2 * P.lc2 * jnp.sin(q1 + q2)
    g2 = -P.g * P.m2 * P.lc2 * jnp.sin(q1 + q2)
    return jnp.array([g1, g2])


def dynamics(x, u):
    u  = u.reshape(2)
    dq = x[2:].reshape(2)
    damping  = jnp.array([P.b1, P.b2]) * dq
    friction = jnp.array([P.f1, P.f2]) * jnp.tanh(dq / FRICTION_EPS)
    rhs = u + G(x) - C(x) @ dq - damping - friction
    ddq = jnp.linalg.solve(M(x), rhs)
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
# Open-loop verification against the real MuJoCo sim
# ─────────────────────────────────────────────
def verify_open_loop(x_traj, u_traj, dt, xml_path=XML_PATH):
    """
    Replay the optimized torques open-loop in MuJoCo and return the resulting
    state trajectory. If the planner matches the simulator, the final state
    should land near x_traj[-1]. This is the sanity check that previously failed.
    """
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    x_traj = np.asarray(x_traj)
    u_traj = np.asarray(u_traj)

    d.qpos[:2] = x_traj[0, :2]
    d.qvel[:2] = x_traj[0, 2:]
    mujoco.mj_forward(m, d)

    n_sub = max(1, int(round(dt / m.opt.timestep)))
    lo = np.array([-P.tau1, -P.tau2])
    hi = np.array([P.tau1, P.tau2])

    sim = []
    for u in u_traj:
        d.ctrl[:] = np.clip(u, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        sim.append(np.concatenate([d.qpos[:2].copy(), d.qvel[:2].copy()]))
    return np.array(sim)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_results(time_span, x_traj, u_traj, out_dir, sim_traj=None):
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(time_span, x_traj[:, 0], label=r'$q_1$ (rad)')
    ax1.plot(time_span, x_traj[:, 1], label=r'$q_2$ (rad)')
    if sim_traj is not None:
        ax1.plot(time_span, sim_traj[:, 0], '--', label=r'$q_1$ MuJoCo')
        ax1.plot(time_span, sim_traj[:, 1], '--', label=r'$q_2$ MuJoCo')
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
    print("Loaded model parameters from dp.xml:")
    for k, v in P._asdict().items():
        print(f"  {k:5s} = {v:.6g}")

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
    u_init = jax.random.uniform(key, (steps, nu), minval=-P.tau1, maxval=P.tau1)
    z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

    # Variable bounds: torque limits come straight from the actuator ctrlrange.
    x_lb = jnp.full((steps, nx), -jnp.inf)
    x_ub = jnp.full((steps, nx),  jnp.inf)
    u_lb = jnp.tile(jnp.array([-P.tau1, -P.tau2]), (steps, 1))
    u_ub = jnp.tile(jnp.array([P.tau1, P.tau2]), (steps, 1))
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
    problem.add_option('max_iter', 300)
    problem.add_option('tol', 1e-3)
    problem.add_option('print_level', 0)

    z_opt, info = problem.solve(z0)
    print("Solver status:", info['status_msg'])

    x_traj = z_opt[:steps * nx].reshape(steps, nx)
    u_traj = z_opt[steps * nx:].reshape(steps, nu)
    print("Planned final state:", np.asarray(x_traj[-1]))

    # ── The critical check: do these torques actually swing up MuJoCo? ──
    sim_traj = verify_open_loop(x_traj, u_traj, dt)
    print("MuJoCo final state (open-loop replay):", sim_traj[-1])
    err = float(np.linalg.norm(sim_traj[-1] - np.asarray(x_traj[-1])))
    goal_err = float(abs((sim_traj[-1, 0] + np.pi) % (2 * np.pi) - np.pi - 0.0)
                     + abs(sim_traj[-1, 1]))
    print(f"Planner-vs-MuJoCo final-state error: {err:.4f}  "
          f"(was ~3+ before; should now be small)")

    # Save results
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.savetxt(os.path.join(results_dir, "trajectory.csv"), np.asarray(x_traj),
               delimiter=",", header="q1,q2,q1_dot,q2_dot", comments="")
    header_full = "time,q1,q2,q1_dot,q2_dot"
    data_full = np.column_stack([np.asarray(time_span), np.asarray(x_traj)])
    np.savetxt(os.path.join(results_dir, "optimal_trajectory_full.csv"), data_full,
               delimiter=",", header=header_full, comments="")
    np.savetxt(os.path.join(results_dir, "inputs.csv"), np.asarray(u_traj),
               delimiter=",", header="u1,u2", comments="")

    plot_results(np.asarray(time_span), np.asarray(x_traj), np.asarray(u_traj),
                 os.path.join(base, "graphs/reference"), sim_traj=sim_traj)


if __name__ == "__main__":
    main()
