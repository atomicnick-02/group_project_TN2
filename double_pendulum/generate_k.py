"""
Reference generator for the double-pendulum swing-up: detailed physics, direct
collocation, and time-varying LQR (TVLQR) gains -- all in a self-contained
numpy/JAX + matplotlib simulator (NO MuJoCo).

WHAT CHANGED vs the old MuJoCo-coupled version
-----------------------------------------------
* The dynamics are now taken DIRECTLY from the cloudpendulum system-identified
  model (dp_fwd_inv_dynamics/fwd_inv_dyn_student.ipynb): a distributed-inertia
  2-link manipulator with viscous + Coulomb (arctan-smoothed) joint friction.
  dp.xml is no longer consulted for the model.
* Verification is done by integrating the SAME equations with RK4 and plotting
  with matplotlib, instead of replaying torques in MuJoCo.
* The TVLQR backward Riccati recursion (linearizing the RK4 step) lives here and
  produces K_matrix.npy. generate_tvlqr_dataset.py imports solve_trajectory(),
  tvlqr_gains() and rollout_tvlqr() from this module to build the dataset.

Manipulator equation (cloudpendulum sys-id):
    M(q) q̈ + C(q,q̇) q̇ + G(q) + F(q̇) = τ
We integrate it in the project's convention where q1 = 0 is hanging DOWN (stable)
and q1 = π is upright (the swing-up goal, x_goal = [π,0,0,0]); concretely
    q̈ = M(q)^{-1} ( τ + G(q) - C(q,q̇) q̇ - F(q̇) ).
The gravity term enters with a "+" here (vs the textbook "-G" on the LHS) so that
this down=0 / up=π convention -- shared by every other file in the project and by
the committed reference trajectory -- holds.
"""

import os
from functools import partial
from collections import namedtuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import cyipopt

jax.config.update("jax_enable_x64", True)

current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"


# ─────────────────────────────────────────────
# Model parameters  (cloudpendulum dp_fwd_inv_dynamics, system-identified)
# ─────────────────────────────────────────────
# l*  : link length            lc* : centre-of-mass distance from the joint
# I*  : link inertia about the joint axis (as used directly in M, per the repo)
# b*  : viscous damping         mu* : Coulomb friction coefficient
# gr* : motor gear ratio        Ir  : rotor inertia (no value in the notebook ->
#                                     defaults to 0; the gr²·Ir + Ir reflected
#                                     rotor-inertia term in M[0,0] is then inert).
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
# Larger -> closer to dry friction but stiffer for the collocation solver.
FRICTION_SCALE = 100.0

X_GOAL = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

# Default cost weights for main()'s single solve. The dataset generator passes
# its own Q/R grid and does not use these.
Q_DEFAULT = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
R_DEFAULT = jnp.diag(jnp.array([1.0, 1.0])) * 0.01


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
    # q1 = 0 hanging down (stable); enters the EOM with a "+" sign (see header).
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


# ─────────────────────────────────────────────
# Discrete linearization of the RK4 step  (for TVLQR)
# ─────────────────────────────────────────────
def get_continuous_jacobians(x, u):
    A = jax.jacfwd(dynamics, argnums=0)(x, u)
    B = jax.jacfwd(dynamics, argnums=1)(x, u)
    return A, B


def get_discrete_matrices(x_k, u_k, dt):
    """A_d, B_d for x_{k+1} = RK4(x_k, u_k), via the chain rule on the 4 stages."""
    nx = x_k.shape[0]
    I = jnp.eye(nx)

    A1, B1 = get_continuous_jacobians(x_k, u_k)
    dk1_dx, dk1_du = A1, B1

    k1 = dynamics(x_k, u_k)
    x2 = x_k + 0.5 * dt * k1
    A2, B2 = get_continuous_jacobians(x2, u_k)
    dk2_dx = A2 @ (I + 0.5 * dt * dk1_dx)
    dk2_du = A2 @ (0.5 * dt * dk1_du) + B2

    k2 = dynamics(x2, u_k)
    x3 = x_k + 0.5 * dt * k2
    A3, B3 = get_continuous_jacobians(x3, u_k)
    dk3_dx = A3 @ (I + 0.5 * dt * dk2_dx)
    dk3_du = A3 @ (0.5 * dt * dk2_du) + B3

    k3 = dynamics(x3, u_k)
    x4 = x_k + dt * k3
    A4, B4 = get_continuous_jacobians(x4, u_k)
    dk4_dx = A4 @ (I + dt * dk3_dx)
    dk4_du = A4 @ (dt * dk3_du) + B4

    A_d = I + (dt / 6.0) * (dk1_dx + 2 * dk2_dx + 2 * dk3_dx + dk4_dx)
    B_d = (dt / 6.0) * (dk1_du + 2 * dk2_du + 2 * dk3_du + dk4_du)
    return A_d, B_d


def tvlqr_backward_pass(P_next, inputs, Q_d, R_d, dt):
    x_k, u_k = inputs
    A_d, B_d = get_discrete_matrices(x_k, u_k, dt)
    S = R_d + B_d.T @ P_next @ B_d
    K_k = jnp.linalg.solve(S, B_d.T @ P_next @ A_d)
    P_k = Q_d + A_d.T @ P_next @ (A_d - B_d @ K_k)
    return P_k, K_k


def tvlqr_gains(x_ref, u_ref, Q, R, Qfin, dt):
    """
    Time-varying LQR gains around the nominal (x_ref, u_ref).

    x_ref: (steps, nx)   u_ref: (steps, nu)
    Returns K: (steps-1, nu, nx)   (gain to apply at each step k = 0..steps-2)
    """
    x_ref = jnp.asarray(x_ref)
    u_ref = jnp.asarray(u_ref)
    # Backward in time from the terminal cost over steps-1 transitions.
    xs = x_ref[:-1][::-1]
    us = u_ref[:-1][::-1]
    scan_fn = lambda Pn, inp: tvlqr_backward_pass(Pn, inp, Q, R, dt)
    _, K_rev = jax.lax.scan(scan_fn, Qfin, (xs, us))
    return np.asarray(K_rev[::-1])


# ─────────────────────────────────────────────
# Direct collocation (trapezoidal) via IPOPT
# ─────────────────────────────────────────────
@partial(jax.jit, static_argnums=(1, 2, 3))
def _objective(z, steps, nx, nu, Q, R, Qfin, x_goal):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    e = X - x_goal
    stage = jax.vmap(lambda ek, uk: ek @ Q @ ek + uk @ R @ uk)(e, U) * 0.5
    terminal = (X[-1] - x_goal) @ Qfin @ (X[-1] - x_goal)
    return jnp.sum(stage.at[-1].set(terminal))


@partial(jax.jit, static_argnums=(1, 2, 3))
def _gradient(z, steps, nx, nu, Q, R, Qfin, x_goal):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    gX, gU = jax.vmap(lambda xk, uk: (2 * Q @ (xk - x_goal), 2 * R @ uk))(X, U)
    gX = (gX * 0.5).at[-1].set(2 * Qfin @ (X[-1] - x_goal))
    gU = gU * 0.5
    return jnp.concatenate([gX.flatten(), gU.flatten()])


@partial(jax.jit, static_argnums=(1, 2, 3))
def _constraints(z, steps, nx, nu, x0, x_goal, dt):
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    c_init = X[0] - x0
    c_final = X[-1] - x_goal

    # RK4 defect (zero-order-hold on u): x_{k+1} = RK4(x_k, u_k, dt). Using the
    # SAME integrator the simulator and the TVLQR linearization use makes the
    # nominal exactly reproducible -- a trapezoidal defect leaves it open-loop-
    # infeasible at this dt (the fast swing-up reaches ~16 rad/s), so the gains
    # have nothing trackable and the closed loop diverges immediately.
    def defect(xk, xkp1, uk):
        return xkp1 - rk4_step(xk, uk, dt)

    c_dyn = jax.vmap(defect)(X[:-1], X[1:], U[:-1]).flatten()
    return jnp.concatenate([c_init, c_dyn, c_final])


@partial(jax.jit, static_argnums=(1, 2, 3))
def _jacobian(z, steps, nx, nu, x0, x_goal, dt):
    jac = jax.jacobian(lambda zz: _constraints(zz, steps, nx, nu, x0, x_goal, dt))
    return jac(z)


class _Problem:
    """cyipopt problem object for one (Q, R, x0, x_goal) collocation solve."""

    def __init__(self, steps, nx, nu, x0, x_goal, dt, Q, R, Qfin):
        self.steps, self.nx, self.nu, self.dt = steps, nx, nu, dt
        self.x0, self.x_goal = jnp.asarray(x0), jnp.asarray(x_goal)
        self.Q, self.R, self.Qfin = jnp.asarray(Q), jnp.asarray(R), jnp.asarray(Qfin)

    def objective(self, z):
        return float(_objective(z, self.steps, self.nx, self.nu,
                                self.Q, self.R, self.Qfin, self.x_goal))

    def gradient(self, z):
        return np.asarray(_gradient(z, self.steps, self.nx, self.nu,
                                    self.Q, self.R, self.Qfin, self.x_goal),
                          dtype=np.float64).ravel()

    def constraints(self, z):
        return np.asarray(_constraints(z, self.steps, self.nx, self.nu,
                                       self.x0, self.x_goal, self.dt),
                          dtype=np.float64)

    def jacobian(self, z):
        return np.asarray(_jacobian(z, self.steps, self.nx, self.nu,
                                    self.x0, self.x_goal, self.dt),
                          dtype=np.float64).ravel()


def solve_trajectory(x0, x_goal=None, Q=None, R=None, Qfin=None,
                     T=2.0, dt=0.05, max_iter=300, tol=1e-3, verbose=False,
                     z0=None, seed=0):
    """
    Solve a swing-up / regulation trajectory with trapezoidal collocation.

    Returns (time (steps,), x_traj (steps, nx), u_traj (steps, nu), info).
    """
    x0 = jnp.asarray(x0)
    x_goal = X_GOAL if x_goal is None else jnp.asarray(x_goal)
    Q = Q_DEFAULT if Q is None else jnp.asarray(Q)
    R = R_DEFAULT if R is None else jnp.asarray(R)
    Qfin = Q if Qfin is None else jnp.asarray(Qfin)

    nx, nu = 4, 2
    steps = int(round(T / dt)) + 1
    time = jnp.linspace(0.0, T, steps)
    tau = P.torque_limit

    if z0 is None:
        key = jax.random.PRNGKey(seed)
        x_init = jnp.zeros((steps, nx)).at[:, 0].set(
            jnp.linspace(float(x0[0]), float(x_goal[0]), steps))
        u_init = jax.random.uniform(key, (steps, nu), minval=-tau, maxval=tau)
        z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

    x_lb = jnp.full((steps, nx), -jnp.inf)
    x_ub = jnp.full((steps, nx), jnp.inf)
    u_lb = jnp.full((steps, nu), -tau)
    u_ub = jnp.full((steps, nu), tau)
    lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
    ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

    num_constraints = nx + (steps - 1) * nx + nx
    cl = cu = jnp.zeros(num_constraints)
    num_vars = steps * nx + steps * nu

    nlp = cyipopt.Problem(
        n=num_vars, m=num_constraints,
        problem_obj=_Problem(steps, nx, nu, x0, x_goal, dt, Q, R, Qfin),
        lb=lb, ub=ub, cl=cl, cu=cu,
    )
    nlp.add_option("max_iter", max_iter)
    nlp.add_option("tol", tol)
    nlp.add_option("print_level", 5 if verbose else 0)

    z_opt, info = nlp.solve(np.asarray(z0))
    x_traj = np.asarray(z_opt[:steps * nx]).reshape(steps, nx)
    u_traj = np.asarray(z_opt[steps * nx:]).reshape(steps, nu)
    return np.asarray(time), x_traj, u_traj, info


# ─────────────────────────────────────────────
# TVLQR closed-loop controller + rollout (numpy simulator)
# ─────────────────────────────────────────────
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_feature_space(x):
    """[q1,q2,qd1,qd2] -> [cos q1, sin q1, cos q2, sin q2, 0.1 qd1, 0.1 qd2]."""
    v_scale = 0.1
    if x.ndim == 1:
        p0, p1 = x[0], x[1]
        v = x[2:] * v_scale
        return np.concatenate(([np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1)], v))
    p0, p1 = x[0, :], x[1, :]
    v = x[2:, :] * v_scale
    return np.vstack((np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1), v))


def _ranked(x, ref, k=1):
    diff = to_feature_space(ref) - to_feature_space(x).reshape(-1, 1)
    dists = np.linalg.norm(diff, axis=0)
    order = np.argsort(dists)
    return order[:k], dists[order[:k]]


def _K_weighted(x, ref_T, K, k=5):
    idx, dists = _ranked(x, ref_T, k=k)
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum()
    Ks = K[np.clip(idx, 0, len(K) - 1)]
    return np.sum(w[:, None, None] * Ks, axis=0)


def rollout_tvlqr(x_ref, u_ref, K, x0, n_steps,
                  dt_control=0.05, dt_sim=None, torque_limit=None,
                  deviation_threshold=2.0, noise_std=None, rng=None):
    """
    Closed-loop TVLQR rollout in the numpy/RK4 simulator.

    Tracks the nominal trajectory index-by-index; if the feature-space deviation
    exceeds `deviation_threshold` (or the nominal has been exhausted -> holding),
    it switches to a nearest-neighbour + distance-weighted-gain recovery mode.
    This is the controller from old_pendulum/simul.ipynb / evaluate_swingup.py.

    x_ref: (steps, nx)  u_ref: (steps, nu)  K: (steps-1, nu, nx)
    Returns (states (n_steps, nx), actions (n_steps, nu), success).
    """
    tau = P.torque_limit if torque_limit is None else torque_limit
    # dt_sim defaults to dt_control: integrating with the SAME single RK4 step the
    # TVLQR gains were derived from keeps the plant and the controller's discrete
    # model identical, so tracking is exact. Pass a smaller dt_sim for a finer
    # (and slightly model-mismatched) continuous-time rollout.
    if dt_sim is None:
        dt_sim = dt_control
    ref_T = np.asarray(x_ref).T          # (nx, steps) for the feature helpers
    uref_T = np.asarray(u_ref).T         # (nu, steps)
    K = np.asarray(K)
    max_idx = ref_T.shape[1] - 1
    n_sub = max(1, int(round(dt_control / dt_sim)))

    x = np.asarray(x0, dtype=np.float64).copy()
    current_idx = 0
    states, actions = [], []
    held = 0

    for _ in range(n_steps):
        target = ref_T[:, current_idx].reshape(4, 1)
        _, d = _ranked(x, target, k=1)
        holding = current_idx >= max_idx
        if d[0] > deviation_threshold or holding:
            best, _ = _ranked(x, ref_T, k=1)
            idx = int(best[0])
            K_gain = _K_weighted(x, ref_T, K, k=5)
        else:
            idx = current_idx
            K_gain = K[idx]
            current_idx += 1

        x_des, u_des = ref_T[:, idx], uref_T[:, idx]
        err = x - x_des
        err[0], err[1] = wrap_to_pi(err[0]), wrap_to_pi(err[1])
        u = np.clip(u_des - K_gain @ err, -tau, tau)

        states.append(x.copy())
        actions.append(u.copy())

        xs = jnp.asarray(x)
        uj = jnp.asarray(u)
        for _ in range(n_sub):
            xs = rk4_step(xs, uj, dt_sim)
        x = np.array(xs, dtype=np.float64)
        if noise_std is not None and rng is not None:
            x = x + rng.normal(0.0, noise_std, size=4)
        x[0] = wrap_to_pi(x[0])
        x[1] = wrap_to_pi(x[1])

        ang = abs(wrap_to_pi(x[0] - np.pi)) + abs(wrap_to_pi(x[1]))
        vel = abs(x[2]) + abs(x[3])
        held = held + 1 if (ang < 0.2 and vel < 1.0) else 0

    # Success requires the rollout to END in a sustained upright hold (the final
    # `held` counts consecutive in-tolerance steps ending at the last step), so
    # swing-ups that reach the top but fall back off are rejected.
    return np.array(states), np.array(actions), held >= 10


# ─────────────────────────────────────────────
# Plotting (matplotlib)
# ─────────────────────────────────────────────
def plot_results(time, x_traj, u_traj, out_dir, sim_traj=None):
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(time, x_traj[:, 0], label=r"$q_1$ (rad)")
    ax1.plot(time, x_traj[:, 1], label=r"$q_2$ (rad)")
    if sim_traj is not None:
        ax1.plot(time, sim_traj[:, 0], "--", label=r"$q_1$ RK4 replay")
        ax1.plot(time, sim_traj[:, 1], "--", label=r"$q_2$ RK4 replay")
    ax1.set(title="Joint Angles", xlabel="Time (s)", ylabel="Angle (rad)")
    ax1.legend(); ax1.grid(True)

    ax2.plot(time, x_traj[:, 2], label=r"$\dot{q}_1$ (rad/s)")
    ax2.plot(time, x_traj[:, 3], label=r"$\dot{q}_2$ (rad/s)")
    ax2.set(title="Joint Velocities", xlabel="Time (s)", ylabel="Velocity (rad/s)")
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "joint_states.png"), dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, u_traj[:, 0], label=r"$u_1$ (Nm)")
    ax.plot(time, u_traj[:, 1], label=r"$u_2$ (Nm)")
    ax.set(title="Control Torques", xlabel="Time (s)", ylabel="Torque (Nm)")
    ax.legend(); ax.grid(True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "torques.png"), dpi=150)
    plt.close(fig2)


def simulate_open_loop(x0, u_traj, dt):
    """Replay the planned torques open-loop with RK4 (planner sanity check)."""
    x = jnp.asarray(x0)
    sim = [np.asarray(x)]
    for u in u_traj[:-1]:
        x = rk4_step(x, jnp.asarray(u), dt)
        sim.append(np.asarray(x))
    return np.array(sim)


# ─────────────────────────────────────────────
# Main: single swing-up reference + TVLQR gains
# ─────────────────────────────────────────────
def main():
    print("Model parameters (cloudpendulum sys-id):")
    for k, v in P._asdict().items():
        print(f"  {k:13s} = {v:.6g}")

    T, dt = 2.0, 0.05
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    x_goal = X_GOAL

    time, x_traj, u_traj, info = solve_trajectory(
        x0, x_goal, Q_DEFAULT, R_DEFAULT, Q_DEFAULT, T=T, dt=dt)
    print("Solver status:", info["status_msg"].decode() if isinstance(
        info["status_msg"], bytes) else info["status_msg"])
    print("Planned final state:", x_traj[-1])

    # Open-loop RK4 replay: do the planned torques actually swing it up?
    sim_traj = simulate_open_loop(x0, u_traj, dt)
    err = float(np.linalg.norm(sim_traj[-1] - x_traj[-1]))
    print(f"Planner-vs-RK4 open-loop final-state error: {err:.4f}")

    # TVLQR gains around the nominal trajectory.
    K = tvlqr_gains(x_traj, u_traj, Q_DEFAULT, R_DEFAULT, Q_DEFAULT, dt)
    print(f"K: {K.shape[0]} gains of shape {K.shape[1:]} ")

    results_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(results_dir / "trajectory.csv", x_traj, delimiter=",",
               header="q1,q2,q1_dot,q2_dot", comments="")
    np.savetxt(results_dir / "inputs.csv", u_traj, delimiter=",",
               header="u1,u2", comments="")
    np.savetxt(results_dir / "optimal_trajectory_full.csv",
               np.column_stack([time, x_traj]), delimiter=",",
               header="time,q1,q2,q1_dot,q2_dot", comments="")
    np.save(results_dir / "K_matrix.npy", K)

    plot_results(time, x_traj, u_traj, current_dir / "graphs/reference",
                 sim_traj=sim_traj)
    print(f"Saved trajectory.csv, inputs.csv, K_matrix.npy to {results_dir}")


if __name__ == "__main__":
    main()
