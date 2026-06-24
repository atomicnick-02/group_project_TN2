from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import scipy.linalg as sla

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


def equilibrium_torque(x_goal):
    """Feed-forward torque that holds `x_goal` fixed: u_eq s.t. dynamics()=0.

    At a goal the velocities are zero, so C@dq and friction vanish and the
    dynamics reduce to ddq = M^-1 (u + G(x_goal)); the equilibrium torque is
    therefore u_eq = -G(x_goal). For the inverted goals here G(x_goal) ~= 0, so
    u_eq ~= 0, but we compute it exactly so the hold also works off the
    cardinal equilibria (and for the mirrored goal, where it flips sign).
    """
    return -np.asarray(G(jnp.asarray(x_goal, dtype=jnp.float64)), dtype=np.float64)


def lqr_gain(x_goal, Q, R, dt=0.05):
    """Infinite-horizon discrete LQR gain that STABILIZES the goal equilibrium.

    The TVLQR gains saved per reference are FINITE-horizon: they terminate at the
    swing-up's last node and were never meant to hold the (highly unstable)
    inverted equilibrium beyond it. This solves the discrete-time algebraic
    Riccati equation for the plant linearized about (x_goal, u_eq) -- the same
    single-RK4-step discretization (dt = control rate) the swing-up gains use --
    and returns the steady-state stabilizing gain K (nu, nx). Feed it to
    `rollout_tvlqr(..., K_hold=K)` so the upright hold is regulated by a controller
    that is actually stabilizing, instead of extrapolating the swing-up gains.

    Q, R may be diagonals (1-D) or full matrices.
    """
    x_goal = np.asarray(x_goal, dtype=np.float64)
    Q = np.diag(np.asarray(Q, dtype=np.float64)) if np.ndim(Q) == 1 else np.asarray(Q, float)
    R = np.diag(np.asarray(R, dtype=np.float64)) if np.ndim(R) == 1 else np.asarray(R, float)
    u_eq = equilibrium_torque(x_goal)

    # CONTINUOUS-time linearization of the plant at (x_goal, u_eq). We deliberately
    # differentiate `dynamics`, NOT one RK4 control step: this plant's arctan-
    # smoothed Coulomb friction makes it stiff (fast ~1e3 /s stable modes), and a
    # single RK4 step at dt=0.05 is numerically unstable for those modes, which
    # corrupts the step Jacobian (spurious ~1e5 eigenvalues) and yields an absurd,
    # diverging gain. The continuous Jacobian is well-conditioned.
    xg, ug = jnp.asarray(x_goal), jnp.asarray(u_eq)
    A_c = np.asarray(jax.jacobian(lambda x: dynamics(x, ug))(xg))
    B_c = np.asarray(jax.jacobian(lambda u: dynamics(xg, u))(ug))

    # Exact zero-order-hold discretization over one control step: expm of the
    # augmented [[A, B], [0, 0]] block gives [[A_d, B_d], [0, I]].
    nx, nu = A_c.shape[0], B_c.shape[1]
    M_aug = np.zeros((nx + nu, nx + nu))
    M_aug[:nx, :nx] = A_c
    M_aug[:nx, nx:] = B_c
    E = sla.expm(M_aug * dt)
    A_d, B_d = E[:nx, :nx], E[:nx, nx:]

    P = sla.solve_discrete_are(A_d, B_d, Q, R)
    return np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)


def rollout_tvlqr(x_ref, u_ref, K, x0, n_steps,
                  dt_control=0.05, dt_sim=None, torque_limit=None,
                  deviation_threshold=2.0, noise_std=None, rng=None,
                  noise_until_upright=False, x_goal=None, K_hold=None,
                  hold_engage_ang=0.6):
    """
    Closed-loop TVLQR rollout in the numpy/RK4 simulator.

    Tracks the nominal trajectory index-by-index; if the feature-space deviation
    exceeds `deviation_threshold` it switches to a nearest-neighbour +
    distance-weighted-gain recovery mode. Once the nominal is exhausted (holding)
    it regulates the goal equilibrium -- with the stabilizing `K_hold` gain when
    one is supplied (see below), otherwise by extrapolating the swing-up gains.
    x_ref: (steps, nx)  u_ref: (steps, nu)  K: (steps-1, nu, nx)

    x_goal: the angular goal [q1, q2, .., ..] the success/latch checks measure
    against; defaults to the [pi, 0] upright (only x_goal[0], x_goal[1] are used).
    Pass the reference's own goal so trajectories that swing up to a DIFFERENT
    target (e.g. [pi, pi]) are still recognized as successful holds.

    K_hold: optional infinite-horizon LQR gain (nu, nx) from `lqr_gain(x_goal, ..)`
    that STABILIZES the goal. The saved per-reference gains are finite-horizon and
    end with the swing-up, so beyond the last node they cannot hold the unstable
    inverted equilibrium; once holding, K_hold regulates x toward `x_goal` (with
    the equilibrium feed-forward torque) so the upright hold actually persists.
    Because the inverted equilibrium is NOT stabilizable at the 20 Hz control rate
    with this actuator, the hold loop is re-closed at the integration rate dt_sim;
    pass a gain designed for dt_sim, e.g. lqr_gain(x_goal, Q, R, dt=dt_sim). The
    hold engages (and latches) as soon as the state is within `hold_engage_ang`
    (summed |q1|+|q2| error) of the goal -- not only when the swing-up nominal is
    exhausted -- so rollouts that START near the goal (goal-perturbed hold
    rollouts) use the stabilizing hold immediately instead of trying to track the
    swing-up from node 0.

    DAgger coverage: pass noise_std + rng to inject Gaussian state noise after
    each step, spreading the off-nominal states the controller must recover from
    (the CLEAN commanded action is still logged as the supervised target). With
    noise_until_upright=True the noise latches OFF once the swing-up first reaches
    the top, so the fragile upright hold stays clean and the rollout still
    succeeds -- we keep the noisy swing-up recovery states without losing the hold.

    Returns (states (n_steps, nx), actions (n_steps, nu), success).
    """
    tau = P.torque_limit if torque_limit is None else torque_limit
    # Goal the success/latch checks measure against (default = [pi, 0] upright).
    goal = (np.array([np.pi, 0.0, 0.0, 0.0]) if x_goal is None
            else np.asarray(x_goal, dtype=np.float64))
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
    # Stabilizing hold controller: regulate the goal with K_hold + the goal's
    # equilibrium feed-forward torque (computed from `goal`, so it is correct for
    # a mirrored goal too). Only used once the swing-up nominal is exhausted.
    K_hold = None if K_hold is None else np.asarray(K_hold)
    u_hold = None if K_hold is None else equilibrium_torque(goal)

    x = np.asarray(x0, dtype=np.float64).copy()
    current_idx = 0
    states, actions = [], []
    held = 0
    reached = False                       # latched once we first hit upright
    hold_engaged = False                  # latched once the LQR hold takes over

    for _ in range(n_steps):
        target = ref_T[:, current_idx].reshape(4, 1)
        _, d = _ranked(x, target, k=1)
        holding = current_idx >= max_idx
        # Engage the stabilizing LQR hold once near the goal (or once the swing-up
        # nominal is exhausted), then latch it. Proximity-engaging is what lets
        # goal-perturbed hold rollouts -- which start near the top at index 0 and
        # would otherwise sit in deviation-recovery forever -- actually hold.
        ang_to_goal = abs(wrap_to_pi(x[0] - goal[0])) + abs(wrap_to_pi(x[1] - goal[1]))
        fast_hold = K_hold is not None and (
            hold_engaged or holding or ang_to_goal < hold_engage_ang)
        if fast_hold:
            hold_engaged = True
            x_des, u_des, K_gain = goal, u_hold, K_hold
        elif d[0] > deviation_threshold or holding:
            best, _ = _ranked(x, ref_T, k=1)
            idx = int(best[0])
            K_gain = _K_weighted(x, ref_T, K, k=5)
            x_des, u_des = ref_T[:, idx], uref_T[:, idx]
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

        if fast_hold:
            # The inverted hold is too fast to stabilize at the 20 Hz control
            # rate (within one 0.05 s step this light pendulum falls past the
            # actuator's catchable basin), so once holding we RE-CLOSE the LQR
            # loop at the integration rate. K_hold must be designed for dt_sim.
            # The logged action above is the first such sub-step command.
            for _ in range(n_sub):
                e = x - goal
                e[0], e[1] = wrap_to_pi(e[0]), wrap_to_pi(e[1])
                u_sub = np.clip(u_hold - K_hold @ e, -tau, tau)
                x = np.array(rk4_step(jnp.asarray(x), jnp.asarray(u_sub), dt_sim),
                             dtype=np.float64)
                x[0], x[1] = wrap_to_pi(x[0]), wrap_to_pi(x[1])
        else:
            xs = jnp.asarray(x)
            uj = jnp.asarray(u)
            for _ in range(n_sub):
                xs = rk4_step(xs, uj, dt_sim)
            x = np.array(xs, dtype=np.float64)
            x[0] = wrap_to_pi(x[0])
            x[1] = wrap_to_pi(x[1])

        # Success / latch are evaluated on the CLEAN post-step state, so noise
        # can't spuriously flip the hold check.
        ang = abs(wrap_to_pi(x[0] - goal[0])) + abs(wrap_to_pi(x[1] - goal[1]))
        vel = abs(x[2]) + abs(x[3])
        reached = reached or (ang < 0.3)
        held = held + 1 if (ang < 0.2 and vel < 1.0) else 0

        if noise_std is not None and rng is not None \
                and not (noise_until_upright and reached):
            x = x + rng.normal(0.0, noise_std, size=4)
            x[0] = wrap_to_pi(x[0])
            x[1] = wrap_to_pi(x[1])

    # Success requires the rollout to END in a sustained upright hold (the final
    # `held` counts consecutive in-tolerance steps ending at the last step), so
    # swing-ups that reach the top but fall back off are rejected.
    return np.array(states), np.array(actions), held >= 10