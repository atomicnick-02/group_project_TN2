from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

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


def rollout_tvlqr(x_ref, u_ref, K, x0, n_steps,
                  dt_control=0.05, dt_sim=None, torque_limit=None,
                  deviation_threshold=2.0, noise_std=None, rng=None,
                  noise_until_upright=False):
    """
    Closed-loop TVLQR rollout in the numpy/RK4 simulator.

    Tracks the nominal trajectory index-by-index; if the feature-space deviation
    exceeds `deviation_threshold` (or the nominal has been exhausted -> holding),
    it switches to a nearest-neighbour + distance-weighted-gain recovery mode.
    x_ref: (steps, nx)  u_ref: (steps, nu)  K: (steps-1, nu, nx)

    DAgger coverage: pass noise_std + rng to inject Gaussian state noise after
    each step, spreading the off-nominal states the controller must recover from
    (the CLEAN commanded action is still logged as the supervised target). With
    noise_until_upright=True the noise latches OFF once the swing-up first reaches
    the top, so the fragile upright hold stays clean and the rollout still
    succeeds -- we keep the noisy swing-up recovery states without losing the hold.

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
    reached = False                       # latched once we first hit upright

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
        x[0] = wrap_to_pi(x[0])
        x[1] = wrap_to_pi(x[1])

        # Success / latch are evaluated on the CLEAN post-step state, so noise
        # can't spuriously flip the hold check.
        ang = abs(wrap_to_pi(x[0] - np.pi)) + abs(wrap_to_pi(x[1]))
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