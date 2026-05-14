import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────
# Fixed parameters
# ─────────────────────────────────────────────
GRAVITY = 9.806781
mass = [0.10548177618443695, 0.07619744360415454]
length = [0.05, 0.05]

m1, m2 = mass
l1, l2 = length

max_torque = 0.07     # physical hard stop

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
    g00 = -GRAVITY * m1 * l1 * jnp.sin(q1) - GRAVITY * m2 * (l1 * jnp.sin(q1) + l2 * jnp.sin(q1 + q2))
    g10 = -GRAVITY * m2 * l2 * jnp.sin(q1 + q2)
    return jnp.array([[g00], [g10]])


def dynamics(x, u):
    u = u.reshape(2, 1)
    dq = x[2:].reshape(2, 1)
    ddq = jnp.linalg.solve(M(x), u - C(x) @ dq + G(x))
    return jnp.concatenate([dq, ddq]).flatten()


def rk4(x, u, dt):
    k1 = dynamics(x, u).flatten()
    k2 = dynamics(x + 0.5 * dt * k1, u).flatten()
    k3 = dynamics(x + 0.5 * dt * k2, u).flatten()
    k4 = dynamics(x + dt * k3, u).flatten()
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ─────────────────────────────────────────────
# RK4-consistent discrete linearization
# ─────────────────────────────────────────────
def get_continuous_jacobians(x, u):
    A = jax.jacfwd(dynamics, argnums=0)(x, u)
    B = jax.jacfwd(dynamics, argnums=1)(x, u)
    return A, B


def get_discrete_matrices(x_k, u_k, dt):
    """A_d, B_d obtained by applying the chain rule through the RK4 step."""
    nx = x_k.shape[0]
    I = jnp.eye(nx)

    # Stage 1
    A1, B1 = get_continuous_jacobians(x_k, u_k)
    dk1_dx, dk1_du = A1, B1

    # Stage 2
    k1 = dynamics(x_k, u_k)
    x2 = x_k + 0.5 * dt * k1
    A2, B2 = get_continuous_jacobians(x2, u_k)
    dk2_dx = A2 @ (I + 0.5 * dt * dk1_dx)
    dk2_du = A2 @ (0.5 * dt * dk1_du) + B2

    # Stage 3
    k2 = dynamics(x2, u_k)
    x3 = x_k + 0.5 * dt * k2
    A3, B3 = get_continuous_jacobians(x3, u_k)
    dk3_dx = A3 @ (I + 0.5 * dt * dk2_dx)
    dk3_du = A3 @ (0.5 * dt * dk2_du) + B3

    # Stage 4
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


# ─────────────────────────────────────────────
# Feature-space reference lookup
# ─────────────────────────────────────────────
def to_feature_space(x):
    """[p0, p1, v0, v1] -> [cos p0, sin p0, cos p1, sin p1, 0.1*v0, 0.1*v1]"""
    v_scale = 0.1
    if x.ndim == 1:
        p0, p1 = x[0], x[1]
        v = x[2:] * v_scale
        return np.concatenate(([np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1)], v))
    p0, p1 = x[0, :], x[1, :]
    v = x[2:, :] * v_scale
    return np.vstack((np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1), v))


def get_ranked_distances(x, x_ref, k=1):
    feat_x = to_feature_space(x)
    feat_ref = to_feature_space(x_ref)
    dists = np.linalg.norm(feat_ref - feat_x.reshape(-1, 1), axis=0)
    idx = np.argsort(dists)
    return idx[:k], dists[idx[:k]]


def get_x_ref(x, x_ref):
    idx, _ = get_ranked_distances(x, x_ref, k=1)
    return int(idx[0])


def get_K_gain_euclidean(x, x_ref, K, k=5):
    """Distance-weighted average of K matrices at the k nearest ref points."""
    indices, dists = get_ranked_distances(x, x_ref, k=k)
    w = 1.0 / (dists + 1e-6)
    weights = w / np.sum(w)
    K_arr = np.asarray(K)
    indices = np.clip(indices, 0, len(K_arr) - 1)
    return np.sum(weights[:, None, None] * K_arr[indices], axis=0)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_results(x_hist, u_hist, x_ref, u_ref, dt, out_dir):
    x_hist = x_hist[:-1]  # match length of u_hist
    actual = x_hist.shape[0]
    t = np.arange(actual) * dt

    # Pad/trim reference to match simulation length
    if actual > x_ref.shape[1]:
        pad = actual - x_ref.shape[1]
        x_ref = np.concatenate([x_ref, np.tile(x_ref[:, -1:], (1, pad))], axis=1)
        u_ref = np.concatenate([u_ref, np.tile(u_ref[:, -1:], (1, pad))], axis=1)
    else:
        x_ref = x_ref[:, :actual]
        u_ref = u_ref[:, :actual]

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(t, x_hist[:, 0], label=r'$q_1$')
    axs[0].plot(t, x_hist[:, 1], label=r'$q_2$')
    axs[0].plot(t, x_ref[0], ':k', linewidth=0.8, label='Ref')
    axs[0].plot(t, x_ref[1], ':k', linewidth=0.8)
    axs[0].set(title='Joint Angles', xlabel='Time (s)', ylabel='Position (rad)')
    axs[0].legend(loc='lower left'); axs[0].grid(True)

    axs[1].plot(t, x_hist[:, 2], label=r'$\dot{q}_1$', color='blue')
    axs[1].plot(t, x_hist[:, 3], label=r'$\dot{q}_2$', color='orange')
    axs[1].plot(t, x_ref[2], ':', linewidth=0.8, color='blue')
    axs[1].plot(t, x_ref[3], ':', linewidth=0.8, color='orange')
    axs[1].set(title='Joint Velocities', xlabel='Time (s)', ylabel='Velocity (rad/s)')
    axs[1].legend(loc='lower left'); axs[1].grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "tracking.png"), dpi=150)

    err = np.linalg.norm(x_hist - x_ref.T, axis=1)
    print(f"Sum of tracking error: {err.sum():.3f}")

    fig2, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(t, err, label='TVLQR error')
    axs[0].set(title='State Tracking Error', xlabel='Time (s)', ylabel='Error norm')
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(t, u_hist[:, 0], label=r'$u_1$ (Nm)')
    axs[1].plot(t, u_hist[:, 1], label=r'$u_2$ (Nm)')
    axs[1].plot(t, u_ref[0], ':k', linewidth=0.8, label=r'Ref $u_1$')
    axs[1].plot(t, u_ref[1], ':', color='grey', linewidth=0.8, label=r'Ref $u_2$')
    axs[1].set(title='Control Inputs (Torques)', xlabel='Time (s)', ylabel='Torque (Nm)')
    axs[1].legend(); axs[1].grid(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "error_torques.png"), dpi=150)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")
    graphs_dir = os.path.join(base, "graphs/simulation")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # Load optimal trajectory
    x_ref = np.loadtxt(os.path.join(results_dir, "trajectory.csv"),
                       delimiter=",", skiprows=1).T
    u_ref = np.loadtxt(os.path.join(results_dir, "inputs.csv"),
                       delimiter=",", skiprows=1).T

    # Sim parameters
    dt_control = 0.05
    dt_sim = 0.01
    sim_substeps = int(dt_control / dt_sim)
    n_ref = x_ref.shape[1]
    max_idx = n_ref - 1
    sim_horizon = n_ref + 200      # extra steps to hold final pose
    DEVIATION_THRESHOLD = 2.0

    # ── TVLQR backward pass (over full reference) ──
    xs_scan = x_ref[:, :-1].T[::-1]
    us_scan = u_ref[:, :-1].T[::-1]
    scan_fn = lambda P, inp: tvlqr_backward_pass(P, inp, Q, R, dt_control)
    _, K_rev = jax.lax.scan(scan_fn, Qfin, (xs_scan, us_scan))
    K = np.asarray(K_rev[::-1])
    np.save(os.path.join(results_dir, "K_matrix.npy"), K)
    print(f"K shape: {K.shape}")

    # ── Measurement noise ──
    key = jax.random.PRNGKey(0)
    base_std = jnp.array([0.002, 0.002, 0.05, 0.05])
    noise_mult = 1.0
    noise_traj = noise_mult * jax.random.normal(key, shape=x_ref.shape) * base_std[:, None]

    # ── Closed-loop simulation ──
    x_current = jnp.array([0.0, 0.0, 0.0, 0.0])
    current_idx = 0
    x_history = [x_current]
    u_history = []

    print("Starting TVLQR simulation...")
    for _ in range(sim_horizon):
        # Detect tracking vs recovery
        target = x_ref[:, current_idx].reshape(4, 1)
        _, dists = get_ranked_distances(x_current, target, k=1)
        recovery = (dists[0] > DEVIATION_THRESHOLD) or (current_idx >= max_idx)

        if recovery:
            current_idx = get_x_ref(x_current, x_ref)
            K_gain = get_K_gain_euclidean(x_current, x_ref, K, k=5)
        else:
            K_gain = K[current_idx]

        x_des = x_ref[:, current_idx]
        u_des = u_ref[:, current_idx]

        # Feedback control with wrapped angle errors
        error = np.array(x_current - x_des)
        error[0] = wrap_to_pi(error[0])
        error[1] = wrap_to_pi(error[1])
        u_total = np.clip(u_des - K_gain @ error, -max_torque, max_torque)

        # Advance the reference index in tracking mode
        if not recovery:
            current_idx += 1

        # Physics sub-stepping
        x_sim = x_current
        for _ in range(sim_substeps):
            x_sim = rk4(x_sim, u_total, dt_sim)

        # Wrap angles for the state used in the next iteration
        x_next = x_sim.at[0].set(wrap_to_pi(x_sim[0])).at[1].set(wrap_to_pi(x_sim[1]))

        # Noisy measurement for plotting / outputs
        meas_idx = min(current_idx, x_ref.shape[1] - 1)
        x_meas = x_sim + noise_traj[:, meas_idx]

        x_history.append(x_meas)
        u_history.append(u_total)
        x_current = x_next

    plot_results(
        np.asarray(x_history),
        np.asarray(u_history).reshape(-1, 2),
        x_ref, u_ref, dt_control, graphs_dir,
    )


if __name__ == "__main__":
    main()