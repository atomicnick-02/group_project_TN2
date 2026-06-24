"""
Batch generator for optimal swing-up trajectories of the system-identified
double pendulum.

For each (Q, R, x_goal) configuration in CONFIGS this script:
  1. solves the direct-collocation trajectory-optimization problem with IPOPT,
     producing the optimal state trajectory x and control trajectory u, then
  2. linearizes about that trajectory and runs the TVLQR backward Riccati pass
     to obtain the time-varying feedback gains K.

The triple (x, u, K) for every configuration is written to its own folder under
optimal_trajectories/, named by the Q and R diagonals and the x_goal, e.g.

    optimal_trajectories/Q_100_100_1_1__R_0p01_0p01__xgoal_3p142_0_0_0/
        trajectory.csv   # x : (steps, 4)   header q1,q2,q1_dot,q2_dot
        inputs.csv       # u : (steps, 2)   header u1,u2
        K_matrix.npy     # K : (steps-1, 2, 4)
        config.json      # the Q/R/x_goal/solver metadata for this folder

trajectory.csv / inputs.csv / K_matrix.npy match the format consumed by
generate_tvlqr_dataset.load_reference(), so any folder is a drop-in reference.
"""

import os
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────
# Model parameters  (cloudpendulum dp_fwd_inv_dynamics, system-identified)
# ─────────────────────────────────────────────
from collections import namedtuple

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
# Trajectory-optimization cost (parameterized by Q, R, Qfin)
# ─────────────────────────────────────────────
def l(xk, uk, x_goal, Q, R):
    se = xk - x_goal
    return se.T @ Q @ se + uk.T @ R @ uk


def total_objective(z, steps, nx, nu, x_goal, Q, R, Qfin, dt=0.05):
    # z is the flattened decision vector [x0, x1, ..., xN, u0, u1, ..., uN-1]
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps, nu))

    point_costs = jax.vmap(lambda x, u: l(x, u, x_goal, Q, R) * 0.5)(X, U)
    # 1. Final State Cost
    final_diff = X[-1] - x_goal
    point_costs = point_costs.at[-1].set(final_diff.T @ Qfin @ final_diff)
    # 2. Sum the (weighted) per-node costs
    cost = jnp.sum(point_costs)

    return cost


def l_deriv(xk, uk, x_goal, Q, R):
    se = xk - x_goal
    grad_x = 2 * Q @ se
    grad_u = 2 * R @ uk
    return grad_x, grad_u


def total_objective_deriv(z, steps, nx, nu, x_goal, Q, R, Qfin, dt=0.05):
    # 1. Reshape z back into state and input trajectories
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps, nu))

    # 2. Compute raw gradients at each time step using l_deriv
    point_grads_X, point_grads_U = jax.vmap(
        lambda x, u: l_deriv(x, u, x_goal, Q, R))(X, U)

    # 3. Apply the weighting
    weights = jnp.ones(steps) * 0.5

    # Broadcast weights to match the shape of gradients
    grad_X_weighted = point_grads_X * weights[:, None]
    grad_U_weighted = point_grads_U * weights[:, None]

    # 4. Add Terminal Cost Gradient (applied only to the last state X[-1])
    terminal_grad_X = 2 * Qfin @ (X[-1] - x_goal)
    grad_X_weighted = grad_X_weighted.at[-1].set(terminal_grad_X)

    # 5. Flatten and concatenate to form the full gradient vector
    grad = jnp.concatenate([grad_X_weighted.flatten(), grad_U_weighted.flatten()])

    return grad


def trapezoidal_collocation(x_k, x_kp1, u_k, u_kp1, dynamics, dt):
    """
    Enforces a linear consistency between two nodes.
    x_k, x_kp1: states at node k and k+1
    u_k, u_kp1: controls at node k and k+1
    returns: Scalar constraint value showing the collocation error
    """
    f_k = dynamics(x_k, u_k)
    f_kp1 = dynamics(x_kp1, u_kp1)
    collocation_constraint = x_kp1 - x_k - (dt / 2.0) * (f_k + f_kp1)
    return collocation_constraint.flatten()


def constraints(z, steps, nx, nu, x0, x_goal, dt=0.05):
    X = z[:steps * nx].reshape((steps, nx))
    U = z[steps * nx:].reshape((steps, nu))

    # Initial state constraint
    c_init = X[0] - x0
    # We have 'steps - 1' intervals to constrain
    x_curr = X[:-1]
    x_next = X[1:]
    u_curr = U[:-1]
    u_next = U[1:]

    hs_vmap = jax.vmap(lambda xk, xkp1, uk, ukp1:
                       trapezoidal_collocation(xk, xkp1, uk, ukp1, dynamics, dt))

    c_dyn = hs_vmap(x_curr, x_next, u_curr, u_next).flatten()

    # Final state constraint
    c_final = X[-1] - x_goal
    return jnp.concatenate([c_init, c_dyn, c_final])


class Problem:
    """
    Defines a minimization problem with the following form:
        min f(z)   (z = [x0, x1, ..., xN, u0, u1, ..., uN-1])
        s.t. h(z) = 0
             lb <= z <= ub
    The cost weights (Q, R, Qfin), the start x0 and the goal xgoal are baked into
    the JIT-compiled callbacks so a fresh Problem is built per configuration.
    """

    def __init__(self, steps, nx, nu, x0, xgoal, dt, Q, R, Qfin):
        self.steps = steps
        self.nx = nx  # Number of states
        self.nu = nu  # Number of inputs
        # f(z) and its gradient (analytic), with the cost weights baked in.
        self._obj_jit = jax.jit(
            lambda z: total_objective(z, steps, nx, nu, xgoal, Q, R, Qfin, dt))
        self._grad_jit = jax.jit(
            lambda z: total_objective_deriv(z, steps, nx, nu, xgoal, Q, R, Qfin, dt))
        # h(z) and its Jacobian, with x0/xgoal baked in.
        self._cons_jit = jax.jit(
            lambda z: constraints(z, steps, nx, nu, x0, xgoal, dt))
        self._jac_jit = jax.jit(jax.jacobian(
            lambda z: constraints(z, steps, nx, nu, x0, xgoal, dt)))

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


# ─────────────────────────────────────────────
# TVLQR gains: linearize the RK4 step, then backward Riccati pass
# (faithful to old_pendulum/simul.ipynb)
# ─────────────────────────────────────────────
def get_continuous_jacobians(x, u):
    f = lambda x, u: dynamics(x, u)
    A = jax.jacfwd(f, argnums=0)(x, u)
    B = jax.jacfwd(f, argnums=1)(x, u)
    return A, B


def get_discrete_matrices(x_k, u_k, dt=0.05):
    """Discrete-time A_d, B_d via the chain rule through the RK4 step."""
    nx = x_k.shape[0]
    I = jnp.eye(nx)

    # Stage 1: k1 = f(x, u)
    A1, B1 = get_continuous_jacobians(x_k, u_k)
    dk1_dx = A1
    dk1_du = B1

    # Stage 2: k2 = f(x + 0.5*dt*k1, u)
    x2 = x_k + 0.5 * dt * dynamics(x_k, u_k)
    A2, B2 = get_continuous_jacobians(x2, u_k)
    dk2_dx = A2 @ (I + 0.5 * dt * dk1_dx)
    dk2_du = A2 @ (0.5 * dt * dk1_du) + B2

    # Stage 3: k3 = f(x + 0.5*dt*k2, u)
    x3 = x_k + 0.5 * dt * dynamics(x_k + 0.5 * dt * dynamics(x_k, u_k), u_k)
    A3, B3 = get_continuous_jacobians(x3, u_k)
    dk3_dx = A3 @ (I + 0.5 * dt * dk2_dx)
    dk3_du = A3 @ (0.5 * dt * dk2_du) + B3

    # Stage 4: k4 = f(x + dt*k3, u)
    k1_val = dynamics(x_k, u_k)
    k2_val = dynamics(x_k + 0.5 * dt * k1_val, u_k)
    k3_val = dynamics(x_k + 0.5 * dt * k2_val, u_k)
    x4 = x_k + dt * k3_val
    A4, B4 = get_continuous_jacobians(x4, u_k)
    dk4_dx = A4 @ (I + dt * dk3_dx)
    dk4_du = A4 @ (dt * dk3_du) + B4

    # A_d = I + dt/6 (dk1_dx + 2 dk2_dx + 2 dk3_dx + dk4_dx)
    A_d = I + (dt / 6.0) * (dk1_dx + 2 * dk2_dx + 2 * dk3_dx + dk4_dx)
    # B_d = dt/6 (dk1_du + 2 dk2_du + 2 dk3_du + dk4_du)
    B_d = (dt / 6.0) * (dk1_du + 2 * dk2_du + 2 * dk3_du + dk4_du)
    return A_d, B_d


def tvlqr_backward_pass(P_next, inputs, Q_d, R_d, dt=0.05):
    x_k, u_k = inputs
    A_d, B_d = get_discrete_matrices(x_k, u_k, dt)
    S = R_d + B_d.T @ P_next @ B_d
    K_k = jnp.linalg.solve(S, B_d.T @ P_next @ A_d)
    P_k = Q_d + A_d.T @ P_next @ (A_d - B_d @ K_k)
    return P_k, K_k


def compute_tvlqr_gains(x_traj, u_traj, Q, R, Qfin, dt):
    """Backward Riccati sweep about (x_traj, u_traj). Returns K (steps-1, nu, nx)."""
    x_traj = jnp.asarray(x_traj)
    u_traj = jnp.asarray(u_traj)
    # Sweep backward over the steps-1 intervals (reverse time order).
    xs_scan = x_traj[:-1][::-1]
    us_scan = u_traj[:-1][::-1]
    scan_fn = lambda p, inp: tvlqr_backward_pass(p, inp, Q, R, dt)
    _, K_rev = jax.lax.scan(scan_fn, Qfin, (xs_scan, us_scan))
    K = K_rev[::-1]                       # back to forward time order
    return np.asarray(K)


# ─────────────────────────────────────────────
# Solve one trajectory-optimization problem
# ─────────────────────────────────────────────
def solve_trajectory(x0, xgoal, Q, R, Qfin, steps, nx, nu, dt,
                     max_iter=100, tol=1e-3, print_level=0):
    num_vars = steps * nx + steps * nu
    num_constraints = (steps - 1) * nx + nx + nx

    # Warm start: linear interpolation x0 -> xgoal per state dim, zero controls.
    x_init = jnp.zeros((steps, nx))
    for i in range(nx):
        x_init = x_init.at[:, i].set(jnp.linspace(x0[i], xgoal[i], steps))
    u_init = jnp.zeros((steps, nu))
    z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

    # Bounds: states unbounded, controls within the actuator torque limit.
    x_lb = jnp.full((steps, nx), -jnp.inf)
    x_ub = jnp.full((steps, nx), jnp.inf)
    u_lb = jnp.full((steps, nu), -P.torque_limit)
    u_ub = jnp.full((steps, nu), P.torque_limit)
    lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
    ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

    # Equality constraints (initial + dynamics + final) => zeros.
    cl = jnp.zeros(num_constraints)
    cu = jnp.zeros(num_constraints)

    problem = cyipopt.Problem(
        n=num_vars,
        m=num_constraints,
        problem_obj=Problem(steps, nx, nu, x0, xgoal, dt, Q, R, Qfin),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    problem.add_option('max_iter', max_iter)
    problem.add_option('tol', tol)
    problem.add_option('print_level', print_level)

    z_opt, info = problem.solve(z0)

    x_traj = np.asarray(z_opt[:steps * nx]).reshape((steps, nx))
    u_traj = np.asarray(z_opt[steps * nx:]).reshape((steps, nu))
    return x_traj, u_traj, info


# ─────────────────────────────────────────────
# Naming + saving
# ─────────────────────────────────────────────
def _fmt(v):
    """Compact, filesystem-safe token for a scalar: '.'->'p', '-'->'m'."""
    v = float(v)
    s = str(int(v)) if v == int(v) else f"{v:.4g}"
    return s.replace("-", "m").replace(".", "p")


def config_dirname(Q, R, xgoal):
    """Folder name encoding the Q and R diagonals and the goal state."""
    q = "_".join(_fmt(v) for v in np.diag(np.asarray(Q)))
    r = "_".join(_fmt(v) for v in np.diag(np.asarray(R)))
    g = "_".join(_fmt(v) for v in np.asarray(xgoal))
    return f"Q_{q}__R_{r}__xgoal_{g}"


def save_trajectory(out_dir, x_traj, u_traj, K, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "trajectory.csv", x_traj, delimiter=",",
               header="q1,q2,q1_dot,q2_dot", comments="")
    np.savetxt(out_dir / "inputs.csv", u_traj, delimiter=",",
               header="u1,u2", comments="")
    np.save(out_dir / "K_matrix.npy", np.asarray(K))
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)


def save_plots(out_dir, time_span, x_traj, u_traj):
    import matplotlib
    matplotlib.use("Agg")              # headless: write PNGs, never block on a window
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(time_span, x_traj[:, 0], label=r'$q_1$ (rad)')
    ax1.plot(time_span, x_traj[:, 1], label=r'$q_2$ (rad)')
    ax1.set_title('Joint States over Time')
    ax1.set_ylabel('Angle (rad)'); ax1.set_xlabel('Time (s)')
    ax1.legend(); ax1.grid(True)
    ax2.plot(time_span, x_traj[:, 2], label=r'$\dot{q}_1$ (rad/s)', linestyle='--')
    ax2.plot(time_span, x_traj[:, 3], label=r'$\dot{q}_2$ (rad/s)', linestyle='--')
    ax2.set_title('Joint Velocities over Time')
    ax2.set_ylabel('Value (rad/s)'); ax2.set_xlabel('Time (s)')
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "states.png", dpi=120)
    plt.close(fig)

    fig2 = plt.figure(figsize=(7, 4))
    plt.plot(time_span, u_traj[:, 0], label=r'$u_1$ (Nm)')
    plt.plot(time_span, u_traj[:, 1], label=r'$u_2$ (Nm)')
    plt.title('Control Inputs (Torques) over Time')
    plt.xlabel('Time (s)'); plt.ylabel('Torque (Nm)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig2.savefig(out_dir / "torques.png", dpi=120)
    plt.close(fig2)


# ─────────────────────────────────────────────
# Configurations to generate
#   Each entry: Q (4 state weights), R (2 control weights), and optionally
#   Qfin (terminal state weights, default = Q) and xgoal (default upright).
# ─────────────────────────────────────────────
UPRIGHT = [float(np.pi), 0.0, 0.0, 0.0]

Q_SWEEP = [
    [10, 10, 1, 1],
    [10, 10, 10, 10],
    [100, 100, 1, 1],
    [100, 100, 10, 10],
    [100, 100, 100, 100],

]

R_SWEEP = [
    [0.01, 0.01],
    [10, 10],
    [50, 50],
    [100, 100],
    [200, 200],
    [500, 500],
    [1000, 1000],


]

XGOAL_SWEEP = [
    [np.pi, 0.0, 0.0, 0.0],
    [np.pi, np.pi, 0.0, 0.0],
]

CONFIGS = [
    dict(Q=Q, R=R, xgoal=xgoal)
    for xgoal in XGOAL_SWEEP
    for Q in Q_SWEEP
    for R in R_SWEEP
]

SAVE_PLOTS = True


def main():
    nx, nu = 4, 2
    T, dt = 2.0, 0.05
    steps = int(T / dt) + 1
    time_span = np.linspace(0, T, steps)
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])

    # optimal_trajectories/ lives in double_pendulum/ (one level up from this generators/ dir).
    out_root = Path(__file__).resolve().parent.parent / "optimal_trajectories"
    out_root.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(CONFIGS, desc="optimal trajectories", unit="traj")
    for i, cfg in enumerate(pbar):
        Q = jnp.diag(jnp.array(cfg["Q"], dtype=float))
        R = jnp.diag(jnp.array(cfg["R"], dtype=float))
        Qfin = jnp.diag(jnp.array(cfg.get("Qfin", cfg["Q"]), dtype=float))
        xgoal = jnp.array(cfg.get("xgoal", UPRIGHT), dtype=float)

        name = config_dirname(Q, R, xgoal)
        out_dir = out_root / name
        pbar.set_postfix_str(name)
        tqdm.write(f"[{i + 1}/{len(CONFIGS)}] {name}")

        # 1. Optimal (x, u) via direct collocation.
        x_traj, u_traj, info = solve_trajectory(
            x0, xgoal, Q, R, Qfin, steps, nx, nu, dt)
        status = int(info["status"])
        msg = info.get("status_msg", b"")
        msg = msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
        tqdm.write(f"    IPOPT status {status}: {msg}")
        tqdm.write(f"    final state {np.round(x_traj[-1], 4)} (goal "
                   f"{np.round(np.asarray(xgoal), 4)})")

        # 2. Time-varying LQR gains about that trajectory.
        K = compute_tvlqr_gains(x_traj, u_traj, Q, R, Qfin, dt)

        # 3. Save x, u, K (+ metadata, + optional plots).
        meta = {
            "Q_diag": [float(v) for v in np.diag(np.asarray(Q))],
            "R_diag": [float(v) for v in np.diag(np.asarray(R))],
            "Qfin_diag": [float(v) for v in np.diag(np.asarray(Qfin))],
            "x0": [float(v) for v in np.asarray(x0)],
            "x_goal": [float(v) for v in np.asarray(xgoal)],
            "T": float(T), "dt": float(dt), "steps": int(steps),
            "nx": nx, "nu": nu,
            "torque_limit": float(P.torque_limit),
            "ipopt_status": status,
            "ipopt_status_msg": msg,
            "ipopt_obj_val": float(info.get("obj_val", np.nan)),
            "final_state": [float(v) for v in np.asarray(x_traj[-1])],
            "x_shape": list(np.asarray(x_traj).shape),
            "u_shape": list(np.asarray(u_traj).shape),
            "K_shape": list(np.asarray(K).shape),
        }
        save_trajectory(out_dir, x_traj, u_traj, K, meta)
        if SAVE_PLOTS:
            save_plots(out_dir, time_span, x_traj, u_traj)
        tqdm.write(f"    saved x{meta['x_shape']}, u{meta['u_shape']}, "
                   f"K{meta['K_shape']} -> {out_dir}")

    print(f"\nDone. {len(CONFIGS)} trajectories written under {out_root}")


if __name__ == "__main__":
    main()
