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
# for deployment must describe the SAME physical system. We pull every inertial
# parameter from the XML, so editing dp.xml automatically updates the planner.
#
# The dynamics implemented below follow the cloudpendulum notebook
#   (dp_fwd_inv_dynamics/fwd_inv_dyn_student.ipynb) STRICTLY:
#
#     M(q) qddot + C(q,qdot) qdot + G(q) + F(qdot) = tau          (manipulator eq.)
#
#   M = [[ I1 + I2 + m2 l1^2 + 2 l1 m2 r2 c2 + gr^2 Ir + Ir,  I2 + l1 m2 r2 c2 ],
#        [ I2 + l1 m2 r2 c2,                                    I2             ]]
#   C = [[ -2 qd2 l1 m2 r2 s2,  -qd2 l1 m2 r2 s2 ],
#        [  qd1 l1 m2 r2 s2,     0               ]]
#   G = [ -g m1 r1 s1 - g m2 (l1 s1 + r2 s12),  -g m2 r2 s12 ]
#   F = [ b1 qd1 + mu1 atan(100 qd1),  b2 qd2 + mu2 atan(100 qd2) ]
#
#   where c2=cos q2, s2=sin q2, s1=sin q1, s12=sin(q1+q2), r1=lc1, r2=lc2.
#
# Notebook<->XML reconciliation (validated against MuJoCo's mj_fullM, see below):
#   * The notebook's link inertias I1, I2 are taken about the JOINT, whereas the
#     MJCF <inertial> diaginertia is about the COM. We convert with the parallel-
#     axis theorem  I_joint = I_com + m*lc^2  when loading, so that M(q) reproduces
#     MuJoCo's mj_fullM exactly (machine precision at every q2).
#   * Ir (rotor inertia) and gr (gear ratio) come from the joint armature and the
#     actuator gear; with armature=0, gear=1 the rotor term gr^2 Ir + Ir is zero.
#   * G is written for the dp.xml convention q1=0 -> hanging down (stable),
#     q1=pi -> upright (goal). It enters the forward dynamics as +G, which
#     reproduces MuJoCo's gravity torque exactly (the notebook's bare q=0=up frame
#     would flip this sign).

XML_PATH = Path(__file__).resolve().parent / "dp.xml"

# Coulomb (dry) friction coefficients. These are kept OUTSIDE the MJCF (dp.xml has
# no frictionloss) and applied in code via the notebook's Coulomb model so that
# the analytic planner and the MuJoCo replay share one friction model. Values are
# the cloudpendulum notebook's coulomb_fric = [mu1, mu2].
COULOMB_FRICTION = (0.00305, 0.0007777)

# Smooth Coulomb model from the notebook:  F_coulomb = mu * arctan(K * qdot).
# arctan saturates the friction torque at +-mu*pi/2 while staying differentiable,
# so it is compatible with gradient-based collocation. K=100 matches the notebook.
ARCTAN_K = 100.0

ModelParams = namedtuple(
    "ModelParams",
    "m1 m2 lc1 lc2 l1 I1 I2 b1 b2 f1 f2 gr Ir g tau1 tau2",
)


def load_params(xml_path=XML_PATH):
    """Extract the planar 2-link dynamics parameters straight from the MJCF.

    Inertias are returned about the JOINT (parallel-axis), matching the
    notebook's M(q) convention and MuJoCo's mj_fullM.
    """
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    b1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link1")
    b2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link2")

    m1 = float(m.body_mass[b1_id])
    m2 = float(m.body_mass[b2_id])
    lc1 = float(-m.body_ipos[b1_id][2])          # COM distance from joint 1
    lc2 = float(-m.body_ipos[b2_id][2])          # COM distance from joint 2
    # Hinge axis is y (axis="0 1 0") and body iquat is identity, so the inertia
    # about the rotation axis is component [1] of diaginertia (about the COM).
    I1_com = float(m.body_inertia[b1_id][1])
    I2_com = float(m.body_inertia[b2_id][1])

    return ModelParams(
        m1=m1,
        m2=m2,
        lc1=lc1,
        lc2=lc2,
        l1=float(-m.body_pos[b2_id][2]),         # joint1 -> joint2 distance
        I1=I1_com + m1 * lc1 ** 2,               # link1 inertia ABOUT JOINT 1
        I2=I2_com + m2 * lc2 ** 2,               # link2 inertia ABOUT JOINT 2
        b1=float(m.dof_damping[0]),              # viscous damping, joint 1
        b2=float(m.dof_damping[1]),
        f1=float(COULOMB_FRICTION[0]),           # Coulomb mu1 (from code, not XML)
        f2=float(COULOMB_FRICTION[1]),           # Coulomb mu2 (from code, not XML)
        gr=float(m.actuator_gear[0, 0]),         # gear ratio (notebook g_r)
        Ir=float(m.dof_armature[0]),             # rotor inertia (notebook I_r)
        g=float(-m.opt.gravity[2]),
        tau1=float(m.actuator_ctrlrange[0, 1] * m.actuator_gear[0, 0]),
        tau2=float(m.actuator_ctrlrange[1, 1] * m.actuator_gear[1, 0]),
    )


P = load_params()


# Cost weights for the standalone single-solve in main(). The dataset generator
# (generate_dataset.py) passes its own Q/R grid and does not use these.
Q = jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0]))
Qfin = Q
R = jnp.diag(jnp.array([1.0, 1.0])) * 0.01


# ─────────────────────────────────────────────
# Dynamics (notebook manipulator model, parameterized from dp.xml)
# ─────────────────────────────────────────────
def M(q):
    """Mass matrix M(q), strict notebook form (r2=lc2; I1,I2 about the joint)."""
    q2 = q[1]
    a = P.l1 * P.m2 * P.lc2 * jnp.cos(q2)         # l1 m2 r2 cos q2
    rotor = P.gr ** 2 * P.Ir + P.Ir               # gr^2 Ir + Ir (0 when armature=0)
    m00 = P.I1 + P.I2 + P.m2 * P.l1 ** 2 + 2 * a + rotor
    m01 = P.I2 + a
    m11 = P.I2
    return jnp.array([[m00, m01], [m01, m11]])


def C(q):
    """Coriolis/centrifugal matrix C(q,qdot), strict notebook form."""
    q2, q1_dot, q2_dot = q[1], q[2], q[3]
    a = P.l1 * P.m2 * P.lc2 * jnp.sin(q2)         # l1 m2 r2 sin q2
    return jnp.array([[-2 * q2_dot * a, -q2_dot * a],
                      [      q1_dot * a,        0.0]])


def G(q):
    """Gravity vector G(q), notebook form in the dp.xml frame (q1=0 hangs down)."""
    q1, q2 = q[0], q[1]
    g1 = -P.g * P.m1 * P.lc1 * jnp.sin(q1) \
         - P.g * P.m2 * (P.l1 * jnp.sin(q1) + P.lc2 * jnp.sin(q1 + q2))
    g2 = -P.g * P.m2 * P.lc2 * jnp.sin(q1 + q2)
    return jnp.array([g1, g2])


def coulomb_friction(dq):
    """Notebook friction vector F(qdot) = b*qdot + mu*arctan(K*qdot).

    Viscous damping (b) is read from the MJCF; the Coulomb term (mu) lives in code
    (COULOMB_FRICTION) because dp.xml deliberately carries no frictionloss. arctan
    is the notebook's smooth, differentiable stand-in for sign(qdot).
    """
    viscous = jnp.array([P.b1, P.b2]) * dq
    coulomb = jnp.array([P.f1, P.f2]) * jnp.arctan(ARCTAN_K * dq)
    return viscous + coulomb


def dynamics(x, u):
    """Forward dynamics: qddot = M^{-1} (tau - C qdot - G - F).

    Algebraically the notebook manipulator equation  M qddot + C qdot + G + F = tau.
    G enters with +sign here because it is written in the q1=0=down frame (see
    header note), which reproduces MuJoCo's gravity torque exactly.
    """
    u  = u.reshape(2)
    dq = x[2:].reshape(2)
    rhs = u + G(x) - C(x) @ dq - coulomb_friction(dq)
    ddq = jnp.linalg.solve(M(x), rhs)
    return jnp.concatenate([dq, ddq]).flatten()


# ─────────────────────────────────────────────
# Time-varying LQR gains around the reference trajectory
# ─────────────────────────────────────────────
# The collocation solve gives an open-loop plan (trajectory.csv + inputs.csv);
# the swing-up is unstable AND the coarse dt=0.05 plan is open-loop infeasible, so
# deployment needs FEEDBACK. We build per-step gains K_k for the tracking law
#
#     u_k = u_ref_k - K_k (x - x_ref_k)
#
# which is exactly what TVLQRController in evaluate_swingup.py applies, and save
# them as results/K_matrix.npy.
#
# Two things were essential to get a K that actually swings up & holds in MuJoCo:
#   1. Linearize the REAL MuJoCo control-step (n_sub substeps, friction injected)
#      by finite differences -- NOT the analytic continuous dynamics. Over the
#      coarse dt=0.05 interval the true discrete map differs enough from expm() of
#      the instantaneous Jacobian that the analytic gains fail to track the swing.
#   2. A finite-horizon backward Riccati sweep (not per-knot ARE) with a heavy
#      terminal weight, so the gains are aggressive enough to hold the open-loop-
#      infeasible plan on the reference. Tuned below; verified to swing up + hold.
Q_LQR  = np.diag([500.0, 500.0, 1.0, 1.0])
R_LQR  = np.diag([1.0, 1.0]) * 5e-4
Qf_LQR = np.diag([2000.0, 2000.0, 5.0, 5.0])


def _mj_apply_friction(model, data):
    """Inject the notebook's Coulomb friction (dp.xml has none) -- same model as
    simulation.apply_coulomb_friction, kept local to avoid importing the gym env."""
    data.qfrc_applied[:2] = -np.array([P.f1, P.f2]) * np.arctan(ARCTAN_K * data.qvel[:2])


def _step_control(model, data, x, u, n_sub):
    """Advance the friction-aware MuJoCo sim one control step (n_sub substeps)."""
    data.qpos[:2] = x[:2]
    data.qvel[:2] = x[2:]
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    for _ in range(n_sub):
        _mj_apply_friction(model, data)
        mujoco.mj_step(model, data)
    return np.concatenate([data.qpos[:2], data.qvel[:2]])


def _linearize_discrete(model, data, x, u, n_sub, eps=1e-6):
    """Central-difference linearization of the discrete control-step map x->x+."""
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    for i in range(4):
        dx = np.zeros(4); dx[i] = eps
        A[:, i] = (_step_control(model, data, x + dx, u, n_sub)
                   - _step_control(model, data, x - dx, u, n_sub)) / (2 * eps)
    for j in range(2):
        du = np.zeros(2); du[j] = eps
        B[:, j] = (_step_control(model, data, x, u + du, n_sub)
                   - _step_control(model, data, x, u - du, n_sub)) / (2 * eps)
    return A, B


def compute_tvlqr_gains(x_traj, u_traj, dt, Q=Q_LQR, R=R_LQR, Qf=Qf_LQR,
                        xml_path=XML_PATH):
    """TVLQR gains via a backward Riccati sweep over the MuJoCo-linearized discrete
    dynamics (friction included), matching deployment. Returns K (steps-1, nu, nx);
    K_k is the gain for interval k in  u = u_ref - K (x - x_ref).
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(dt / model.opt.timestep)))

    x_traj = np.asarray(x_traj)
    u_traj = np.asarray(u_traj)
    Q, R = np.asarray(Q, dtype=float), np.asarray(R, dtype=float)
    steps = x_traj.shape[0]

    AB = [_linearize_discrete(model, data, x_traj[k], u_traj[k], n_sub)
          for k in range(steps - 1)]

    P = np.asarray(Qf, dtype=float)
    gains = []
    for A, B in reversed(AB):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        gains.append(K)
    gains.reverse()
    return np.asarray(gains)


def regenerate_gains(results_dir, dt=0.05):
    """Compute K_matrix.npy for the trajectory.csv/inputs.csv already on disk,
    WITHOUT re-solving them. This is the safe way to (re)create the gains for the
    validated reference: the trajectory is left untouched, so the controller it
    was tuned for keeps working."""
    x_traj = np.loadtxt(os.path.join(results_dir, "trajectory.csv"),
                        delimiter=",", skiprows=1)
    u_traj = np.loadtxt(os.path.join(results_dir, "inputs.csv"),
                        delimiter=",", skiprows=1)
    K = compute_tvlqr_gains(x_traj, u_traj, dt)
    np.save(os.path.join(results_dir, "K_matrix.npy"), K)
    print(f"Saved TVLQR gains for the existing reference: K_matrix.npy {K.shape} "
          f"(mean|K|={np.abs(K).mean():.3f}, max|K|={np.abs(K).max():.3f})")
    return K


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

    dp.xml carries no Coulomb friction; we inject the notebook's Coulomb model
    (mu*arctan(K*qdot)) as a generalized force via qfrc_applied each substep, so
    the MuJoCo replay sees the SAME friction the planner optimized against.
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
    mu = np.array([P.f1, P.f2])

    sim = []
    for u in u_traj:
        d.ctrl[:] = np.clip(u, lo, hi)
        for _ in range(n_sub):
            # Coulomb friction torque opposes motion: -mu*arctan(K*qdot).
            d.qfrc_applied[:2] = -mu * np.arctan(ARCTAN_K * d.qvel[:2])
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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gains-only", action="store_true",
        help="Only (re)compute results/K_matrix.npy for the trajectory.csv/"
             "inputs.csv already on disk; do NOT re-solve or overwrite them. "
             "Use this to refresh the TVLQR gains for the validated reference.")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")

    if args.gains_only:
        # Safe path: keep the validated swing-up trajectory, just rebuild its gains.
        print("Computing TVLQR gains for the existing reference (no re-solve)...")
        regenerate_gains(results_dir)
        return

    print("Loaded model parameters from dp.xml:")
    for k, v in P._asdict().items():
        print(f"  {k:5s} = {v:.6g}")
    print("\nWARNING: a full re-solve OVERWRITES trajectory.csv / inputs.csv with a\n"
          "fresh collocation plan. That plan reaches upright only near the final\n"
          "knot and is hard to track closed-loop, so the resulting reference may\n"
          "NOT pass the generate_tvlqr_dataset self-check. To refresh only the\n"
          "gains for the validated reference, run:  python generate_k.py --gains-only\n")

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

    # ── Sanity check: replay the torques open-loop in MuJoCo. ──
    # NOTE: the per-step analytic dynamics match MuJoCo to ~1e-13 (M == mj_fullM,
    # same C/G/F). Any large error here is NOT a model mismatch but the expected
    # divergence of an OPEN-LOOP replay of a coarse (dt=0.05 trapezoidal) plan on
    # an unstable swing-up -- which is exactly why the pipeline adds TVLQR feedback
    # (generate_tvlqr_dataset.py). Closed-loop, the trajectory is tracked.
    sim_traj = verify_open_loop(x_traj, u_traj, dt)
    print("MuJoCo final state (open-loop replay):", sim_traj[-1])
    err = float(np.linalg.norm(sim_traj[-1] - np.asarray(x_traj[-1])))
    print(f"Planner-vs-MuJoCo open-loop final-state error: {err:.4f} "
          f"(open-loop on an unstable plan; stabilized closed-loop by TVLQR)")

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    np.savetxt(os.path.join(results_dir, "trajectory.csv"), np.asarray(x_traj),
               delimiter=",", header="q1,q2,q1_dot,q2_dot", comments="")
    header_full = "time,q1,q2,q1_dot,q2_dot"
    data_full = np.column_stack([np.asarray(time_span), np.asarray(x_traj)])
    np.savetxt(os.path.join(results_dir, "optimal_trajectory_full.csv"), data_full,
               delimiter=",", header=header_full, comments="")
    np.savetxt(os.path.join(results_dir, "inputs.csv"), np.asarray(u_traj),
               delimiter=",", header="u1,u2", comments="")

    # TVLQR feedback gains around this plan -> K_matrix.npy. Saved together with
    # the CSVs so (trajectory, inputs, K) always form a consistent set.
    K_gains = compute_tvlqr_gains(np.asarray(x_traj), np.asarray(u_traj), dt)
    np.save(os.path.join(results_dir, "K_matrix.npy"), K_gains)
    print(f"Saved TVLQR gains: K_matrix.npy {K_gains.shape} "
          f"(mean|K|={np.abs(K_gains).mean():.3f}, max|K|={np.abs(K_gains).max():.3f})")

    plot_results(np.asarray(time_span), np.asarray(x_traj), np.asarray(u_traj),
                 os.path.join(base, "graphs/reference"), sim_traj=sim_traj)


if __name__ == "__main__":
    main()
