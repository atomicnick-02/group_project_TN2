import os

# These NLPs are tiny and CPU-bound; keep JAX on CPU so GPU dispatch overhead
# doesn't slow down the many small solves. Must be set before `import jax`.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
import h5py
from pathlib import Path
from functools import partial
from tqdm import tqdm

import generate_k

jax.config.update("jax_enable_x64", True)


# ── Compiled-once objective / gradient ──────────────────────────────────────────
# KEY FIX vs the original: the weight matrices are ARGUMENTS, not closed-over
# Python values. JAX traces/compiles each function exactly once and reuses it for
# every (Q, R) in the grid, instead of recompiling on all ~4320 iterations.

@partial(jax.jit, static_argnums=(4, 5, 6))
def total_objective(z, q_diag, r_diag, x_goal, steps, nx, nu):
    Q = jnp.diag(q_diag)
    R = jnp.diag(r_diag)
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)

    def stage(x, u):
        e = x - x_goal
        return e @ Q @ e + u @ R @ u

    costs    = jax.vmap(stage)(X, U) * 0.5
    terminal = (X[-1] - x_goal) @ Q @ (X[-1] - x_goal)
    return jnp.sum(costs.at[-1].set(terminal))


@partial(jax.jit, static_argnums=(4, 5, 6))
def total_objective_grad(z, q_diag, r_diag, x_goal, steps, nx, nu):
    Q = jnp.diag(q_diag)
    R = jnp.diag(r_diag)
    X = z[:steps * nx].reshape(steps, nx)
    U = z[steps * nx:].reshape(steps, nu)
    gX, gU = jax.vmap(lambda x, u: (2 * Q @ (x - x_goal), 2 * R @ u))(X, U)
    gX = (gX * 0.5).at[-1].set(2 * Q @ (X[-1] - x_goal))
    gU = gU * 0.5
    return jnp.concatenate([gX.flatten(), gU.flatten()])


# Constraints/jacobian don't depend on the weights, so they compile once too.
@partial(jax.jit, static_argnums=(1, 2, 3))
def constraints_fn(z, steps, nx, nu, x0, x_goal, dt):
    return generate_k.constraints(z, steps, nx, nu, x0, x_goal, dt)


@partial(jax.jit, static_argnums=(1, 2, 3))
def constraints_jac(z, steps, nx, nu, x0, x_goal, dt):
    return jax.jacobian(
        lambda zz: generate_k.constraints(zz, steps, nx, nu, x0, x_goal, dt)
    )(z)


class IpoptProblem:
    """Thin cyipopt adapter that calls the pre-compiled JAX functions."""

    def __init__(self, q_diag, r_diag, x0, x_goal, steps, nx, nu, dt):
        self.q_diag, self.r_diag = q_diag, r_diag
        self.x0, self.x_goal     = x0, x_goal
        self.steps, self.nx, self.nu, self.dt = steps, nx, nu, dt

    def objective(self, z):
        return float(total_objective(z, self.q_diag, self.r_diag, self.x_goal,
                                      self.steps, self.nx, self.nu))

    def gradient(self, z):
        g = total_objective_grad(z, self.q_diag, self.r_diag, self.x_goal,
                                  self.steps, self.nx, self.nu)
        return np.asarray(g, dtype=np.float64).ravel()

    def constraints(self, z):
        c = constraints_fn(z, self.steps, self.nx, self.nu, self.x0, self.x_goal, self.dt)
        return np.asarray(c, dtype=np.float64)

    def jacobian(self, z):
        j = constraints_jac(z, self.steps, self.nx, self.nu, self.x0, self.x_goal, self.dt)
        return np.asarray(j, dtype=np.float64).ravel()


def solve_one(q_diag, r_val, peak_t, steps, nx, nu, x0, x_goal, dt, z_guess):
    """Solve a single NLP starting from z_guess (warm start). Returns (x, u, status, z_opt)."""
    q_diag_j = jnp.array(q_diag)
    r_diag_j = jnp.array([1.0, 1.0]) * r_val

    x_lb = jnp.full((steps, nx), -jnp.inf)
    x_ub = jnp.full((steps, nx),  jnp.inf)
    u_lb = jnp.full((steps, nu), -peak_t)
    u_ub = jnp.full((steps, nu),  peak_t)
    lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
    ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

    num_constraints = nx + (steps - 1) * nx + nx
    cl = cu = jnp.zeros(num_constraints)

    problem = cyipopt.Problem(
        n=len(z_guess), m=num_constraints,
        problem_obj=IpoptProblem(q_diag_j, r_diag_j, x0, x_goal, steps, nx, nu, dt),
        lb=lb, ub=ub, cl=cl, cu=cu,
    )
    problem.add_option('max_iter', 200)
    problem.add_option('tol', 1e-3)
    problem.add_option('print_level', 2)

    z_opt, info = problem.solve(np.asarray(z_guess, dtype=np.float64))
    x_traj = z_opt[:steps * nx].reshape(steps, nx)
    u_traj = z_opt[steps * nx:].reshape(steps, nu)
    return x_traj, u_traj, info['status_msg'], z_opt


def is_success(status):
    s = status.decode('utf-8') if isinstance(status, bytes) else str(status)
    return ("Success" in s
            or "Solved To Acceptable Level" in s
            or "locally optimal point" in s.lower()), s


def generate_dataset():
    Q_values = [
        [10.0, 10.0, 10.0, 10.0],
        [100.0, 100.0, 1.0, 1.0],
        [500.0, 500.0, 10.0, 10.0],
        [10.0, 10.0, 0.1, 0.1],
    ]
    R_values     = [0.001, 0.01, 0.1, 1, 10, 100]
    Peak_Torques = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15]

    T  = 2.0
    dt = 0.05
    steps  = int(T / dt) + 1
    nx, nu = 4, 2
    x_goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    possible_x0 = [
        jnp.array([0.0, 0.0, 0.0, 0.0]),
        *[jnp.array([angle, 0.0, 0.0, 0.0]) for angle in jnp.linspace(-jnp.pi, jnp.pi, 5)],
        *[jnp.array([0.0, angle, 0.0, 0.0]) for angle in jnp.linspace(-jnp.pi, jnp.pi, 5)],
        *[jnp.array([0.0, 0.0, vel, 0.0])   for vel   in jnp.linspace(-2.0, 2.0, 5)],
        *[jnp.array([0.0, 0.0, 0.0, vel])   for vel   in jnp.linspace(-2.0, 2.0, 5)],
    ]

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    h5_path = results_dir / "expert_trajectories.h5"
    print(f"Generating dataset at {h5_path}...")

    total_iters = len(possible_x0) * len(Q_values) * len(R_values) * len(Peak_Torques)
    count = 0
    with h5py.File(h5_path, 'w') as f:
        with tqdm(total=total_iters, desc="Overall progress") as pbar:
            for x0 in possible_x0:
                x0_j = jnp.asarray(x0)

                # Cold guess for the first solve at this x0.
                x_init = jnp.zeros((steps, nx)).at[:, 0].set(
                    jnp.linspace(float(x0_j[0]), float(x_goal[0]), steps))
                cold_guess = jnp.concatenate([x_init.flatten(), jnp.zeros((steps, nu)).flatten()])

                for q_diag in Q_values:
                    for r_val in R_values:
                        # Warm-start the peak_t sweep: adjacent torque caps give
                        # similar trajectories, cutting Ipopt iterations.
                        z_chain = cold_guess
                        for peak_t in Peak_Torques:
                            try:
                                x_traj, u_traj, status, z_opt = solve_one(
                                    q_diag, r_val, peak_t, steps, nx, nu,
                                    x0_j, x_goal, dt, z_chain
                                )
                                ok, status_str = is_success(status)
                                if ok:
                                    grp = f.create_group(f"traj_{count}")
                                    grp.create_dataset("states",  data=np.asarray(x_traj))
                                    grp.create_dataset("actions", data=np.asarray(u_traj))
                                    grp.attrs["Q_weights"]   = np.asarray(q_diag, dtype=np.float64)
                                    grp.attrs["R_weight"]    = float(r_val)
                                    grp.attrs["peak_torque"] = float(peak_t)
                                    grp.attrs["status"]      = status_str.encode('ascii', 'ignore')
                                    count += 1
                                    z_chain = jnp.asarray(z_opt)  # warm-start next peak_t
                            except Exception as e:
                                print(f"Skipping x0={np.asarray(x0_j)}, R={r_val}, T={peak_t}: {e}")
                            finally:
                                pbar.update(1)

    print(f"\nDone! Successfully saved {count} trajectories to {h5_path}")


if __name__ == "__main__":
    generate_dataset()