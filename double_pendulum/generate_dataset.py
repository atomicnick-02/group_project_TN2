import os
import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
import h5py
from pathlib import Path
from tqdm import tqdm

# Import dynamics and problem functions from generate_k
import generate_k
from generate_k import Problem, dynamics, torque_limits, Q, R, Qfin

jax.config.update("jax_enable_x64", True)

def run_optimization(q_diag, r_val, peak_t, steps, nx, nu, x0, x_goal, dt):
    """
    Runs a single optimization with custom weights and torque limits.
    """
    # Override global-like parameters for this run
    q_matrix = jnp.diag(jnp.array(q_diag))
    r_matrix = jnp.diag(jnp.array([1.0, 1.0])) * r_val
    t_limits = jnp.array([-peak_t, peak_t])

    # Re-define objective functions for these specific weights
    def local_stage_cost(x, u, x_goal):
        e = x - x_goal
        return e @ q_matrix @ e + u @ r_matrix @ u

    def local_total_objective(z, steps, nx, nu, x_goal):
        X = z[:steps * nx].reshape(steps, nx)
        U = z[steps * nx:].reshape(steps, nu)
        costs = jax.vmap(lambda x, u: local_stage_cost(x, u, x_goal))(X, U) * 0.5
        terminal = (X[-1] - x_goal) @ q_matrix @ (X[-1] - x_goal)
        return jnp.sum(costs.at[-1].set(terminal))

    def local_total_objective_grad(z, steps, nx, nu, x_goal):
        X = z[:steps * nx].reshape(steps, nx)
        U = z[steps * nx:].reshape(steps, nu)
        gX, gU = jax.vmap(lambda x, u: (2 * q_matrix @ (x - x_goal), 2 * r_matrix @ u))(X, U)
        gX = (gX * 0.5).at[-1].set(2 * q_matrix @ (X[-1] - x_goal))
        gU = gU * 0.5
        return jnp.concatenate([gX.flatten(), gU.flatten()])

    class LocalProblem:
        def __init__(self, steps, nx, nu, x0, x_goal, dt):
            self._obj = jax.jit(lambda z: local_total_objective(z, steps, nx, nu, x_goal))
            self._grad = lambda z: local_total_objective_grad(z, steps, nx, nu, x_goal)
            self._cons = jax.jit(lambda z: generate_k.constraints(z, steps, nx, nu, x0, x_goal, dt))
            self._jac = jax.jit(jax.jacobian(
                lambda z: generate_k.constraints(z, steps, nx, nu, x0, x_goal, dt)
            ))

        def objective(self, z):
            return float(self._obj(z))

        def gradient(self, z):
            return np.asarray(self._grad(z), dtype=np.float64).ravel()

        def constraints(self, z):
            return np.asarray(self._cons(z), dtype=np.float64)

        def jacobian(self, z):
            return np.asarray(self._jac(z), dtype=np.float64).ravel()

    # Initial guess
    key = jax.random.PRNGKey(0)
    x_init = jnp.zeros((steps, nx)).at[:, 0].set(jnp.linspace(x0[0], x_goal[0], steps))
    u_init = jax.random.uniform(key, (steps, nu), minval=t_limits[0], maxval=t_limits[1])
    z0 = jnp.concatenate([x_init.flatten(), u_init.flatten()])

    # Bounds
    x_lb, x_ub = jnp.full((steps, nx), -jnp.inf), jnp.full((steps, nx), jnp.inf)
    u_lb, u_ub = jnp.full((steps, nu), t_limits[0]), jnp.full((steps, nu), t_limits[1])
    lb = jnp.concatenate([x_lb.flatten(), u_lb.flatten()])
    ub = jnp.concatenate([x_ub.flatten(), u_ub.flatten()])

    num_constraints = nx + (steps - 1) * nx + nx
    cl = cu = jnp.zeros(num_constraints)

    problem = cyipopt.Problem(
        n=len(z0), m=num_constraints,
        problem_obj=LocalProblem(steps, nx, nu, x0, x_goal, dt),
        lb=lb, ub=ub, cl=cl, cu=cu,
    )
    problem.add_option('max_iter', 200)
    problem.add_option('tol', 1e-3)
    problem.add_option('print_level', 0)

    z_opt, info = problem.solve(z0)
    
    x_traj = z_opt[:steps * nx].reshape(steps, nx)
    u_traj = z_opt[steps * nx:].reshape(steps, nu)
    
    return x_traj, u_traj, info['status_msg']

def generate_dataset():
    # Grid of parameters to vary
    Q_values = [
        [100.0, 100.0, 1.0, 1.0],
        [500.0, 500.0, 10.0, 10.0],
        [10.0, 10.0, 0.1, 0.1]
    ]
    R_values = [0.01, 0.1, 0.001]
    Peak_Torques = [0.02, 0.04, 0.06]

    # Setup
    T = 2.0
    dt = 0.05
    steps = int(T / dt) + 1
    nx, nu = 4, 2
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    x_goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    h5_path = results_dir / "expert_trajectories.h5"

    print(f"Generating dataset at {h5_path}...")

    with h5py.File(h5_path, 'w') as f:
        count = 0
        for qi, q_diag in enumerate(tqdm(Q_values, desc="Q variants")):
            for ri, r_val in enumerate(R_values):
                for ti, peak_t in enumerate(Peak_Torques):
                    try:
                        x_traj, u_traj, status = run_optimization(
                            q_diag, r_val, peak_t, steps, nx, nu, x0, x_goal, dt
                        )
                        
                        # Ensure status is a string before checking content
                        status_str = status.decode('utf-8') if isinstance(status, bytes) else str(status)
                        print(f"Status for Q={qi}, R={ri}, T={ti}: {status_str}")
                        
                        # Use a more robust check for success as some versions return long descriptive strings
                        if "Success" in status_str or "Solved To Acceptable Level" in status_str or "locally optimal point" in status_str.lower():
                            grp = f.create_group(f"traj_{count}")
                            grp.create_dataset("states", data=np.array(x_traj))
                            grp.create_dataset("actions", data=np.array(u_traj))
                            
                            # Store metadata
                            grp.attrs["Q_weights"] = np.array(q_diag)
                            grp.attrs["R_weight"] = float(r_val)
                            grp.attrs["peak_torque"] = float(peak_t)
                            
                            # Encode status as ascii bytes for HDF5 compatibility
                            if isinstance(status_str, str):
                                grp.attrs["status"] = status_str.encode('ascii', 'ignore')
                            else:
                                grp.attrs["status"] = status_str
                            
                            count += 1
                    except Exception as e:
                        import traceback
                        print(f"Skipping Q={qi}, R={ri}, T={ti} due to error: {e}")
                        traceback.print_exc()

    print(f"\nDone! Successfully saved {count} trajectories to {h5_path}")

if __name__ == "__main__":
    generate_dataset()
