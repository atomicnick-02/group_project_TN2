import os
import numpy as np
import jax
import jax.numpy as jnp
from simulation import DoublePendulumEnv
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Load optimal trajectory and K matrices
current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"

x_ref = np.loadtxt(results_dir / "trajectory.csv", delimiter=",", skiprows=1).T
u_ref = np.loadtxt(results_dir / "inputs.csv", delimiter=",", skiprows=1).T
K_matrices = np.load(results_dir / "K_matrix.npy")

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def to_feature_space(x):
    # Weighting factor for velocity
    v_scale = 0.1 
    if x.ndim == 1:
        p0, p1 = x[0], x[1]
        v = x[2:] * v_scale
        return np.concatenate(([np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1)], v))
    else:
        p0, p1 = x[0, :], x[1, :]
        v = x[2:, :] * v_scale
        return np.vstack((np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1), v))

def get_ranked_distances(x, x_ref, k=1):
    feat_x = to_feature_space(x)
    feat_ref = to_feature_space(x_ref)
    diff = feat_ref - feat_x.reshape(-1, 1)
    dists = np.linalg.norm(diff, axis=0)
    sorted_indices = np.argsort(dists)
    return sorted_indices[:k], dists[sorted_indices[:k]]

def get_K_gain_weighted(x, x_ref, K, k=5):
    indices, dists = get_ranked_distances(x, x_ref, k=k)
    eps = 1e-6
    w = 1.0 / (dists + eps)
    weights = w / np.sum(w)
    
    indices_clipped = np.clip(indices, 0, len(K) - 1)
    Ks = K[indices_clipped]
    K_weighted = np.sum(weights[:, None, None] * Ks, axis=0)
    return K_weighted

def main():
    env = DoublePendulumEnv(render_mode="human", frame_skip=1)
    obs, _ = env.reset()
    
    dt_control = 0.05
    dt_sim = env.model.opt.timestep # usually 0.002 or 0.01
    sim_steps_per_control = int(dt_control / dt_sim)
    
    max_idx = x_ref.shape[1] - 1
    current_idx = 0
    DEVIATION_THRESHOLD = 2.0
    
    # Torque limits from MJCF/Env
    max_torque = env.action_space.high[0]

    print(f"Starting simulation with TVLQR control (dt_control={dt_control}, sim_steps={sim_steps_per_control})...")
    
    # State history for plotting
    sim_history = []
    ref_history = []
    u_history = []
    u_ref_history = []
    time_history = []

    try:
        # Run for the length of trajectory plus some stabilization time
        for step in range(x_ref.shape[1] + 500):
            x_current = obs # obs is [q1, q2, qd1, qd2] from env
            
            # 1. Determine mode and index
            target_state = x_ref[:, current_idx].reshape(4, 1)
            _, dists = get_ranked_distances(x_current, target_state, k=1)
            dist_to_target = dists[0]
            
            holding = (current_idx >= max_idx)
            
            if dist_to_target > DEVIATION_THRESHOLD or holding:
                # Recovery or Holding: find closest point in trajectory
                mode = "RECOVERY/HOLDING"
                best_idx, _ = get_ranked_distances(x_current, x_ref, k=1)
                idx = best_idx[0]
                K_gain = get_K_gain_weighted(x_current, x_ref, K_matrices, k=5)
            else:
                # Normal Tracking
                mode = "TRACKING"
                idx = current_idx
                K_gain = K_matrices[idx]
                current_idx += 1

            x_des = x_ref[:, idx]
            u_des = u_ref[:, idx]

            # 2. Control calculation
            error = x_current - x_des
            error[0] = wrap_to_pi(error[0])
            error[1] = wrap_to_pi(error[1])
            
            u_feedback = -K_gain @ error
            u_total = u_des + u_feedback
            u_total = np.clip(u_total, -max_torque, max_torque)

            # Save to history
            sim_history.append(x_current.copy())
            ref_history.append(x_des.copy())
            u_history.append(u_total.copy())
            u_ref_history.append(u_des.copy())
            time_history.append(step * dt_control)
            
            # 3. Step environment with sub-stepping to match dt_control
            for _ in range(sim_steps_per_control):
                obs, reward, terminated, truncated, _ = env.step(u_total)
            
            if step % 10 == 0:
                print(f"Step: {step} | Mode: {mode} | Index: {idx} | Error Norm: {np.linalg.norm(error):.3f}")

            if terminated or truncated:
                break
            # time.sleep(0.01) # Optional: added back if you want to see simulation slower
                
    except KeyboardInterrupt:
        pass
    finally:
        # Close viewer explicitly before env.close to avoid Segfault
        if hasattr(env, '_viewer') and env._viewer is not None:
            env._viewer.close()
        env.close()
        print("Simulation finished.")

        # Plot comparison
        sim_history = np.array(sim_history)
        ref_history = np.array(ref_history)
        u_history = np.array(u_history)
        u_ref_history = np.array(u_ref_history)
        time_history = np.array(time_history)

        # Plot States
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # q1
        axs[0, 0].plot(time_history, ref_history[:, 0], 'r--', label='Reference q1')
        axs[0, 0].plot(time_history, sim_history[:, 0], 'b-', label='Simulation q1')
        axs[0, 0].set_title('Shoulder Angle (q1)')
        axs[0, 0].set_ylabel('Angle (rad)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # q2
        axs[0, 1].plot(time_history, ref_history[:, 1], 'r--', label='Reference q2')
        axs[0, 1].plot(time_history, sim_history[:, 1], 'b-', label='Simulation q2')
        axs[0, 1].set_title('Elbow Angle (q2)')
        axs[0, 1].set_ylabel('Angle (rad)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # q1_dot
        axs[1, 0].plot(time_history, ref_history[:, 2], 'r--', label='Reference q1_dot')
        axs[1, 0].plot(time_history, sim_history[:, 2], 'b-', label='Simulation q1_dot')
        axs[1, 0].set_title('Shoulder Velocity (q1_dot)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Velocity (rad/s)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # q2_dot
        axs[1, 1].plot(time_history, ref_history[:, 3], 'r--', label='Reference q2_dot')
        axs[1, 1].plot(time_history, sim_history[:, 3], 'b-', label='Simulation q2_dot')
        axs[1, 1].set_title('Elbow Velocity (q2_dot)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (rad/s)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plots_dir = current_dir / "graphs" / "simulation"
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(plots_dir / "tracking_performance_states.png")
        
        # Plot Controls
        fig_u, axs_u = plt.subplots(1, 2, figsize=(15, 5))
        
        # u1
        axs_u[0].plot(time_history, u_ref_history[:, 0], 'r--', label='Reference u1')
        axs_u[0].plot(time_history, u_history[:, 0], 'b-', label='Actual u1')
        axs_u[0].set_title('Shoulder Torque (u1)')
        axs_u[0].set_xlabel('Time (s)')
        axs_u[0].set_ylabel('Torque (Nm)')
        axs_u[0].legend()
        axs_u[0].grid(True)

        # u2
        axs_u[1].plot(time_history, u_ref_history[:, 1], 'r--', label='Reference u2')
        axs_u[1].plot(time_history, u_history[:, 1], 'b-', label='Actual u2')
        axs_u[1].set_title('Elbow Torque (u2)')
        axs_u[1].set_xlabel('Time (s)')
        axs_u[1].set_ylabel('Torque (Nm)')
        axs_u[1].legend()
        axs_u[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "tracking_performance_controls.png")
        
        print(f"Tracking performance plots saved to: {plots_dir}")
        plt.show()

if __name__ == "__main__":
    main()
