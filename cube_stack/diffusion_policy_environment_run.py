import torch
import numpy as np
import robosuite as suite
import argparse
from robosuite import load_controller_config

import mimicgen  # registers StackThree and other MimicGen environments with robosuite
from diffusion_policy_model import TemporalUNet1D, DDPMScheduler
from state_utils import get_state_from_obs, get_state_from_obs_three
from metric_utils import TrajectoryMetricsTracker
from normalization_utils import unnormalize_actions, NormalizationResults

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
if args.dataset not in ["stack_d0", "stack_d1", "stack_d0_stack_d1", "stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1"]:
    print('ERROR: dataset name not in available datasets ["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1"]')
    exit(1)

# --- 1. Load Configurations and Weights ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIFFUSION_STEPS = 100
NUM_INFERENCE_STEPS = 50    # DDIM allows far fewer steps than DDPM (100 -> 20)
EXECUTE_STEPS = 8           # Receding horizon: only execute the most confident actions
BLEND_ALPHA = 0.7           # Temporal ensembling weight for blending overlapping horizons

# initialize trajectory metrics class
metrics_tracker = TrajectoryMetricsTracker()

# Initialize the trained denoiser architecture
DATASET = args.dataset

if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"]:
    model = TemporalUNet1D(state_dim=32, horizon=8, action_dim=7).to(DEVICE)
elif DATASET in ["stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1"]:
    model = TemporalUNet1D(state_dim=48, horizon=8, action_dim=7).to(DEVICE)
model.load_state_dict(torch.load(f"./checkpoints/temporal_unet_{DATASET}_best.pth", map_location=DEVICE))
model.eval()

# calculated from `compute_dataset_statistics.py`
normalization_results = NormalizationResults()
ACTION_MIN = np.array(normalization_results.norm_results[DATASET]["ACTION_MIN"])
ACTION_MAX = np.array(normalization_results.norm_results[DATASET]["ACTION_MAX"])

# Initialize our companion inference noise scheduler (must match training schedule)
noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS, beta_schedule="cosine").to(DEVICE)

# Pre-compute the DDIM timestep schedule: evenly spaced subset of the full 100 steps
# E.g., with 20 inference steps: [99, 94, 89, 84, ..., 9, 4]
step_ratio = NUM_DIFFUSION_STEPS // NUM_INFERENCE_STEPS
ddim_timesteps = list(range(NUM_DIFFUSION_STEPS - 1, -1, -step_ratio))

# Setup standard Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

print("Launching native robosuite Stack environment for Diffusion Policy rollouts...")
print(f"Using DDIM sampling with {NUM_INFERENCE_STEPS} steps (stride={step_ratio})")
print(f"Receding horizon: executing {EXECUTE_STEPS} of 8 predicted steps")
env_name = "Stack" if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"] else "StackThree"
env = suite.make(
    env_name=env_name,
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=500,
)

# Run multiple test episodes to calculate statistical success rate
NUM_EPISODES = 50
total_successes = 0
all_episode_summaries = []

for episode in range(1, NUM_EPISODES + 1):
    obs = env.reset()
    done = False
    success = False
    metrics_tracker.reset()
    prev_horizon = None  # Track previous horizon for temporal ensembling
    
    print(f"\n--- Starting Evaluation Episode {episode}/{NUM_EPISODES} ---")
    
    while not done:
        # 1. Pull a fresh state at the start of the horizon window
        if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"]:
            current_state = get_state_from_obs(obs)
        else:
            current_state = get_state_from_obs_three(obs)
        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 2. Run the DDIM reverse diffusion process to get an 8-step plan
        #    DDIM is deterministic (no noise injection) → smoother trajectories
        current_sample = torch.randn((1, 8, 7), device=DEVICE)
        for i, t in enumerate(ddim_timesteps):
            t_tensor = torch.tensor([t], device=DEVICE).long()
            with torch.no_grad():
                noise_pred = model(current_sample, t_tensor, state_tensor)
            # Determine the previous timestep to jump to
            prev_t = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            current_sample = noise_scheduler.ddim_step(noise_pred, t, current_sample, prev_t)
            
        # Convert the fully denoised 8-step trajectory to a numpy matrix
        clean_horizon = current_sample.squeeze(0).cpu().numpy()

        # keeps track of metrics calculations for quality
        metrics_tracker.update_step_metrics(predicted_horizon=clean_horizon)

        clean_horizon = unnormalize_actions(clean_horizon, ACTION_MIN, ACTION_MAX)

        # 3. Apply temporal ensembling: blend overlapping portions with previous plan
        #    This smooths the transition between consecutive planning windows
        if prev_horizon is not None:
            overlap_len = min(EXECUTE_STEPS, prev_horizon.shape[0] - EXECUTE_STEPS)
            if overlap_len > 0:
                for j in range(overlap_len):
                    # Gradually increase trust in the new plan across the overlap region
                    weight = BLEND_ALPHA * ((j + 1) / overlap_len)
                    clean_horizon[j] = ((1.0 - weight) * prev_horizon[j + EXECUTE_STEPS] + 
                                        weight * clean_horizon[j])

        prev_horizon = clean_horizon.copy()

        # 4. Execute only the first EXECUTE_STEPS actions (receding horizon control)
        for i in range(EXECUTE_STEPS):
            action = np.clip(clean_horizon[i], -1.0, 1.0)

            try:
                obs, reward, done, info = env.step(action)
                env.render()
                metrics_tracker.update_grasp_alignment(
                    eef_quat=obs["robot0_eef_quat"],
                    gripper_to_cube_vec=obs["gripper_to_cubeA"],
                )
                if env._check_success():
                    success = True
                    done = True
                    total_successes += 1
                    print("--> Success!")
                    break
            except Exception as e:
                pass
                    
    ep_metrics = metrics_tracker.get_episode_summary()
    all_episode_summaries.append(ep_metrics)
    align_str = f"{ep_metrics['mean_approach_alignment_deg']:.1f}° (best {ep_metrics['best_approach_alignment_deg']:.1f}°)" if "mean_approach_alignment_deg" in ep_metrics else "n/a"
    print(f"Episode {episode} Metrics -> Jitter: {ep_metrics['mean_jitter']:.6f} | Path Effort: {ep_metrics['total_path_effort']:.2f} | Joint Cost: {ep_metrics['total_joint_cost']:.2f} | Approach Alignment: {align_str}")
    print(f"Episode {episode} Finished. Success Outcome: {success}")

print("\n================ FINAL ROLLOUT SUMMARY ================")
print(f"Final Empirical Success Rate: {(total_successes / NUM_EPISODES) * 100:.1f}%")
print("=======================================================")

avg_jitter     = np.mean([x["mean_jitter"]       for x in all_episode_summaries])
avg_effort     = np.mean([x["total_path_effort"] for x in all_episode_summaries])
avg_joint_cost = np.mean([x["total_joint_cost"]  for x in all_episode_summaries])
align_values   = [x["mean_approach_alignment_deg"] for x in all_episode_summaries if "mean_approach_alignment_deg" in x]
print(f"\n=== OVERALL MODEL QUALITY ===")
print(f"Average Trajectory Jitter   : {avg_jitter:.6f}")
print(f"Average Path Effort Score   : {avg_effort:.2f}")
print(f"Average Joint Cost          : {avg_joint_cost:.2f}")
if align_values:
    print(f"Average Approach Alignment  : {np.mean(align_values):.1f}° (lower = gripper points toward cube)")

env.close()