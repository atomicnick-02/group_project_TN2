import os
import csv
import json
import torch
import numpy as np
import h5py
import robosuite as suite
import argparse

import mimicgen  # registers the MimicGen StackThree_D0/etc. environments with robosuite
from diffusion_transformer_model import DiffusionPolicy
from state_utils import (
    get_state_from_obs, get_state_from_obs_three,
    synthesize_stack_goal, synthesize_stack_three_goal,
    STACK_GOAL_STATE_DIM, STACK_THREE_GOAL_STATE_DIM,
)
from metric_utils import TrajectoryMetricsTracker
from normalization_utils import unnormalize_actions, NormalizationResults

VALID_DATASETS = [
    "stack_d0", "stack_d1", "stack_d0_stack_d1",
    "stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1",
]

parser = argparse.ArgumentParser(description="Rollout for the (updated) transformer DiffusionPolicy.")
parser.add_argument("dataset", choices=VALID_DATASETS)
parser.add_argument("--sampler", choices=["ddpm", "ddim", "ode"], default="ddpm",
                    help="Reverse sampler for the transformer DiffusionPolicy.")
parser.add_argument("--ode-solver", choices=["euler", "heun"], default="heun",
                    help="ODE integrator used when --sampler=ode (heun = 2nd order).")
parser.add_argument("--goal", action="store_true",
                    help="Goal-conditioned model: state includes joint proprioception "
                         "+ a synthesized stacked-cube goal.")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint path. Defaults to the training script's output for this dataset.")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--no-render", action="store_true", help="Disable the on-screen viewer.")
parser.add_argument("--seed", type=int, default=None,
                    help="Fix RNG (torch/numpy/env) for a reproducible, PAIRED comparison "
                         "(both models see identical cube layouts). Leave unset for a "
                         "representative success rate over random layouts.")
args = parser.parse_args()

if args.seed is not None:
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIFFUSION_STEPS = 100
NUM_INFERENCE_STEPS = 50
EXECUTE_STEPS = 8
MAX_ENV_STEPS = 500    # per-episode cap: if not solved within this many env steps -> failure

DATASET = args.dataset
IS_STACK_TWO = DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"]

if args.goal:
    state_dim = STACK_GOAL_STATE_DIM if IS_STACK_TWO else STACK_THREE_GOAL_STATE_DIM
else:
    state_dim = 32 if IS_STACK_TWO else 48
    

# --- Build the policy and load the trained (EMA) denoiser weights ---
model = DiffusionPolicy(action_dim=7, obs_dim=state_dim, horizon=8,
                        num_train_timesteps=NUM_DIFFUSION_STEPS, device=DEVICE)

ckpt_tag = "with_goal" if args.goal else "with_goal"

ckpt_path = args.checkpoint or f"./checkpoints/diffusion_transformer_no_goal_stack_three_d0_best_V4.pth"
if not os.path.exists(ckpt_path):
    print(f"ERROR: checkpoint not found: {ckpt_path}")
    raise SystemExit(1)
# save_checkpoint() stores the denoiser's state_dict, so load into model.transformer.
model.transformer.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.transformer.eval()
print(f"Loaded transformer DiffusionPolicy from {ckpt_path} (goal={args.goal})")

normalization_results = NormalizationResults()
ACTION_MIN = np.array(normalization_results.norm_results[DATASET]["ACTION_MIN"])
ACTION_MAX = np.array(normalization_results.norm_results[DATASET]["ACTION_MAX"])

metrics_tracker = TrajectoryMetricsTracker()

# Rebuild the EXACT env the dataset was generated with (env_name, controller,
# control_freq, placement distribution) from the HDF5's stored env_args, so the
# test-time cube distribution and dynamics match training. For combined datasets
# (e.g. stack_three_d0_stack_three_d1) use the first component's metadata.
SINGLE_DATASETS = ["stack_three_d0", "stack_three_d1", "stack_d0", "stack_d1"]
first_dataset = next(s for s in SINGLE_DATASETS if DATASET.startswith(s))
with h5py.File(f"datasets/core/{first_dataset}.hdf5", "r") as _f:
    env_args = json.loads(_f["data"].attrs["env_args"])
env_kwargs = env_args["env_kwargs"]

print(f"Launching {env_args['env_name']} with {args.sampler.upper()} sampling "
      f"({NUM_INFERENCE_STEPS} steps), executing {EXECUTE_STEPS}/8 predicted steps")
renderer = True
env = suite.make(
    env_name=env_args["env_name"],                      # e.g. "StackThree_D0"
    robots=env_kwargs["robots"],
    controller_configs=env_kwargs["controller_configs"],
    control_freq=env_kwargs["control_freq"],            # 20
    has_renderer=renderer,
    has_offscreen_renderer=True,
    use_camera_obs=False,                               # eval uses object/proprio obs, not images
    use_object_obs=True,
    ignore_done=env_kwargs.get("ignore_done", True),
    horizon=500,
)


def build_state(obs):
    """Assemble the conditioning vector exactly as the matching dataset did."""
    if args.goal:
        if IS_STACK_TWO:
            base = get_state_from_obs(obs)
            goal = synthesize_stack_goal(obs)
        else:
            base = get_state_from_obs_three(obs)
            goal = synthesize_stack_three_goal(obs)
        return np.concatenate([base, goal])
    if IS_STACK_TWO:
        return get_state_from_obs(obs)
    return get_state_from_obs_three(obs)


NUM_EPISODES = args.episodes
total_successes = 0
all_episode_summaries = []

for episode in range(1, NUM_EPISODES + 1):
    # Per-episode reseed so episode i gets an identical cube layout across runs,
    # independent of how many sampler/step calls happened earlier (paired eval).
    if args.seed is not None:
        np.random.seed(args.seed + episode)
        torch.manual_seed(args.seed + episode)
    obs = env.reset()
    done = False
    success = False
    step_count = 0
    metrics_tracker.reset()

    print(f"\n--- Starting Evaluation Episode {episode}/{NUM_EPISODES} ---")

    while not done and step_count < MAX_ENV_STEPS:
        current_state = build_state(obs)

        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            current_sample = model.sample(state_tensor, sampler=args.sampler,
                                          num_steps=NUM_INFERENCE_STEPS, eta=0.0,
                                          ode_solver=args.ode_solver)

        clean_horizon = current_sample.squeeze(0).cpu().numpy()
        metrics_tracker.update_step_metrics(predicted_horizon=clean_horizon)
        clean_horizon = unnormalize_actions(clean_horizon, ACTION_MIN, ACTION_MAX)

        for i in range(EXECUTE_STEPS):
            action = np.clip(clean_horizon[i], -1.0, 1.0)
            try:
                obs, reward, done, info = env.step(action)
                step_count += 1
                if renderer:
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
                if step_count >= MAX_ENV_STEPS:
                    break
            except Exception:
                pass

    if not success:
        print(f"--> Failed (no success within {step_count} steps)")

    ep_metrics = metrics_tracker.get_episode_summary()
    all_episode_summaries.append(ep_metrics)
    align_str = (f"{ep_metrics['mean_approach_alignment_deg']:.1f}° (best {ep_metrics['best_approach_alignment_deg']:.1f}°)"
                 if "mean_approach_alignment_deg" in ep_metrics else "n/a")
    print(f"Episode {episode} Metrics -> Jitter: {ep_metrics['mean_jitter']:.6f} | "
          f"Path Effort: {ep_metrics['total_path_effort']:.2f} | "
          f"Joint Cost: {ep_metrics['total_joint_cost']:.2f} | Approach Alignment: {align_str}")
    print(f"Episode {episode} Finished. Success Outcome: {success}")

print("\n================ FINAL ROLLOUT SUMMARY ================")
print(f"Final Empirical Success Rate: {(total_successes / NUM_EPISODES) * 100:.1f}%")
print("=======================================================")

avg_jitter     = np.mean([x["mean_jitter"]       for x in all_episode_summaries])
avg_effort     = np.mean([x["total_path_effort"] for x in all_episode_summaries])
avg_joint_cost = np.mean([x["total_joint_cost"]  for x in all_episode_summaries])
median_jitter     = np.median([x["mean_jitter"]       for x in all_episode_summaries])
median_effort     = np.median([x["total_path_effort"] for x in all_episode_summaries])
median_joint_cost = np.median([x["total_joint_cost"]  for x in all_episode_summaries])
align_values   = [x["mean_approach_alignment_deg"] for x in all_episode_summaries if "mean_approach_alignment_deg" in x]
print(f"\n=== OVERALL MODEL QUALITY ===")
print(f"Average Trajectory Jitter   : {avg_jitter:.6f}")
print(f"Average Path Effort Score   : {avg_effort:.2f}")
print(f"Average Joint Cost          : {avg_joint_cost:.2f}")
print(f"Median Trajectory Jitter   : {median_jitter:.6f}")
print(f"Median Path Effort Score   : {median_effort:.2f}")
print(f"Median Joint Cost          : {median_joint_cost:.2f}")
if align_values:
    print(f"Average Approach Alignment  : {np.mean(align_values):.1f}° (lower = gripper points toward cube)")
    print(f"Median Approach Alignment  : {np.median(align_values):.1f}° (lower = gripper points toward cube)")

# --- Append the rollout summary to a CSV (one row per run) ---
csv_path = "rollout_results.csv"
row = {
    "dataset": DATASET,
    "goal": "with_goal" if args.goal else "no_goal",
    "sampler": args.sampler,
    "execute_steps": EXECUTE_STEPS,
    "episodes": NUM_EPISODES,
    "success_rate": (total_successes / NUM_EPISODES) * 100,
    "avg_jitter": avg_jitter,
    "median_jitter": median_jitter,
    "avg_path_effort": avg_effort,
    "median_path_effort": median_effort,
    "avg_joint_cost": avg_joint_cost,
    "median_joint_cost": median_joint_cost,
    "avg_approach_alignment_deg": np.mean(align_values) if align_values else "",
    "median_approach_alignment_deg": np.median(align_values) if align_values else "",
}
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"\nSaved rollout summary to {csv_path}")

env.close()
