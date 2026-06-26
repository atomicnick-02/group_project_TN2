import torch
import numpy as np
import argparse
import robosuite as suite
from robosuite import load_controller_config
import mimicgen  # registers StackThree and other MimicGen environments with robosuite
from behaviour_clone_model import BehaviorCloningBaseline
from state_utils import get_state_from_obs, get_state_from_obs_three
from metric_utils import TrajectoryMetricsTracker

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
if args.dataset not in ["stack_d0", "stack_d1", "stack_d0_stack_d1", "stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1"]:
    print('ERROR: dataset name not in available datasets ["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1"]')
    exit(1)


# --- 1. Load Device and Model ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initial metrics tracker
metrics_tracker = TrajectoryMetricsTracker()

# Re-initialize the model with the correct input size
DATASET = args.dataset
if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"]:
    model = BehaviorCloningBaseline(input_dim=32, horizon=8, action_dim=7).to(DEVICE)
elif DATASET in ["stack_three_d0", "stack_three_d1", "stack_three_d0_stack_three_d1"]:
    model = BehaviorCloningBaseline(input_dim=48, horizon=8, action_dim=7).to(DEVICE)
model.load_state_dict(torch.load(f"./checkpoints/bc_baseline_{DATASET}_best.pth", map_location=DEVICE))
model.eval()


# Load the standard Operational Space Control (OSC) configuration
controller_config = load_controller_config(default_controller="OSC_POSE")

# --- 2. Initialize the Standard Stack Environment ---
env_name = "Stack" if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"] else "StackThree"
print(f"Launching native robosuite {env_name} environment...")
env = suite.make(
    env_name=env_name,
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,          # Set to True to open the visual window!
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,            # 20 Hz control rate matching the paper design
    horizon=500,                # Max steps allowed for a single rollout
)

success_count = 0
total_count = 0
all_episode_summaries = []

NUM_EPISODES = 50
for episode in range(1,NUM_EPISODES+1):

    # --- 3. The Receding Horizon Execution Loop ---
    obs = env.reset()
    done = False
    success = False
    metrics_tracker.reset()

    EXECUTE_STEPS = 8
    print(f"Starting evaluation rollout with a {EXECUTE_STEPS}-step receding horizon...")

    while not done:
        if DATASET in ["stack_d0", "stack_d1", "stack_d0_stack_d1"]:
            current_state = get_state_from_obs(obs)       # (32,)
        else:
            current_state = get_state_from_obs_three(obs) # (48,)

        state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            predicted_horizon = model(state_tensor).squeeze(0).cpu().numpy()

        metrics_tracker.update_step_metrics(predicted_horizon=predicted_horizon)

        for i in range(EXECUTE_STEPS):
            action = predicted_horizon[i]
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
                    success_count += 1
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
print(f"Final Empirical Success Rate: {(success_count / NUM_EPISODES) * 100:.1f}%")
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