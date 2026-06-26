import h5py
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("dataset")
args = argparser.parse_args()

# config
DATASET = args.dataset  # [stack_d0, stack_d1, stack_three_d0, stack_three_d1]
FILE_PATH = f"datasets/core/{DATASET}.hdf5"
HORIZON = 8

if DATASET not in ["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1"]:
    print('ERROR: dataset name not in available datasets ["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1"]')
    exit(1)

print(f"\nComputing expert trajectory metrics from: {FILE_PATH}")
print(f"Horizon window size: {HORIZON} steps\n")

with h5py.File(FILE_PATH, "r") as f:
    demos = sorted(f["data"].keys())
    print(f"Total demonstrations: {len(demos)}")

    ep_jitters = []
    ep_efforts = []
    ep_joint_costs = []

    for demo_key in demos:
        actions = f["data"][demo_key]["actions"][:]  # (T, 7)

        T = actions.shape[0]
        window_jitters = []
        window_efforts = []
        window_joint_costs = []

        # Non-overlapping HORIZON-step windows (mirrors receding-horizon execution)
        for start in range(0, T - HORIZON + 1, HORIZON):
            w = actions[start : start + HORIZON]  # (8, 7)
            window_jitters.append(np.mean(np.diff(w, axis=0) ** 2))
            window_efforts.append(np.sum(np.abs(w[:, :3])))
            window_joint_costs.append(np.sum(np.abs(w)))

        if not window_jitters:
            continue  # demo shorter than one full horizon window

        ep_jitters.append(np.mean(window_jitters))
        ep_efforts.append(np.sum(window_efforts))
        ep_joint_costs.append(np.sum(window_joint_costs))

avg_jitter = np.mean(ep_jitters)
avg_effort = np.mean(ep_efforts)
avg_joint_cost = np.mean(ep_joint_costs)

print(f"\n================ DATASET EXPERT METRICS ================")
print(f"Dataset : {DATASET}  ({len(ep_jitters)} demonstrations evaluated)")
print(f"========================================================")
print(f"  Average Trajectory Jitter  : {avg_jitter:.6f}")
print(f"  Average Path Effort Score  : {avg_effort:.2f}")
print(f"  Average Joint Cost         : {avg_joint_cost:.2f}")
print(f"========================================================\n")
