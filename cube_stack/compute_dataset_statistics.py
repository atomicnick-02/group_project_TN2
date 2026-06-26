import h5py
import numpy as np

dataset_path = "datasets/core/stack_three_d1.hdf5"

all_actions = []
with h5py.File(dataset_path, "r") as f:
    for demo_key in f["data"].keys():
        all_actions.append(f[f"data/{demo_key}/actions"][:])

all_actions = np.concatenate(all_actions, axis=0)

# Calculate min and max per column (dimension) across the entire dataset
action_min = np.min(all_actions, axis=0)
action_max = np.max(all_actions, axis=0)

print("--- COPY THESE ARRAYS FOR YOUR SCRIPTS ---")
print(f"ACTION_MIN = {repr(action_min.tolist())}")
print(f"ACTION_MAX = {repr(action_max.tolist())}")