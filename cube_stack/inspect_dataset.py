import h5py
import numpy as np

# Path pointed directly to your new standard stack dataset
dataset_path = "./datasets/core/stack_three_d0.hdf5"

with h5py.File(dataset_path, "r") as f:
    print("==================================================")
    print(f"Inspecting Dataset: {dataset_path}")
    print("==================================================")
    
    # 1. Check top-level structure
    data_group = f["data"]
    demos = list(data_group.keys())
    print(f"Total number of demonstration trajectories: {len(demos)}")
    
    # Peek into the first demonstration (demo_0)
    first_demo_key = demos[0]
    first_demo = data_group[first_demo_key]
    
    # 2. Extract specific robotics variables
    actions = first_demo["actions"][:]
    obs_group = first_demo["obs"]
    
    
    print(f"\nTotal time steps (T) in this trajectory: {actions.shape[0]}")
    print(f"Action vector shape (T x action_dim): {actions.shape}")
    
    print("\nAvailable Observations (States):")
    for obs_key in obs_group.keys():
        print(f"  - {obs_key}: shape {obs_group[obs_key][:].shape}")
        
    # 3. Print out sample values at step 0
    print("\n--- Sample Values at Time Step t=0 ---")
    if "object" in obs_group:
        print(f"Object states at step 0: {obs_group['object'][0]}")