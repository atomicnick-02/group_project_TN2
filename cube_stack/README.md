# Robot control with Diffusion policy and Behaviour clone models

## Installation
Create virtual environment (linux tested, python3.12 test - 3.8 tested and it is NOT supported):
```bash
python3.12 -m venv robot_control_venv
source robot_control_venv/bin/activate
pip install --upgrade pip
```

Install dependencies:
```bash
pip install -r requirements.txt
```
<br>

## Datasets
By default the datasets for the training are not provided and uploaded to the repository. You can download them using the `dataset_download.py` script and specifying the dataset name. The datasets should be placed in the following path as follows:
```directory
datasets/
└── core/
    ├── stack_d0.hdf5
    ├── stack_d1.hdf5
    ├── stack_three_d0.hdf5
    └── stack_three_d1.hdf5
```

<br>

## Environment run
For behaviour clone model run the following defining the dataset (available: `["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1"]`):
```bash
python3 behaviour_clone_environment_run.py stack_d0
```

For diffusion policy model run the following defining the dataset (available: `["stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1"]`):
```bash
python3 diffusion_policy_environment_run.py stack_d0
```

<br>

## Scripts cheatsheet

| Script | Description |
|--------|-------------|
| `diffusion_policy_model.py` | Defines the `TemporalUNet1D` denoiser, `FiLMConvBlock1D`, legacy `DiffusionTrajectoryDenoiser`, and `DDPMScheduler` with DDPM/DDIM sampling. |
| `diffusion_policy_train.py` | Trains the diffusion policy with EMA, cosine noise schedule, and AdamW; saves the best checkpoint based on validation loss. |
| `diffusion_policy_environment_run.py` | Runs trained diffusion policy rollouts in robosuite using DDIM inference and receding-horizon control; reports success rate and trajectory metrics. |
| `behaviour_clone_model.py` | Defines the `BehaviorCloningBaseline` MLP that maps a state vector directly to an 8-step action horizon. |
| `behaviour_clone_train.py` | Trains the behaviour cloning model with MSE loss and saves the best checkpoint based on validation loss. |
| `behaviour_clone_environment_run.py` | Runs trained behaviour clone rollouts in robosuite with receding-horizon control; reports success rate and trajectory metrics. |
| `dataset_loader.py` | Provides `ShortHorizonRoboticsDataset` (Stack, 32D state) and `ShortHorizonRoboticsDatasetThree` (StackThree, 48D state) for loading sliding-window HDF5 demonstrations. |
| `state_utils.py` | Builds the state vector from raw observation arrays (training) or robosuite obs dicts (inference) for both the 2-cube and 3-cube tasks. |
| `normalization_utils.py` | Provides `normalize_actions`/`unnormalize_actions` helpers and `NormalizationResults` with precomputed action bounds for all four datasets. |
| `metric_utils.py` | Tracks per-episode trajectory quality metrics: mean jitter, total path effort, and optional ground-truth MSE. |
| `compute_dataset_statistics.py` | Scans an HDF5 dataset and prints per-dimension action min/max values for copy-pasting into `normalization_utils.py`. |
| `dataset_download.py` | Downloads MimicGen HDF5 datasets from Hugging Face for a specified task and difficulty level. |
| `compute_dataset_metrics.py` | Computes expert-demonstration mean jitter and total path effort from an HDF5 dataset for direct comparison against environment rollout metrics. |
| `inspect_dataset.py` | Prints demonstration count, trajectory length, observation shapes, and sample values from an HDF5 dataset file.
