# Double-Pendulum Swing-Up — Diffusion Policy vs. Behavioral Cloning

This directory contains the full pipeline for the double-pendulum swing-up task:
plan an expert swing-up controller, roll it out to build a training set, clone it
with a Diffusion Policy (and a BC baseline), then evaluate and visualize the
results.

The physical system is defined once in [`dp.xml`](dp.xml) and simulated by the
`DoublePendulumEnv` MuJoCo wrapper in [`simulation.py`](simulation.py). Every
script reads the plant from the same XML, so they cannot silently drift apart.

## Environment

Dependencies (MuJoCo, JAX, cyipopt, PyTorch, h5py, dash, …) live in the
`school-env` pyenv. Run every script with that interpreter, **from the repo
root** so that `diffusion_models` and `simulation` import correctly:

```bash
/home/nickv/.pyenv/versions/school-env/bin/python double_pendulum/<script>.py
```

(JAX is pinned to CPU on purpose; PyTorch uses the GPU if available.)

---

## Docker
Download the docker image:
```bash
sudo docker pull xaristeidou/double-pendulum:latest
```

Run the following:
```bash
xhost +local:docker
```

Run docker container:
```bash
sudo docker run -it --gpus all --rm -e DISPLAY=$DISPLAY -e MUJOCO_GL=glx -v /tmp/.X11-unix:/tmp/.X11-unix:rw xaristeidou/double-pendulum:latest
```

Then move to scripts folder:
```bash
cd double_pendulum
```

## Run order

The scripts form a pipeline. Steps 1–3 produce the artifacts (in `results/`)
that every later step consumes, so run them in order the first time. Once the
artifacts exist, the evaluation/visualization scripts (steps 4+) can be run in
any order and as often as you like.

| # | Script | Produces / Consumes |
|---|--------|---------------------|
| 1 | `generate_k.py` | → `trajectory.csv`, `inputs.csv`, `K_matrix.npy` |
| 2 | `generate_tvlqr_dataset.py` | → `expert_trajectories.h5` |
| 3a | `train_diffusion_policy.py` | h5 → `diffusion_policy.pt` (+ sidecars) |
| 3b | `../behaviour_cloning/behavioral_cloning.py` | h5 → `bc_policy_*.pt` (baseline) |
| 4 | `evaluate_swingup.py` | interactive single-policy rollout |
| 5 | `evaluate_methods.py` | BC vs. Diffusion benchmark |
| 6 | `compare_diffusion_models.py` | Diffusion-vs-Diffusion hyperparameter sweep |
| 7 | `compare_architectures_random_init.py` | architecture shoot-out, random starts |
| 8 | `visualize_diffusion_process.py` | noising/denoising figures |
| — | `visualize_h5.py` | inspect the dataset (any time after step 2) |

---

## What each script does

### 1. `generate_k.py` — plan the expert swing-up
Solves a direct-collocation trajectory-optimization problem (IPOPT via cyipopt,
JAX-differentiated dynamics) to swing the pendulum from hanging-down
`[0,0,0,0]` to upright `[π,0,0,0]`, then computes the TVLQR feedback gains
around that trajectory. Verifies the plan by replaying the torques open-loop in
MuJoCo. **Outputs** to `results/`: `trajectory.csv` (reference states),
`inputs.csv` (reference torques), `optimal_trajectory_full.csv`, and the LQR
`K_matrix.npy`, plus reference plots under `graphs/`.

### 2. `generate_tvlqr_dataset.py` — build the training set
Rolls out the working TVLQR swing-up-and-hold controller (using the step-1
artifacts) inside MuJoCo from many randomized initial conditions, with DAgger-
style action-noise perturbations for state coverage, and logs
`(observed_state, commanded_torque)` at the control rate. Every sample is
dynamically valid by construction. **Output:** `results/expert_trajectories.h5`
(groups `traj_*`, each with `states (T,4)` and `actions (T,2)`).
Key flags: `--n-episodes`, `--perturb`, `--seed`, `--out`.

### 3a. `train_diffusion_policy.py` — train the Diffusion Policy
Slices the h5 trajectories into `(state-history → future-action-chunk)` windows,
converts states to wrap-safe angular features, normalizes, and trains a
conditional Diffusion Policy (noise-prediction MSE). The architecture is
selectable and self-describing — the full network spec is saved into the
checkpoint so evaluation can rebuild it with no manual edits.
**Output:** `results/diffusion_policy.pt` plus `*_hparams.json` / `*_stats.json`
sidecars and a loss history. Key flags:
`--arch {mlp,transformer}`, `--epochs`, `--timesteps`, `--horizon`, `--k`,
`--lr`, `--ckpt`.

### 3b. `../behaviour_cloning/behavioral_cloning.py` — BC baseline
Trains a plain MLP to directly regress the expert action from the state history
(no diffusion), consuming the **same** `expert_trajectories.h5`. This is the
head-to-head baseline for the Diffusion Policy. **Output:**
`behaviour_cloning/bc_policy_*.pt` + `norm_stats.json`.
(Note: paths inside the file are currently hard-coded — adjust `H5_PATH` /
`OUT_DIR` to your machine before running.)

### 4. `evaluate_swingup.py` — interactive single-policy rollout
Launches a live MuJoCo viewer and runs one selected controller from the
hanging-down start, reporting whether it swings up and holds upright.
`--controller {diffusion,bc}`, `--ckpt`, `--hold-steps`, `--angle-tol`,
`--vel-tol`, `--max-steps`, `--no-realtime`.

### 5. `evaluate_methods.py` — BC vs. Diffusion benchmark
Runs both controllers over an identical battery of trials in the same
environment/start/success criteria and reports success rate, trajectory quality
(IAE, time-to-success), stability, control smoothness, robustness to observation
noise, and per-step inference cost. **Outputs** CSVs + comparison plots under
`results/method_comparison/`. Key flags: `--quick` (smoke test), `--n-nominal`,
`--noise-levels`, `--device`.

### 6. `compare_diffusion_models.py` — Diffusion-vs-Diffusion sweep
Compares several Diffusion checkpoints (and/or eval-time replan rates `--n-exec`)
against each other to see how hyperparameters — denoising timesteps, transformer
depth/width, action horizon, replan rate — trade off. Auto-discovers checkpoints
under `results/checkpoints/`. **Outputs** to `results/diffusion_comparison/`.
Key flags: `--ckpts`, `--n-exec`, `--sweep-key`, `--quick`.

### 7. `compare_architectures_random_init.py` — random-init shoot-out
The random-start companion to step 6. Stage A scores every final checkpoint from
fully random joint angles (uniform in `[-π, π]`) under light observation noise;
Stage B sweeps the `--n-exec` replan rate for the chosen transformer. Reports a
single success rate per architecture. **Outputs** to
`results/architecture_random_init/`. Key flags: `--n-runs`, `--noise`,
`--n-exec`, `--quick`.

### 8. `visualize_diffusion_process.py` — diffusion figures
Takes one action chunk from the middle of an expert trajectory and plots the
forward **noising** (closed-form `q(aₜ|a₀)`) and reverse **denoising** (the
trained net's DDPM loop) side by side, in the model's normalized action space
(`--denorm` to show physical torque instead). **Outputs** figures to
`graphs/diffusion_process/`. Key flags: `--traj`, `--t-index`, `--stochastic`.

### Utility: `visualize_h5.py` — inspect the dataset
A Dash web app to browse `expert_trajectories.h5`. Run any time after step 2 and
open the printed `localhost` URL. Flags: `--file`, `--port`. Supports range
expressions like `1,3,5-10` to select trajectories.

### Module: `simulation.py`
The `DoublePendulumEnv` Gymnasium/MuJoCo wrapper imported by the scripts above.
Running it directly opens a viewer and steps the plant with random actions — a
quick sanity check that the plant/rendering work.

<br>
<br>


# Stack and StackThree Behaviour clone and Diffusion model

## Docker image

First download the docker image:
```bash
sudo docker pull xaristeidou/cube-stack:latest
```

Run the following:
```bash
xhost +local:docker
```

Run the container:
```bash
sudo docker run -it --gpus all --rm -e DISPLAY=$DISPLAY -e MUJOCO_GL=glx -v /tmp/.X11-unix:/tmp/.X11-unix:rw xaristeidou/cube-stack:latest
```

## Run the environments

Single Stack BC:
```bash
python3 behaviour_clone_environment_run.py stack_d0
```

Single Stack Diffusion (no goal):
```bash
python3 diffusion_policy_env_run.py stack_d0 --seed 42  --no-render --checkpoint ./checkpoints/diffusion_transformer_no_goal_stack_d0_best.pth
```

<br>

Single Stack Diffusion (with goal): 
```bash
python3 diffusion_policy_env_run.py stack_d0 --seed 42  --no-render --goal --checkpoint ./checkpoints/diffusion_transformer_with_goal_stack_d0_best.pth
```

StackThree BC:
```bash
python3 behaviour_clone_environment_run.py stack_three_d0
```

<br>

StackThree Diffusion (no goal):
```bash
python3 diffusion_policy_env_run.py stack_three_d0 --seed 42  --no-render --checkpoint ./checkpoints/diffusion_transformer_no_goal_stack_three_d0_best_V4.pth
```
<br>


StackThree Diffusion (with goal):
```bash
python3 diffusion_policy_env_run.py stack_three_d0 --seed 42 --goal  --no-render --checkpoint ./checkpoints/diffusion_transformer_with_goal_stack_three_d0_best_V2.pth
```
<br>
