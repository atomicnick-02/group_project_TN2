"""
Train the conditional Diffusion Policy on the double-pendulum expert
trajectories produced by generate_dataset.py.

Pipeline:
  1. Load expert_trajectories.h5  (groups traj_*, each with states (T,4), actions (T,2))
  2. Slice every trajectory into (cond, action_seq) windows:
        cond       = last k states           -> (k*nx,)   [+ goal if enabled]
        action_seq = next H actions          -> (H, nu)
  3. Normalize states and actions to ~[-1, 1] (stats saved for inference).
  4. Train via the DiffusionPolicy class (noise-prediction MSE).

Run inside the robotics container:
    python train_diffusion_policy.py
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# The classes built earlier. Keep diffusion_policy.py on the path / same dir.
from diffusion_models.pendulum_model import Scheduler, MLP, DiffusionPolicy


# ── Config ───────────────────────────────────────────────────────────────────
H5_PATH      = "double_pendulum/results/expert_trajectories.h5"
OUT_DIR      = "double_pendulum/results"
CKPT_PATH    = os.path.join(OUT_DIR, "diffusion_policy.pt")
STATS_PATH   = os.path.join(OUT_DIR, "norm_stats.json")

NX, NU       = 4, 2          # state dim, action dim
K            = 2             # observation-history length
H            = 8             # action prediction horizon
USE_GOAL     = False         # data has a single fixed x_goal -> a constant col
                             # teaches nothing; leave off. Flip on only if you
                             # later add trajectories with varied goals.
X_GOAL       = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)

TIMESTEPS    = 100
HIDDEN_DIM   = 256
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 1e-3
SEED         = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ──────────────────────────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """
    Slices expert trajectories into (cond, action_seq) supervised windows and
    applies per-dimension normalization.

    Normalization (min-max -> [-1, 1]) is computed over the WHOLE dataset and
    stored so it can be inverted at inference. Without this, unit-variance
    diffusion noise swamps the tiny (~0.02-0.15 Nm) torque targets.
    """

    def __init__(self, h5_path, k=K, horizon=H, nx=NX, nu=NU,
                 use_goal=USE_GOAL, x_goal=X_GOAL):
        import h5py

        self.k, self.horizon = k, horizon
        self.nx, self.nu     = nx, nu
        self.use_goal        = use_goal
        self.x_goal          = np.asarray(x_goal, dtype=np.float32)

        # --- load all trajectories ---
        trajs = []
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                states  = np.asarray(grp["states"],  dtype=np.float32)   # (T, nx)
                actions = np.asarray(grp["actions"], dtype=np.float32)   # (T, nu)
                trajs.append((states, actions))
        if not trajs:
            raise RuntimeError(f"No trajectories found in {h5_path}")

        # --- compute normalization stats over all states / actions ---
        all_states  = np.concatenate([s for s, _ in trajs], axis=0)
        all_actions = np.concatenate([a for _, a in trajs], axis=0)
        self.state_min,  self.state_max  = all_states.min(0),  all_states.max(0)
        self.action_min, self.action_max = all_actions.min(0), all_actions.max(0)
        # guard against zero-range dims (avoids divide-by-zero)
        self._state_range  = np.where((self.state_max - self.state_min) > 1e-8,
                                      self.state_max - self.state_min, 1.0)
        self._action_range = np.where((self.action_max - self.action_min) > 1e-8,
                                      self.action_max - self.action_min, 1.0)

        # --- slice into (cond, action_seq) windows ---
        self.samples = []  # list of (cond_vec, action_seq)
        for states, actions in trajs:
            states_n  = self._norm_state(states)
            actions_n = self._norm_action(actions)
            T = len(states_n)
            for i in range(T):
                # last k states, pad at the FRONT by repeating the first state
                lo = i - k + 1
                if lo < 0:
                    pad = np.repeat(states_n[:1], -lo, axis=0)
                    hist = np.concatenate([pad, states_n[:i + 1]], axis=0)
                else:
                    hist = states_n[lo:i + 1]
                cond = hist.reshape(-1)  # (k*nx,)
                if use_goal:
                    cond = np.concatenate([cond, self._norm_state(self.x_goal[None])[0]])

                # next H actions, pad at the END by repeating the last action
                acts = actions_n[i:i + horizon]
                if len(acts) < horizon:
                    pad = np.repeat(acts[-1:], horizon - len(acts), axis=0)
                    acts = np.concatenate([acts, pad], axis=0)

                self.samples.append((cond.astype(np.float32), acts.astype(np.float32)))

        self.cond_dim = self.samples[0][0].shape[0]

    # --- normalization helpers ---
    def _norm_state(self, x):
        return 2.0 * (x - self.state_min) / self._state_range - 1.0

    def _norm_action(self, a):
        return 2.0 * (a - self.action_min) / self._action_range - 1.0

    def denorm_action(self, a_norm):
        return (a_norm + 1.0) * 0.5 * self._action_range + self.action_min

    def save_stats(self, path):
        with open(path, "w") as fp:
            json.dump({
                "state_min":  self.state_min.tolist(),
                "state_max":  self.state_max.tolist(),
                "action_min": self.action_min.tolist(),
                "action_max": self.action_max.tolist(),
                "k": self.k, "horizon": self.horizon,
                "nx": self.nx, "nu": self.nu,
                "use_goal": self.use_goal,
                "cond_dim": self.cond_dim,
            }, fp, indent=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond, acts = self.samples[idx]
        return torch.from_numpy(cond), torch.from_numpy(acts)


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    dataset = DiffusionPolicyDataset(H5_PATH)
    dataset.save_stats(STATS_PATH)
    print(f"Dataset: {len(dataset)} windows | cond_dim={dataset.cond_dim} "
          f"| action_seq=({H},{NU})")
    print(f"Normalization stats saved to {STATS_PATH}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True)

    scheduler = Scheduler(num_steps=TIMESTEPS, device=DEVICE)
    network   = MLP(horizon=H, action_dim=NU, cond_dim=dataset.cond_dim,
                    hidden_dim=HIDDEN_DIM)
    policy    = DiffusionPolicy(
        scheduler=scheduler, network=network, device=DEVICE,
        timesteps=TIMESTEPS, horizon=H, action_dim=NU, learning_rate=LR,
    )

    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    policy.train(loader, epochs=EPOCHS)

    torch.save({
        "model_state": policy.model.state_dict(),
        "config": {
            "timesteps": TIMESTEPS, "horizon": H, "action_dim": NU,
            "cond_dim": dataset.cond_dim, "hidden_dim": HIDDEN_DIM,
            "k": K, "nx": NX, "use_goal": USE_GOAL,
        },
    }, CKPT_PATH)
    print(f"Saved checkpoint to {CKPT_PATH}")

    # quick sanity sample from the first batch's conditioning
    cond0, _ = next(iter(loader))
    a_norm = policy.sample(cond0[:4].to(DEVICE)).cpu().numpy()  # (4, H, nu)
    a_real = dataset.denorm_action(a_norm)
    print("Sample action seq (denormalized), first item:\n", a_real[0])


if __name__ == "__main__":
    main()