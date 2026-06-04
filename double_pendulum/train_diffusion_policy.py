"""
Train the conditional Diffusion Policy on the double-pendulum expert
trajectories produced by generate_dataset.py.

Architecture is selectable and SELF-DESCRIBING in the checkpoint:
    python train_diffusion_policy.py --arch mlp
    python train_diffusion_policy.py --arch transformer

The full network spec (class name + hyperparameters) is saved into the
checkpoint's "config", so the evaluation script can rebuild the exact same
network with no manual edits.

Pipeline:
  1. Load expert_trajectories.h5  (groups traj_*, each with states (T,4), actions (T,2))
  2. Slice every trajectory into (cond, action_seq) windows:
        cond       = last k states           -> (k*nx,)   [+ goal if enabled]
        action_seq = next H actions          -> (H, nu)
  3. Normalize states and actions to ~[-1, 1] (stats saved for inference).
  4. Train via the DiffusionPolicy class (noise-prediction MSE).
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from diffusion_models.diffusion_policy import (
    Scheduler, MLP, TrajectoryTransformer, DiffusionPolicy,
)


# ── Config (defaults; some overridable via CLI) ──────────────────────────────
H5_PATH      = "double_pendulum/results/expert_trajectories.h5"
OUT_DIR      = "double_pendulum/results"
CKPT_PATH    = os.path.join(OUT_DIR, "diffusion_policy.pt")
STATS_PATH   = os.path.join(OUT_DIR, "norm_stats.json")

NX, NU       = 4, 2          # state dim, action dim
K            = 2             # observation-history length
H            = 8             # action prediction horizon
USE_GOAL     = False         # data has a single fixed x_goal -> a constant col
X_GOAL       = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)

TIMESTEPS    = 100
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 1e-3
SEED         = 42

# MLP-specific
MLP_HIDDEN   = 256

# Transformer-specific
TF_D_MODEL   = 128
TF_HEADS     = 4
TF_LAYERS    = 4
TF_FF        = 256
TF_DROPOUT   = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ──────────────────────────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """Slices expert trajectories into (cond, action_seq) windows + normalizes."""

    def __init__(self, h5_path, k=K, horizon=H, nx=NX, nu=NU,
                 use_goal=USE_GOAL, x_goal=X_GOAL):
        import h5py

        self.k, self.horizon = k, horizon
        self.nx, self.nu     = nx, nu
        self.use_goal        = use_goal
        self.x_goal          = np.asarray(x_goal, dtype=np.float32)

        trajs = []
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                states  = np.asarray(grp["states"],  dtype=np.float32)
                actions = np.asarray(grp["actions"], dtype=np.float32)
                trajs.append((states, actions))
        if not trajs:
            raise RuntimeError(f"No trajectories found in {h5_path}")

        all_states  = np.concatenate([s for s, _ in trajs], axis=0)
        all_actions = np.concatenate([a for _, a in trajs], axis=0)
        self.state_min,  self.state_max  = all_states.min(0),  all_states.max(0)
        self.action_min, self.action_max = all_actions.min(0), all_actions.max(0)
        self._state_range  = np.where((self.state_max - self.state_min) > 1e-8,
                                      self.state_max - self.state_min, 1.0)
        self._action_range = np.where((self.action_max - self.action_min) > 1e-8,
                                      self.action_max - self.action_min, 1.0)

        self.samples = []
        for states, actions in trajs:
            states_n  = self._norm_state(states)
            actions_n = self._norm_action(actions)
            T = len(states_n)
            for i in range(T):
                lo = i - k + 1
                if lo < 0:
                    pad  = np.repeat(states_n[:1], -lo, axis=0)
                    hist = np.concatenate([pad, states_n[:i + 1]], axis=0)
                else:
                    hist = states_n[lo:i + 1]
                cond = hist.reshape(-1)
                if use_goal:
                    cond = np.concatenate([cond, self._norm_state(self.x_goal[None])[0]])

                acts = actions_n[i:i + horizon]
                if len(acts) < horizon:
                    pad  = np.repeat(acts[-1:], horizon - len(acts), axis=0)
                    acts = np.concatenate([acts, pad], axis=0)

                self.samples.append((cond.astype(np.float32), acts.astype(np.float32)))

        self.cond_dim = self.samples[0][0].shape[0]

    def _norm_state(self, x):   return 2.0 * (x - self.state_min) / self._state_range - 1.0
    def _norm_action(self, a):  return 2.0 * (a - self.action_min) / self._action_range - 1.0
    def denorm_action(self, a): return (a + 1.0) * 0.5 * self._action_range + self.action_min

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


# ── Network builder (single source of truth, shared with eval) ───────────────
def build_network(arch, cond_dim, horizon=H, action_dim=NU):
    """
    Returns (network, net_kwargs). net_kwargs is everything needed to rebuild
    the SAME architecture later -- it gets stored in the checkpoint.
    """
    if arch == "mlp":
        kwargs = {"horizon": horizon, "action_dim": action_dim,
                  "cond_dim": cond_dim, "hidden_dim": MLP_HIDDEN}
        return MLP(**kwargs), kwargs
    elif arch == "transformer":
        kwargs = {"horizon": horizon, "action_dim": action_dim,
                  "cond_dim": cond_dim, "d_model": TF_D_MODEL,
                  "n_heads": TF_HEADS, "n_layers": TF_LAYERS,
                  "dim_feedforward": TF_FF, "dropout": TF_DROPOUT}
        return TrajectoryTransformer(**kwargs), kwargs
    raise ValueError(f"unknown arch: {arch}")


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ckpt", default=CKPT_PATH,
                    help="checkpoint output path (use distinct names per arch)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    dataset = DiffusionPolicyDataset(H5_PATH)
    dataset.save_stats(STATS_PATH)
    print(f"Dataset: {len(dataset)} windows | cond_dim={dataset.cond_dim} "
          f"| action_seq=({H},{NU})")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True)

    scheduler = Scheduler(num_steps=TIMESTEPS, device=DEVICE)
    network, net_kwargs = build_network(args.arch, dataset.cond_dim)
    policy = DiffusionPolicy(
        scheduler=scheduler, network=network, device=DEVICE,
        timesteps=TIMESTEPS, horizon=H, action_dim=NU, learning_rate=LR,
    )

    print(f"Training arch='{args.arch}' on {DEVICE} for {args.epochs} epochs "
          f"({sum(p.numel() for p in network.parameters()):,} params)...")
    policy.train(loader, epochs=args.epochs)

    # SELF-DESCRIBING checkpoint: store arch + exact net kwargs so eval can
    # rebuild the matching class with zero manual edits.
    torch.save({
        "model_state": policy.model.state_dict(),
        "config": {
            "arch": args.arch,
            "net_kwargs": net_kwargs,
            "timesteps": TIMESTEPS, "horizon": H, "action_dim": NU,
            "cond_dim": dataset.cond_dim,
            "k": K, "nx": NX, "use_goal": USE_GOAL,
        },
    }, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")


if __name__ == "__main__":
    main()