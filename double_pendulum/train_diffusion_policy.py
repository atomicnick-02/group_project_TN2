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
        cond       = last k state features   -> (k*NX_FEAT,)  [+ NX_FEAT if USE_GOAL]
        action_seq = next H actions          -> (H, nu)
  3. States are converted to angular features [sin(q1),cos(q1),sin(q2),cos(q2),dq1_n,dq2_n]
     to avoid angle-wrap discontinuities. Velocities are min-max normalized to [-1,1].
     Actions are min-max normalized to [-1,1]. Stats saved for inference.
  4. Train via the DiffusionPolicy class (noise-prediction MSE).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Make the repo root importable so `diffusion_models` resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffusion_models.diffusion_policy import (
    Scheduler, MLP, TrajectoryTransformer, DiffusionPolicy,
)


# ── Config (defaults; some overridable via CLI) ──────────────────────────────
H5_PATH      = "double_pendulum/results/expert_trajectories.h5"
OUT_DIR      = "double_pendulum/results"
CKPT_PATH    = os.path.join(OUT_DIR, "diffusion_policy.pt")
STATS_PATH   = os.path.join(OUT_DIR, "norm_stats.json")

NX, NU       = 4, 2          # raw state dim (from HDF5), action dim
NX_FEAT      = 6             # feature dim: [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
K            = 6             # observation-history length
H            = 8             # action prediction horizon
USE_GOAL     = True          # append goal features to conditioning vector
X_GOAL       = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)  # upright

TIMESTEPS    = 100
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 1e-4
SEED         = 42
# Real upright-hold rollouts now come from generate_tvlqr_dataset.py (--n-hold),
# which capture the deviation->corrective-torque map. The old synthetic hack just
# repeated each trajectory's final state/action, teaching the exact fixed point
# but NOT how to recover -- so it's disabled (0) in favor of the real data.
HOLD_STEPS   = 0

# MLP-specific
MLP_HIDDEN   = 256

# Transformer-specific
TF_D_MODEL   = 128
TF_HEADS     = 3
TF_LAYERS    = 4
TF_FF        = 256
TF_DROPOUT   = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ──────────────────────────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """Slices expert trajectories into (cond, action_seq) windows + normalizes."""

    def __init__(self, h5_path, k=K, horizon=H, nx=NX, nx_feat=NX_FEAT, nu=NU,
                 use_goal=USE_GOAL, x_goal=X_GOAL, hold_steps=HOLD_STEPS,
                 keys=None, stats=None):
        import h5py

        self.k, self.horizon = k, horizon
        self.nx, self.nx_feat, self.nu = nx, nx_feat, nu
        self.use_goal        = use_goal
        self.hold_steps      = hold_steps
        self.x_goal          = np.asarray(x_goal, dtype=np.float32)

        trajs = []
        with h5py.File(h5_path, "r") as f:
            # `keys` restricts loading to a trajectory subset (e.g. a CV fold);
            # None loads everything. Sorted -> deterministic ordering.
            load_keys = sorted(f.keys()) if keys is None else list(keys)
            for key in load_keys:
                grp = f[key]
                states  = np.asarray(grp["states"],  dtype=np.float32)
                actions = np.asarray(grp["actions"], dtype=np.float32)
                trajs.append((states, actions))
        if not trajs:
            raise RuntimeError(f"No trajectories found in {h5_path}")

        if stats is None:
            # Compute normalization stats from THIS subset (the train fold).
            all_vels    = np.concatenate([s[:, 2:] for s, _ in trajs], axis=0)
            all_actions = np.concatenate([a for _, a in trajs], axis=0)
            self.vel_min,    self.vel_max    = all_vels.min(0),    all_vels.max(0)
            self.action_min, self.action_max = all_actions.min(0), all_actions.max(0)
        else:
            # Reuse externally supplied stats (the val fold must normalize with
            # the TRAIN fold's stats -- computing its own would leak val data).
            self.vel_min    = np.asarray(stats["vel_min"],    dtype=np.float32)
            self.vel_max    = np.asarray(stats["vel_max"],    dtype=np.float32)
            self.action_min = np.asarray(stats["action_min"], dtype=np.float32)
            self.action_max = np.asarray(stats["action_max"], dtype=np.float32)
        self._vel_range    = np.where((self.vel_max - self.vel_min) > 1e-8,
                                      self.vel_max - self.vel_min, 1.0)
        self._action_range = np.where((self.action_max - self.action_min) > 1e-8,
                                      self.action_max - self.action_min, 1.0)

        goal_feat = self._state_to_features(self.x_goal)      # (NX_FEAT,), computed once

        self.samples = []
        for states, actions in trajs:
            states_f  = self._state_to_features(states)       # (T, NX_FEAT)
            actions_n = self._norm_action(actions)
            T = len(states_f)
            for i in range(T):
                lo = i - k + 1
                if lo < 0:
                    pad  = np.repeat(states_f[:1], -lo, axis=0)
                    hist = np.concatenate([pad, states_f[:i + 1]], axis=0)
                else:
                    hist = states_f[lo:i + 1]
                cond = hist.reshape(-1)
                if use_goal:
                    cond = np.concatenate([cond, goal_feat])

                acts = actions_n[i:i + horizon]
                if len(acts) < horizon:
                    pad  = np.repeat(acts[-1:], horizon - len(acts), axis=0)
                    acts = np.concatenate([acts, pad], axis=0)

                self.samples.append((cond.astype(np.float32), acts.astype(np.float32)))

            # Append synthetic holding samples using the trajectory's final
            # state and final action (the IPOPT-computed stabilizing torque).
            # This teaches the model to maintain the upright position.
            if hold_steps > 0:
                final_feat = states_f[-1]                        # (NX_FEAT,)
                final_act  = actions_n[-1]                       # (nu,)
                hold_hist  = np.repeat(final_feat[None], k, axis=0)  # (k, NX_FEAT)
                hold_cond  = hold_hist.reshape(-1)
                if use_goal:
                    hold_cond = np.concatenate([hold_cond, goal_feat])
                hold_acts = np.repeat(final_act[None], horizon, axis=0)  # (H, nu)
                for _ in range(hold_steps):
                    self.samples.append((
                        hold_cond.astype(np.float32),
                        hold_acts.astype(np.float32),
                    ))

        self.cond_dim = self.samples[0][0].shape[0]

    def _state_to_features(self, x):
        """(T, 4) or (4,) -> (T, NX_FEAT) or (NX_FEAT,): sin/cos angles + normed velocities."""
        single = x.ndim == 1
        if single:
            x = x[None]
        q1, q2  = x[:, 0], x[:, 1]
        v_norm  = 2.0 * (x[:, 2:] - self.vel_min) / self._vel_range - 1.0
        feat    = np.column_stack([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), v_norm])
        return feat[0] if single else feat

    def _norm_action(self, a):  return 2.0 * (a - self.action_min) / self._action_range - 1.0
    def denorm_action(self, a): return (a + 1.0) * 0.5 * self._action_range + self.action_min

    def get_stats(self):
        """Normalization stats to share with a held-out (val) dataset."""
        return {
            "vel_min":    self.vel_min,    "vel_max":    self.vel_max,
            "action_min": self.action_min, "action_max": self.action_max,
        }

    def save_stats(self, path):
        with open(path, "w") as fp:
            json.dump({
                "vel_min":  self.vel_min.tolist(),
                "vel_max":  self.vel_max.tolist(),
                "action_min": self.action_min.tolist(),
                "action_max": self.action_max.tolist(),
                "k": self.k, "horizon": self.horizon,
                "nx": self.nx, "nx_feat": NX_FEAT, "nu": self.nu,
                "use_goal": self.use_goal,
                "use_angular_features": True,
                "cond_dim": self.cond_dim,
            }, fp, indent=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond, acts = self.samples[idx]
        return torch.from_numpy(cond), torch.from_numpy(acts)


# ── K-fold split (at the trajectory level, before any window slicing) ────────
def kfold_train_val_keys(h5_path, n_folds, fold, seed):
    """Return (train_keys, val_keys) for one fold of a trajectory-level split.

    Splitting by trajectory (not by sliced window) keeps every window from a
    given trajectory on the same side of the train/val boundary -> no leakage.
    Deterministic given (n_folds, fold, seed).
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        keys = np.asarray(sorted(f.keys()))
    if n_folds < 2:
        return keys.tolist(), []                      # no split: train on all
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0,{n_folds}), got {fold}")
    if len(keys) < n_folds:
        raise ValueError(f"{len(keys)} trajectories < n_folds={n_folds}")

    perm = np.random.default_rng(seed).permutation(len(keys))
    val_idx = np.array_split(perm, n_folds)[fold]     # near-equal folds
    val_keys = sorted(keys[val_idx].tolist())
    train_keys = sorted(set(keys.tolist()) - set(val_keys))
    return train_keys, val_keys


@torch.no_grad()
def validation_loss(policy, loader, scheduler, timesteps, seed=0):
    """Mean noise-prediction MSE over `loader`, mirroring the train objective.

    Uses the EMA model (the one saved for inference) in eval mode. A fixed
    generator makes the diffusion timesteps/noise reproducible across folds.
    """
    if loader is None or len(loader.dataset) == 0:
        return float("nan")
    model = policy.ema_model
    model.eval()
    # Seed global RNG so both t (randint) and the noise (randn_like inside
    # add_noise) are reproducible across folds. Safe: val runs after training.
    torch.manual_seed(seed)
    total, n = 0.0, 0
    for cond, action_seq in loader:
        cond       = cond.to(policy.device)
        action_seq = action_seq.to(policy.device)
        t = torch.randint(1, timesteps, (action_seq.size(0),), device=policy.device)
        z_t, epsilon = scheduler.add_noise(action_seq, t)
        eps_pred = model(z_t, t.float() / timesteps, cond)
        total += torch.nn.functional.mse_loss(eps_pred, epsilon).item() * action_seq.size(0)
        n += action_seq.size(0)
    return total / max(n, 1)


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


def build_checkpoint(policy, arch, net_kwargs, cond_dim):
    """Self-describing checkpoint dict shared by periodic + final saves.

    Stores arch + exact net kwargs so eval rebuilds the matching class with no
    manual edits. model_state = EMA weights (used for inference).
    """
    return {
        "model_state": policy.ema_model.state_dict(),
        "model_state_raw": policy.model.state_dict(),
        "config": {
            "arch": arch,
            "net_kwargs": net_kwargs,
            "timesteps": TIMESTEPS, "horizon": H, "action_dim": NU,
            "cond_dim": cond_dim,
            "k": K, "nx": NX, "nx_feat": NX_FEAT, "use_goal": USE_GOAL,
            "use_angular_features": True,
        },
    }


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="transformer",)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ckpt", default=CKPT_PATH,
                    help="checkpoint output path (use distinct names per arch)")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="K-fold count for the trajectory-level split (1 = no split, train on all)")
    ap.add_argument("--fold", type=int, default=0,
                    help="which fold (0-based) is held out for validation")
    ap.add_argument("--save-every", type=int, default=20,
                    help="save an intermediate checkpoint every N epochs (0 = off)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Split BEFORE loading: pick the train/val trajectory keys for this fold ──
    train_keys, val_keys = kfold_train_val_keys(H5_PATH, args.n_folds, args.fold, SEED)
    if val_keys:
        print(f"K-fold: n_folds={args.n_folds} fold={args.fold} -> "
              f"{len(train_keys)} train / {len(val_keys)} val trajectories")
    else:
        print("K-fold: disabled (n_folds<2) -> training on all trajectories")

    dataset = DiffusionPolicyDataset(H5_PATH, keys=train_keys)
    dataset.save_stats(STATS_PATH)
    print(f"Train dataset: {len(dataset)} windows | cond_dim={dataset.cond_dim} "
          f"| action_seq=({H},{NU})")

    # Val fold reuses the train fold's normalization stats (no leakage).
    val_loader = None
    if val_keys:
        val_ds = DiffusionPolicyDataset(H5_PATH, keys=val_keys, stats=dataset.get_stats())
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=(DEVICE == "cuda"))
        print(f"Val dataset:   {len(val_ds)} windows")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=(DEVICE == "cuda"))

    scheduler = Scheduler(num_steps=TIMESTEPS, device=DEVICE)
    network, net_kwargs = build_network(args.arch, dataset.cond_dim)
    policy = DiffusionPolicy(
        scheduler=scheduler, network=network, device=DEVICE,
        timesteps=TIMESTEPS, horizon=H, action_dim=NU, learning_rate=LR,
    )

    print(f"Training arch='{args.arch}' on {DEVICE} for {args.epochs} epochs "
          f"({sum(p.numel() for p in network.parameters()):,} params)...")

    # Periodic checkpointing: every --save-every epochs, write a numbered ckpt
    # alongside the final one (e.g. diffusion_policy_epoch020.pt).
    ckpt = Path(args.ckpt)
    def on_epoch_end(epoch, avg_loss):
        if args.save_every > 0 and epoch % args.save_every == 0 and epoch < args.epochs:
            path = ckpt.with_name(f"{ckpt.stem}_epoch{epoch:03d}{ckpt.suffix}")
            torch.save(build_checkpoint(policy, args.arch, net_kwargs, dataset.cond_dim), path)
            print(f"  [checkpoint] saved {path}")

    policy.train(loader, epochs=args.epochs, on_epoch_end=on_epoch_end)

    if val_loader is not None:
        vloss = validation_loss(policy, val_loader, scheduler, TIMESTEPS)
        print(f"Held-out fold {args.fold} val loss (EMA, noise-pred MSE): {vloss:.5f}")

    # Final SELF-DESCRIBING checkpoint (EMA weights -> smoother inference
    # controller; eval loads model_state unchanged).
    torch.save(build_checkpoint(policy, args.arch, net_kwargs, dataset.cond_dim), args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")


if __name__ == "__main__":
    main()