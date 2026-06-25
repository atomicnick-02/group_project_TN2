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
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_models.diffusion_policy import (
    Scheduler, MLP, TrajectoryTransformer, DiffusionPolicy,
)


# ── Config (defaults; some overridable via CLI) ──────────────────────────────
# H5_PATH      = "double_pendulum/results/expert_trajectories.h5"
H5_PATH      = "double_pendulum/optimal_trajectories/expert_trajectories_kux_swingup_hold_mirrored.h5"

OUT_DIR      = "double_pendulum/results"
CKPT_PATH    = os.path.join(OUT_DIR, "diffusion_policy.pt")
STATS_PATH   = os.path.join(OUT_DIR, "norm_stats.json")
CKPT_DIR     = os.path.join(OUT_DIR, "checkpoints")  # periodic per-epoch snapshots
CKPT_EVERY   = 20                                    # save a checkpoint every N epochs

NX, NU       = 4, 2          # raw state dim (from HDF5), action dim
NX_FEAT      = 6             # feature dim: [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
K            = 6             # observation-history length
H            = 10            # action prediction horizon
USE_GOAL     = True          # append goal features to conditioning vector
X_GOAL       = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)  # upright

TIMESTEPS    = 100
EPOCHS       = 200
BATCH_SIZE   = 258
LR           = 1e-4
SEED         = 42

# Train/val/test split. The expert trajectories are divided 70/15/15 at the
# TRAJECTORY level (a whole traj_* group goes entirely into one split, so windows
# from the same demo never leak across sets). The split is STRATIFIED BY CONFIG:
# each expert (Q/R/xgoal) is split 70/15/15 independently, so all three sets see
# every expert in proportion. Deterministic given SPLIT_SEED.
SPLIT_FRACS  = (0.70, 0.15, 0.15)   # (train, val, test)
SPLIT_SEED   = 42

# Real upright-hold rollouts now come from generate_tvlqr_dataset.py (--n-hold),
# which capture the deviation->corrective-torque map. The old synthetic hack just
# repeated each trajectory's final state/action, teaching the exact fixed point
# but NOT how to recover -- so it's disabled (0) in favor of the real data.
HOLD_STEPS   = 0

# MLP-specific
MLP_HIDDEN   = 512

# Transformer-specific
TF_D_MODEL   = 192
TF_HEADS     = 4
TF_LAYERS    = 5
TF_FF        = 256
TF_DROPOUT   = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Loading & splitting ──────────────────────────────────────────────────────
def load_trajectories(h5_path, x_goal=X_GOAL):
    """Read every traj_* group as a (states, actions, goal, config) tuple.

    Per-trajectory goal: prefer the optimum THIS demo was generated for (stored in
    the HDF5 `xgoal` attr) so the goal feature varies across the dataset; fall back
    to the global x_goal if absent. `config` (the Q/R/xgoal expert id) is kept so
    the split can stratify by it.
    """
    import h5py

    x_goal = np.asarray(x_goal, dtype=np.float32)
    trajs = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            grp = f[key]
            states  = np.asarray(grp["states"],  dtype=np.float32)
            actions = np.asarray(grp["actions"], dtype=np.float32)
            goal    = (np.asarray(grp.attrs["xgoal"], dtype=np.float32)
                       if "xgoal" in grp.attrs else x_goal)
            config  = str(grp.attrs.get("config", "default"))
            trajs.append((states, actions, goal, config))
    if not trajs:
        raise RuntimeError(f"No trajectories found in {h5_path}")
    return trajs


def split_trajectories(trajs, fracs=SPLIT_FRACS, seed=SPLIT_SEED):
    """Stratified-by-config train/val/test split at the trajectory level.

    Each config's trajectories are shuffled (seeded) and sliced by `fracs`, so a
    whole demo lands in exactly one split (no window leakage) and every split sees
    each expert in proportion. Returns three lists of (states, actions, goal).
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    by_config = defaultdict(list)
    for t in trajs:
        by_config[t[3]].append(t)

    train, val, test = [], [], []
    for config in sorted(by_config):                 # sorted -> deterministic
        group = by_config[config]
        n     = len(group)
        n_tr  = int(round(fracs[0] * n))
        n_va  = min(int(round(fracs[1] * n)), n - n_tr)
        order = rng.permutation(n)
        for split, sl in ((train, order[:n_tr]),
                          (val,   order[n_tr:n_tr + n_va]),
                          (test,  order[n_tr + n_va:])):
            split.extend(group[i][:3] for i in sl)   # drop config; keep (s, a, goal)
    return train, val, test


# ── Dataset ──────────────────────────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """Slices expert trajectories into (cond, action_seq) windows + normalizes.

    `trajs` is a list of (states, actions, goal) tuples (see load_trajectories /
    split_trajectories). `stats` lets val/test reuse the TRAIN split's
    normalization ranges instead of refitting on their own data -- the correct
    practice, since the model only ever sees train-derived scaling at inference.
    Pass stats=None (default) to fit the ranges from `trajs` (the train split).
    """

    def __init__(self, trajs, k=K, horizon=H, nx=NX, nx_feat=NX_FEAT, nu=NU,
                 use_goal=USE_GOAL, x_goal=X_GOAL, hold_steps=HOLD_STEPS, stats=None):
        self.k, self.horizon = k, horizon
        self.nx, self.nx_feat, self.nu = nx, nx_feat, nu
        self.use_goal        = use_goal
        self.hold_steps      = hold_steps
        self.x_goal          = np.asarray(x_goal, dtype=np.float32)

        if not trajs:
            raise RuntimeError("DiffusionPolicyDataset got an empty trajectory list")

        if stats is None:
            all_vels    = np.concatenate([s[:, 2:] for s, _, _ in trajs], axis=0)
            all_actions = np.concatenate([a for _, a, _ in trajs], axis=0)
            self.vel_min,    self.vel_max    = all_vels.min(0),    all_vels.max(0)
            self.action_min, self.action_max = all_actions.min(0), all_actions.max(0)
        else:
            self.vel_min    = np.asarray(stats["vel_min"],    dtype=np.float32)
            self.vel_max    = np.asarray(stats["vel_max"],    dtype=np.float32)
            self.action_min = np.asarray(stats["action_min"], dtype=np.float32)
            self.action_max = np.asarray(stats["action_max"], dtype=np.float32)

        # to avoid division by zero i precomputed the ranges
        self._vel_range    = np.where((self.vel_max - self.vel_min) > 1e-8,
                                      self.vel_max - self.vel_min, 1.0)
        self._action_range = np.where((self.action_max - self.action_min) > 1e-8,
                                      self.action_max - self.action_min, 1.0)

        self.samples = []
        for states, actions, goal in trajs:
            goal_feat = self._state_to_features(goal)         # (NX_FEAT,), per-trajectory
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

    @property
    def stats(self):
        """Normalization ranges, to be reused by the val/test datasets."""
        return {"vel_min": self.vel_min, "vel_max": self.vel_max,
                "action_min": self.action_min, "action_max": self.action_max}

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


# ── Checkpoint serialization (single source of truth) ────────────────────────
def save_checkpoint(path, policy, arch, net_kwargs, cond_dim):
    """
    Write a SELF-DESCRIBING checkpoint: store arch + exact net kwargs so eval can
    rebuild the matching class with zero manual edits. Saves the EMA weights as
    model_state -> inference uses the smoother, more consistent controller.

    Used for both the periodic per-epoch snapshots and the final checkpoint, so
    every saved file has an identical, eval-loadable format.
    """
    torch.save({
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
    }, path)


# ── Held-out evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def eval_loss(policy, loader, seed=0):
    """Average noise-prediction MSE over a loader (held-out val/test).

    Uses the EMA weights (what the checkpoint saves) and a FIXED RNG so the same
    (timestep, noise) draws are used every call -- otherwise the diffusion loss is
    too noisy to compare epoch-to-epoch. The loader must have shuffle=False for
    the fixed-seed comparison to line up across epochs.
    """
    if loader is None:
        return float("nan")
    sched = policy.scheduler
    gen   = torch.Generator(device=policy.device).manual_seed(seed)
    policy.ema_model.eval()
    total, count = 0.0, 0
    for cond, action_seq in loader:
        cond       = cond.to(policy.device)
        action_seq = action_seq.to(policy.device)
        bsz = action_seq.size(0)
        t   = torch.randint(1, policy.timesteps, (bsz,),
                            device=policy.device, generator=gen)
        noise   = torch.randn(action_seq.shape, device=policy.device, generator=gen)
        sqrt_ab     = torch.sqrt(sched.alpha_bar[t]).view(-1, 1, 1)
        sqrt_1m_ab  = sched.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        z_t         = sqrt_ab * action_seq + sqrt_1m_ab * noise
        eps_pred    = policy.ema_model(z_t, t.float() / policy.timesteps, cond)
        total += torch.nn.functional.mse_loss(eps_pred, noise).item() * bsz
        count += bsz
    return total / max(count, 1)


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="transformer",)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ckpt", default=None,
                    help="final checkpoint output path "
                         "(default: <results>/<arch>/diffusion_policy.pt)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="folder for periodic per-epoch checkpoints "
                         "(default: <results>/<arch>/checkpoints)")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY,
                    help="save a checkpoint every N epochs (0 disables periodic saves)")
    args = ap.parse_args()

    # Keep mlp and transformer runs from overwriting each other: unless the user
    # passes explicit paths, nest checkpoints under an arch-specific subdirectory.
    arch_dir = os.path.join(OUT_DIR, args.arch)
    if args.ckpt is None:
        args.ckpt = os.path.join(arch_dir, "diffusion_policy.pt")
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(arch_dir, "checkpoints")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Stratified-by-config 70/15/15 split at the trajectory level. Whole demos go
    # into one split (no window leakage); each expert appears in all three sets.
    trajs = load_trajectories(H5_PATH)
    train_trajs, val_trajs, test_trajs = split_trajectories(trajs)

    # Normalization is fit ONLY on the train split; val/test reuse those ranges
    # (so held-out scaling matches what inference will see). Stats saved from train.
    train_ds = DiffusionPolicyDataset(train_trajs)
    val_ds   = DiffusionPolicyDataset(val_trajs,  stats=train_ds.stats)
    test_ds  = DiffusionPolicyDataset(test_trajs, stats=train_ds.stats)
    train_ds.save_stats(STATS_PATH)
    dataset = train_ds  # what gets trained on

    print(f"Split (stratified by config, seed={SPLIT_SEED}): "
          f"{len(trajs)} trajectories -> "
          f"train {len(train_trajs)} / val {len(val_trajs)} / test {len(test_trajs)}")
    print(f"Windows: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} "
          f"| cond_dim={train_ds.cond_dim} | action_seq=({H},{NU})")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=(DEVICE == "cuda"))

    scheduler = Scheduler(num_steps=TIMESTEPS, device=DEVICE)
    network, net_kwargs = build_network(args.arch, dataset.cond_dim)
    policy = DiffusionPolicy(
        scheduler=scheduler, network=network, device=DEVICE,
        timesteps=TIMESTEPS, horizon=H, action_dim=NU, learning_rate=LR,
    )

    print(f"Training arch='{args.arch}' on {DEVICE} for {args.epochs} epochs "
          f"({sum(p.numel() for p in network.parameters()):,} params)...")
    if args.ckpt_every > 0:
        print(f"Periodic checkpoints every {args.ckpt_every} epochs -> {args.ckpt_dir}/")

    # After each epoch: report held-out validation loss (EMA weights) and, every
    # N epochs, save a snapshot. The final epoch is skipped here -- it's written
    # once below as the canonical --ckpt path.
    def on_epoch_end(epoch, avg_loss):
        val = eval_loss(policy, val_loader)
        print(f"  ↳ val_loss={val:.5f}")
        if args.ckpt_every > 0 and epoch % args.ckpt_every == 0 and epoch != args.epochs:
            snap = os.path.join(args.ckpt_dir, f"{args.arch}_epoch{epoch:04d}.pt")
            save_checkpoint(snap, policy, args.arch, net_kwargs, dataset.cond_dim)
            print(f"  ↳ checkpoint saved: {snap}")

    policy.train(loader, epochs=args.epochs, on_epoch_end=on_epoch_end)

    # Final checkpoint (canonical path used by eval). Same self-describing format
    # as the periodic snapshots, written via the shared helper.
    save_checkpoint(args.ckpt, policy, args.arch, net_kwargs, dataset.cond_dim)
    print(f"Saved checkpoint to {args.ckpt}")

    # Final held-out report on the untouched test split.
    print(f"Final  val_loss={eval_loss(policy, val_loader):.5f}  "
          f"test_loss={eval_loss(policy, test_loader):.5f}")


if __name__ == "__main__":
    main()