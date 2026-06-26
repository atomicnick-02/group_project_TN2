"""
Visualize the forward (noising) and reverse (denoising) diffusion of a single
action sequence taken from the middle of an expert trajectory.

The Diffusion Policy never diffuses a whole trajectory -- it diffuses a short
H-step action *chunk* conditioned on the recent state history. This script picks
one such chunk from the middle of an expert rollout and shows the two halves of
the diffusion process side by side:

  * Noising  (forward q(a_t | a_0)): start from the clean expert action chunk and
    progressively corrupt it toward N(0, I). We use the closed-form marginal
    a_t = sqrt(alpha_bar_t)*a_0 + sqrt(1-alpha_bar_t)*eps with one fixed eps so
    the snapshots form a smooth, cumulative corruption (not 20 independent draws).

  * Denoising (reverse): start from pure Gaussian noise and run the trained
    network's DDPM reverse loop, conditioned on the same state history, capturing
    the action chunk after every denoising step. The final chunk is compared to
    the ground-truth expert chunk to show reconstruction quality.

Everything is plotted in the model's own normalized action space ([-1, 1]),
because that is where the diffusion math lives -- clean signal sits in [-1, 1]
and the terminal state is ~N(0, I). Pass --denorm to instead plot physical
torque (Nm), which rescales by the +/-0.10 Nm action range.

Run from the repo root (so `diffusion_models` is importable):
    python double_pendulum/visualize_diffusion_process.py
    python double_pendulum/visualize_diffusion_process.py --traj traj_42 --t-index 90
    python double_pendulum/visualize_diffusion_process.py --stochastic --denorm
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.colors import Normalize

# Make the repo root importable so `diffusion_models` resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CURRENT_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _CURRENT_DIR / "results"


# Normalization helpers (mirror train_diffusion_policy.DiffusionPolicyDataset)
class Featurizer:
    """Reproduces the dataset's state->feature and action normalization exactly,
    so the conditioning vector we build matches what the network was trained on."""

    def __init__(self, stats):
        self.vel_min = np.asarray(stats["vel_min"], dtype=np.float32)
        self.vel_max = np.asarray(stats["vel_max"], dtype=np.float32)
        self.v_rng   = np.where((self.vel_max - self.vel_min) > 1e-8,
                                self.vel_max - self.vel_min, 1.0)
        self.a_min = np.asarray(stats["action_min"], dtype=np.float32)
        self.a_max = np.asarray(stats["action_max"], dtype=np.float32)
        self.a_rng = np.where((self.a_max - self.a_min) > 1e-8,
                              self.a_max - self.a_min, 1.0)
        self.k        = int(stats["k"])
        self.horizon  = int(stats["horizon"])
        self.use_goal = bool(stats["use_goal"])

    def state_to_features(self, x):
        """(T,4) or (4,) -> angular features [sin q1, cos q1, sin q2, cos q2, dq1_n, dq2_n]."""
        single = x.ndim == 1
        x = np.atleast_2d(x)
        q1, q2 = x[:, 0], x[:, 1]
        v = 2.0 * (x[:, 2:] - self.vel_min) / self.v_rng - 1.0
        feat = np.column_stack([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2),
                                v[:, 0], v[:, 1]]).astype(np.float32)
        return feat[0] if single else feat

    def norm_action(self, a):
        return (2.0 * (a - self.a_min) / self.a_rng - 1.0).astype(np.float32)

    def denorm_action(self, a):
        return (a + 1.0) * 0.5 * self.a_rng + self.a_min

    def build_cond(self, states, i, x_goal=(np.pi, 0.0, 0.0, 0.0)):
        """Conditioning vector for the window ending at index i: k-state history
        features (left-padded with the first state if needed) + optional goal."""
        feats = self.state_to_features(states)            # (T, NX_FEAT)
        lo = i - self.k + 1
        if lo < 0:
            pad  = np.repeat(feats[:1], -lo, axis=0)
            hist = np.concatenate([pad, feats[:i + 1]], axis=0)
        else:
            hist = feats[lo:i + 1]
        cond = hist.reshape(-1)
        if self.use_goal:
            goal_feat = self.state_to_features(np.asarray(x_goal, dtype=np.float32))
            cond = np.concatenate([cond, goal_feat])
        return cond.astype(np.float32)


# Load the trained diffusion policy (network + scheduler)
def load_policy(ckpt_path, device):
    import torch
    from diffusion_models.diffusion_policy import (
        Scheduler, MLP, TrajectoryTransformer, DiffusionPolicy,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    arch = cfg.get("arch", "mlp")
    net_kwargs = cfg.get("net_kwargs", {
        "horizon": cfg["horizon"], "action_dim": cfg["action_dim"],
        "cond_dim": cfg["cond_dim"], "hidden_dim": cfg.get("hidden_dim", 256),
    })
    net_cls = {"mlp": MLP, "transformer": TrajectoryTransformer}[arch]
    network = net_cls(**net_kwargs)

    scheduler = Scheduler(num_steps=cfg["timesteps"], device=device)
    policy = DiffusionPolicy(scheduler, network, device,
                             cfg["timesteps"], cfg["horizon"], cfg["action_dim"])
    policy.model.load_state_dict(ckpt["model_state"])
    policy.model.eval()
    print(f"[viz] loaded arch='{arch}' (T={cfg['timesteps']}, H={cfg['horizon']}, "
          f"nu={cfg['action_dim']}) from {Path(ckpt_path).name}")
    return policy, scheduler, cfg


# Forward process: capture a_t for every t with one fixed noise draw
def forward_snapshots(scheduler, a0, torch, seed=0):
    """Returns (T, H, nu) array of the noised chunk at every diffusion step
    t = 0..T-1, using a single fixed epsilon so corruption is cumulative."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(a0.shape, generator=g)               # fixed across all t
    T = scheduler.num_steps
    snaps = []
    for t in range(T):
        sqrt_ab  = torch.sqrt(scheduler.alpha_bar[t]).cpu()
        sqrt_1ab = scheduler.sqrt_one_minus_alpha_bar[t].cpu()
        a_t = sqrt_ab * a0 + sqrt_1ab * eps
        snaps.append(a_t.numpy())
    return np.stack(snaps, axis=0)                          # (T, H, nu)


# Reverse process: capture a_t after every DDPM denoising step
def reverse_snapshots(policy, scheduler, cond, torch, stochastic=False, seed=0):
    """Returns (n_steps, H, nu): the action chunk starting from pure noise (step
    T) down to the final denoised chunk, conditioned on `cond`."""
    device = policy.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_t = torch.randn((1, policy.horizon, policy.action_dim), generator=g).to(device)
    cond_t = torch.as_tensor(cond, dtype=torch.float32, device=device).reshape(1, -1)

    snaps = [a_t.squeeze(0).cpu().numpy()]                  # initial pure noise (t=T)
    with torch.no_grad():
        for t in reversed(range(1, policy.timesteps)):
            t_norm  = torch.full((1,), t / policy.timesteps, device=device, dtype=torch.float32)
            beta_t  = scheduler.beta_array[t]
            alpha_t = scheduler.alpha_bar[t]
            first   = (1 / torch.sqrt(1 - beta_t)) * a_t
            second  = (beta_t / (torch.sqrt(1 - beta_t) * torch.sqrt(1 - alpha_t))) \
                      * policy.model(a_t, t_norm, cond_t)
            a_t = first - second
            if stochastic and t > 1:
                a_t = a_t + torch.sqrt(beta_t) * torch.randn(a_t.shape).to(device)
            snaps.append(a_t.squeeze(0).cpu().numpy())
    return np.stack(snaps, axis=0)                          # (T, H, nu)


# Plotting
def _maybe_denorm(arr, feat, denorm):
    return feat.denorm_action(arr) if denorm else arr


def plot_overlay(fwd, rev, a0, feat, out_path, denorm, info):
    """2 rows (noising / denoising) x nu cols. Each panel overlays the chunk at
    every diffusion step, colored by step, with the clean expert chunk dashed."""
    nu  = a0.shape[1]
    H   = a0.shape[0]
    Tf  = fwd.shape[0]
    Tr  = rev.shape[0]
    steps_x = np.arange(H)
    a0_p  = _maybe_denorm(a0,  feat, denorm)
    fwd_p = _maybe_denorm(fwd, feat, denorm)
    rev_p = _maybe_denorm(rev, feat, denorm)
    ylab  = "torque (Nm)" if denorm else "normalized action [-1, 1]"

    fig, axs = plt.subplots(2, nu, figsize=(6.0 * nu, 8.5), squeeze=False,
                            constrained_layout=True)
    cmap_f = matplotlib.colormaps["viridis"]
    cmap_r = matplotlib.colormaps["plasma"]

    for j in range(nu):
        # --- noising row ---
        ax = axs[0][j]
        for s in range(Tf):
            ax.plot(steps_x, fwd_p[s, :, j], color=cmap_f(s / max(Tf - 1, 1)),
                    lw=1.2, alpha=0.85)
        ax.plot(steps_x, a0_p[:, j], "k--", lw=2.5, label="expert (clean)")
        ax.set_title(f"Noising u{j+1} (clean -> noise)")
        ax.set_xlabel("action step in chunk"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)

        # --- denoising row ---
        ax = axs[1][j]
        for s in range(Tr):
            ax.plot(steps_x, rev_p[s, :, j], color=cmap_r(s / max(Tr - 1, 1)),
                    lw=1.2, alpha=0.85)
        ax.plot(steps_x, a0_p[:, j], "k--", lw=2.5, label="expert (target)")
        ax.plot(steps_x, rev_p[-1, :, j], "c-", lw=2.5, label="final denoised")
        ax.set_title(f"Denoising u{j+1} (noise -> clean)")
        ax.set_xlabel("action step in chunk"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    # colorbars indicating the diffusion timestep direction
    sm_f = mcm.ScalarMappable(cmap=cmap_f, norm=Normalize(0, Tf - 1))
    sm_r = mcm.ScalarMappable(cmap=cmap_r, norm=Normalize(0, Tr - 1))
    fig.colorbar(sm_f, ax=list(axs[0]), label="forward step t (0=clean)")
    fig.colorbar(sm_r, ax=list(axs[1]), label="reverse step (0=pure noise)")

    fig.suptitle(
        f"Diffusion of a mid-trajectory action chunk  |  {info['traj']}  "
        f"t-index={info['t_index']}/{info['traj_len']}  |  H={H}, T={Tf}"
        f"{'  |  stochastic' if info['stochastic'] else '  |  deterministic'}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def plot_grid(snaps, a0, feat, out_path, denorm, title, n_show=8, reverse_labels=False):
    """Snapshot grid: a few evenly-spaced diffusion steps, each panel showing the
    full (H, nu) chunk at that step with the clean expert chunk for reference."""
    T  = snaps.shape[0]
    H  = a0.shape[0]
    nu = a0.shape[1]
    idx = np.unique(np.linspace(0, T - 1, min(n_show, T)).round().astype(int))
    a0_p    = _maybe_denorm(a0,    feat, denorm)
    snaps_p = _maybe_denorm(snaps, feat, denorm)
    steps_x = np.arange(H)
    ylab = "torque (Nm)" if denorm else "norm. action"

    ncol = 4
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow),
                            squeeze=False, sharex=True, sharey=True)
    colors = ["#1f77b4", "#d62728"]
    for ax_pos, s in enumerate(idx):
        ax = axs[ax_pos // ncol][ax_pos % ncol]
        for j in range(nu):
            ax.plot(steps_x, snaps_p[s, :, j], "-", color=colors[j % len(colors)],
                    lw=1.8, label=f"u{j+1}")
            ax.plot(steps_x, a0_p[:, j], "--", color=colors[j % len(colors)],
                    lw=1.0, alpha=0.5)
        label = f"reverse step {s}/{T-1}" if reverse_labels else f"forward t={s}"
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if ax_pos % ncol == 0:
            ax.set_ylabel(ylab, fontsize=8)
    for ax_pos in range(len(idx), nrow * ncol):           # hide unused panels
        axs[ax_pos // ncol][ax_pos % ncol].axis("off")
    axs[0][0].legend(loc="upper right", fontsize=7)
    fig.suptitle(title + "   (dashed = clean expert chunk)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# Main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(_RESULTS_DIR / "diffusion_policy.pt"),
                   help="diffusion checkpoint (.pt with config+model_state)")
    p.add_argument("--stats", default=str(_RESULTS_DIR / "norm_stats.json"),
                   help="normalization stats json")
    p.add_argument("--h5", default=str(_RESULTS_DIR / "expert_trajectories.h5"),
                   help="expert trajectory dataset")
    p.add_argument("--traj", default=None,
                   help="trajectory key (default: a middle trajectory)")
    p.add_argument("--t-index", type=int, default=None,
                   help="time index of the chunk start (default: middle of the trajectory)")
    p.add_argument("--stochastic", action="store_true",
                   help="use ancestral sampling noise in the reverse loop (default: deterministic)")
    p.add_argument("--denorm", action="store_true",
                   help="plot physical torque (Nm) instead of normalized action space")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the forward epsilon and the reverse initial noise")
    p.add_argument("--n-show", type=int, default=8,
                   help="number of snapshots in each grid figure")
    p.add_argument("--out-dir", default=str(_CURRENT_DIR / "graphs" / "diffusion_process"),
                   help="output directory for the figures")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.stats) as fp:
        stats = json.load(fp)
    feat = Featurizer(stats)
    policy, scheduler, cfg = load_policy(args.ckpt, device)
    H = policy.horizon

    # Pick the trajectory + a chunk from the middle of it
    with h5py.File(args.h5, "r") as f:
        keys = sorted(f.keys())
        traj_key = args.traj if args.traj is not None else keys[len(keys) // 2]
        if traj_key not in f:
            raise SystemExit(f"trajectory '{traj_key}' not in {args.h5}")
        states  = np.asarray(f[traj_key]["states"],  dtype=np.float32)
        actions = np.asarray(f[traj_key]["actions"], dtype=np.float32)
    Tlen = len(states)
    i = args.t_index if args.t_index is not None else Tlen // 2
    i = int(np.clip(i, 0, Tlen - 1))
    if i + H > Tlen:                                        # keep the chunk in-bounds
        i = Tlen - H
        print(f"[viz] t-index clipped to {i} so the {H}-step chunk fits")

    # Clean (normalized) expert action chunk + matching conditioning.
    a0_np = feat.norm_action(actions[i:i + H])             # (H, nu)
    cond  = feat.build_cond(states, i)
    a0_t  = torch.from_numpy(a0_np)
    print(f"[viz] {traj_key}: chunk start t-index={i}/{Tlen} (state "
          f"q=[{states[i,0]:.2f},{states[i,1]:.2f}])")

    # Run both halves of the diffusion process, capturing all intermediates
    fwd = forward_snapshots(scheduler, a0_t, torch, seed=args.seed)
    rev = reverse_snapshots(policy, scheduler, cond, torch,
                            stochastic=args.stochastic, seed=args.seed)

    recon_mse = float(np.mean((rev[-1] - a0_np) ** 2))
    print(f"[viz] reverse reconstruction MSE vs expert (normalized): {recon_mse:.4f}")

    # Plot
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"traj": traj_key, "t_index": i, "traj_len": Tlen,
            "stochastic": args.stochastic}

    plot_overlay(fwd, rev, a0_np, feat, out_dir / "noising_denoising_overlay.png",
                 args.denorm, info)
    plot_grid(fwd, a0_np, feat, out_dir / "noising_grid.png", args.denorm,
              "Forward diffusion (noising) of the expert action chunk",
              n_show=args.n_show, reverse_labels=False)
    plot_grid(rev, a0_np, feat, out_dir / "denoising_grid.png", args.denorm,
              "Reverse diffusion (denoising) toward the action chunk",
              n_show=args.n_show, reverse_labels=True)

    with open(out_dir / "info.json", "w") as fp:
        json.dump({**info, "horizon": H, "timesteps": int(scheduler.num_steps),
                   "reconstruction_mse_normalized": recon_mse,
                   "denorm_units": bool(args.denorm)}, fp, indent=2)
    print(f"[viz] all figures + info.json saved to {out_dir}")


if __name__ == "__main__":
    main()
