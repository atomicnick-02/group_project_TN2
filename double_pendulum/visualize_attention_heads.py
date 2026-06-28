"""
Visualize what each self-attention HEAD of the Transformer backbone does.

The TrajectoryTransformer (diffusion_models/diffusion_policy.py) feeds a length
H+2 token sequence  [ t_token, cond_token, a_1, ..., a_H ]  through a stack of
TransformerEncoder layers, each with multi-head self-attention. The most telling
thing to look at per head is its ATTENTION MAP: an (H+2)x(H+2) matrix whose entry
(i, j) is how much query token i attends to key token j. That shows whether a
head, e.g., spreads attention along the action chunk (temporal mixing), pulls
every action toward the cond token (conditioning), or focuses on the timestep
token (noise-level gating).

PyTorch's nn.TransformerEncoder calls self-attention with need_weights=False, so
the per-head weights are never returned. We therefore temporarily wrap each
layer's `self_attn` with a capture shim that forces
need_weights=True, average_attn_weights=False and stores the (B, heads, L, L)
weights, then plot one heatmap per (layer, head).

The network's attention depends on the noisy action chunk and the diffusion
timestep, so we feed a REAL expert chunk noised to a chosen step t (q(a_t|a_0)),
conditioned on the matching state history -- i.e. the same distribution the
attention was trained on.

Run from the repo root:
    python double_pendulum/visualize_attention_heads.py
    python double_pendulum/visualize_attention_heads.py --t 10 --traj traj_42
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the repo root importable so `diffusion_models` resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CURRENT_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _CURRENT_DIR / "results"

# Reuse the exact featurizer + checkpoint loader the other viz script uses, so
# the conditioning vector and network match training with zero duplication.
from double_pendulum.visualize_diffusion_process import Featurizer, load_policy


class _CaptureAttn(nn.Module):
    """Drop-in wrapper around an nn.MultiheadAttention that records per-head
    attention weights. nn.TransformerEncoderLayer calls self.self_attn(...)[0],
    so forward() just returns a tuple whose [0] is the attn output; we force
    need_weights=True / average_attn_weights=False to also grab the
    (B, heads, L, L) weight tensor the encoder normally discards.

    Must be an nn.Module (the encoder layer registers it as a child module), and
    the forward pass must run in train() mode -- in eval() the encoder takes a
    fused C++ fast path that never calls self_attn at all. dropout is 0.0 here,
    so train() is numerically identical, it just forces the Python path."""

    def __init__(self, mha):
        super().__init__()
        self.mha = mha
        self.weights = None

    def __getattr__(self, name):
        # The encoder probes attributes of self_attn (batch_first, in_proj_bias,
        # ...) directly; forward those to the wrapped module. self.mha is fetched
        # via nn.Module's own lookup (it lives in _modules) to avoid recursion.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("mha"), name)

    def forward(self, query, key, value, **kw):
        kw.pop("need_weights", None)
        kw.pop("average_attn_weights", None)
        out, w = self.mha(query, key, value,
                          need_weights=True, average_attn_weights=False, **kw)
        self.weights = w.detach().float().cpu().numpy()[0]   # (heads, L, L)
        return out, w


def capture_attention(policy, scheduler, cond, a0_np, t_step, seed=0):
    """Forward one noised expert chunk through the transformer and return a
    (n_layers, n_heads, L, L) array of attention maps, L = H + 2."""
    model = policy.model
    if not hasattr(model, "encoder"):
        raise SystemExit("checkpoint is not a TrajectoryTransformer "
                         "(no .encoder) -- attention viz needs --arch transformer")
    device = policy.device

    # Noise the clean expert chunk to diffusion step t: q(a_t | a_0).
    a0 = torch.from_numpy(a0_np).unsqueeze(0).to(device)          # (1, H, nu)
    t_idx = torch.full((1,), int(t_step), dtype=torch.long, device=device)
    torch.manual_seed(seed)
    a_noisy, _ = scheduler.add_noise(a0, t_idx)
    t_norm = t_idx.float() / scheduler.num_steps
    cond_t = torch.as_tensor(cond, dtype=torch.float32, device=device).reshape(1, -1)

    # Temporarily wrap every layer's attention, run a forward pass, restore.
    shims, originals = [], []
    for layer in model.encoder.layers:
        originals.append(layer.self_attn)
        shim = _CaptureAttn(layer.self_attn)
        layer.self_attn = shim
        shims.append(shim)
    try:
        # train() (not eval) so the encoder takes the Python path that calls
        # self_attn; dropout is 0.0 so this is numerically identical.
        model.train()
        with torch.no_grad():
            model(a_noisy, t_norm, cond_t)
    finally:
        for layer, orig in zip(model.encoder.layers, originals):
            layer.self_attn = orig
        model.eval()

    return np.stack([s.weights for s in shims], axis=0)          # (L_layers, heads, L, L)


def plot_attention(attn, horizon, out_path, title):
    """Grid of heatmaps: rows = encoder layers, cols = heads. Token axis order is
    [t, cond, a_1..a_H]; row i / col j of each map = query i attends to key j."""
    n_layers, n_heads, L, _ = attn.shape
    labels = ["t", "c"] + [f"a{i+1}" for i in range(horizon)]
    vmax = float(attn.max())

    fig, axs = plt.subplots(n_layers, n_heads,
                            figsize=(2.7 * n_heads, 2.7 * n_layers + 1.2),
                            squeeze=False)
    im = None
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axs[li][hi]
            im = ax.imshow(attn[li, hi], cmap="magma", vmin=0.0, vmax=vmax,
                           aspect="equal")
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_yticklabels(labels, fontsize=6)
            # Mark the two context-token rows/cols (t, cond) with light gridlines.
            for k in (1.5,):
                ax.axhline(k, color="#5fd0ff", lw=0.8, alpha=0.7)
                ax.axvline(k, color="#5fd0ff", lw=0.8, alpha=0.7)
            if li == 0:
                ax.set_title(f"head {hi+1}", fontsize=10, fontweight="bold")
            if hi == 0:
                ax.set_ylabel(f"layer {li+1}\nquery (from)", fontsize=8)
            if li == n_layers - 1:
                ax.set_xlabel("key (to)", fontsize=8)

    fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.04, 0.92, 0.96])
    cax = fig.add_axes([0.94, 0.12, 0.015, 0.72])
    fig.colorbar(im, cax=cax, label="attention weight")
    fig.text(0.5, 0.005,
             "each row sums to 1 across keys · 't','c' = timestep & conditioning "
             "tokens (left of blue lines) · 'a*' = action-chunk tokens",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",
                   default=str(_RESULTS_DIR / "checkpoints/" / "T020_transformer"
                              / "diffusion.pt"),
                   help="transformer checkpoint (.pt with config+model_state)")
    p.add_argument("--stats", default=str(_RESULTS_DIR / "norm_stats.json"),
                   help="normalization stats json")
    p.add_argument("--h5", default=str(_RESULTS_DIR / "expert_trajectories.h5"),
                   help="expert trajectory dataset")
    p.add_argument("--traj", default=None,
                   help="trajectory key (default: a middle trajectory)")
    p.add_argument("--t-index", type=int, default=None,
                   help="chunk start index in the trajectory (default: middle)")
    p.add_argument("--t", type=int, default=None,
                   help="diffusion step to noise the chunk to (default: T//2)")
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.add_argument("--out-dir", default=str(_CURRENT_DIR / "graphs" / "architecture"),
                   help="output directory for the figure")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.stats) as fp:
        stats = json.load(fp)
    feat = Featurizer(stats)
    policy, scheduler, cfg = load_policy(args.ckpt, device)
    if cfg.get("arch") != "transformer":
        raise SystemExit(f"attention viz needs a transformer checkpoint, "
                         f"got arch='{cfg.get('arch')}'")
    H = policy.horizon
    T = int(scheduler.num_steps)
    t_step = args.t if args.t is not None else max(1, T // 2)
    t_step = int(np.clip(t_step, 1, T - 1))

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
    if i + H > Tlen:
        i = Tlen - H
        print(f"[viz] t-index clipped to {i} so the {H}-step chunk fits")

    a0_np = feat.norm_action(actions[i:i + H])          # (H, nu)
    cond  = feat.build_cond(states, i)
    print(f"[viz] {traj_key}: chunk i={i}/{Tlen}, diffusion step t={t_step}/{T}")

    attn = capture_attention(policy, scheduler, cond, a0_np, t_step,
                             seed=args.seed)
    print(f"[viz] captured attention {attn.shape}  (layers, heads, L, L)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f"Transformer self-attention per head\n"
             f"{traj_key}, chunk i={i}, diffusion step t={t_step}/{T}")
    plot_attention(attn, H, out_dir / "attention_heads.png", title)

    with open(out_dir / "attention_heads_info.json", "w") as fp:
        json.dump({"traj": traj_key, "t_index": i, "traj_len": Tlen,
                   "diffusion_step": t_step, "timesteps": T, "horizon": H,
                   "n_layers": int(attn.shape[0]), "n_heads": int(attn.shape[1]),
                   "token_order": ["t", "cond"] + [f"a{k+1}" for k in range(H)]},
                  fp, indent=2)
    print(f"[viz] figure + info.json saved to {out_dir}")


if __name__ == "__main__":
    main()
