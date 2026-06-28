"""
Animate how the Transformer backbone's per-head self-attention evolves.

Two animations are produced, both in the SAME heatmap-grid format as
visualize_attention_heads.py (rows = encoder layers, cols = heads, token axis
[ t, c, a_1..a_H ], magma colormap, blue lines marking the two context tokens):

  * attention_vs_diffusion_step
        Fix one trajectory chunk; sweep the DIFFUSION STEP t from least to most
        noised (q(a_t|a_0) with a single fixed noise draw, so only t changes).
        Shows how attention re-routes as the action chunk is corrupted.

  * attention_vs_trajectory_pos
        Fix the diffusion step; slide the chunk's START INDEX i along the
        trajectory (hang-down -> swing-up -> upright hold). Shows how attention
        depends on where in the motion the policy is planning from.

Each is written in three forms, into SEPARATE subfolders of --out-dir:
    png/    a representative still frame (the middle frame)
    gif/    the animated GIF
    videos/ the MP4 video
Pick a subset with --formats (e.g. --formats gif,video).

Each frame is one full layer x head grid. The colour scale (vmax) is held FIXED
across all frames so brightness is comparable frame-to-frame.

Attention extraction, conditioning, and normalization are reused verbatim from
the existing viz scripts (no duplicated network/featurizer logic).

Run from the repo root:
    python double_pendulum/animate_attention_heads.py
    python double_pendulum/animate_attention_heads.py --traj traj_42 --fps 4
    python double_pendulum/animate_attention_heads.py --which diffusion
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# Make the repo root importable so `diffusion_models` resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CURRENT_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _CURRENT_DIR / "results"

# Reuse the exact featurizer, loader, and attention-capture shim -- no dup logic.
from double_pendulum.visualize_diffusion_process import Featurizer, load_policy
from double_pendulum.visualize_attention_heads import _CaptureAttn


# ── Attention capture ────────────────────────────────────────────────────────────
def _noise_chunk(scheduler, a0, t, eps):
    """q(a_t | a_0) for integer step t with a CALLER-SUPPLIED eps, so the only
    thing that varies across frames is t (no per-frame noise jitter)."""
    sqrt_ab  = torch.sqrt(scheduler.alpha_bar[t])
    sqrt_1ab = scheduler.sqrt_one_minus_alpha_bar[t]
    return sqrt_ab * a0 + sqrt_1ab * eps


def _noise_chunk_frac(scheduler, a0, t_float, eps):
    """Same as _noise_chunk but for a FRACTIONAL step, linearly interpolating the
    schedule between the bracketing integer steps. t_norm is already continuous,
    so fractional noise levels give extra in-between frames -> a smoother GIF."""
    lo   = int(np.floor(t_float))
    hi   = min(lo + 1, scheduler.num_steps - 1)
    frac = float(t_float - lo)
    sqrt_ab  = (torch.sqrt(scheduler.alpha_bar[lo]) * (1.0 - frac)
                + torch.sqrt(scheduler.alpha_bar[hi]) * frac)
    sqrt_1ab = (scheduler.sqrt_one_minus_alpha_bar[lo] * (1.0 - frac)
                + scheduler.sqrt_one_minus_alpha_bar[hi] * frac)
    return sqrt_ab * a0 + sqrt_1ab * eps


def capture(policy, a_noisy, t_norm, cond_t):
    """Forward one noised chunk through the transformer and return the per-head
    attention maps as a (n_layers, n_heads, L, L) array, L = H + 2."""
    model = policy.model
    if not hasattr(model, "encoder"):
        raise SystemExit("checkpoint is not a TrajectoryTransformer (no .encoder) "
                         "-- attention viz needs --arch transformer")
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
    return np.stack([s.weights for s in shims], axis=0)   # (n_layers, heads, L, L)


# ── Animation ────────────────────────────────────────────────────────────────────
def render(frames, title_for, horizon, base_name, out_dirs, fps, formats):
    """Render a list of (n_layers, n_heads, L, L) attention frames in the same
    grid layout as visualize_attention_heads.plot_attention, and write it as a
    still PNG, a GIF, and/or an MP4 video -- each into its own folder.

    title_for(frame_index) -> the per-frame suptitle string.
    out_dirs:  {"png": Path, "gif": Path, "video": Path}.
    formats:   any subset of {"png", "gif", "video"}."""
    n_layers, n_heads, L, _ = frames[0].shape
    vmax = max(float(f.max()) for f in frames)     # fixed scale across all frames
    labels = ["t", "c"] + [f"a{i+1}" for i in range(horizon)]

    fig, axs = plt.subplots(n_layers, n_heads,
                            figsize=(2.7 * n_heads, 2.7 * n_layers + 1.4),
                            squeeze=False)
    ims = [[None] * n_heads for _ in range(n_layers)]
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axs[li][hi]
            ims[li][hi] = ax.imshow(frames[0][li, hi], cmap="magma",
                                    vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_yticklabels(labels, fontsize=6)
            # Mark the two context-token rows/cols (t, cond) with light gridlines.
            ax.axhline(1.5, color="#5fd0ff", lw=0.8, alpha=0.7)
            ax.axvline(1.5, color="#5fd0ff", lw=0.8, alpha=0.7)
            if li == 0:
                ax.set_title(f"head {hi+1}", fontsize=10, fontweight="bold")
            if hi == 0:
                ax.set_ylabel(f"layer {li+1}\nquery (from)", fontsize=8)
            if li == n_layers - 1:
                ax.set_xlabel("key (to)", fontsize=8)

    sup = fig.suptitle(title_for(0), fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 0.92, 0.95])
    cax = fig.add_axes([0.94, 0.12, 0.015, 0.72])
    fig.colorbar(ims[0][0], cax=cax, label="attention weight")
    fig.text(0.5, 0.01,
             "each row sums to 1 across keys · 't','c' = timestep & conditioning "
             "tokens (left of blue lines) · 'a*' = action-chunk tokens",
             ha="center", fontsize=8, color="#555555")

    def update(fi):
        f = frames[fi]
        for li in range(n_layers):
            for hi in range(n_heads):
                ims[li][hi].set_data(f[li, hi])
        sup.set_text(title_for(fi))
        return [sup]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000.0 / fps, blit=False)

    if "gif" in formats:
        out = out_dirs["gif"] / f"{base_name}.gif"
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"[viz] wrote {out}")
    if "video" in formats:
        out = out_dirs["video"] / f"{base_name}.mp4"
        anim.save(out, writer=FFMpegWriter(fps=fps, codec="h264",
                                           extra_args=["-pix_fmt", "yuv420p"]))
        print(f"[viz] wrote {out}")
    if "png" in formats:
        still = len(frames) // 2                       # representative mid frame
        update(still)
        out = out_dirs["png"] / f"{base_name}.png"
        fig.savefig(out, dpi=150)
        print(f"[viz] wrote {out}  (still frame {still}/{len(frames)})")

    plt.close(fig)
    print(f"[viz] {base_name}: {len(frames)} frames @ {fps} fps")


# ── Frame builders ───────────────────────────────────────────────────────────────
def frames_vs_diffusion_step(policy, scheduler, feat, states, actions, i,
                             H, T, device, seed, substeps):
    """Fix chunk i; sweep diffusion step t = 1..T-1 with one fixed eps draw.
    `substeps` fractional frames are inserted between integer steps for smoothness."""
    a0   = torch.from_numpy(feat.norm_action(actions[i:i + H]))[None].to(device)
    cond = feat.build_cond(states, i)
    cond_t = torch.as_tensor(cond, dtype=torch.float32, device=device).reshape(1, -1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(a0.shape, generator=g).to(device)           # fixed across t

    n_int = T - 1                                                 # integer steps 1..T-1
    if substeps > 1 and n_int > 1:
        steps = list(np.linspace(1.0, T - 1, (n_int - 1) * substeps + 1))
    else:
        steps = [float(t) for t in range(1, T)]

    frames = []
    for tf in steps:
        a_noisy = _noise_chunk_frac(scheduler, a0, tf, eps)
        t_norm  = torch.full((1,), float(tf) / T, dtype=torch.float32, device=device)
        frames.append(capture(policy, a_noisy, t_norm, cond_t))
    return steps, frames


def frames_vs_trajectory_pos(policy, scheduler, feat, states, actions, t_step,
                             H, T, device, seed, max_frames):
    """Fix diffusion step t_step; slide chunk start i along the trajectory."""
    Tlen   = len(states)
    starts = np.arange(0, Tlen - H + 1)
    if len(starts) > max_frames:                                  # evenly subsample
        starts = starts[np.linspace(0, len(starts) - 1, max_frames).round().astype(int)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn((1, H, policy.action_dim), generator=g).to(device)
    t_norm = torch.full((1,), t_step / T, dtype=torch.float32, device=device)

    idxs, frames = [], []
    for i in starts:
        a0 = torch.from_numpy(feat.norm_action(actions[i:i + H]))[None].to(device)
        a_noisy = _noise_chunk(scheduler, a0, t_step, eps)
        cond = feat.build_cond(states, int(i))
        cond_t = torch.as_tensor(cond, dtype=torch.float32, device=device).reshape(1, -1)
        frames.append(capture(policy, a_noisy, t_norm, cond_t))
        idxs.append(int(i))
    return idxs, frames


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",
                   default=str(_RESULTS_DIR / "checkpoints" / "T020_transformer"
                              / "diffusion.pt"),
                   help="transformer checkpoint (.pt with config+model_state)")
    p.add_argument("--stats", default=str(_RESULTS_DIR / "norm_stats.json"),
                   help="normalization stats json")
    p.add_argument("--h5", default=str(_RESULTS_DIR / "expert_trajectories.h5"),
                   help="expert trajectory dataset")
    p.add_argument("--traj", default=None,
                   help="trajectory key (default: a middle trajectory)")
    p.add_argument("--t-index", type=int, default=None,
                   help="chunk start index for the diffusion-step GIF (default: middle)")
    p.add_argument("--t", type=int, default=None,
                   help="fixed diffusion step for the trajectory-position GIF (default: T//2)")
    p.add_argument("--max-frames", type=int, default=200,
                   help="max frames for the trajectory-position GIF (subsampled)")
    p.add_argument("--diffusion-substeps", type=int, default=4,
                   help="fractional frames inserted between integer diffusion "
                        "steps (1 = integer steps only)")
    p.add_argument("--fps", type=int, default=10, help="frame rate (gif + video)")
    p.add_argument("--seed", type=int, default=0, help="fixed noise seed")
    p.add_argument("--which", choices=["both", "diffusion", "position"],
                   default="both", help="which animation(s) to render")
    p.add_argument("--formats", default="png,gif,video",
                   help="comma-separated outputs: any of png,gif,video")
    p.add_argument("--out-dir", default=str(_CURRENT_DIR / "graphs" / "architecture"),
                   help="base output dir; png/gif/videos subfolders are created in it")
    args = p.parse_args()

    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    if not formats <= {"png", "gif", "video"}:
        raise SystemExit(f"--formats must be a subset of png,gif,video (got {formats})")
    if "video" in formats and not FFMpegWriter.isAvailable():
        print("[viz] ffmpeg not found -- skipping mp4 video output")
        formats.discard("video")

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
    if Tlen < H + 1:
        raise SystemExit(f"trajectory '{traj_key}' too short ({Tlen}) for horizon {H}")

    i = args.t_index if args.t_index is not None else Tlen // 2
    i = int(np.clip(i, 0, Tlen - H))

    out_dir = Path(args.out_dir)
    out_dirs = {"png": out_dir / "png", "gif": out_dir / "gif",
                "video": out_dir / "videos"}
    for f in formats:
        out_dirs[f].mkdir(parents=True, exist_ok=True)

    if args.which in ("both", "diffusion"):
        print(f"[viz] {traj_key}: diffusion-step sweep at chunk i={i}/{Tlen}")
        steps, frames = frames_vs_diffusion_step(
            policy, scheduler, feat, states, actions, i, H, T, device, args.seed,
            args.diffusion_substeps)
        render(frames,
               lambda fi: ("Transformer self-attention per head — diffusion step\n"
                           f"{traj_key}, chunk i={i}  ·  "
                           f"t = {steps[fi]:.1f}/{T}  (higher = noisier)"),
               H, "attention_vs_diffusion_step", out_dirs, args.fps, formats)

    if args.which in ("both", "position"):
        print(f"[viz] {traj_key}: trajectory-position sweep at diffusion step t={t_step}/{T}")
        idxs, frames = frames_vs_trajectory_pos(
            policy, scheduler, feat, states, actions, t_step, H, T, device,
            args.seed, args.max_frames)
        render(frames,
               lambda fi: ("Transformer self-attention per head — trajectory position\n"
                           f"{traj_key}, t = {t_step}/{T}  ·  "
                           f"chunk i = {idxs[fi]}/{Tlen}"),
               H, "attention_vs_trajectory_pos", out_dirs, args.fps, formats)

    print(f"[viz] done -> {out_dir}  (formats: {', '.join(sorted(formats))})")


if __name__ == "__main__":
    main()
