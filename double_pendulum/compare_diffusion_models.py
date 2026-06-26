"""
Compare several DIFFUSION-POLICY checkpoints (different hyperparameters) on the
double-pendulum swing-up task, head-to-head, under one identical battery.

Where evaluate_methods.py answers "BC vs Diffusion", this script answers
"Diffusion-A vs Diffusion-B vs ..." -- i.e. how the diffusion policy's own
hyperparameters (denoising timesteps, transformer depth/width, action horizon,
the receding-horizon replan rate n_exec, training epochs/LR, ...) trade off on
success rate, trajectory quality, smoothness, robustness and inference cost.

Each "model" in the comparison is a (checkpoint, n_exec) pair, so you can sweep
BOTH the trained network AND the eval-time replan rate from one command:

    # every checkpoint found under results/checkpoints/, executed n_exec=1
    python double_pendulum/compare_diffusion_models.py

    # two specific checkpoints
    python double_pendulum/compare_diffusion_models.py \
        --ckpts results/checkpoints/transformer_T20/diffusion_policy_transformer.pt \
                results/checkpoints/transformer_T100/diffusion_policy_transformer.pt

    # one checkpoint, sweep the receding-horizon replan rate
    python double_pendulum/compare_diffusion_models.py \
        --ckpts results/checkpoints/transformer/diffusion_policy_transformer.pt \
        --n-exec 1 2 4 8

Every checkpoint is labelled by the hyperparameters that produced it, read from
the <ckpt>_hparams.json sidecar that train_diffusion_policy.py now writes (it
falls back to the config embedded in the .pt if the sidecar is absent). The
matching <ckpt>_stats.json sidecar (also written by training) is used for
normalization so each model is evaluated with the exact stats it was trained on;
results/norm_stats.json is the fallback.

Outputs (default results/diffusion_comparison/):
    per_trial_metrics.csv ... every trial of every model
    summary.csv ............. one aggregated row per model
    robustness_success_rate.csv
    plots/metrics_comparison.png   bar charts across models
    plots/robustness_vs_noise.png  success rate vs observation noise
    plots/sweep_<key>.png          metric vs the single varying hyperparameter
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Make the script dir (for `simulation`, `evaluate_*`) and the repo root (for
# `diffusion_models`) importable regardless of the launch directory.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
for _p in (_THIS, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import DoublePendulumEnv
from evaluate_swingup import DiffusionController
# Reuse the EXACT rollout physics + metric definitions from the BC-vs-diffusion
# harness, so these numbers are directly comparable to evaluate_methods.py.
from evaluate_methods import rollout, compute_metrics, GOAL, VEL_NOISE_FACTOR

CKPT_DIR = _THIS / "results" / "checkpoints"
SHARED_STATS = _THIS / "results" / "norm_stats.json"


# ── Per-checkpoint metadata + stats discovery ────────────────────────────────
def read_meta(ckpt_path):
    """Hyperparameters describing a checkpoint.

    Prefers the <ckpt>_hparams.json sidecar (cheap, no torch); falls back to the
    `config` dict embedded in the .pt for checkpoints trained before sidecars.
    """
    sidecar = ckpt_path.with_name(f"{ckpt_path.stem}_hparams.json")
    if sidecar.exists():
        with open(sidecar) as fp:
            return json.load(fp)
    import torch
    cfg = torch.load(ckpt_path, map_location="cpu").get("config", {})
    return dict(cfg)


def stats_for(ckpt_path, override):
    """Normalization-stats path to evaluate this checkpoint with.

    Priority: explicit --stats override > per-checkpoint <ckpt>_stats.json sidecar
    > shared results/norm_stats.json. The sidecar matters because each training
    run overwrites the shared file, so only the newest model matches it.
    """
    if override:
        return Path(override)
    sidecar = ckpt_path.with_name(f"{ckpt_path.stem}_stats.json")
    return sidecar if sidecar.exists() else SHARED_STATS


def discover_checkpoints():
    """All final diffusion checkpoints under results/checkpoints/, newest first.

    Skips the periodic *_epoch### snapshots -- only the converged checkpoints are
    meaningful comparison points.
    """
    if not CKPT_DIR.exists():
        return []
    cands = [p for p in CKPT_DIR.rglob("*.pt") if "_epoch" not in p.stem]
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)


def short_label(meta, n_exec, multi_nexec):
    """Compact, human-readable tag built from the distinguishing hyperparameters."""
    nk = meta.get("net_kwargs", {}) or {}
    arch = meta.get("arch", "?")
    parts = [f"T{meta.get('timesteps', '?')}"]
    if arch == "transformer":
        parts.append(f"L{nk.get('n_layers', '?')}"
                     f"d{nk.get('d_model', '?')}"
                     f"ff{nk.get('dim_feedforward', '?')}")
    elif arch == "mlp":
        parts.append(f"mlp{nk.get('hidden_dim', '?')}")
    parts.append(f"H{nk.get('horizon', meta.get('horizon', '?'))}")
    if multi_nexec:
        parts.append(f"ne{n_exec}")
    return "_".join(str(p) for p in parts)


# ── Controller construction (mirrors evaluate_methods.build_controller) ──────
def build_diffusion(ckpt_path, stats_path, device, n_exec, compile_diff):
    """Build a DiffusionController in EVAL mode, honoring a forced device."""
    import torch
    if device == "cpu":                      # force CPU even on a CUDA box
        _orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    try:
        ctrl = DiffusionController(str(ckpt_path), str(stats_path), n_exec=n_exec)
        if compile_diff and device == "cuda":
            # Diffusion sampling is many tiny sequential forward passes dominated
            # by launch overhead; CUDA-graph compile is a large, lossless speedup.
            try:
                ctrl.policy.model = torch.compile(ctrl.policy.model,
                                                  mode="reduce-overhead")
                print("  [diffusion] torch.compile(reduce-overhead) enabled")
            except Exception as e:
                print(f"  [diffusion] torch.compile failed ({e}); eager mode")
    finally:
        if device == "cpu":
            torch.cuda.is_available = _orig
    return ctrl


# ── Run one model (checkpoint + n_exec) over the whole battery ───────────────
def run_model(label, ctrl, env, noise_levels, n_nominal, n_robust,
              max_steps, dt_control, hold_steps, angle_tol, vel_tol, seed):
    rows, examples = [], {}
    for sigma in noise_levels:
        n_trials = n_nominal if sigma == 0.0 else n_robust
        for trial in range(n_trials):
            # Deterministic per (label, sigma, trial) so reruns reproduce, and so
            # every model sees the SAME initial conditions at each (sigma, trial).
            rng = np.random.default_rng(
                abs(hash(("diff", round(sigma, 4), trial, seed))) % (2**32))
            q0 = rng.normal(0.0, 0.10, size=2)
            v0 = rng.normal(0.0, 0.20, size=2)

            roll = rollout(env, ctrl, q0, v0, sigma, rng,
                           max_steps, dt_control, hold_steps, angle_tol, vel_tol)
            m = compute_metrics(roll, dt_control, hold_steps, angle_tol, vel_tol)
            m.update({"model": label, "noise_sigma": sigma, "trial": trial})
            rows.append(m)
            if sigma == 0.0:
                examples[trial] = roll
            print(f"  [{label}] sigma={sigma:0.3f} trial {trial:02d} | "
                  f"success={m['success']} IAE={m['iae_angle']:.2f} "
                  f"|du|={m['mean_abs_du']:.4f} infer={m['mean_infer_ms']:.1f}ms")
    return rows, examples


# ── Aggregation ──────────────────────────────────────────────────────────────
def build_summary(df, meta_by_label):
    rows = []
    for label, sub in df.groupby("model", sort=False):
        nom = sub[sub["noise_sigma"] == 0.0]
        succ = nom[nom["success"] == 1]
        meta = meta_by_label.get(label, {})
        rows.append({
            "model": label,
            "arch": meta.get("arch"),
            "timesteps": meta.get("timesteps"),
            "horizon": (meta.get("net_kwargs", {}) or {}).get("horizon",
                                                              meta.get("horizon")),
            "n_exec": meta.get("_n_exec"),
            "epochs": meta.get("epochs"),
            "learning_rate": meta.get("learning_rate"),
            "n_params": meta.get("n_params"),
            "test_loss": meta.get("test_loss"),
            "n_nominal_trials": len(nom),
            "success_rate": float(nom["success"].mean()) if len(nom) else np.nan,
            "time_to_success_s_mean": float(np.nanmean(succ["time_to_success_s"])) if len(succ) else np.nan,
            "iae_angle_mean": float(np.nanmean(nom["iae_angle"])) if len(nom) else np.nan,
            "iae_angle_std": float(np.nanstd(nom["iae_angle"])) if len(nom) else np.nan,
            "final_angle_err_mean": float(np.nanmean(nom["final_angle_err"])) if len(nom) else np.nan,
            "ss_angle_err_mean": float(np.nanmean(succ["ss_angle_err"])) if len(succ) else np.nan,
            "held_to_end_rate": float(succ["held_to_end"].mean()) if len(succ) else np.nan,
            "mean_abs_du_mean": float(np.nanmean(nom["mean_abs_du"])) if len(nom) else np.nan,
            "control_tv_mean": float(np.nanmean(nom["control_tv"])) if len(nom) else np.nan,
            "rms_jerk_mean": float(np.nanmean(nom["rms_jerk"])) if len(nom) else np.nan,
            "mean_abs_u_mean": float(np.nanmean(nom["mean_abs_u"])) if len(nom) else np.nan,
            "infer_ms_mean": float(np.nanmean(nom["mean_infer_ms"])) if len(nom) else np.nan,
            "infer_ms_std": float(np.nanstd(nom["mean_infer_ms"])) if len(nom) else np.nan,
            "control_hz_mean": float(np.nanmean(nom["control_hz"])) if len(nom) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────
def _bars(ax, labels, vals, errs, title, ylabel, log=False):
    xs = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))
    ax.bar(xs, vals, yerr=errs, capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3g}", (x, v), ha="center", va="bottom", fontsize=7)


def make_plots(df, summary, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["model"].tolist()

    # 1) headline metric bars across all models
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    _bars(axs[0, 0], labels, summary["success_rate"], None,
          "Success rate (nominal)\n(higher = better)", "fraction")
    _bars(axs[0, 1], labels, summary["time_to_success_s_mean"], None,
          "Time to success\n(lower = better)", "seconds")
    _bars(axs[0, 2], labels, summary["iae_angle_mean"], summary["iae_angle_std"],
          "Trajectory quality: IAE angle\n(lower = better)", "rad·s")
    _bars(axs[1, 0], labels, summary["ss_angle_err_mean"], None,
          "Stability: steady-state angle err\n(lower = better)", "rad")
    _bars(axs[1, 1], labels, summary["mean_abs_du_mean"], None,
          "Smoothness: mean |Δu|\n(lower = smoother)", "Nm/step")
    _bars(axs[1, 2], labels, summary["infer_ms_mean"], summary["infer_ms_std"],
          "Inference cost / control step", "ms (log)", log=True)
    fig.suptitle("Diffusion-policy hyperparameter comparison — swing-up", fontsize=15)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=130)
    plt.close(fig)

    # 2) robustness: success rate vs observation-noise level, one line per model
    rob = (df.groupby(["model", "noise_sigma"])["success"].mean()
           .reset_index().rename(columns={"success": "success_rate"}))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label in labels:
        sub = rob[rob["model"] == label].sort_values("noise_sigma")
        ax.plot(sub["noise_sigma"], sub["success_rate"], "o-", lw=2, ms=6, label=label)
    ax.set_xlabel("observation-noise sigma (rad on angles, ×%g on velocities)" % VEL_NOISE_FACTOR)
    ax.set_ylabel("success rate")
    ax.set_title("Robustness to observation noise")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "robustness_vs_noise.png", dpi=130)
    plt.close(fig)
    return rob


def make_sweep_plot(summary, plots_dir, sweep_key):
    """If exactly one numeric hyperparameter varies, plot metrics against it."""
    candidates = ["timesteps", "n_exec", "horizon", "epochs", "learning_rate", "n_params"]
    if sweep_key is None:
        varying = [k for k in candidates
                   if k in summary and summary[k].nunique(dropna=True) > 1]
        # auto-pick only when a single axis varies -> an unambiguous sweep
        if len(varying) != 1:
            return None
        sweep_key = varying[0]
    if sweep_key not in summary or summary[sweep_key].nunique(dropna=True) < 2:
        return None

    sub = summary.dropna(subset=[sweep_key]).sort_values(sweep_key)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sub[sweep_key], sub["success_rate"], "o-", color="tab:green",
             lw=2, ms=7, label="success rate")
    ax1.set_xlabel(sweep_key); ax1.set_ylabel("success rate", color="tab:green")
    ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sub[sweep_key], sub["iae_angle_mean"], "s--", color="tab:red",
             lw=2, ms=6, label="IAE angle")
    ax2.set_ylabel("IAE angle (rad·s)", color="tab:red")
    fig.suptitle(f"Diffusion swing-up vs {sweep_key}")
    fig.tight_layout()
    out = plots_dir / f"sweep_{sweep_key}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="diffusion checkpoint paths (default: auto-discover under "
                        "results/checkpoints/, excluding *_epoch### snapshots)")
    p.add_argument("--n-exec", type=int, nargs="+", default=[1],
                   help="receding-horizon replan rate(s); each checkpoint is run "
                        "once per value (n_exec=1 = re-plan every control step)")
    p.add_argument("--stats", default=None,
                   help="force one norm-stats JSON for ALL checkpoints (default: "
                        "each checkpoint's <ckpt>_stats.json sidecar, else shared)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n-nominal", type=int, default=8,
                   help="trials at noise=0 (success rate & per-metric stats)")
    p.add_argument("--n-robust", type=int, default=4,
                   help="trials per non-zero noise level")
    p.add_argument("--noise-levels", type=float, nargs="+",
                   default=[0.0, 0.02, 0.03, 0.04, 0.05],
                   help="observation-noise sigmas to sweep (must include 0.0)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--hold-steps", type=int, default=40)
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile for the diffusion models (CUDA only)")
    p.add_argument("--sweep-key", default=None,
                   help="force the x-axis hyperparameter for the sweep plot "
                        "(default: auto-detect the single varying one)")
    p.add_argument("--out", default=None,
                   help="output dir (default: results/diffusion_comparison)")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test: 2 nominal, 1 robust, 150 steps, [0,0.05]")
    args = p.parse_args()

    if args.quick:
        args.n_nominal, args.n_robust = 2, 1
        args.max_steps, args.noise_levels = 150, [0.0, 0.05]
    if 0.0 not in args.noise_levels:
        args.noise_levels = [0.0] + args.noise_levels

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve checkpoints (relative paths are taken from the repo root, matching
    # how train_diffusion_policy.py prints them).
    if args.ckpts:
        ckpts = []
        for c in args.ckpts:
            cp = Path(c)
            if not cp.is_absolute():
                cp = (_REPO / cp) if (_REPO / cp).exists() else (_THIS / cp)
            ckpts.append(cp)
    else:
        ckpts = discover_checkpoints()
    ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        raise SystemExit("No diffusion checkpoints found. Pass --ckpts or train "
                         "a model so results/checkpoints/ is populated.")

    out_root = Path(args.out) if args.out else (_THIS / "results" / "diffusion_comparison")
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"

    multi_nexec = len(args.n_exec) > 1
    print(f"Device     : {device}")
    print(f"Checkpoints: {len(ckpts)}  x  n_exec {args.n_exec}  "
          f"= {len(ckpts) * len(args.n_exec)} models")
    print(f"Noise sweep: {args.noise_levels} | nominal={args.n_nominal} "
          f"robust/level={args.n_robust} max_steps={args.max_steps}")
    print(f"Output dir : {out_root}")

    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    all_rows, meta_by_label = [], {}
    used_labels = set()

    for ckpt in ckpts:
        meta = read_meta(ckpt)
        stats_path = stats_for(ckpt, args.stats)
        if not Path(stats_path).exists():
            print(f"[skip] {ckpt}: no stats file ({stats_path})")
            continue
        print(f"\n=== {ckpt}  (stats: {stats_path.name}) ===")
        ctrl = build_diffusion(ckpt, stats_path, device, args.n_exec[0],
                               compile_diff=not args.no_compile)
        for ne in args.n_exec:
            ctrl.n_exec = ne                       # runtime knob; no rebuild needed
            label = short_label(meta, ne, multi_nexec)
            # disambiguate identical labels from different files
            base, i = label, 2
            while label in used_labels:
                label = f"{base}#{i}"; i += 1
            used_labels.add(label)

            m = dict(meta); m["_n_exec"] = ne; m["_ckpt"] = str(ckpt)
            meta_by_label[label] = m

            print(f"-- model '{label}' (n_exec={ne}) --")
            rows, _ = run_model(label, ctrl, env, args.noise_levels,
                                args.n_nominal, args.n_robust, args.max_steps,
                                args.dt_control, args.hold_steps,
                                args.angle_tol, args.vel_tol, args.seed)
            all_rows.extend(rows)
    env.close()

    if not all_rows:
        raise SystemExit("No models were evaluated (all skipped for missing stats).")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "per_trial_metrics.csv", index=False)

    summary = build_summary(df, meta_by_label)
    summary.to_csv(out_root / "summary.csv", index=False)

    rob = make_plots(df, summary, plots_dir)
    rob.to_csv(out_root / "robustness_success_rate.csv", index=False)
    sweep_png = make_sweep_plot(summary, plots_dir, args.sweep_key)

    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n================  SUMMARY  ================")
    cols = ["model", "success_rate", "iae_angle_mean", "mean_abs_du_mean",
            "infer_ms_mean", "test_loss"]
    print(summary[cols].to_string(index=False))
    print(f"\nPer-trial   -> {out_root / 'per_trial_metrics.csv'}")
    print(f"Summary     -> {out_root / 'summary.csv'}")
    print(f"Plots       -> {plots_dir}")
    if sweep_png:
        print(f"Sweep plot  -> {sweep_png}")
    print("Done.")


if __name__ == "__main__":
    main()
