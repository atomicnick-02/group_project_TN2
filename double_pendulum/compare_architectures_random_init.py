"""
Architecture shoot-out under RANDOM initial conditions.

This is the random-init companion to compare_diffusion_models.py. Where that
script scores diffusion checkpoints from the near-hanging-down start they were
trained on, this one stresses every FINAL checkpoint from fully random initial
joint positions (uniform in [-pi, pi]) under a fixed light observation noise,
and reports a single number per architecture: the success rate over N runs.

Two stages, one command:

  Stage A -- architecture comparison
    Every final checkpoint under results/checkpoints/ (the converged *.pt files,
    excluding the periodic *_epoch### snapshots) is run for `--n-runs` swing-ups
    from random start angles at `--noise` observation noise, n_exec=1. The
    per-architecture success rate is bar-charted.

  Stage B -- execution-step sweep
    The transformer trained with `--sweep-timesteps` denoising steps (default 20)
    is then re-run at each of `--n-exec` (default 1 2 4), and the success rate is
    plotted against the receding-horizon replan rate.

Both stages reuse the EXACT rollout physics + success criteria from
evaluate_methods.py, so these numbers are comparable to the other harnesses.

    python double_pendulum/compare_architectures_random_init.py
    python double_pendulum/compare_architectures_random_init.py --n-runs 50 --noise 0.02
    python double_pendulum/compare_architectures_random_init.py --quick

Outputs (default results/architecture_random_init/):
    architecture_success_rate.csv   one row per architecture
    architecture_per_trial.csv      every random-init trial of Stage A
    nexec_sweep_success_rate.csv    success rate per n_exec (Stage B)
    nexec_sweep_per_trial.csv       every trial of Stage B
    plots/architecture_success_rate.png
    plots/nexec_sweep_success_rate.png
"""

import sys
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
from evaluate_methods import rollout, compute_metrics, VEL_NOISE_FACTOR
# Reuse the checkpoint-discovery / metadata / controller-building helpers so this
# script stays in lockstep with compare_diffusion_models.py.
from compare_diffusion_models import (
    read_meta, stats_for, discover_checkpoints, short_label, build_diffusion,
)


# ── One random-init battery for a single controller ──────────────────────────
def run_random_init(label, ctrl, env, n_runs, noise_sigma, max_steps,
                    dt_control, hold_steps, angle_tol, vel_tol, seed):
    """`n_runs` swing-ups from uniformly random start angles at fixed noise.

    Each trial is seeded by (label-independent) (trial, seed) so EVERY model sees
    the SAME random initial conditions -- the success-rate differences are then
    purely the architecture/n_exec, not luck of the draw.
    """
    rows = []
    for trial in range(n_runs):
        rng = np.random.default_rng(
            abs(hash(("random_init", trial, seed))) % (2**32))
        q0 = rng.uniform(-np.pi, np.pi, size=2)        # fully random start position
        v0 = rng.normal(0.0, 0.20, size=2)             # small random start velocity

        roll = rollout(env, ctrl, q0, v0, noise_sigma, rng,
                       max_steps, dt_control, hold_steps, angle_tol, vel_tol)
        m = compute_metrics(roll, dt_control, hold_steps, angle_tol, vel_tol)
        m.update({"model": label, "noise_sigma": noise_sigma, "trial": trial,
                  "q1_0": float(q0[0]), "q2_0": float(q0[1])})
        rows.append(m)
        print(f"  [{label}] trial {trial:02d} | q0=({q0[0]:+.2f},{q0[1]:+.2f}) "
              f"success={m['success']} IAE={m['iae_angle']:.2f}")
    sr = float(np.mean([r["success"] for r in rows]))
    print(f"  [{label}] success rate = {sr:.1%} "
          f"({int(sum(r['success'] for r in rows))}/{len(rows)})")
    return rows


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_architecture_bars(summary, plots_dir, n_runs, noise):
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["model"].tolist()
    rates = summary["success_rate"].to_numpy()
    xs = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 5.5))
    ax.bar(xs, rates, color=colors, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)
    for x, v in zip(xs, rates):
        if np.isfinite(v):
            ax.annotate(f"{v:.0%}", (x, v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Architecture success rate — random init\n"
                 f"(n={n_runs} runs, noise sigma={noise}, n_exec=1)")
    fig.tight_layout()
    out = plots_dir / "architecture_success_rate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_nexec_sweep(sweep, plots_dir, n_runs, noise, timesteps):
    plots_dir.mkdir(parents=True, exist_ok=True)
    sweep = sweep.sort_values("n_exec")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["n_exec"], sweep["success_rate"], "o-", color="tab:green",
            lw=2, ms=9)
    for x, v in zip(sweep["n_exec"], sweep["success_rate"]):
        ax.annotate(f"{v:.0%}", (x, v), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("execution steps per replan (n_exec)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(sweep["n_exec"].tolist())
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Transformer (T{timesteps}) success rate vs execution steps\n"
                 f"(random init, n={n_runs} runs, noise sigma={noise})")
    fig.tight_layout()
    out = plots_dir / "nexec_sweep_success_rate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="checkpoints for Stage A (default: auto-discover all final "
                        "checkpoints under results/checkpoints/)")
    p.add_argument("--n-runs", type=int, default=50,
                   help="random-init swing-ups per model (default: 50)")
    p.add_argument("--noise", type=float, default=0.02,
                   help="observation-noise sigma on angles (default: 0.02)")
    p.add_argument("--n-exec", type=int, nargs="+", default=[1, 2, 4],
                   help="execution-step sweep for Stage B (default: 1 2 4)")
    p.add_argument("--sweep-timesteps", type=int, default=20,
                   help="denoising steps of the transformer used in Stage B "
                        "(default: 20)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--hold-steps", type=int, default=40)
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile for the diffusion models (CUDA only)")
    p.add_argument("--no-sweep", action="store_true",
                   help="skip Stage B (the n_exec sweep)")
    p.add_argument("--sweep-only", action="store_true",
                   help="skip Stage A; run only the n_exec sweep on the "
                        "T<sweep-timesteps> transformer (leaves Stage A outputs "
                        "untouched)")
    p.add_argument("--out", default=None,
                   help="output dir (default: results/architecture_random_init)")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test: 3 runs, 150 steps")
    args = p.parse_args()

    if args.quick:
        args.n_runs, args.max_steps = 3, 150

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve Stage-A checkpoints (relative paths from repo root, like training prints).
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

    out_root = Path(args.out) if args.out else (_THIS / "results" / "architecture_random_init")
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"

    print(f"Device     : {device}")
    print(f"Stage A    : {len(ckpts)} checkpoints x {args.n_runs} random-init runs "
          f"@ noise={args.noise}, n_exec=1")
    print(f"Stage B    : T{args.sweep_timesteps} transformer x n_exec {args.n_exec}")
    print(f"Output dir : {out_root}")

    env = DoublePendulumEnv(render_mode=None, frame_skip=1)

    # ── Stage A: one success rate per architecture ────────────────────────────
    all_rows, summary_rows, used_labels = [], [], set()
    # Remember the T<sweep> transformer checkpoint so Stage B reuses it.
    sweep_ckpt = sweep_stats = sweep_meta = None
    arch_png = None

    for ckpt in ckpts:
        meta = read_meta(ckpt)
        stats_path = stats_for(ckpt, None)
        if not Path(stats_path).exists():
            print(f"[skip] {ckpt}: no stats file ({stats_path})")
            continue

        # Latch the transformer matching --sweep-timesteps for Stage B (done for
        # every checkpoint so --sweep-only can find it without running Stage A).
        if (sweep_ckpt is None and meta.get("arch") == "transformer"
                and meta.get("timesteps") == args.sweep_timesteps):
            sweep_ckpt, sweep_stats, sweep_meta = ckpt, stats_path, meta

        if args.sweep_only:
            continue

        label = short_label(meta, 1, multi_nexec=False)
        base, i = label, 2
        while label in used_labels:
            label = f"{base}#{i}"; i += 1
        used_labels.add(label)

        print(f"\n=== Stage A: {label}  ({ckpt}, stats: {stats_path.name}) ===")
        ctrl = build_diffusion(ckpt, stats_path, device, n_exec=1,
                               compile_diff=not args.no_compile)
        rows = run_random_init(label, ctrl, env, args.n_runs, args.noise,
                               args.max_steps, args.dt_control, args.hold_steps,
                               args.angle_tol, args.vel_tol, args.seed)
        all_rows.extend(rows)
        summary_rows.append({
            "model": label,
            "arch": meta.get("arch"),
            "timesteps": meta.get("timesteps"),
            "n_params": meta.get("n_params"),
            "n_runs": len(rows),
            "n_success": int(sum(r["success"] for r in rows)),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "ckpt": str(ckpt),
        })

    if not args.sweep_only:
        pd.DataFrame(all_rows).to_csv(out_root / "architecture_per_trial.csv", index=False)
        summary = pd.DataFrame(summary_rows).sort_values("success_rate", ascending=False)
        summary.to_csv(out_root / "architecture_success_rate.csv", index=False)
        arch_png = plot_architecture_bars(summary, plots_dir, args.n_runs, args.noise)

        pd.set_option("display.float_format", lambda v: f"{v:.4g}")
        print("\n==========  STAGE A: ARCHITECTURE SUCCESS RATE  ==========")
        print(summary[["model", "arch", "timesteps", "n_success", "success_rate"]]
              .to_string(index=False))
    else:
        print("\n[sweep-only] Stage A skipped; existing architecture_* outputs kept.")

    # ── Stage B: execution-step sweep on the T<sweep> transformer ─────────────
    sweep_summary = None
    if not args.no_sweep:
        if sweep_ckpt is None:
            print(f"\n[skip] Stage B: no transformer with timesteps="
                  f"{args.sweep_timesteps} among the checkpoints. "
                  f"Pass --sweep-timesteps to match one.")
        else:
            label0 = short_label(sweep_meta, 1, multi_nexec=False)
            print(f"\n=== Stage B: n_exec sweep on '{label0}' ({sweep_ckpt}) ===")
            ctrl = build_diffusion(sweep_ckpt, sweep_stats, device,
                                   n_exec=args.n_exec[0],
                                   compile_diff=not args.no_compile)
            sweep_rows, sweep_pts = [], []
            for ne in args.n_exec:
                ctrl.n_exec = ne                       # runtime knob; no rebuild
                label = f"{label0}_ne{ne}"
                print(f"-- n_exec={ne} --")
                rows = run_random_init(label, ctrl, env, args.n_runs, args.noise,
                                       args.max_steps, args.dt_control,
                                       args.hold_steps, args.angle_tol,
                                       args.vel_tol, args.seed)
                sweep_pts.extend(rows)
                sweep_rows.append({
                    "n_exec": ne,
                    "n_runs": len(rows),
                    "n_success": int(sum(r["success"] for r in rows)),
                    "success_rate": float(np.mean([r["success"] for r in rows])),
                })
            pd.DataFrame(sweep_pts).to_csv(out_root / "nexec_sweep_per_trial.csv",
                                           index=False)
            sweep_summary = pd.DataFrame(sweep_rows)
            sweep_summary.to_csv(out_root / "nexec_sweep_success_rate.csv", index=False)
            sweep_png = plot_nexec_sweep(sweep_summary, plots_dir, args.n_runs,
                                         args.noise, args.sweep_timesteps)
            print("\n==========  STAGE B: SUCCESS RATE vs n_exec  ==========")
            print(sweep_summary.to_string(index=False))

    env.close()

    if arch_png is not None:
        print(f"\nArchitecture plot -> {arch_png}")
    if sweep_summary is not None:
        print(f"n_exec sweep plot -> {sweep_png}")
    print(f"CSVs              -> {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
