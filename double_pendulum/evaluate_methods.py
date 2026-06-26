"""
Head-to-head evaluation of the two imitation-learning controllers on the
double-pendulum swing-up task:

    * Behavioral Cloning (BC)   -- behaviour_cloning/bc_policy_3.pt   (plain MLP)
    * Diffusion Policy           -- double_pendulum/results/diffusion_policy.pt

Both nets are run in EVALUATION MODE (model.eval(), no_grad) inside the identical
MuJoCo environment, start state and success criteria, so their numbers are
directly comparable.

What it does, in order:
    1. Run the BC controller over a battery of trials -> per-step CSVs + a
       per-trial metrics CSV  (results/method_comparison/bc/...).
    2. Run the Diffusion controller over the same battery -> its own CSVs.
    3. Combine both methods' metrics into comparison plots + a summary CSV.

Evaluation metrics recorded (per trial, then aggregated):
    success rate ........ fraction of trials that reach & hold upright
    trajectory quality .. integrated angle error (IAE) + time-to-success
    stability ........... post-success steady-state angle error & state std
    smoothness .......... control increment |du|, total variation, jerk, effort
    robustness to noise . success rate vs injected observation-noise level
    computational cost ... wall-clock inference time per control step (ms / Hz)

Run from the repo root or from double_pendulum/ (both are put on sys.path):
    python double_pendulum/evaluate_methods.py
    python double_pendulum/evaluate_methods.py --quick           # tiny smoke test
    python double_pendulum/evaluate_methods.py --n-nominal 10 --device cuda
    python double_pendulum/evaluate_methods.py --no-success-rate # only random-init
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Make both the script dir (for `simulation`, `evaluate_swingup`) and the repo
# root (for `diffusion_models`) importable regardless of the launch directory.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
for _p in (_THIS, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import DoublePendulumEnv
from evaluate_swingup import DiffusionController, BCController, wrap_to_pi

GOAL = np.array([np.pi, 0.0, 0.0, 0.0])          # upright target [q1, q2, dq1, dq2]
VEL_NOISE_FACTOR = 5.0                             # obs-noise on velocities = factor * angle-noise sigma
BC_COLOR, DIFF_COLOR = "tab:blue", "tab:orange"


# ── BC controller with the device fix ────────────────────────────────────────
class BCControllerGPU(BCController):
    """
    BCController.action() builds the input tensor on CPU but the model may live
    on CUDA, which raises a device-mismatch on GPU machines. This override moves
    the input to the model's device; everything else (loading, normalization,
    eval-mode) is inherited unchanged.
    """

    def action(self, x):
        f  = self._feat(np.asarray(x, dtype=np.float32))
        ft = self.torch.from_numpy(f[None]).to(self.device)        # (1, 6)
        with self.torch.no_grad():
            a_seq = self.model(ft).cpu().numpy()[0]                 # (H, nu) normalized
        return self._denorm_a(a_seq[0])


# ── Controller construction ──────────────────────────────────────────────────
def build_controller(name, device, paths, compile_diff=True):
    """Build a fresh controller in EVAL mode. `device` forces cuda/cpu for both."""
    import torch
    # Both controllers pick their device from torch.cuda.is_available(); to honor
    # --device we temporarily mask cuda so a forced 'cpu' run really runs on CPU.
    if device == "cpu":
        _orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    try:
        if name == "bc":
            ctrl = BCControllerGPU(paths["bc_ckpt"], paths["bc_stats"])
        else:
            ctrl = DiffusionController(paths["diff_ckpt"], paths["diff_stats"], n_exec=1)
            # Diffusion sampling is 100 sequential tiny forward passes -> dominated
            # by kernel-launch overhead. CUDA-graph compile is ~3x and numerically
            # lossless here; warmup happens on the first few action() calls.
            if compile_diff and device == "cuda":
                try:
                    ctrl.policy.model = torch.compile(ctrl.policy.model,
                                                      mode="reduce-overhead")
                    print("[diffusion] torch.compile(reduce-overhead) enabled")
                except Exception as e:                       # fall back to eager
                    print(f"[diffusion] torch.compile failed ({e}); eager mode")
    finally:
        if device == "cpu":
            torch.cuda.is_available = _orig
    return ctrl


# ── One rollout ──────────────────────────────────────────────────────────────
def rollout(env, ctrl, q0, v0, noise_sigma, rng,
            max_steps, dt_control, hold_steps, angle_tol, vel_tol):
    """
    Simulate one swing-up attempt through the headless DoublePendulumEnv gym API.

    The whole episode is driven by env.reset()/env.step(): the env is a headless
    MuJoCo environment (render_mode=None) whose frame_skip is set so a single
    step() advances exactly one dt_control window. The controller observes a NOISY
    copy of the env's state (Gaussian obs-noise with std `noise_sigma` on angles
    and VEL_NOISE_FACTOR*noise_sigma on velocities); the clean state the env
    returns is what we record and score. The env returns angles wrapped to
    [-pi, pi]; both controllers use sin/cos angular features, so that wrapping is
    transparent to them (success/metrics handle it via wrap_to_pi / np.unwrap).
    """
    # One env.step() == one control window: frame_skip = dt_control / dt_sim.
    dt_sim = env.model.opt.timestep
    env.frame_skip = max(1, int(round(dt_control / dt_sim)))

    obs, _ = env.reset(options={"qpos": np.asarray(q0, dtype=np.float64),
                                "qvel": np.asarray(v0, dtype=np.float64)})
    x = np.asarray(obs[:4], dtype=np.float64)
    ctrl.reset(x)

    max_tau = float(env.action_space.high[0])
    sigma_v = np.array([noise_sigma, noise_sigma,
                        noise_sigma * VEL_NOISE_FACTOR,
                        noise_sigma * VEL_NOISE_FACTOR], dtype=np.float64)

    t_log, x_log, u_log, dt_log = [], [], [], []
    consecutive_hold, success_step = 0, None

    for step in range(max_steps):
        x_obs = x + rng.normal(0.0, sigma_v) if noise_sigma > 0 else x

        t0 = time.perf_counter()
        u  = np.asarray(ctrl.action(x_obs), dtype=np.float64).ravel()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        u  = np.clip(u, -max_tau, max_tau)

        t_log.append(step * dt_control)
        x_log.append(x.copy())
        u_log.append(u.copy())
        dt_log.append(infer_ms)

        ang_err = abs(wrap_to_pi(x[0] - GOAL[0])) + abs(wrap_to_pi(x[1] - GOAL[1]))
        vel_err = abs(x[2]) + abs(x[3])
        if ang_err < angle_tol and vel_err < vel_tol:
            consecutive_hold += 1
            if consecutive_hold >= hold_steps and success_step is None:
                success_step = step
        else:
            consecutive_hold = 0

        obs, _, _, _, _ = env.step(u)
        x = np.asarray(obs[:4], dtype=np.float64)

    return {
        "t": np.array(t_log), "x": np.array(x_log), "u": np.array(u_log),
        "infer_ms": np.array(dt_log), "success_step": success_step,
    }


# ── Metrics for one rollout ──────────────────────────────────────────────────
def compute_metrics(roll, dt_control, hold_steps, angle_tol, vel_tol):
    t, x, u = roll["t"], roll["x"], roll["u"]
    success_step = roll["success_step"]
    success = success_step is not None

    ang_err = np.abs(wrap_to_pi(x[:, 0] - GOAL[0])) + np.abs(wrap_to_pi(x[:, 1] - GOAL[1]))
    vel_err = np.abs(x[:, 2]) + np.abs(x[:, 3])

    # trajectory quality
    iae_angle  = float(np.sum(ang_err) * dt_control)          # integrated abs angle error
    tail       = max(1, int(round(1.0 / dt_control)))          # last ~1 s
    final_ang  = float(np.mean(ang_err[-tail:]))               # where it ended up

    # stability (only meaningful once upright is reached)
    if success:
        post = ang_err[success_step:]
        ss_angle_err = float(np.mean(post))
        # Positional jitter measured around the goal in wrap-safe coords: the env
        # returns angles wrapped to [-pi, pi], so the raw std of q1 at the upright
        # (q1=pi) boundary would spuriously blow up as it flips between +/-pi.
        ss_state_std = float(np.mean([np.std(wrap_to_pi(x[success_step:, 0] - GOAL[0])),
                                      np.std(wrap_to_pi(x[success_step:, 1] - GOAL[1]))]))
        held_to_end  = bool(ang_err[-1] < angle_tol and vel_err[-1] < vel_tol)
        t_success    = float(success_step * dt_control)
    else:
        ss_angle_err = ss_state_std = np.nan
        held_to_end  = False
        t_success    = np.nan

    # smoothness (control)
    if len(u) > 1:
        du          = np.diff(u, axis=0)
        mean_abs_du = float(np.mean(np.abs(du)))               # avg per-step torque change
        control_tv  = float(np.sum(np.abs(du)))               # total variation
        rms_jerk    = float(np.sqrt(np.mean((du / dt_control) ** 2)))
    else:
        mean_abs_du = control_tv = rms_jerk = np.nan
    mean_abs_u = float(np.mean(np.abs(u)))                     # control effort

    # computational cost
    mean_infer_ms = float(np.mean(roll["infer_ms"]))
    ctrl_hz       = float(1000.0 / mean_infer_ms) if mean_infer_ms > 0 else np.nan

    return {
        "success": int(success),
        "time_to_success_s": t_success,
        "iae_angle": iae_angle,
        "final_angle_err": final_ang,
        "ss_angle_err": ss_angle_err,
        "ss_state_std": ss_state_std,
        "held_to_end": int(held_to_end),
        "mean_abs_du": mean_abs_du,
        "control_tv": control_tv,
        "rms_jerk": rms_jerk,
        "mean_abs_u": mean_abs_u,
        "mean_infer_ms": mean_infer_ms,
        "control_hz": ctrl_hz,
        "total_steps": int(len(t)),
    }


# ── Per-trial trajectory CSV ─────────────────────────────────────────────────
def save_trajectory_csv(path, roll):
    df = pd.DataFrame({
        "t": roll["t"],
        "q1": roll["x"][:, 0], "q2": roll["x"][:, 1],
        "dq1": roll["x"][:, 2], "dq2": roll["x"][:, 3],
        "u1": roll["u"][:, 0], "u2": roll["u"][:, 1],
        "infer_ms": roll["infer_ms"],
    })
    df.to_csv(path, index=False)


# ── Run a whole battery for one method ───────────────────────────────────────
def run_method(name, device, paths, out_root, noise_levels, n_nominal, n_robust,
               max_steps, dt_control, hold_steps, angle_tol, vel_tol, seed,
               compile_diff=True):
    print(f"\n=== {name.upper()} : building controller (device={device}) ===")
    ctrl = build_controller(name, device, paths, compile_diff=compile_diff)
    env  = DoublePendulumEnv(render_mode=None, frame_skip=1)

    method_dir = out_root / name
    method_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    examples = {}     # noise=0 trajectories kept in memory for the overlay plots

    for sigma in noise_levels:
        n_trials = n_nominal if sigma == 0.0 else n_robust
        for trial in range(n_trials):
            # deterministic per (method, sigma, trial) so runs are reproducible
            rng = np.random.default_rng(
                abs(hash((name, round(sigma, 4), trial, seed))) % (2**32))
            q0 = rng.normal(0.0, 0.10, size=2)        # start near hanging-down, varied
            v0 = rng.normal(0.0, 0.20, size=2)

            roll = rollout(env, ctrl, q0, v0, sigma, rng,
                           max_steps, dt_control, hold_steps, angle_tol, vel_tol)
            m = compute_metrics(roll, dt_control, hold_steps, angle_tol, vel_tol)
            m.update({"method": name, "noise_sigma": sigma, "trial": trial})
            rows.append(m)

            tag = f"sigma{sigma:0.3f}_trial{trial:02d}"
            save_trajectory_csv(method_dir / f"{tag}.csv", roll)
            if sigma == 0.0:
                examples[trial] = roll

            print(f"  [{name}] sigma={sigma:0.3f} trial {trial:02d} | "
                  f"success={m['success']} IAE={m['iae_angle']:.2f} "
                  f"|du|={m['mean_abs_du']:.4f} infer={m['mean_infer_ms']:.1f}ms")

    env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"{name}_metrics.csv", index=False)
    print(f"  [{name}] per-trial metrics -> {out_root / f'{name}_metrics.csv'}")
    return df, examples


# ── Aggregation + summary CSV ────────────────────────────────────────────────
def _agg(df_nom, col):
    return float(np.nanmean(df_nom[col])), float(np.nanstd(df_nom[col]))


def build_summary(bc_df, diff_df, out_root):
    rows = []
    for name, df in (("bc", bc_df), ("diffusion", diff_df)):
        nom = df[df["noise_sigma"] == 0.0]
        succ = nom[nom["success"] == 1]
        row = {
            "method": name,
            "n_nominal_trials": len(nom),
            "success_rate": float(nom["success"].mean()),
            "time_to_success_s_mean": float(np.nanmean(succ["time_to_success_s"])) if len(succ) else np.nan,
            "iae_angle_mean": _agg(nom, "iae_angle")[0],
            "iae_angle_std":  _agg(nom, "iae_angle")[1],
            "final_angle_err_mean": _agg(nom, "final_angle_err")[0],
            "ss_angle_err_mean":    float(np.nanmean(succ["ss_angle_err"])) if len(succ) else np.nan,
            "ss_state_std_mean":    float(np.nanmean(succ["ss_state_std"])) if len(succ) else np.nan,
            "held_to_end_rate":     float(succ["held_to_end"].mean()) if len(succ) else np.nan,
            "mean_abs_du_mean": _agg(nom, "mean_abs_du")[0],
            "control_tv_mean":  _agg(nom, "control_tv")[0],
            "rms_jerk_mean":    _agg(nom, "rms_jerk")[0],
            "mean_abs_u_mean":  _agg(nom, "mean_abs_u")[0],
            "infer_ms_mean":  _agg(nom, "mean_infer_ms")[0],
            "infer_ms_std":   _agg(nom, "mean_infer_ms")[1],
            "control_hz_mean": _agg(nom, "control_hz")[0],
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "comparison_summary.csv", index=False)
    print(f"\nComparison summary -> {out_root / 'comparison_summary.csv'}")

    # robustness table: success rate per noise level per method
    rob = (pd.concat([bc_df, diff_df])
           .groupby(["method", "noise_sigma"])["success"].mean()
           .reset_index().rename(columns={"success": "success_rate"}))
    rob.to_csv(out_root / "robustness_success_rate.csv", index=False)
    print(f"Robustness table  -> {out_root / 'robustness_success_rate.csv'}")
    return summary, rob


# ── Plots combining both methods ─────────────────────────────────────────────
def _bar(ax, vals, errs, title, ylabel, log=False):
    xs = [0, 1]
    ax.bar(xs, vals, yerr=errs, capsize=5, color=[BC_COLOR, DIFF_COLOR],
           alpha=0.85, width=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(["BC", "Diffusion"])
    ax.set_title(title); ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3g}", (x, v), ha="center", va="bottom", fontsize=9)


def _success_panel(ax, bc_nom, diff_nom, rand_summary=None, random_noise=None):
    """Success-rate panel for the headline figure. Given a random-init summary it
    shows nominal vs random-init as grouped bars per method (so the off-
    distribution collapse is visible in one panel); otherwise it falls back to
    the nominal-only bar."""
    xs = np.arange(2)
    ax.set_xticks(xs); ax.set_xticklabels(["BC", "Diffusion"])
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)

    def _annotate(rects):
        for r in rects:
            h = r.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2g}", (r.get_x() + r.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)

    if rand_summary is None:
        _annotate(ax.bar(xs, [bc_nom, diff_nom], width=0.6,
                         color=[BC_COLOR, DIFF_COLOR], alpha=0.85))
        ax.set_title("Success rate (nominal)")
        return

    rs = rand_summary.set_index("method")["success_rate"]
    rnd = [float(rs.get("bc", np.nan)), float(rs.get("diffusion", np.nan))]
    w = 0.38
    _annotate(ax.bar(xs - w / 2, [bc_nom, diff_nom], w,
                     color=[BC_COLOR, DIFF_COLOR], alpha=0.85))
    _annotate(ax.bar(xs + w / 2, rnd, w,
                     color=[BC_COLOR, DIFF_COLOR], alpha=0.45, hatch="//"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.85, label="nominal"),
                       Patch(facecolor="gray", alpha=0.45, hatch="//",
                             label="random-init")],
              fontsize=8, loc="upper right")
    n = int(rand_summary["n_runs"].iloc[0])
    extra = f", noise={random_noise}" if random_noise is not None else ""
    ax.set_title(f"Success rate: nominal vs random-init\n"
                 f"(random-init: n={n}{extra})")


def make_plots(bc_df, diff_df, summary, rob, bc_ex, diff_ex, plots_dir, dt_control,
               rand_summary=None, random_noise=None):
    plots_dir.mkdir(parents=True, exist_ok=True)
    s_bc   = summary[summary["method"] == "bc"].iloc[0]
    s_diff = summary[summary["method"] == "diffusion"].iloc[0]

    # 1) metric comparison bars (the headline figure)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    _success_panel(axs[0, 0], s_bc["success_rate"], s_diff["success_rate"],
                   rand_summary, random_noise)
    _bar(axs[0, 1], [s_bc["time_to_success_s_mean"], s_diff["time_to_success_s_mean"]],
         None, "Time to success", "seconds")
    _bar(axs[0, 2],
         [s_bc["iae_angle_mean"], s_diff["iae_angle_mean"]],
         [s_bc["iae_angle_std"], s_diff["iae_angle_std"]],
         "Trajectory quality: IAE angle\n(lower = better)", "rad·s")
    _bar(axs[1, 0], [s_bc["ss_angle_err_mean"], s_diff["ss_angle_err_mean"]], None,
         "Stability: steady-state angle err\n(lower = better)", "rad")
    _bar(axs[1, 1], [s_bc["mean_abs_du_mean"], s_diff["mean_abs_du_mean"]], None,
         "Smoothness: mean |Δu|\n(lower = smoother)", "Nm/step")
    _bar(axs[1, 2], [s_bc["infer_ms_mean"], s_diff["infer_ms_mean"]],
         [s_bc["infer_ms_std"], s_diff["infer_ms_std"]],
         "Computational cost\n(inference / control step)", "ms (log)", log=True)
    fig.suptitle("BC vs Diffusion Policy — double-pendulum swing-up", fontsize=15)
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=130)
    plt.close(fig)

    # 2) robustness: success rate vs observation-noise level
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in (("bc", BC_COLOR), ("diffusion", DIFF_COLOR)):
        sub = rob[rob["method"] == name].sort_values("noise_sigma")
        ax.plot(sub["noise_sigma"], sub["success_rate"], "o-", color=color,
                label=name.upper(), lw=2, ms=7)
    ax.set_xlabel("observation-noise sigma (rad on angles, ×%g on velocities)" % VEL_NOISE_FACTOR)
    ax.set_ylabel("success rate")
    ax.set_title("Robustness to observation noise")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "robustness_vs_noise.png", dpi=130)
    plt.close(fig)

    # 3) example state trajectories (best nominal trial of each method)
    def pick(df, ex):
        nom = df[df["noise_sigma"] == 0.0]
        if len(nom) == 0 or len(ex) == 0:
            return None
        best = nom.loc[nom["iae_angle"].idxmin(), "trial"]
        return ex.get(int(best))

    rb, rd = pick(bc_df, bc_ex), pick(diff_df, diff_ex)
    if rb is not None and rd is not None:
        labels = ["q1 (shoulder)", "q2 (elbow)", "q1_dot", "q2_dot"]
        fig, axs = plt.subplots(2, 2, figsize=(15, 9))
        for i, ax in enumerate(axs.ravel()):
            # angles come back wrapped to [-pi, pi]; unwrap the two position
            # traces so the swing-up reads continuously against the goal at pi.
            yb = np.unwrap(rb["x"][:, i]) if i < 2 else rb["x"][:, i]
            yd = np.unwrap(rd["x"][:, i]) if i < 2 else rd["x"][:, i]
            ax.plot(rb["t"], yb, color=BC_COLOR, label="BC")
            ax.plot(rd["t"], yd, color=DIFF_COLOR, label="Diffusion")
            ax.axhline(GOAL[i], color="r", ls="--", alpha=0.6, label="goal")
            ax.set_title(labels[i]); ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle("Example state trajectories (best nominal trial)", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / "example_states.png", dpi=130)
        plt.close(fig)

        # 4) example control trajectories
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
        for j, ax in enumerate(axs):
            ax.plot(rb["t"], rb["u"][:, j], color=BC_COLOR, label="BC")
            ax.plot(rd["t"], rd["u"][:, j], color=DIFF_COLOR, label="Diffusion")
            ax.set_title(f"Control torque u{j+1}"); ax.set_xlabel("Time (s)")
            ax.set_ylabel("Nm"); ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle("Example control trajectories (best nominal trial)", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / "example_controls.png", dpi=130)
        plt.close(fig)

    print(f"Plots -> {plots_dir}")


# ── Random-initial-position stress battery ───────────────────────────────────
def run_random_init_battery(name, device, paths, out_root, n_runs, noise_sigma,
                            max_steps, dt_control, hold_steps, angle_tol, vel_tol,
                            seed, compile_diff=True):
    """
    Stress test: `n_runs` swing-ups from FULLY RANDOM initial joint positions
    (uniform in [-pi, pi]) under a fixed heavy observation noise (`noise_sigma`),
    to estimate a success rate well outside the near-hanging-down start the
    policies were trained on. Returns the per-trial metrics DataFrame.
    """
    print(f"\n=== {name.upper()} : random-init battery "
          f"(n={n_runs}, noise={noise_sigma}, device={device}) ===")
    ctrl = build_controller(name, device, paths, compile_diff=compile_diff)
    env  = DoublePendulumEnv(render_mode=None, frame_skip=1)

    rand_dir = out_root / name / "random_init"
    rand_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in range(n_runs):
        # deterministic per (method, trial, seed) so the battery is reproducible
        rng = np.random.default_rng(
            abs(hash((name, "random_init", trial, seed))) % (2**32))
        q0 = rng.uniform(-np.pi, np.pi, size=2)        # fully random start position
        v0 = rng.normal(0.0, 0.20, size=2)             # small random start velocity

        roll = rollout(env, ctrl, q0, v0, noise_sigma, rng,
                       max_steps, dt_control, hold_steps, angle_tol, vel_tol)
        m = compute_metrics(roll, dt_control, hold_steps, angle_tol, vel_tol)
        m.update({"method": name, "noise_sigma": noise_sigma, "trial": trial,
                  "q1_0": float(q0[0]), "q2_0": float(q0[1])})
        rows.append(m)

        save_trajectory_csv(rand_dir / f"trial{trial:02d}.csv", roll)
        print(f"  [{name}] random trial {trial:02d} | "
              f"q0=({q0[0]:+.2f},{q0[1]:+.2f}) success={m['success']} "
              f"IAE={m['iae_angle']:.2f}")

    env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"{name}_random_init_metrics.csv", index=False)
    sr = float(df["success"].mean())
    print(f"  [{name}] random-init success rate = {sr:.1%} "
          f"({int(df['success'].sum())}/{len(df)})  -> "
          f"{out_root / f'{name}_random_init_metrics.csv'}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="run both nets on this device (default: auto)")
    p.add_argument("--n-nominal", type=int, default=8,
                   help="trials at noise=0 (success rate & per-metric stats)")
    p.add_argument("--n-robust", type=int, default=4,
                   help="trials per non-zero noise level")
    p.add_argument("--noise-levels", type=float, nargs="+",
                   default=[0.0, 0.02, 0.03, 0.04, 0.05],
                   help="observation-noise sigmas to sweep (must include 0.0)")
    p.add_argument("--max-steps", type=int, default=500,
                   help="control steps per trial (dt_control=0.05 -> 25 s)")
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--hold-steps", type=int, default=40)
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=10,
                   help="random-initial-position stress-test runs per method")
    p.add_argument("--random-noise", type=float, default=0.025,
                   help="observation-noise sigma for the random-init battery")
    p.add_argument("--no-random-init", action="store_true",
                   help="skip the random-initial-position stress test")
    p.add_argument("--no-success-rate", action="store_true",
                   help="skip the noise-sweep battery (nominal success rate + "
                        "robustness-vs-noise + comparison summary/plots)")
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile for the diffusion model (CUDA only)")
    p.add_argument("--out", default=None,
                   help="output dir (default: double_pendulum/results/method_comparison)")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test: 2 nominal, 1 robust, 150 steps, [0,0.05]")
    args = p.parse_args()

    if args.quick:
        args.n_nominal, args.n_robust = 2, 1
        args.max_steps, args.noise_levels = 150, [0.0, 0.05]
        args.n_random = 3

    if 0.0 not in args.noise_levels:
        args.noise_levels = [0.0] + args.noise_levels

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out) if args.out else (_THIS / "results" / "method_comparison")
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"

    paths = {
        "bc_ckpt":   _REPO / "behaviour_cloning" / "bc_policy_3.pt",
        "bc_stats":  _REPO / "behaviour_cloning" / "norm_stats.json",
        "diff_ckpt": _THIS / "results" / "diffusion_policy.pt",
        "diff_stats": _THIS / "results" / "norm_stats.json",
    }

    print("Output dir:", out_root)
    print("Device    :", device)
    print("Noise sweep:", args.noise_levels,
          f"| nominal={args.n_nominal} robust/level={args.n_robust} "
          f"max_steps={args.max_steps}")

    # A checkpoint can be transiently absent (e.g. a training run is regenerating
    # it), so each method is skipped (not crashed) if its files are missing.
    ckpts = {"bc": (paths["bc_ckpt"], paths["bc_stats"]),
             "diffusion": (paths["diff_ckpt"], paths["diff_stats"])}

    # Success-rate noise-sweep battery: run 1) BC then 2) Diffusion over the same
    # battery/env/criteria, then combine into the comparison summary. Plotting is
    # deferred to the end so the headline figure can fold in the random-init
    # success rate. Toggle the whole battery off with --no-success-rate.
    dfs, exs = {}, {}
    summary = rob = None
    if not args.no_success_rate:
        for name in ("bc", "diffusion"):
            ckpt, stats = ckpts[name]
            missing = [str(p) for p in (ckpt, stats) if not Path(p).exists()]
            if missing:
                print(f"\n[skip] {name.upper()}: missing {missing}")
                continue
            dfs[name], exs[name] = run_method(
                name, device, paths, out_root,
                args.noise_levels, args.n_nominal, args.n_robust,
                args.max_steps, args.dt_control, args.hold_steps,
                args.angle_tol, args.vel_tol, args.seed,
                compile_diff=not args.no_compile)

        # combine -- only when BOTH methods ran (the comparison needs both)
        if "bc" in dfs and "diffusion" in dfs:
            summary, rob = build_summary(dfs["bc"], dfs["diffusion"], out_root)
            pd.set_option("display.float_format", lambda v: f"{v:.4g}")
            print("\n================  SUMMARY  ================")
            print(summary.to_string(index=False))
            print("\n========  SUCCESS RATE vs NOISE  =========")
            print(rob.pivot(index="noise_sigma", columns="method",
                            values="success_rate").to_string())
        else:
            ran = sorted(dfs) or ["nothing"]
            missing = [m for m in ("bc", "diffusion") if m not in dfs]
            print(f"\n[partial] Ran {ran}; per-trial CSVs saved under {out_root}.")
            print(f"[partial] The combined comparison needs BOTH methods; missing: {missing}.")
            if "diffusion" in missing:
                print("          results/diffusion_policy.pt is absent -- a diffusion "
                      "training run may be regenerating it.\n"
                      "          Re-run this script once that file exists to get the "
                      "full BC-vs-Diffusion comparison.")
    else:
        print("\n[skip] success-rate noise-sweep battery (--no-success-rate)")

    # Random-initial-position stress test: n_random runs per method from uniformly
    # random start angles at a fixed heavy noise -> success rate. Its per-method
    # rate is folded into the headline comparison figure below.
    rand_summary = None
    if not args.no_random_init:
        rand_rows = []
        for name in ("bc", "diffusion"):
            ckpt, stats = ckpts[name]
            missing = [str(p) for p in (ckpt, stats) if not Path(p).exists()]
            if missing:
                print(f"\n[skip] {name.upper()} random-init: missing {missing}")
                continue
            rdf = run_random_init_battery(
                name, device, paths, out_root,
                args.n_random, args.random_noise,
                args.max_steps, args.dt_control, args.hold_steps,
                args.angle_tol, args.vel_tol, args.seed,
                compile_diff=not args.no_compile)
            rand_rows.append({
                "method": name,
                "n_runs": len(rdf),
                "noise_sigma": args.random_noise,
                "n_success": int(rdf["success"].sum()),
                "success_rate": float(rdf["success"].mean()),
            })
        if rand_rows:
            rand_summary = pd.DataFrame(rand_rows)
            rand_summary.to_csv(out_root / "random_init_success_rate.csv", index=False)
            pd.set_option("display.float_format", lambda v: f"{v:.4g}")
            print(f"\n=====  RANDOM-INIT SUCCESS RATE  "
                  f"(n={args.n_random}, noise={args.random_noise})  =====")
            print(rand_summary.to_string(index=False))
            print(f"\nRandom-init summary -> "
                  f"{out_root / 'random_init_success_rate.csv'}")

    # Plots last: the headline figure folds in the random-init success rate when
    # both batteries produced a per-method rate (otherwise it shows nominal only).
    if summary is not None:
        rand_for_plot = (rand_summary
                         if rand_summary is not None
                         and set(rand_summary["method"]) >= {"bc", "diffusion"}
                         else None)
        make_plots(dfs["bc"], dfs["diffusion"], summary, rob,
                   exs["bc"], exs["diffusion"], plots_dir, args.dt_control,
                   rand_summary=rand_for_plot, random_noise=args.random_noise)
        print("\nDone.")


if __name__ == "__main__":
    main()
