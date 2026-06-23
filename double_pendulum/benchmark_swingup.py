"""
Headless batch evaluation of swing-up controllers under disturbances.

This implements section 10 (Diffusion-Based Control from Demonstrations), Part B
item (x): run the policy in the environment and

    "Record success rate, trajectory quality, stability, smoothness,
     robustness to noise, and computational cost."

It runs MANY MuJoCo rollouts with NO viewer (render_mode=None), across several
disturbance conditions and random seeds, and writes aggregate metrics + plots.

Conditions evaluated (each over --n-rollouts seeds):
    clean        : swing up from hanging-down, no disturbance.
    push         : an external force impulse is applied to the pendulum TIP at a
                   random time (pushing it mid-swing / mid-hold).
    random_init  : the episode STARTS from a random point in state space.
    state_jump   : mid-episode the state is teleported to a random configuration
                   (forcing the controller to recover from an arbitrary state).
    noise_<std>  : Gaussian observation noise of the given std is injected into
                   the state the controller sees (robustness-to-noise sweep).

Controllers (reused verbatim from visualize_swingup.py):
    diffusion    : trained Diffusion Policy checkpoint.
    tvlqr        : TVLQR trajectory-tracking baseline.

Run from the repo root (school-env has mujoco + torch):
    python double_pendulum/benchmark_swingup.py --controller diffusion
    python double_pendulum/benchmark_swingup.py --controller both --n-rollouts 20

Outputs go to double_pendulum/graphs/evaluation/benchmark/:
    benchmark_per_rollout.csv   one row per rollout (every raw metric)
    benchmark_summary.json      per-condition / per-controller aggregates
    *.png                       success-rate, robustness, smoothness, examples
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

# Make both the repo root (for diffusion_models) and this dir importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import DoublePendulumEnv
# Reuse the EXACT controllers used by the interactive viewer so the headless
# numbers correspond to what you watch on screen.
from visualize_swingup import DiffusionController, TVLQRController, wrap_to_pi

results_dir = _HERE / "results"
OUT_DIR = _HERE / "graphs" / "evaluation" / "benchmark"

GOAL = np.array([np.pi, 0.0, 0.0, 0.0])          # upright


# ── controller construction ──────────────────────────────────────────────────
def make_controller(name, ckpt, n_exec):
    if name == "tvlqr":
        c = TVLQRController()
        c.reset()
        return c
    c = DiffusionController(results_dir / ckpt,
                            results_dir / "norm_stats.json",
                            n_exec=n_exec)
    return c


def reset_controller(ctrl, name, x0):
    if name == "diffusion":
        ctrl.reset(x0)
    else:
        ctrl.reset()


# ── disturbance schedule ──────────────────────────────────────────────────────
def make_init_state(init_mode, rng):
    """Return the (qpos, qvel) the episode starts from."""
    if init_mode == "random":
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-4.0, 4.0, size=2)
        return np.array([q[0], q[1], qd[0], qd[1]])
    return np.array([0.0, 0.0, 0.0, 0.0])         # hanging down


def make_disturbance(kind, rng, max_steps, push_force, jump_vel):
    """
    Build a per-rollout disturbance description.

    push -> a force impulse on the tip, active over [step, step+dur) control
            steps, direction uniform in the XZ swing-plane.
    jump -> a one-shot state teleport at `step`.
    """
    if kind == "push":
        step = int(rng.uniform(0.30, 0.80) * max_steps)
        dur = 1                                     # control steps the force is held
        theta = rng.uniform(0.0, 2 * np.pi)
        mag = push_force * rng.uniform(0.7, 1.3)
        force = np.array([mag * np.cos(theta), 0.0, mag * np.sin(theta)])
        return {"type": "push", "step": step, "dur": dur,
                "force": force, "mag": float(mag)}
    if kind == "jump":
        step = int(rng.uniform(0.40, 0.80) * max_steps)
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-jump_vel, jump_vel, size=2)
        return {"type": "jump", "step": step,
                "state": np.array([q[0], q[1], qd[0], qd[1]])}
    return None


# ── one headless rollout ──────────────────────────────────────────────────────
def run_rollout(env, ctrl, ctrl_name, x0, *, max_steps, dt_control, hold_steps,
                angle_tol, vel_tol, obs_noise_std, disturb, rng, max_tau):
    """
    Step the MuJoCo plant by hand (no viewer), exactly mirroring the physics loop
    in visualize_swingup.evaluate(), while injecting the requested disturbance.

    Success / metrics are evaluated on the CLEAN simulator state; observation
    noise only corrupts the state handed to the controller (robustness test).

    Returns a dict with the time series and per-step inference timings.
    """
    # Reset plant to x0.
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:2] = x0[:2]
    env.data.qvel[:2] = x0[2:]
    mujoco.mj_forward(env.model, env.data)
    reset_controller(ctrl, ctrl_name, x0)

    tip_id = env.model.nbody - 1
    dt_sim = env.model.opt.timestep
    n_sub = max(1, int(round(dt_control / dt_sim)))

    ts, xs, us, infer_ms = [], [], [], []
    consecutive_hold = 0
    success_step = None
    reach_step = None

    for step in range(max_steps):
        # apply a state-jump disturbance just before observing
        if disturb and disturb["type"] == "jump" and step == disturb["step"]:
            env.data.qpos[:2] = disturb["state"][:2]
            env.data.qvel[:2] = disturb["state"][2:]
            mujoco.mj_forward(env.model, env.data)

        x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])

        # observation handed to the controller (optionally noisy)
        x_obs = x.copy()
        if obs_noise_std > 0.0:
            x_obs = x_obs + rng.normal(0.0, obs_noise_std, size=4)

        t0 = time.perf_counter()
        u = np.asarray(ctrl.action(x_obs), dtype=np.float64).ravel()
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        u = np.clip(u, -max_tau, max_tau)

        ts.append(step * dt_control)
        xs.append(x.copy())
        us.append(u.copy())

        # success bookkeeping on the CLEAN state
        ang_err = abs(wrap_to_pi(x[0] - GOAL[0])) + abs(wrap_to_pi(x[1] - GOAL[1]))
        vel_err = abs(x[2]) + abs(x[3])
        if reach_step is None and ang_err < 0.30:
            reach_step = step
        if ang_err < angle_tol and vel_err < vel_tol:
            consecutive_hold += 1
            if consecutive_hold >= hold_steps and success_step is None:
                success_step = step
        else:
            consecutive_hold = 0

        # is a push active on this control step?
        push_force = None
        if disturb and disturb["type"] == "push":
            if disturb["step"] <= step < disturb["step"] + disturb["dur"]:
                push_force = disturb["force"]

        env.data.ctrl[:] = u
        for _ in range(n_sub):
            env.data.xfrc_applied[:] = 0.0
            if push_force is not None:
                env.data.xfrc_applied[tip_id, :3] = push_force
            mujoco.mj_step(env.model, env.data)
        env.data.xfrc_applied[:] = 0.0

    return {
        "t": np.array(ts), "x": np.array(xs), "u": np.array(us),
        "infer_ms": np.array(infer_ms),
        "success_step": success_step, "reach_step": reach_step,
        "dt_control": dt_control, "hold_steps": hold_steps,
        "angle_tol": angle_tol, "vel_tol": vel_tol,
        "disturb": disturb,
    }


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(roll):
    """Reduce one rollout to the six metric families the project asks for."""
    t, x, u = roll["t"], roll["x"], roll["u"]
    dt = roll["dt_control"]
    ang_tol, vel_tol = roll["angle_tol"], roll["vel_tol"]

    ang_err = np.abs(wrap_to_pi(x[:, 0] - GOAL[0])) + np.abs(wrap_to_pi(x[:, 1] - GOAL[1]))
    vel_mag = np.abs(x[:, 2]) + np.abs(x[:, 3])

    success = roll["success_step"] is not None
    reached = roll["reach_step"] is not None

    # --- trajectory quality: how close to the goal, on average / at the end ---
    last_window = max(1, int(round(1.0 / dt)))                 # final ~1 s
    mean_ang_err = float(np.mean(ang_err))
    rms_ang_err = float(np.sqrt(np.mean(ang_err ** 2)))
    final_ang_err = float(np.mean(ang_err[-last_window:]))

    # --- stability: once upright, did it STAY up? ---
    if reached:
        post = slice(roll["reach_step"], None)
        in_tol = (ang_err[post] < ang_tol) & (vel_mag[post] < vel_tol)
        held_fraction = float(np.mean(in_tol))
        post_max_dev = float(np.max(ang_err[post]))
    else:
        held_fraction = 0.0
        post_max_dev = float(np.max(ang_err)) if len(ang_err) else float("nan")

    # --- smoothness: variation / jerk of the commanded torque ---
    if len(u) > 1:
        du = np.diff(u, axis=0)
        control_tv = float(np.mean(np.sum(np.abs(du), axis=1)))      # mean |Δu|_1 / step
    else:
        control_tv = 0.0
    if len(u) > 2:
        ddu = np.diff(u, n=2, axis=0)
        control_jerk = float(np.mean(np.sum(np.abs(ddu), axis=1)))
    else:
        control_jerk = 0.0
    control_effort = float(np.mean(np.sum(np.abs(u), axis=1)))       # mean |u|_1

    # --- computational cost: per-step controller inference time ---
    im = roll["infer_ms"]
    infer_mean_ms = float(np.mean(im))
    infer_p95_ms = float(np.percentile(im, 95))
    steps_per_s = float(1000.0 / infer_mean_ms) if infer_mean_ms > 0 else float("inf")

    return {
        # success rate
        "success": int(success),
        "time_to_success_s": float(roll["success_step"] * dt) if success else None,
        "reached_top": int(reached),
        # trajectory quality
        "mean_ang_err": mean_ang_err,
        "rms_ang_err": rms_ang_err,
        "final_ang_err": final_ang_err,
        # stability
        "held_fraction": held_fraction,
        "post_reach_max_dev": post_max_dev,
        # smoothness
        "control_tv": control_tv,
        "control_jerk": control_jerk,
        "control_effort": control_effort,
        # computational cost
        "infer_mean_ms": infer_mean_ms,
        "infer_p95_ms": infer_p95_ms,
        "steps_per_s": steps_per_s,
    }


# ── aggregation helpers ───────────────────────────────────────────────────────
def aggregate(rows):
    """Mean/std (and success rate) over a list of per-rollout metric dicts."""
    keys = ["success", "reached_top", "mean_ang_err", "rms_ang_err",
            "final_ang_err", "held_fraction", "post_reach_max_dev",
            "control_tv", "control_jerk", "control_effort",
            "infer_mean_ms", "infer_p95_ms", "steps_per_s"]
    out = {"n": len(rows)}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    out["success_rate"] = float(np.mean([r["success"] for r in rows]))
    tts = [r["time_to_success_s"] for r in rows if r["time_to_success_s"] is not None]
    out["time_to_success_s_mean"] = float(np.mean(tts)) if tts else None
    return out


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(summary, examples, controllers, base_conditions, noise_levels):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"diffusion": "tab:blue", "tvlqr": "tab:orange"}

    # 1. success rate by (base) condition
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(controllers))
    xpos = np.arange(len(base_conditions))
    for ci, ctrl in enumerate(controllers):
        rates = [summary[ctrl][c]["success_rate"] for c in base_conditions]
        ax.bar(xpos + ci * width, rates, width, label=ctrl,
               color=colors.get(ctrl))
    ax.set_xticks(xpos + width * (len(controllers) - 1) / 2)
    ax.set_xticklabels(base_conditions, rotation=15)
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by disturbance condition"); ax.legend(); ax.grid(True, axis="y")
    fig.tight_layout(); fig.savefig(OUT_DIR / "success_rate_by_condition.png", dpi=130)
    plt.close(fig)

    # 2. robustness to noise (success rate vs obs-noise std)
    if noise_levels:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ctrl in controllers:
            rates = [summary[ctrl][f"noise_{s:g}"]["success_rate"] for s in noise_levels]
            ax.plot(noise_levels, rates, "o-", label=ctrl, color=colors.get(ctrl))
        ax.set_xlabel("observation noise std"); ax.set_ylabel("success rate")
        ax.set_ylim(0, 1.05); ax.set_title("Robustness to observation noise")
        ax.legend(); ax.grid(True)
        fig.tight_layout(); fig.savefig(OUT_DIR / "robustness_noise.png", dpi=130)
        plt.close(fig)

    # 3. smoothness / stability / cost by condition
    panels = [("control_tv_mean", "smoothness  (mean |Δu|/step, lower=better)"),
              ("post_reach_max_dev_mean", "instability  (max dev after reaching top)"),
              ("infer_mean_ms_mean", "compute cost  (ms / control step)")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axs, panels):
        for ci, ctrl in enumerate(controllers):
            vals = [summary[ctrl][c][key] for c in base_conditions]
            ax.bar(xpos + ci * width, vals, width, label=ctrl, color=colors.get(ctrl))
        ax.set_xticks(xpos + width * (len(controllers) - 1) / 2)
        ax.set_xticklabels(base_conditions, rotation=20)
        ax.set_title(title); ax.grid(True, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "metrics_by_condition.png", dpi=130)
    plt.close(fig)

    # 4. example trajectories (angle error vs time), one per base condition
    fig, axs = plt.subplots(1, len(base_conditions),
                            figsize=(4 * len(base_conditions), 4), squeeze=False)
    for j, cond in enumerate(base_conditions):
        ax = axs[0][j]
        for ctrl in controllers:
            roll = examples.get((ctrl, cond))
            if roll is None:
                continue
            x = roll["x"]
            ang_err = np.abs(wrap_to_pi(x[:, 0] - GOAL[0])) + np.abs(wrap_to_pi(x[:, 1] - GOAL[1]))
            ax.plot(roll["t"], ang_err, label=ctrl, color=colors.get(ctrl))
            if roll["disturb"]:
                ax.axvline(roll["disturb"]["step"] * roll["dt_control"],
                           color="r", ls=":", alpha=0.6)
        ax.set_title(cond); ax.set_xlabel("t (s)"); ax.set_ylabel("angle error (rad)")
        ax.grid(True); ax.legend()
    fig.suptitle("Example rollouts: angle error to upright (red = disturbance)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "example_trajectories.png", dpi=130)
    plt.close(fig)

    print(f"[plots] saved to {OUT_DIR}")


# ── main driver ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=["diffusion", "tvlqr", "both"],
                   default="diffusion")
    p.add_argument("--ckpt", default="diffusion_policy.pt",
                   help="checkpoint filename inside results/ (diffusion)")
    p.add_argument("--n-rollouts", type=int, default=10,
                   help="rollouts (random seeds) per condition")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--hold-steps", type=int, default=40)
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--n-exec", type=int, default=4,
                   help="diffusion: actions executed per re-plan (cost vs accuracy)")
    p.add_argument("--push-force", type=float, default=1.0,
                   help="nominal tip-push force magnitude (N)")
    p.add_argument("--jump-vel", type=float, default=4.0,
                   help="max |velocity| of a random state-jump (rad/s)")
    p.add_argument("--noise-levels", type=float, nargs="*",
                   default=[0.0, 0.02, 0.05, 0.10, 0.20],
                   help="obs-noise stds for the robustness sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="*",
                   default=["clean", "push", "random_init", "state_jump"],
                   help="disturbance conditions to run (besides the noise sweep)")
    args = p.parse_args()

    controllers = ["diffusion", "tvlqr"] if args.controller == "both" else [args.controller]

    # map a condition name -> (init_mode, disturbance_kind, obs_noise_std)
    base_specs = {
        "clean":       ("bottom", None,   0.0),
        "push":        ("bottom", "push", 0.0),
        "random_init": ("random", None,   0.0),
        "state_jump":  ("bottom", "jump", 0.0),
    }
    base_conditions = [c for c in args.conditions if c in base_specs]
    noise_conditions = {f"noise_{s:g}": ("bottom", None, s) for s in args.noise_levels}
    all_conditions = {**{c: base_specs[c] for c in base_conditions}, **noise_conditions}

    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    max_tau = float(env.action_space.high[0])

    summary = {}
    per_rollout_rows = []
    examples = {}                       # (controller, base_condition) -> one rollout

    for ctrl_name in controllers:
        print(f"\n=== controller: {ctrl_name} ===")
        ctrl = make_controller(ctrl_name, args.ckpt, args.n_exec)
        summary[ctrl_name] = {}

        for cond_name, (init_mode, kind, noise_std) in all_conditions.items():
            rows = []
            for i in range(args.n_rollouts):
                # deterministic, condition-independent seed per rollout index
                rng = np.random.default_rng(
                    args.seed + 1000 * i + hash(cond_name) % 1000)
                x0 = make_init_state(init_mode, rng)
                disturb = make_disturbance(kind, rng, args.max_steps,
                                           args.push_force, args.jump_vel)
                roll = run_rollout(
                    env, ctrl, ctrl_name, x0,
                    max_steps=args.max_steps, dt_control=args.dt_control,
                    hold_steps=args.hold_steps, angle_tol=args.angle_tol,
                    vel_tol=args.vel_tol, obs_noise_std=noise_std,
                    disturb=disturb, rng=rng, max_tau=max_tau)
                m = compute_metrics(roll)
                rows.append(m)
                per_rollout_rows.append(
                    {"controller": ctrl_name, "condition": cond_name,
                     "rollout": i, **m})
                if cond_name in base_specs and (ctrl_name, cond_name) not in examples:
                    examples[(ctrl_name, cond_name)] = roll

            agg = aggregate(rows)
            summary[ctrl_name][cond_name] = agg
            print(f"  {cond_name:14s} | success {agg['success_rate']*100:5.1f}% "
                  f"| held {agg['held_fraction_mean']:.2f} "
                  f"| smooth(|Δu|) {agg['control_tv_mean']:.4f} "
                  f"| {agg['infer_mean_ms_mean']:.2f} ms/step")

    env.close()

    # ── write artifacts ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "benchmark_summary.json", "w") as fp:
        json.dump({"args": vars(args), "summary": summary}, fp, indent=2)
    if per_rollout_rows:
        with open(OUT_DIR / "benchmark_per_rollout.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(per_rollout_rows[0].keys()))
            w.writeheader(); w.writerows(per_rollout_rows)

    plot_results(summary, examples, controllers, base_conditions, args.noise_levels)
    print(f"\n[done] summary + csv + plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
