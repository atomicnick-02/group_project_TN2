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

NOTE on cost: the diffusion policy denoises 100 steps every re-plan, so on CPU
a single rollout is ~40 s at the default --n-exec 2 (~80 s at --n-exec 1). The
full default sweep (9 conditions) is therefore minutes-to-an-hour of CPU per
controller -- use a GPU, fewer --n-rollouts, or fewer --conditions for a quick
look. TVLQR is ~0.1 ms/step and effectively free.

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
import zlib
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ── quiet the noise ───────────────────────────────────────────────────────────
# Silence library warnings (Torch AMP deprecations, dm_control/MuJoCo import
# chatter, etc.). Scoped to the main process here; _worker_init re-applies it in
# every spawned worker since spawn starts a fresh interpreter.
warnings.filterwarnings("ignore")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Make both the repo root (for diffusion_models) and this dir importable.
# This MUST run before the local `simulation` / `visualize_swingup` imports.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

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


# ── parallel workers ──────────────────────────────────────────────────────────
# Each rollout is one "job". MuJoCo's MjData and a Torch policy are NOT safe to
# share across threads, so we parallelize at the PROCESS level: every worker owns
# its own env + controller(s) and pulls jobs off the pool. A worker builds its
# env/controller ONCE (lazily) and caches them, so the expensive checkpoint load
# is paid once per process, not once per rollout.
_CFG = None          # shared run config, set by the pool initializer
_ENV = None          # this worker's MuJoCo env
_CTRL_CACHE = {}     # this worker's controllers, keyed by name


def _worker_init(cfg, torch_threads):
    """Pool initializer: store the shared config and cap Torch's own threading.

    With many worker processes already saturating the cores, letting each one
    also spin up an intra-op thread pool oversubscribes the CPU and slows things
    down -- so by default each worker runs Torch single-threaded.

    Also re-applies the warning filter: spawned workers start a fresh interpreter
    so the main-process filter does NOT carry over.
    """
    global _CFG, _ENV, _CTRL_CACHE
    warnings.filterwarnings("ignore")
    _CFG, _ENV, _CTRL_CACHE = cfg, None, {}
    if torch_threads and torch_threads > 0:
        try:
            import torch
            torch.set_num_threads(torch_threads)
            try:
                torch.set_num_interop_threads(torch_threads)
            except Exception:
                pass            # can only be set once per process
        except Exception:
            pass


def _get_env():
    global _ENV
    if _ENV is None:
        _ENV = DoublePendulumEnv(render_mode=None, frame_skip=1)
    return _ENV


def _get_ctrl(name):
    if name not in _CTRL_CACHE:
        _CTRL_CACHE[name] = make_controller(name, _CFG["ckpt"], _CFG["n_exec"])
    return _CTRL_CACHE[name]


def _run_job(job):
    """Execute one rollout. Runs in a worker process (or inline if --workers 1)."""
    cfg = _CFG
    env = _get_env()
    ctrl = _get_ctrl(job["controller"])
    max_tau = float(env.action_space.high[0])

    rng = np.random.default_rng(job["seed"])
    x0 = make_init_state(job["init_mode"], rng)
    disturb = make_disturbance(job["kind"], rng, cfg["max_steps"],
                               cfg["push_force"], cfg["jump_vel"])
    roll = run_rollout(
        env, ctrl, job["controller"], x0,
        max_steps=cfg["max_steps"], dt_control=cfg["dt_control"],
        hold_steps=cfg["hold_steps"], angle_tol=cfg["angle_tol"],
        vel_tol=cfg["vel_tol"], obs_noise_std=job["noise_std"],
        disturb=disturb, rng=rng, max_tau=max_tau)
    m = compute_metrics(roll)
    row = {"controller": job["controller"], "condition": job["cond_name"],
           "rollout": job["rollout_idx"], **m}
    example = None
    if job["keep_example"]:
        example = {"key": (job["controller"], job["cond_name"]),
                   "t": roll["t"], "x": roll["x"],
                   "disturb": roll["disturb"], "dt_control": roll["dt_control"]}
    return row, example


def _stable_seed(base, i, cond):
    """Reproducible per-(rollout, condition) seed (crc32, not hash(), so it is
    identical across runs and across worker processes)."""
    return (base + 1000 * i + zlib.crc32(cond.encode()) % 1000) & 0x7FFFFFFF


def _progress_str(n, total, row):
    """One-line per-rollout detail, routed through tqdm.write so it doesn't
    clobber the live progress bar."""
    tts = row["time_to_success_s"]
    tts_s = f"tts={tts:5.2f}s" if tts is not None else "tts=  --  "
    return (f"[{n:>4}/{total}] {row['controller']:9s} {row['condition']:12s} "
            f"#{row['rollout']:<2d} success={row['success']} {tts_s} "
            f"({row['infer_mean_ms']:.0f} ms/step)")


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
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--hold-steps", type=int, default=40)
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--n-exec", type=int, default=2,
                   help="diffusion: actions executed per re-plan (cost vs accuracy; "
                        "2 holds the top at ~2x speed, 1 is most faithful, >=3 falls)")
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
    p.add_argument("--workers", type=int, default=4,
                   help="parallel worker PROCESSES (each = its own env+policy). "
                        "1 = sequential. Try your physical core count.")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Torch threads PER worker (default: 1 when --workers>1, "
                        "else unrestricted). Keep workers*threads <= cores.")
    p.add_argument("--quiet", action="store_true",
                   help="show only the progress bar, suppress per-rollout lines")
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

    # config every worker needs (besides the per-job fields)
    cfg = {"ckpt": args.ckpt, "n_exec": args.n_exec, "max_steps": args.max_steps,
           "dt_control": args.dt_control, "hold_steps": args.hold_steps,
           "angle_tol": args.angle_tol, "vel_tol": args.vel_tol,
           "push_force": args.push_force, "jump_vel": args.jump_vel}

    # flat list of independent rollouts (the unit of parallelism)
    jobs = []
    for ctrl_name in controllers:
        for cond_name, (init_mode, kind, noise_std) in all_conditions.items():
            for i in range(args.n_rollouts):
                jobs.append({
                    "controller": ctrl_name, "cond_name": cond_name,
                    "init_mode": init_mode, "kind": kind, "noise_std": noise_std,
                    "rollout_idx": i, "seed": _stable_seed(args.seed, i, cond_name),
                    "keep_example": (cond_name in base_specs and i == 0),
                })

    workers = max(1, args.workers)
    tthreads = args.torch_threads
    if tthreads is None:
        tthreads = 1 if workers > 1 else 0          # 0 = leave Torch default
    total = len(jobs)
    print(f"running {total} rollouts on {workers} worker(s) "
          f"({tthreads or 'default'} torch thread(s) each)\n")

    results = []
    t_start = time.time()
    if workers <= 1:
        _worker_init(cfg, tthreads)                 # set up the single process
        bar = tqdm(jobs, total=total, desc="rollouts", unit="roll")
        for job in bar:
            row, ex = _run_job(job)
            results.append((row, ex))
            if not args.quiet:
                tqdm.write(_progress_str(len(results), total, row))
    else:
        # spawn (not fork): a clean interpreter per worker avoids Torch/OpenMP
        # state being copied across a fork, which can deadlock.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(cfg, tthreads)) as ex_pool:
            bar = tqdm(ex_pool.map(_run_job, jobs, chunksize=1),
                       total=total, desc="rollouts", unit="roll")
            for row, ex in bar:
                results.append((row, ex))
                if not args.quiet:
                    tqdm.write(_progress_str(len(results), total, row))

    # ── reduce: per-rollout rows, example trajectories, per-condition aggregates ─
    per_rollout_rows = [row for row, _ in results]
    examples = {ex["key"]: ex for _, ex in results if ex is not None}

    grouped = defaultdict(list)
    for row in per_rollout_rows:
        grouped[(row["controller"], row["condition"])].append(row)

    summary = {c: {} for c in controllers}
    for ctrl_name in controllers:
        print(f"\n=== controller: {ctrl_name} ===")
        for cond_name in all_conditions:
            agg = aggregate(grouped[(ctrl_name, cond_name)])
            summary[ctrl_name][cond_name] = agg
            print(f"  {cond_name:14s} | success {agg['success_rate']*100:5.1f}% "
                  f"| held {agg['held_fraction_mean']:.2f} "
                  f"| smooth(|Δu|) {agg['control_tv_mean']:.4f} "
                  f"| {agg['infer_mean_ms_mean']:.2f} ms/step")
    print(f"\nwall time: {time.time() - t_start:.1f}s")

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