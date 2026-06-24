"""
Generate a Diffusion-Policy training set by ROLLING OUT the TVLQR controllers
produced by kux_generation.py and logging (observed_state, commanded_torque) at
the control rate.

DIFFERENCE FROM generate_tvlqr_dataset.py:
    generate_tvlqr_dataset.py rolls out a SINGLE committed reference
    (optimal_trajectories/working_optimal_trajectory/). This script instead
    sweeps EVERY (Q, R, x_goal) reference kux_generation.py wrote under
    optimal_trajectories/ -- each folder holds a trajectory.csv / inputs.csv /
    K_matrix.npy / config.json -- and rolls each one out. The resulting dataset
    therefore mixes swing-ups produced by many different cost weightings and even
    different goals (e.g. [pi, 0] vs [pi, pi]). For EVERY trajectory group we tag
    the metadata with that reference's Q, R and x_goal, so a downstream
    (goal/cost)-conditioned policy knows which expert generated each sample.

WHICH MODEL (and why): identical to generate_tvlqr_dataset.py -- every (state,
    action) pair is rolled out through rollout_tvlqr.rollout_tvlqr in the
    notebook's system-identified double-pendulum model, so each pair is
    dynamically valid for that model BY CONSTRUCTION.

DAgger detail: on a fraction of the swing-up rollouts we inject state noise for
    coverage of the off-nominal states the controller must recover from, but we
    LOG the CLEAN commanded torque as the supervised target. The noise latches
    OFF once the rollout first reaches the goal (noise_until_upright=True), so the
    fragile hold stays clean and the rollout still succeeds.

BOTH SIDES: as in generate_tvlqr_dataset.py we exploit the plant's left-right
    reflection symmetry -- (x(t), u(t)) valid => (-x(t), -u(t)) valid, reusing the
    same TVLQR gains K -- to ALSO roll out each mirrored reference, swinging up to
    the SAME goal from the other side. Episodes alternate sides so the kept set is
    left-right balanced; each rollout records a "side" attribute (+1/-1).

Output: a single results .h5 with groups traj_*, each holding states (T, 4) and
    actions (T, 2) -- the SAME format train_diffusion_policy.py consumes -- plus
    per-group attrs Q (2,)/(4,), R (2,), xgoal (4,), and the source config folder
    name. Group indices are GLOBAL across all configs, so traj_0..traj_{N-1} span
    the whole sweep.

Each reference is self-checked (and its mirror, when mirroring) before use: a
    reference that no longer swings up and holds in the notebook model is SKIPPED
    with a warning rather than aborting the whole sweep.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py

# `rollout_tvlqr` lives in double_pendulum/ (one level up from this generators/ dir).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rollout_tvlqr import P, rollout_tvlqr, lqr_gain

current_dir = Path(__file__).resolve().parent
opt_traj_root = current_dir.parent / "optimal_trajectories"

# Control rate: ONE reference node + gain per step. Must stay 0.05 s -- the kux
# references are 41 nodes at 0.05 s spacing and K is the discrete gain for a 0.05 s
# step. For a finer *integration* step (which makes these references trackable),
# use --dt-sim, which sub-steps RK4 while still commanding/logging at this rate.
DT_CTRL = 0.05


def discover_configs(root):
    """Every kux_generation.py output folder under `root`.

    A folder qualifies when it holds the full reference quartet
    (trajectory.csv, inputs.csv, K_matrix.npy, config.json). The committed
    working_optimal_trajectory/ folder has no config.json, so it is skipped.
    """
    refs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        needed = ["trajectory.csv", "inputs.csv", "K_matrix.npy", "config.json"]
        if all((d / n).exists() for n in needed):
            refs.append(d)
    return refs


def load_reference(ref_dir):
    """Load one kux reference: (x_ref, u_ref, K, Q, R, x_goal)."""
    x_ref = np.loadtxt(ref_dir / "trajectory.csv", delimiter=",", skiprows=1)
    u_ref = np.loadtxt(ref_dir / "inputs.csv", delimiter=",", skiprows=1)
    K = np.load(ref_dir / "K_matrix.npy")
    with open(ref_dir / "config.json") as fh:
        cfg = json.load(fh)
    Q = np.asarray(cfg["Q_diag"], dtype=np.float64)
    R = np.asarray(cfg["R_diag"], dtype=np.float64)
    xgoal = np.asarray(cfg["x_goal"], dtype=np.float64)
    return x_ref, u_ref, K, Q, R, xgoal


def mirror(x_ref, u_ref):
    """Reflect a reference across the plant's left-right symmetry: (-x, -u).

    For this symmetric plant (cos/sin kinematics, friction odd in velocity)
    f(-x, -u) = -f(x, u), so (x(t), u(t)) valid implies (-x(t), -u(t)) valid -- a
    swing-up to the SAME goal from the other side. The TVLQR gains K are reused
    unchanged: A, B are identical at the mirrored point, so K is the same.
    """
    return -np.asarray(x_ref), -np.asarray(u_ref)


def sample_x0(rng):
    """Start near hanging-down with a spread for swing-up coverage."""
    return np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                     rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])


def sample_goal_x0(rng, xgoal):
    """
    Start PERTURBED around this reference's GOAL equilibrium. Rolling out the
    controller from here records the deviation->corrective-torque map -- i.e. the
    stabilizing gain. Without it the policy only ever sees the exact fixed point
    and never learns how to RECOVER from small deviations, so it can't balance.
    """
    return np.asarray(xgoal, dtype=np.float64) + np.array(
        [rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
         rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])


def make_out_path(mode, mirror_on):
    """Output filename records the dataset's composition.

    e.g. expert_trajectories_kux_swingup_hold_mirrored.h5
    """
    content = "swingup" if mode == "swingup" else "swingup_hold"
    mir = "mirrored" if mirror_on else "singleside"
    return opt_traj_root / f"expert_trajectories_kux_{content}_{mir}.h5"


def write_traj(f, idx, s, a, *, Q, R, xgoal, config, x0, noise_std, side, kind):
    """Write one rollout as group traj_{idx}, tagged with its reference's metadata."""
    grp = f.create_group(f"traj_{idx}")
    grp.create_dataset("states",  data=s.astype(np.float64))
    grp.create_dataset("actions", data=a.astype(np.float64))
    # The requested per-trajectory expert identity: which (Q, R, x_goal) produced it.
    grp.attrs["Q"] = np.asarray(Q, dtype=np.float64)
    grp.attrs["R"] = np.asarray(R, dtype=np.float64)
    grp.attrs["xgoal"] = np.asarray(xgoal, dtype=np.float64)
    grp.attrs["config"] = config
    grp.attrs["x0"] = x0
    grp.attrs["noise_std"] = noise_std
    grp.attrs["side"] = side
    grp.attrs["kind"] = kind


def generate_for_config(f, kept, ref_dir, args, rng):
    """Roll out one kux reference into the open file `f`, starting at group `kept`.

    Returns (new_kept, n_swing_kept, n_hold_kept, mirror_used) or None if the
    reference fails its swing-up-and-hold self-check (the config is skipped).
    """
    x_ref, u_ref, K, Q, R, xgoal = load_reference(ref_dir)
    name = ref_dir.name

    # ── Stabilizing hold gain: the saved K is finite-horizon and ends with the
    #    swing-up, so it cannot hold the (unstable) inverted goal past the last
    #    node. Solve the infinite-horizon LQR at the goal once and hand it to
    #    every rollout below as K_hold so the upright hold actually persists. By
    #    the plant's left-right symmetry the SAME gain stabilizes the mirrored
    #    goal -xgoal, so one K_hold serves both sides. The inverted hold is not
    #    stabilizable at the 20 Hz control rate, so rollout re-closes this loop at
    #    dt_sim -- the gain is therefore designed for dt_sim, not DT_CTRL. ──
    K_hold = lqr_gain(xgoal, Q, R, dt=args.dt_sim)

    # ── Self-check: does this reference still swing up & hold in the notebook
    #    model? Measure success against the reference's OWN goal. ──
    _, _, ok = rollout_tvlqr(x_ref, u_ref, K, np.zeros(4), args.episode_steps,
                             dt_sim=args.dt_sim, x_goal=xgoal, K_hold=K_hold)
    if not ok:
        print(f"  [skip] {name}: reference does not swing up & hold (self-check).")
        return None

    # ── Mirror self-check: only when mirroring. If the mirrored reference fails,
    #    fall back to a single-side dataset for THIS config. ──
    mirror_used = args.mirror
    if mirror_used:
        x_ref_m, u_ref_m = mirror(x_ref, u_ref)
        _, _, ok_m = rollout_tvlqr(x_ref_m, u_ref_m, K, np.zeros(4),
                                   args.episode_steps, dt_sim=args.dt_sim,
                                   x_goal=xgoal, K_hold=K_hold)
        if not ok_m:
            print(f"  [warn] {name}: mirrored reference fails self-check; "
                  "using single side for this config.")
            mirror_used = False

    swing_sides = {1: 0, -1: 0}
    hold_sides = {1: 0, -1: 0}

    # ── Swing-up rollouts ──
    for ep in range(args.n_episodes):
        if mirror_used:
            side = 1 if ep % 2 == 0 else -1
            clean_key = ep // 2          # key clean cadence on the side-pair index
        else:
            side = 1
            clean_key = ep
        xr, ur = (x_ref, u_ref) if side == 1 else mirror(x_ref, u_ref)
        goal_side = xgoal if side == 1 else -xgoal

        x0 = np.zeros(4) if ep == 0 else sample_x0(rng)
        if side == -1:
            x0 = -x0                     # mirror the start to match the ref
        noise = None if clean_key % 3 == 0 else args.noise_std
        s, a, ok = rollout_tvlqr(
            xr, ur, K, x0, args.episode_steps,
            dt_control=DT_CTRL, dt_sim=args.dt_sim, noise_std=noise, rng=rng,
            noise_until_upright=True, x_goal=goal_side, K_hold=K_hold)
        if not ok:
            continue                     # keep only successful rollouts
        write_traj(f, kept, s, a, Q=Q, R=R, xgoal=xgoal, config=name,
                   x0=x0, noise_std=0.0 if noise is None else noise,
                   side=side, kind="swingup")
        kept += 1
        swing_sides[side] += 1
    n_swing = swing_sides[1] + swing_sides[-1]

    # ── Upright/goal-hold rollouts: teach the stabilizing gain around the goal. ──
    n_hold = 0
    if args.mode == "both":
        for i in range(args.n_hold):
            side = (1 if i % 2 == 0 else -1) if mirror_used else 1
            xr, ur = (x_ref, u_ref) if side == 1 else mirror(x_ref, u_ref)
            goal_side = xgoal if side == 1 else -xgoal
            x0 = sample_goal_x0(rng, xgoal)
            if side == -1:
                x0 = -x0
            s, a, ok = rollout_tvlqr(xr, ur, K, x0, args.hold_steps,
                                     dt_control=DT_CTRL, dt_sim=args.dt_sim,
                                     x_goal=goal_side, K_hold=K_hold)
            if not ok:
                continue
            write_traj(f, kept, s, a, Q=Q, R=R, xgoal=xgoal, config=name,
                       x0=x0, noise_std=0.0, side=side, kind="hold")
            kept += 1
            hold_sides[side] += 1
        n_hold = hold_sides[1] + hold_sides[-1]

    print(f"  [ok]   {name}: {n_swing} swing-up (+{swing_sides[1]}/-{swing_sides[-1]})"
          f" + {n_hold} hold (+{hold_sides[1]}/-{hold_sides[-1]})"
          + ("" if mirror_used else "  [single-side]"))
    return kept, n_swing, n_hold, mirror_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=300,
                    help="swing-up rollouts attempted PER reference config")
    ap.add_argument("--mode", choices=["swingup", "both"], default="both",
                    help="'swingup' = swing-up rollouts only; 'both' = swing-up"
                         " rollouts plus the goal-hold rollouts")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="include mirrored (-x_ref, -u_ref) trajectories so the"
                         " policy swings up from BOTH sides; --no-mirror keeps only"
                         " each reference's own side")
    ap.add_argument("--n-hold", type=int, default=100,
                    help="goal-hold rollouts attempted PER reference config"
                         " (mode=both only); started perturbed around x_goal")
    ap.add_argument("--hold-steps", type=int, default=60,   # 60*0.05 = 3 s
                    help="length of each goal-hold rollout")
    ap.add_argument("--episode-steps", type=int, default=120)   # 120*0.05 = 6 s
    ap.add_argument("--dt-sim", type=float, default=0.005,
                    help="inner RK4 integration step (control stays at DT_CTRL=0.05"
                         " s; targets/actions are still logged at the control rate)."
                         " kux swing-ups are violent (q2_dot > 40 rad/s), so a 0.05"
                         " s step destabilizes closed-loop tracking; 0.005 s"
                         " sub-stepping makes the smoother references trackable."
                         " Pass 0.05 to integrate at the control rate.")
    ap.add_argument("--noise-std", type=float, default=0.05,
                    help="swing-up-phase state-noise std (DAgger coverage); latches"
                         " off once the goal is reached so the hold stays clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs-root", default=None,
                    help="folder of kux_generation.py reference folders"
                         " (default: double_pendulum/optimal_trajectories)")
    ap.add_argument("--out", default=None,
                    help="output .h5 path; if omitted, derived from --mode/--mirror"
                         " (e.g. expert_trajectories_kux_swingup_hold_mirrored.h5)")
    args = ap.parse_args()

    root = Path(args.configs_root) if args.configs_root else opt_traj_root
    out_path = Path(args.out) if args.out else make_out_path(args.mode, args.mirror)

    refs = discover_configs(root)
    if not refs:
        raise SystemExit(
            f"ABORT: no kux reference folders found under {root}.\n"
            "       Run generators/kux_generation.py first to produce the\n"
            "       (Q, R, x_goal) trajectory folders this script rolls out.")
    print(f"Found {len(refs)} reference config(s) under {root}; "
          f"torque_limit={P.torque_limit} Nm.")
    print(f"Mode={args.mode}, mirror={args.mirror}, "
          f"{args.n_episodes} swing-up + "
          f"{args.n_hold if args.mode == 'both' else 0} hold attempts per config.\n")

    rng = np.random.default_rng(args.seed)
    kept = 0
    n_swing_total = n_hold_total = 0
    used_configs = 0
    with h5py.File(out_path, "w") as f:
        for ref_dir in refs:
            res = generate_for_config(f, kept, ref_dir, args, rng)
            if res is None:
                continue
            kept, n_swing, n_hold, _ = res
            n_swing_total += n_swing
            n_hold_total += n_hold
            used_configs += 1

    n_swing_samples = n_swing_total * args.episode_steps
    n_hold_samples = n_hold_total * args.hold_steps
    print(f"\nDone! Saved {kept} rollouts from {used_configs}/{len(refs)} configs "
          f"to {out_path}")
    print(f"  ({n_swing_total} swing-up + {n_hold_total} hold).")
    print(f"~{n_swing_samples + n_hold_samples} (state, action) samples "
          f"(~{n_hold_samples} from the hold regime).")


if __name__ == "__main__":
    main()
