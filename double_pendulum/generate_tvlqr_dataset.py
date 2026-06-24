"""
Generate the Diffusion-Policy training set by ROLLING OUT the working TVLQR
swing-up controller in the NOTEBOOK's double-pendulum model and logging
(observed_state, commanded_torque) at the control rate.

WHICH MODEL (and why it changed):
    The expert trajectories are generated for the SAME system the cloudpendulum
    notebook defines -- the system-identified, distributed-inertia 2-link
    manipulator with viscous + arctan-smoothed Coulomb friction
    (dp_fwd_inv_dynamics/fwd_inv_dyn_student.ipynb). That model, its parameters,
    and the RK4 integrator live in generate_k.py; we roll out through
    generate_k.rollout_tvlqr, so every (state, action) pair is dynamically valid
    for the notebook model BY CONSTRUCTION.

DAgger detail: on a fraction of the swing-up rollouts we inject state noise for
    coverage of the off-nominal states the controller must recover from, but we
    LOG the CLEAN commanded torque as the supervised target. The noise latches
    OFF once the rollout first reaches upright (noise_until_upright=True), so the
    fragile hold stays clean and the rollout still succeeds -- we keep the noisy
    swing-up recovery states without sacrificing the hold.

BOTH SIDES: the committed reference swings up in ONE rotational direction, so on
    its own it teaches a left/right-biased policy. We exploit the plant's
    left-right reflection symmetry -- for this pendulum (cos/sin kinematics,
    friction odd in velocity) (x(t), u(t)) valid => (-x(t), -u(t)) valid -- to
    ALSO roll out the mirrored reference (-x_ref, -u_ref) from mirrored starts,
    swinging up to the SAME upright from the other side. The TVLQR gains K are
    reused unchanged: linearizing about -x_ref gives the same (A, B), hence the
    same Riccati solution. Episodes alternate between the two sides so the kept
    set is left-right balanced; each rollout records a "side" attribute (+1/-1).

Output: results/expert_trajectories_*.h5 with groups traj_*, each holding
    states  (T, 4) and actions (T, 2)  -- the SAME format
    train_diffusion_policy.py already consumes, so nothing downstream changes.
    The filename records the dataset's composition: --mode {swingup,both} chooses
    swing-up only vs swing-up + upright-hold, and --mirror/--no-mirror chooses
    whether mirrored (both-sides) trajectories are included --
    e.g. expert_trajectories_swingup_hold_mirrored.h5.

The controller uses the committed, working reference (trajectory.csv, inputs.csv,
K_matrix.npy) produced by generate_k.py. A self-check aborts if that reference on
disk no longer swings up and holds in the notebook model.
"""

import argparse
from pathlib import Path

import numpy as np
import h5py

from rollout_tvlqr import P, rollout_tvlqr, wrap_to_pi

current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"

X_GOAL = np.array([0.0, 0.0, 0.0, 0.0])
DT_CTRL = 0.05


def load_reference():
    """Load the committed notebook-model TVLQR reference (built by generate_k.py)."""
    x_ref = np.loadtxt(results_dir / "trajectory.csv", delimiter=",", skiprows=1)
    u_ref = np.loadtxt(results_dir / "inputs.csv", delimiter=",", skiprows=1)
    K = np.load(results_dir / "K_matrix.npy")
    return x_ref, u_ref, K


def mirror(x_ref, u_ref):
    """
    Reflect a reference across the pendulum's left-right symmetry.

    For this symmetric plant (cos/sin kinematics, friction odd in velocity) the
    dynamics satisfy f(-x, -u) = -f(x, u), so (x(t), u(t)) valid implies
    (-x(t), -u(t)) valid -- a swing-up to the SAME upright from the other side.
    The TVLQR gains K need NO change: A, B are identical at the mirrored point,
    so the Riccati solution (hence K) is the same -- reuse the original K.
    """
    return -np.asarray(x_ref), -np.asarray(u_ref)


def sample_x0(rng):
    """Start near hanging-down with a spread for swing-up coverage."""
    return np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                     rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])


def sample_upright_x0(rng):
    """
    Start PERTURBED around the upright equilibrium. Rolling out the controller
    from here records the deviation->corrective-torque map -- i.e. the stabilizing
    gain. Without this, the policy only ever sees the exact upright fixed point
    and never learns how to RECOVER from small deviations, so it can't balance.
    """
    return np.array([np.pi + rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
                     rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)])


def make_out_path(mode, mirror):
    """Derive the output filename so it records the dataset's composition.

    Base name expert_trajectories.h5 gains suffixes for the chosen content
    (swingup vs swingup_hold) and whether mirrored trajectories are included
    (mirrored vs singleside), e.g. expert_trajectories_swingup_hold_mirrored.h5.
    """
    content = "swingup" if mode == "swingup" else "swingup_hold"
    mir = "mirrored" if mirror else "singleside"
    return results_dir / f"expert_trajectories_{content}_{mir}.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--mode", choices=["swingup", "both"], default="both",
                    help="'swingup' = swing-up rollouts only; 'both' = swing-up"
                         " rollouts plus the upright-hold rollouts")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="include mirrored (-x_ref, -u_ref) trajectories so the"
                         " policy swings up from BOTH sides; --no-mirror keeps only"
                         " the committed reference's side")
    ap.add_argument("--n-hold", type=int, default=200,
                    help="extra rollouts started perturbed around upright, to teach"
                         " the stabilizing (deviation->torque) gain (mode=both only)")
    ap.add_argument("--hold-steps", type=int, default=60,   # 60*0.05 = 3 s
                    help="length of each upright-hold rollout")
    ap.add_argument("--episode-steps", type=int, default=120)   # 120*0.05 = 6 s
    ap.add_argument("--noise-std", type=float, default=0.03,
                    help="swing-up-phase state-noise std (DAgger coverage); latches"
                         " off once upright is reached so the hold stays clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output .h5 path; if omitted, derived from --mode/--mirror"
                         " (e.g. expert_trajectories_swingup_hold_mirrored.h5)")
    args = ap.parse_args()

    include_hold = args.mode == "both"
    out_path = Path(args.out) if args.out else make_out_path(args.mode, args.mirror)

    rng = np.random.default_rng(args.seed)
    x_ref, u_ref, K = load_reference()
    print(f"Notebook-model TVLQR reference loaded: x_ref{x_ref.shape}, "
          f"u_ref{u_ref.shape}, K{K.shape}; torque_limit={P.torque_limit} Nm.")

    # ── Guard: confirm the reference on disk still swings up & holds in the
    #    notebook model (generate_k.py's RK4 integrator of the notebook EOM). ──
    _, _, ok = rollout_tvlqr(x_ref, u_ref, K, np.zeros(4), args.episode_steps)
    if not ok:
        raise SystemExit(
            "ABORT: the TVLQR reference in results/ does not swing up and hold in\n"
            "       the notebook model. The working reference is the committed\n"
            "       trajectory.csv + inputs.csv + K_matrix.npy produced by\n"
            "       generate_k.py. Re-run `python generate_k.py` (or restore the\n"
            "       committed versions with git checkout) before generating data.")
    print("Reference self-check: swing-up + hold OK in the notebook model.")

    # ── Guard: confirm the MIRRORED reference also swings up & holds. By the
    #    reflection symmetry it should always pass (the model is symmetric and
    #    RK4 of an odd ODE stays odd); checking it here catches any model
    #    asymmetry before we generate a whole side's worth of bad data. Only
    #    relevant when mirrored trajectories are actually included. ──
    if args.mirror:
        x_ref_m, u_ref_m = mirror(x_ref, u_ref)
        _, _, ok_m = rollout_tvlqr(x_ref_m, u_ref_m, K, np.zeros(4),
                                   args.episode_steps)
        if not ok_m:
            raise SystemExit(
                "ABORT: the MIRRORED TVLQR reference does not swing up and hold in\n"
                "       the notebook model, so the plant is not left-right symmetric\n"
                "       and both-sides reflection augmentation is invalid here.")
        print("Mirror self-check: swing-up + hold OK from the other side too. "
              "Generating rollouts...")
    else:
        print("Mirroring disabled (--no-mirror): single-side dataset. "
              "Generating rollouts...")

    kept = 0
    swing_sides = {1: 0, -1: 0}
    hold_sides = {1: 0, -1: 0}
    with h5py.File(out_path, "w") as f:
        for ep in range(args.n_episodes):
            # Alternate the swing-up direction when mirroring: even episodes track
            # the committed reference, odd episodes track its mirror (-x_ref,
            # -u_ref) so the policy sees swing-ups to the SAME upright from BOTH
            # sides (same K). With --no-mirror every episode stays on side +1.
            if args.mirror:
                side = 1 if ep % 2 == 0 else -1
                clean_key = ep // 2     # key clean cadence on the side-pair index
            else:
                side = 1
                clean_key = ep          # no pairing -> key on the raw episode
            xr, ur = (x_ref, u_ref) if side == 1 else mirror(x_ref, u_ref)

            x0 = np.zeros(4) if ep == 0 else sample_x0(rng)
            if side == -1:
                x0 = -x0                       # mirror the start to match the ref
            # Mix clean and noisy rollouts (~1 in 4 clean), balanced across sides
            # by keying the clean cadence on clean_key, not raw ep.
            noise = None if clean_key % 3 == 0 else args.noise_std
            s, a, ok = rollout_tvlqr(
                xr, ur, K, x0, args.episode_steps,
                dt_control=DT_CTRL, noise_std=noise, rng=rng,
                noise_until_upright=True)
            if not ok:
                continue                       # keep only successful rollouts
            grp = f.create_group(f"traj_{kept}")
            grp.create_dataset("states",  data=s.astype(np.float64))
            grp.create_dataset("actions", data=a.astype(np.float64))
            grp.attrs["x0"] = x0
            grp.attrs["noise_std"] = 0.0 if noise is None else noise
            grp.attrs["side"] = side
            kept += 1
            swing_sides[side] += 1
            if (ep + 1) % 50 == 0:
                print(f"  {ep+1}/{args.n_episodes} episodes, {kept} kept "
                      f"(+side {swing_sides[1]} / -side {swing_sides[-1]})")
        swing_kept = kept
        if include_hold:
            print(f"Swing-up rollouts: {swing_kept} kept "
                  f"(+side {swing_sides[1]}, -side {swing_sides[-1]}). "
                  "Generating upright-hold rollouts...")

            # ── Upright-hold rollouts: teach the stabilizing gain around the top.
            # Alternate sides when mirroring: -side perturbs around the q1=-pi
            # representation of the SAME upright, balancing the
            # deviation->corrective-torque coverage. With --no-mirror, side +1. ──
            for i in range(args.n_hold):
                side = (1 if i % 2 == 0 else -1) if args.mirror else 1
                xr, ur = (x_ref, u_ref) if side == 1 else mirror(x_ref, u_ref)
                x0 = sample_upright_x0(rng)
                if side == -1:
                    x0 = -x0
                s, a, ok = rollout_tvlqr(xr, ur, K, x0, args.hold_steps,
                                         dt_control=DT_CTRL)        # no noise
                if not ok:
                    continue
                grp = f.create_group(f"traj_{kept}")
                grp.create_dataset("states",  data=s.astype(np.float64))
                grp.create_dataset("actions", data=a.astype(np.float64))
                grp.attrs["x0"] = x0
                grp.attrs["noise_std"] = 0.0
                grp.attrs["kind"] = "hold"
                grp.attrs["side"] = side
                kept += 1
                hold_sides[side] += 1
        else:
            print(f"Swing-up rollouts: {swing_kept} kept "
                  f"(+side {swing_sides[1]}, -side {swing_sides[-1]}). "
                  "Skipping upright-hold rollouts (mode=swingup).")

    n_swing_samples = swing_kept * args.episode_steps
    n_hold_samples  = (kept - swing_kept) * args.hold_steps
    print(f"\nDone! Saved {kept} rollouts to {out_path} "
          f"({swing_kept} swing-up + {kept - swing_kept} hold).")
    print(f"  sides -- swing-up: +{swing_sides[1]}/-{swing_sides[-1]}, "
          f"hold: +{hold_sides[1]}/-{hold_sides[-1]}.")
    print(f"~{n_swing_samples + n_hold_samples} (state, action) samples "
          f"(~{n_hold_samples} from the hold regime).")


if __name__ == "__main__":
    main()
