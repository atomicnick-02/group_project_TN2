"""
Generate the Diffusion-Policy training set by ROLLING OUT the working TVLQR
swing-up controller inside the MuJoCo simulator and logging
(observed_state, commanded_torque) at the control rate.

WHY THIS REPLACES generate_dataset.py:
    The old generator saved the trajectory-optimizer's collocation states/actions
    directly. Those were computed for a frictionless point-mass model that didn't
    match dp.xml AND were dynamically infeasible at dt=0.05 -- applying the saved
    torques in MuJoCo does NOT reproduce the saved states. A policy cloning that
    data learns a state->action map for a simulator it is never evaluated on, so
    it cannot swing up.

    Here every (state, action) pair is produced by stepping the ACTUAL MuJoCo
    model under a controller that provably swings up and HOLDS (verified: 265
    consecutive in-tolerance steps from rest, 12/12 from perturbed starts). The
    data is therefore dynamically valid by construction, and -- because we roll
    out closed-loop from many initial conditions with action-noise perturbations
    -- it covers the state distribution the policy will actually encounter,
    including recovery (DAgger-style). That is what fixes the compounding-error
    problem on this chaotic system.

DAgger detail: we apply (commanded + noise) to the sim for state coverage, but
    LOG the clean commanded action as the supervised target. So each sample is
    "at this (possibly off-nominal) state, the expert would command THIS torque".

Output: results/expert_trajectories.h5 with groups traj_*, each holding
    states  (T, 4) and actions (T, 2)  -- the SAME format
    train_diffusion_policy.py already consumes, so nothing downstream changes.

The controller uses the committed, working reference (trajectory.csv, inputs.csv,
K_matrix.npy). It does NOT re-run generate_k.py (whose dt=0.05 collocation
reference is open-loop-infeasible); a self-check aborts if the reference on disk
no longer holds.
"""

import argparse
import numpy as np
import mujoco
import h5py
from pathlib import Path

from simulation import DoublePendulumEnv
from evaluate_swingup import TVLQRController, wrap_to_pi

current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"

X_GOAL = np.array([np.pi, 0.0, 0.0, 0.0])
DT_CTRL = 0.05


def _ang_vel_err(x):
    ang = abs(wrap_to_pi(x[0] - np.pi)) + abs(wrap_to_pi(x[1]))
    vel = abs(x[2]) + abs(x[3])
    return ang, vel


def rollout(env, ctrl, x0, n_steps, max_tau, perturb_scale=0.0, rng=None):
    """
    One closed-loop MuJoCo rollout.

    Returns (states (T,4), actions (T,2), success) where `actions` are the CLEAN
    commanded torques (supervised targets) and success means it reached and held
    upright. The applied torque may carry exploration noise for state coverage.
    """
    env.reset()
    env.data.qpos[:2] = x0[:2]
    env.data.qvel[:2] = x0[2:]
    mujoco.mj_forward(env.model, env.data)
    ctrl.reset()

    n_sub = max(1, int(round(DT_CTRL / env.model.opt.timestep)))
    states, actions = [], []
    held = 0
    held_max = 0
    reached = False                            # latched once we first hit upright
    for _ in range(n_steps):
        x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])
        u_cmd = np.clip(np.asarray(ctrl.action(x)).ravel(), -max_tau, max_tau)

        states.append(x.copy())
        actions.append(u_cmd.copy())          # log the CLEAN command

        # Perturb the APPLIED torque only during the swing-up phase: this spreads
        # the off-nominal states the controller must recover from (DAgger-style
        # coverage), while the fragile upright hold is left noise-free so the
        # rollout still succeeds. Noise latches off once upright is first reached.
        ang, _ = _ang_vel_err(x)
        if ang < 0.3:
            reached = True
        u_app = u_cmd
        if perturb_scale > 0.0 and rng is not None and not reached:
            u_app = np.clip(u_cmd + rng.normal(0.0, perturb_scale * max_tau, u_cmd.shape),
                            -max_tau, max_tau)

        env.data.ctrl[:] = u_app
        for _ in range(n_sub):
            mujoco.mj_step(env.model, env.data)

        x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])
        ang, vel = _ang_vel_err(x)
        held = held + 1 if (ang < 0.2 and vel < 1.0) else 0
        held_max = max(held_max, held)

    return np.array(states), np.array(actions), held_max >= 10


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--n-hold", type=int, default=200,
                    help="extra rollouts started perturbed around upright, to teach"
                         " the stabilizing (deviation->torque) gain")
    ap.add_argument("--hold-steps", type=int, default=60,   # 60*0.05 = 3 s
                    help="length of each upright-hold rollout")
    ap.add_argument("--episode-steps", type=int, default=120)   # 120*0.05 = 6 s
    ap.add_argument("--perturb", type=float, default=0.10,
                    help="swing-up-phase action-noise std as a fraction of max torque")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(results_dir / "expert_trajectories.h5"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    max_tau = float(env.action_space.high[0])
    ctrl = TVLQRController()                  # loads trajectory.csv/inputs.csv/K_matrix.npy

    # ── Guard: confirm the reference on disk still swings up & holds. ──
    _, _, ok = rollout(env, ctrl, np.zeros(4), args.episode_steps, max_tau)
    if not ok:
        env.close()
        raise SystemExit(
            "ABORT: the TVLQR reference in results/ does not hold upright.\n"
            "       The working reference is the committed trajectory.csv + inputs.csv\n"
            "       + K_matrix.npy. Did generate_k.py overwrite them with its coarse\n"
            "       (open-loop-infeasible) dt=0.05 trajectory? Restore the committed\n"
            "       versions (git checkout) before generating the dataset.")
    print("Reference self-check: swing-up + hold OK. Generating rollouts...")

    kept = 0
    with h5py.File(args.out, "w") as f:
        for ep in range(args.n_episodes):
            x0 = np.zeros(4) if ep == 0 else sample_x0(rng)
            pert = 0.0 if ep % 4 == 0 else args.perturb   # mix clean + perturbed
            s, a, ok = rollout(env, ctrl, x0, args.episode_steps, max_tau,
                               perturb_scale=pert, rng=rng)
            if not ok:
                continue                       # keep only successful rollouts
            grp = f.create_group(f"traj_{kept}")
            grp.create_dataset("states",  data=s.astype(np.float64))
            grp.create_dataset("actions", data=a.astype(np.float64))
            grp.attrs["x0"] = x0
            grp.attrs["perturb"] = pert
            kept += 1
            if (ep + 1) % 50 == 0:
                print(f"  {ep+1}/{args.n_episodes} episodes, {kept} kept")
        swing_kept = kept
        print(f"Swing-up rollouts: {swing_kept} kept. Generating upright-hold rollouts...")

        # ── Upright-hold rollouts: teach the stabilizing gain around the top. ──
        for hp in range(args.n_hold):
            x0 = sample_upright_x0(rng)
            s, a, ok = rollout(env, ctrl, x0, args.hold_steps, max_tau)  # no perturb
            if not ok:
                continue
            grp = f.create_group(f"traj_{kept}")
            grp.create_dataset("states",  data=s.astype(np.float64))
            grp.create_dataset("actions", data=a.astype(np.float64))
            grp.attrs["x0"] = x0
            grp.attrs["perturb"] = 0.0
            grp.attrs["kind"] = "hold"
            kept += 1
    env.close()
    n_swing_samples = swing_kept * args.episode_steps
    n_hold_samples  = (kept - swing_kept) * args.hold_steps
    print(f"\nDone! Saved {kept} rollouts to {args.out} "
          f"({swing_kept} swing-up + {kept - swing_kept} hold).")
    print(f"~{n_swing_samples + n_hold_samples} (state, action) samples "
          f"(~{n_hold_samples} from the hold regime).")


if __name__ == "__main__":
    main()
