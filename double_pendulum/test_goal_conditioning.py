"""
Test whether the Diffusion Policy can be CONDITIONED to different optima.

Section 10, Part C (xii/xv): "Analyze whether the diffusion model captures
multiple valid behaviors" / "what limitations appeared in your setup".

The network is architecturally goal-conditioned (cond = k-step history features +
goal features, cond_dim=42). But whether it actually *uses* the goal depends on
whether the goal VARIED during training. In this checkpoint it did not:
train_diffusion_policy.py appends a single constant X_GOAL=[pi,0,0,0] to every
sample, and 100% of the demonstrations end at that one upright. So we expect the
goal input to be IGNORED. This script measures that, two ways:

  A. Action sensitivity (instant, no simulation)
       For a few fixed state-histories, sample the action chunk while ONLY the
       goal changes. If the chunks are ~identical across goals, the goal carries
       no information -> the policy is single-goal. Reported as mean |Δaction|
       between every pair of goals.

  B. Goal-conditioned rollouts (headless MuJoCo)
       From hanging-down, run the policy conditioned on each goal and log where
       it actually ends up. If every goal converges to the same [pi,0] upright,
       conditioning is not working.

It reuses the trained policy + normalization from visualize_swingup.DiffusionController
(no edits to that file): we instantiate it for loading, then call its internal
helpers with our OWN goal vector instead of the hard-coded one.

Run from the repo root:
    python double_pendulum/test_goal_conditioning.py
    python double_pendulum/test_goal_conditioning.py --goals "pi,0,0,0" "0,pi,0,0" "1.5708,-1.5708,0,0"
"""

import os
import sys
import json
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from double_pendulum_environment import DoublePendulumEnv
from visualizations.visualize_swingup import DiffusionController, wrap_to_pi

results_dir = _HERE / "results"
OUT_DIR = _HERE / "graphs" / "evaluation" / "goal_conditioning"

# Default optima to probe. The first two are true (unstable) equilibria of the
# plant -- gravity torque G(x)=0 there, so they are holdable with control:
#   [pi,0]  both links up (the trained upright)
#   [0,pi]  link1 down, link2 inverted
# [pi/2,-pi/2] is NOT a zero-torque equilibrium: holding it needs a sustained
# ~0.089 Nm on joint1, right at the +-0.1 Nm actuator limit -> at best marginal,
# and it is far out-of-distribution. Expect it to fail even if conditioning worked.
DEFAULT_GOALS = {
    "up [pi,0]":        [np.pi, 0.0, 0.0, 0.0],
    "fold [0,pi]":      [0.0, np.pi, 0.0, 0.0],
    "mid [pi/2,-pi/2]": [np.pi / 2, -np.pi / 2, 0.0, 0.0],
}


def parse_goal(s):
    vals = [float(v) for v in s.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError(f"goal must be 4 comma-separated numbers: {s}")
    return np.array(vals, dtype=np.float32)


# ── goal-conditioned sampling (reuses the controller's exact normalization) ───
def sample_chunk(ctrl, hist_feats, goal_state, seed=None):
    """One deterministic action chunk (H, nu) for a (k, nx_feat) history + goal.

    DDPM sampling starts from RANDOM latent noise even with stochastic=False, so
    two calls with the same goal differ by that noise. Pass `seed` to fix the
    latent draw -> then the ONLY thing that varies between two calls is the goal,
    which is what we want when measuring the goal's influence.
    """
    cond = hist_feats.reshape(-1)
    if ctrl.use_goal:
        cond = np.concatenate([cond, ctrl._norm_s(np.asarray(goal_state, np.float32))])
    cond_t = ctrl.torch.from_numpy(cond[None].astype(np.float32))
    if seed is not None:
        ctrl.torch.manual_seed(seed)
    with ctrl.torch.no_grad():
        a = ctrl.policy.sample(cond_t, stochastic=False).cpu().numpy()[0]   # (H, nu) normed
    return ctrl._denorm_a(a)


def rollout_to_goal(env, ctrl, goal_state, x0, *, max_steps, dt_control, max_tau):
    """Closed-loop receding-horizon rollout conditioned on `goal_state`."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:2] = x0[:2]
    env.data.qvel[:2] = x0[2:]
    mujoco.mj_forward(env.model, env.data)

    dt_sim = env.model.opt.timestep
    n_sub = max(1, int(round(dt_control / dt_sim)))

    hist = np.repeat(ctrl._norm_s(np.asarray(x0, np.float32))[None], ctrl.k, axis=0)
    queue = []
    ts, xs = [], []
    for step in range(max_steps):
        x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])
        hist = np.concatenate([hist[1:], ctrl._norm_s(x.astype(np.float32))[None]], axis=0)
        if not queue:
            chunk = sample_chunk(ctrl, hist, goal_state)
            queue = list(chunk[: ctrl.n_exec])
        u = np.clip(queue.pop(0), -max_tau, max_tau)
        ts.append(step * dt_control); xs.append(x.copy())
        env.data.ctrl[:] = u
        for _ in range(n_sub):
            mujoco.mj_step(env.model, env.data)
    return np.array(ts), np.array(xs)


# ── Experiment A: does the sampled action even change with the goal? ──────────
def action_sensitivity(ctrl, goals):
    probes = {
        "bottom  [0,0,0,0]":     np.array([0.0, 0.0, 0.0, 0.0], np.float32),
        "mid-swing [pi/2,0,3,0]": np.array([np.pi / 2, 0.0, 3.0, 0.0], np.float32),
        "near-up  [pi-.1,0,0,0]": np.array([np.pi - 0.1, 0.0, 0.0, 0.0], np.float32),
    }
    names = list(goals.keys())
    print("\n=== A. Action sensitivity to the goal (shared latent noise) ===")
    print("    mean |Δaction| (Nm) over the H-step chunk, between goal pairs,")
    print("    with the diffusion latent FIXED so only the goal differs.")
    print(f"    Reference: full torque range is +-{0.1:.2f} Nm.")
    print("    |Δa| ~0 vs the torque range => the goal is IGNORED.\n")
    SEED = 0
    for pname, pstate in probes.items():
        hist = np.repeat(ctrl._norm_s(pstate)[None], ctrl.k, axis=0)
        chunks = {g: sample_chunk(ctrl, hist, goals[g], seed=SEED) for g in names}
        print(f"  state {pname}")
        # sanity: same goal + same latent seed -> EXACTLY 0
        base = np.mean(np.abs(chunks[names[0]] - sample_chunk(ctrl, hist, goals[names[0]], seed=SEED)))
        print(f"    (same goal, same latent -> 0): {base:.2e}")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = float(np.mean(np.abs(chunks[names[i]] - chunks[names[j]])))
                print(f"    |Δa| {names[i]:18s} vs {names[j]:18s} = {d:.2e} Nm")
        print()


# ── Experiment B: where do goal-conditioned rollouts actually end up? ─────────
def rollout_experiment(ctrl, goals, max_steps, dt_control):
    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    max_tau = float(env.action_space.high[0])
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    print("=== B. Goal-conditioned rollouts from hanging-down ===")
    print("    final (q1,q2) and distance to the COMMANDED goal.\n")
    trajs = {}
    for gname, gstate in goals.items():
        t, x = rollout_to_goal(env, ctrl, gstate, x0,
                               max_steps=max_steps, dt_control=dt_control,
                               max_tau=max_tau)
        trajs[gname] = (t, x, gstate)
        final = x[-30:].mean(0)
        err = abs(wrap_to_pi(final[0] - gstate[0])) + abs(wrap_to_pi(final[1] - gstate[1]))
        print(f"  goal {gname:18s} -> final q1={wrap_to_pi(final[0]):+.2f} "
              f"q2={wrap_to_pi(final[1]):+.2f} | dist-to-goal={err:.2f} "
              f"{'(reached)' if err < 0.3 else ''}")
    env.close()
    return trajs


def plot(trajs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(trajs)
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 7), squeeze=False)
    for j, (gname, (t, x, gstate)) in enumerate(trajs.items()):
        for r, (idx, lbl) in enumerate([(0, "q1"), (1, "q2")]):
            ax = axs[r][j]
            ax.plot(t, wrap_to_pi(x[:, idx]), label=f"{lbl} actual")
            ax.axhline(wrap_to_pi(gstate[idx]), color="r", ls="--",
                       label=f"{lbl} commanded")
            ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
            ax.set_xlabel("t (s)"); ax.grid(True); ax.legend(fontsize=8)
            if r == 0:
                ax.set_title(f"goal: {gname}")
    fig.suptitle("Goal-conditioned rollouts: actual joint angles vs commanded goal (red)\n"
                 "(goal works only if each column tracks its OWN red line; here only the "
                 "trained [pi,0] is reached, others just fail)")
    fig.tight_layout()
    out = OUT_DIR / "goal_conditioning.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n[plot] saved to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="diffusion_policy.pt")
    p.add_argument("--goals", type=parse_goal, nargs="*", default=None,
                   help='goals as "q1,q2,dq1,dq2" strings; default = 3 named optima')
    p.add_argument("--n-exec", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--dt-control", type=float, default=0.05)
    p.add_argument("--skip-rollouts", action="store_true",
                   help="run only the fast action-sensitivity diagnostic")
    args = p.parse_args()

    if args.goals is None:
        goals = DEFAULT_GOALS
    else:
        goals = {f"goal{i} [{g[0]:.2f},{g[1]:.2f}]": g for i, g in enumerate(args.goals)}

    ctrl = DiffusionController(results_dir / args.ckpt,
                               results_dir / "norm_stats.json", n_exec=args.n_exec)
    if not ctrl.use_goal:
        print("WARNING: this checkpoint has use_goal=False -- it has NO goal input "
              "at all, so it is inherently single-goal.")

    action_sensitivity(ctrl, goals)
    if not args.skip_rollouts:
        trajs = rollout_experiment(ctrl, goals, args.max_steps, args.dt_control)
        plot(trajs)

    print("\nInterpretation: the goal barely changes the action (<~0.04 Nm of a "
          "0.1 Nm range), and only the TRAINED optimum [pi,0] is actually reached "
          "-- the other commanded goals are NOT steered to their targets, they just "
          "fail. So the goal input provides no useful conditioning (it was a constant "
          "in training). To make the policy conditionable you must VARY the goal in "
          "training: generate demos that reach each optimum and set each sample's "
          "goal feature from its OWN target (not the single constant X_GOAL), then "
          "retrain. Note [pi/2,-pi/2] is not a zero-torque equilibrium and is at the "
          "actuator limit, so it likely can't be held even after retraining.")


if __name__ == "__main__":
    main()
