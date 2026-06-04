"""
Interactive swing-up evaluation for the double pendulum.

Starts the pendulum hanging at the bottom [0,0,0,0] and tries to swing it up to
upright [pi,0,0,0], holding there. Two controllers are selectable:

    --controller diffusion   # the trained Diffusion Policy checkpoint
    --controller tvlqr        # the TVLQR trajectory-tracking baseline

The diffusion controller rebuilds whatever architecture the checkpoint was
trained with (MLP or Transformer) automatically -- no manual edits needed.

INTERACTION (native MuJoCo viewer):
    * Double-click a body (a pendulum link) to select it.
    * Ctrl + right-drag  -> apply an external FORCE to the selected body.
    * Ctrl + left-drag   -> apply an external TORQUE.
    * The controller keeps running, so you can shove the pendulum mid-swing and
      watch whether it recovers.

Run from the repo root (so diffusion_models is importable):
    python double_pendulum/evaluate_swingup.py --controller diffusion
    python double_pendulum/evaluate_swingup.py --controller tvlqr
"""

import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import DoublePendulumEnv

current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"


# ── Helpers ─────────────────────────────────────────────────────────────────
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── TVLQR controller (adapted from the reference script) ─────────────────────
class TVLQRController:
    """Trajectory-tracking baseline with deviation-triggered recovery."""

    DEVIATION_THRESHOLD = 2.0

    def __init__(self):
        self.x_ref = np.loadtxt(results_dir / "trajectory.csv", delimiter=",", skiprows=1).T
        self.u_ref = np.loadtxt(results_dir / "inputs.csv", delimiter=",", skiprows=1).T
        self.K     = np.load(results_dir / "K_matrix.npy")
        self.max_idx     = self.x_ref.shape[1] - 1
        self.current_idx = 0

    @staticmethod
    def _feat(x, v_scale=0.1):
        if x.ndim == 1:
            p0, p1 = x[0], x[1]
            v = x[2:] * v_scale
            return np.concatenate(([np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1)], v))
        p0, p1 = x[0, :], x[1, :]
        v = x[2:, :] * v_scale
        return np.vstack((np.cos(p0), np.sin(p0), np.cos(p1), np.sin(p1), v))

    def _ranked(self, x, ref, k=1):
        diff  = self._feat(ref) - self._feat(x).reshape(-1, 1)
        dists = np.linalg.norm(diff, axis=0)
        order = np.argsort(dists)
        return order[:k], dists[order[:k]]

    def _K_weighted(self, x, k=5):
        idx, dists = self._ranked(x, self.x_ref, k=k)
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum()
        Ks = self.K[np.clip(idx, 0, len(self.K) - 1)]
        return np.sum(w[:, None, None] * Ks, axis=0)

    def reset(self):
        self.current_idx = 0

    def action(self, x):
        target = self.x_ref[:, self.current_idx].reshape(4, 1)
        _, d = self._ranked(x, target, k=1)
        holding = self.current_idx >= self.max_idx
        if d[0] > self.DEVIATION_THRESHOLD or holding:
            best, _ = self._ranked(x, self.x_ref, k=1)
            idx = best[0]
            K = self._K_weighted(x, k=5)
        else:
            idx = self.current_idx
            K = self.K[idx]
            self.current_idx += 1

        x_des, u_des = self.x_ref[:, idx], self.u_ref[:, idx]
        err = x - x_des
        err[0], err[1] = wrap_to_pi(err[0]), wrap_to_pi(err[1])
        return u_des - K @ err


# ── Diffusion-policy controller ──────────────────────────────────────────────
class DiffusionController:
    """
    Receding-horizon Diffusion Policy.
    Observes last k states -> samples an H-step action chunk -> executes the
    first `n_exec` actions before re-planning.
    """

    def __init__(self, ckpt_path, stats_path, n_exec=2):
        import torch
        from diffusion_models.diffusion_policy import (
            Scheduler, MLP, TrajectoryTransformer, DiffusionPolicy,
        )
        self.torch = torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        with open(stats_path) as fp:
            self.stats = json.load(fp)
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt["config"]

        self.k        = cfg["k"]
        self.nx       = cfg["nx"]
        self.nu       = cfg["action_dim"]
        self.horizon  = cfg["horizon"]
        self.use_goal = cfg["use_goal"]
        self.n_exec   = n_exec

        # Rebuild the EXACT architecture the checkpoint was trained with.
        # The checkpoint stores arch + the kwargs the network was built from,
        # so MLP vs Transformer (and their sizes) round-trip automatically.
        arch       = cfg.get("arch", "mlp")
        net_kwargs = cfg.get("net_kwargs", {
            "horizon": cfg["horizon"], "action_dim": cfg["action_dim"],
            "cond_dim": cfg["cond_dim"], "hidden_dim": cfg.get("hidden_dim", 256),
        })
        net_cls    = {"mlp": MLP, "transformer": TrajectoryTransformer}[arch]
        network    = net_cls(**net_kwargs)
        print(f"[diffusion] loaded arch='{arch}' from {Path(ckpt_path).name}")

        scheduler   = Scheduler(num_steps=cfg["timesteps"], device=device)
        self.policy = DiffusionPolicy(scheduler, network, device,
                                      cfg["timesteps"], cfg["horizon"],
                                      cfg["action_dim"])
        self.policy.model.load_state_dict(ckpt["model_state"])
        self.policy.model.eval()

        self.s_min = np.array(self.stats["state_min"],  dtype=np.float32)
        self.s_max = np.array(self.stats["state_max"],  dtype=np.float32)
        self.a_min = np.array(self.stats["action_min"], dtype=np.float32)
        self.a_max = np.array(self.stats["action_max"], dtype=np.float32)
        self.s_rng = np.where((self.s_max - self.s_min) > 1e-8, self.s_max - self.s_min, 1.0)
        self.a_rng = np.where((self.a_max - self.a_min) > 1e-8, self.a_max - self.a_min, 1.0)

        self._hist  = None          # rolling k-state history (normalized)
        self._queue = []            # remaining actions from the current chunk

    def _norm_s(self, x):  return 2.0 * (x - self.s_min) / self.s_rng - 1.0
    def _denorm_a(self, a): return (a + 1.0) * 0.5 * self.a_rng + self.a_min

    def reset(self, x0):
        x0n = self._norm_s(np.asarray(x0, dtype=np.float32))
        self._hist  = np.repeat(x0n[None], self.k, axis=0)   # (k, nx)
        self._queue = []

    def action(self, x):
        # update rolling history with the latest observation
        xn = self._norm_s(np.asarray(x, dtype=np.float32))
        self._hist = np.concatenate([self._hist[1:], xn[None]], axis=0)

        if not self._queue:
            cond = self._hist.reshape(-1)
            if self.use_goal:
                goal_n = self._norm_s(np.array([np.pi, 0, 0, 0], dtype=np.float32))
                cond = np.concatenate([cond, goal_n])
            cond_t = self.torch.from_numpy(cond[None].astype(np.float32))
            with self.torch.no_grad():
                a_seq = self.policy.sample(cond_t).cpu().numpy()[0]   # (H, nu) normalized
            a_seq = self._denorm_a(a_seq)
            self._queue = list(a_seq[: self.n_exec])

        return self._queue.pop(0)


# ── Evaluation loop ──────────────────────────────────────────────────────────
def evaluate(controller_name, hold_steps=40, angle_tol=0.20, vel_tol=1.0,
             max_steps=1200, dt_control=0.05, realtime=True, ckpt="diffusion_policy.pt"):
    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    obs, _ = env.reset()

    # Force the start state to the hanging-down configuration.
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    env.data.qpos[:2] = x0[:2]
    env.data.qvel[:2] = x0[2:]
    mujoco.mj_forward(env.model, env.data)
    obs = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])

    # Build the chosen controller.
    if controller_name == "tvlqr":
        ctrl = TVLQRController(); ctrl.reset()
    else:
        ctrl = DiffusionController(
            results_dir / ckpt,
            results_dir / "norm_stats.json",
            n_exec=2,
        )
        ctrl.reset(x0)

    dt_sim   = env.model.opt.timestep
    n_sub    = max(1, int(round(dt_control / dt_sim)))
    max_tau  = env.action_space.high[0]
    x_goal   = np.array([np.pi, 0.0, 0.0, 0.0])

    hist = {"t": [], "x": [], "u": []}
    consecutive_hold = 0
    success_step = None

    print(f"[{controller_name}] swing-up from bottom. "
          f"Drag the pendulum in the viewer (Ctrl+right-drag = force).")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Match the env's fixed "side" camera instead of the default free view.
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        if cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        for step in range(max_steps):
            x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])

            u = np.asarray(ctrl.action(x), dtype=np.float64).ravel()
            u = np.clip(u, -max_tau, max_tau)

            hist["t"].append(step * dt_control)
            hist["x"].append(x.copy())
            hist["u"].append(u.copy())

            # success check: near upright (angle + velocity) for N consecutive steps
            ang_err = abs(wrap_to_pi(x[0] - x_goal[0])) + abs(wrap_to_pi(x[1] - x_goal[1]))
            vel_err = abs(x[2]) + abs(x[3])
            if ang_err < angle_tol and vel_err < vel_tol:
                consecutive_hold += 1
                if consecutive_hold >= hold_steps and success_step is None:
                    success_step = step
                    print(f"[{controller_name}] SUCCESS: held upright "
                          f"{hold_steps} steps, first at step {step} "
                          f"(t={step * dt_control:.2f}s)")
            else:
                consecutive_hold = 0

            env.data.ctrl[:] = u
            for _ in range(n_sub):
                # Inject the viewer's mouse perturbation (Ctrl+drag) into the
                # dynamics. launch_passive hands physics to us, so the drag is
                # only recorded in viewer.perturb until WE apply it: this writes
                # the force/torque into data.xfrc_applied, which mj_step consumes.
                if viewer.perturb.select > 0:
                    mujoco.mjv_applyPerturbForce(env.model, env.data, viewer.perturb)
                else:
                    # nothing selected -> clear any leftover applied force,
                    # otherwise a past drag would keep pushing forever.
                    env.data.xfrc_applied[:] = 0.0
                mujoco.mj_step(env.model, env.data)
            viewer.sync()

            if realtime:
                time.sleep(dt_control)

            if not viewer.is_running():
                break

    env.close()
    return _finish(controller_name, hist, success_step, hold_steps)


def _finish(name, hist, success_step, hold_steps):
    t = np.array(hist["t"]); x = np.array(hist["x"]); u = np.array(hist["u"])
    success = success_step is not None
    summary = {
        "controller": name,
        "success": bool(success),
        "success_step": int(success_step) if success else None,
        "time_to_success_s": float(success_step * 0.05) if success else None,
        "hold_steps_required": hold_steps,
        "total_steps": int(len(t)),
    }

    out = current_dir / "graphs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    labels = ["q1 (shoulder)", "q2 (elbow)", "q1_dot", "q2_dot"]
    goal   = [np.pi, 0.0, 0.0, 0.0]
    for i, ax in enumerate(axs.ravel()):
        ax.plot(t, x[:, i], "b-", label=labels[i])
        ax.axhline(goal[i], color="r", ls="--", alpha=0.6, label="goal")
        if success:
            ax.axvline(success_step * 0.05, color="g", ls=":", label="success")
        ax.set_title(labels[i]); ax.set_xlabel("Time (s)"); ax.grid(True); ax.legend()
    fig.suptitle(f"Swing-up evaluation: {name}  |  success={success}")
    fig.tight_layout()
    fig.savefig(out / f"swingup_states_{name}.png")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(t, u[:, 0], label="u1"); ax2.plot(t, u[:, 1], label="u2")
    ax2.set_title(f"Control torques: {name}"); ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (Nm)"); ax2.grid(True); ax2.legend()
    fig2.tight_layout(); fig2.savefig(out / f"swingup_controls_{name}.png")

    with open(out / f"swingup_summary_{name}.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[{name}] summary: {summary}")
    print(f"[{name}] plots + summary saved to {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["diffusion", "tvlqr"], default="diffusion")
    p.add_argument("--ckpt", default="diffusion_policy.pt",
                   help="checkpoint filename inside results/ (diffusion only)")
    p.add_argument("--hold-steps", type=int, default=40,
                   help="consecutive in-tolerance control steps to count as success")
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--no-realtime", action="store_true",
                   help="run as fast as possible instead of wall-clock paced")
    args = p.parse_args()

    evaluate(
        controller_name=args.controller,
        hold_steps=args.hold_steps,
        angle_tol=args.angle_tol,
        vel_tol=args.vel_tol,
        max_steps=args.max_steps,
        realtime=not args.no_realtime,
        ckpt=args.ckpt,
    )