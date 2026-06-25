"""
Interactive swing-up evaluation for the double pendulum.

Starts the pendulum hanging at the bottom [0,0,0,0] and tries to swing it up to
upright [pi,0,0,0], holding there, using the trained Diffusion Policy checkpoint.

The diffusion controller rebuilds whatever architecture the checkpoint was
trained with (MLP or Transformer) automatically -- no manual edits needed.

INTERACTION (native MuJoCo viewer):
    * Click the viewer window to focus it, then push the tip with the keyboard:
        W / Up arrow  -> +Z        S / Down arrow -> -Z
        A             -> -X        D              -> +X        Space -> clear
      NOTE: MuJoCo's viewer reserves the Left/Right arrows for stepping through
      simulation history, so they never reach this script -- use A / D for the
      horizontal push (Up/Down arrows are unbound and do work).
    * Or Ctrl + right-drag for a mouse FORCE / Ctrl + left-drag for a TORQUE.
    * The controller keeps running, so you can shove the pendulum mid-swing and
      watch whether it recovers.

Run from the repo root (so diffusion_models is importable):
    python double_pendulum/evaluate_swingup.py
"""

import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)                                   # for `diffusion_models`
sys.path.insert(0, os.path.join(_REPO_ROOT, "double_pendulum"))  # for `double_pendulum_environment`
import time
import json
import select
import signal
import termios
import threading
import tty
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from double_pendulum_environment import DoublePendulumEnv

current_dir = Path(__file__).resolve().parent
results_dir = current_dir.parent / "double_pendulum" / "results"


# ── Helpers ─────────────────────────────────────────────────────────────────
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ── New Helper for Key Interaction ──────────────────────────────────────────
class ArrowKeyForce:
    """
    Reads arrow keys from the terminal stdin (works in Docker/remote VSCode)
    AND from the GLFW viewer window as a fallback.

    Terminal arrow keys send ESC sequences: ESC [ A/B/C/D
    Pendulum swings in XZ plane (hinge axis = Y):
      Left/Right → force[0] (X)   Up/Down → force[2] (Z)
    """

    def __init__(self, model, force_magnitude=5):
        self.force = np.zeros(3)
        self.mag = force_magnitude
        self.tip_id = model.nbody - 1
        self._running = True
        self._old_term = None
        t = threading.Thread(target=self._stdin_listener, daemon=True)
        t.start()

    def _apply(self, axis, sign):
        self.force[:] = 0.0
        self.force[axis] = sign * self.mag

    def _stdin_listener(self):
        fd = sys.stdin.fileno()
        try:
            self._old_term = termios.tcgetattr(fd)
            tty.setraw(fd)
        except termios.error:
            return  # stdin is not a tty (e.g. piped input) – skip
        try:
            while self._running:
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                b = sys.stdin.buffer.read(1)
                if b == b'\x1b':
                    # read the rest of the escape sequence with a short timeout
                    if select.select([sys.stdin], [], [], 0.02)[0]:
                        b2 = sys.stdin.buffer.read(1)
                        if b2 == b'[' and select.select([sys.stdin], [], [], 0.02)[0]:
                            b3 = sys.stdin.buffer.read(1)
                            if   b3 == b'A': self._apply(2, +1)   # Up    → +Z
                            elif b3 == b'B': self._apply(2, -1)   # Down  → -Z
                            elif b3 == b'C': self._apply(0, +1)   # Right → +X
                            elif b3 == b'D': self._apply(0, -1)   # Left  → -X
                elif b == b' ':
                    self.force[:] = 0.0
                elif b == b'\x03':  # Ctrl+C – restore terminal then re-raise
                    self.stop()
                    os.kill(os.getpid(), signal.SIGINT)
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._old_term is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
            except termios.error:
                pass
            self._old_term = None

    def key_callback(self, keycode):
        # Fires when the viewer WINDOW has focus. MuJoCo's built-in simulate UI
        # consumes Left/Right arrows (history step back/forward) *before* this
        # callback runs, so those two never arrive here -- use A/D for the
        # horizontal (X) push. Up/Down arrows are unbound and do pass through.
        #   GLFW codes: Up 265, Down 264, Right 262, Left 263,
        #               W 87, S 83, A 65, D 68, Space 32
        if   keycode in (265, 87): self._apply(2, +1)   # Up   / W → +Z
        elif keycode in (264, 83): self._apply(2, -1)   # Down / S → -Z
        elif keycode in (262, 68): self._apply(0, +1)   # Right/ D → +X
        elif keycode in (263, 65): self._apply(0, -1)   # Left / A → -X
        elif keycode == 32:        self.force[:] = 0.0  # Space → clear

# ── Diffusion-policy controller ──────────────────────────────────────────────
class DiffusionController:
    """
    Receding-horizon Diffusion Policy.
    Observes last k states -> samples an H-step action chunk -> executes the
    first `n_exec` actions before re-planning.
    """

    def __init__(self, ckpt_path, stats_path, n_exec=2, ddim_steps=0, use_compile=True):
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
        # ddim_steps>0 switches to fast DDIM sampling, BUT for this model that
        # converges to a different (worse-for-control) solution than the
        # deterministic-DDPM sampler and roughly halves the swing-up success
        # rate -- so it is OFF by default. The real speedup is torch.compile
        # below, which keeps the DDPM sampler's output bit-identical (~5x faster).
        self.ddim_steps = ddim_steps

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

        # The dominant inference cost is the 99-step reverse loop calling the net
        # once per step at batch=1, which is kernel-launch bound. torch.compile
        # with CUDA graphs ("reduce-overhead") removes that overhead for a ~5x
        # speedup with BIT-IDENTICAL output (verified maxΔ=0). CUDA-only; falls
        # back to eager if compile/Triton is unavailable. We warm it up here so
        # the one-time compile+graph-capture cost is paid at construction rather
        # than stalling the first control step.
        if use_compile and device == "cuda":
            try:
                self.policy.model = torch.compile(self.policy.model,
                                                  mode="reduce-overhead")
                warm = torch.zeros((1, cfg["cond_dim"]), device=device)
                for _ in range(3):
                    self.policy.sample(warm, stochastic=False)
                print("[diffusion] torch.compile (reduce-overhead) enabled")
            except Exception as e:
                print(f"[diffusion] torch.compile unavailable, using eager: {e}")

        self.use_angular_features = self.stats.get("use_angular_features", False)
        if self.use_angular_features:
            self.vel_min = np.array(self.stats["vel_min"], dtype=np.float32)
            self.vel_max = np.array(self.stats["vel_max"], dtype=np.float32)
            self.v_rng   = np.where((self.vel_max - self.vel_min) > 1e-8,
                                    self.vel_max - self.vel_min, 1.0)
        else:
            self.s_min = np.array(self.stats["state_min"], dtype=np.float32)
            self.s_max = np.array(self.stats["state_max"], dtype=np.float32)
            self.s_rng = np.where((self.s_max - self.s_min) > 1e-8,
                                  self.s_max - self.s_min, 1.0)
        self.a_min = np.array(self.stats["action_min"], dtype=np.float32)
        self.a_max = np.array(self.stats["action_max"], dtype=np.float32)
        self.a_rng = np.where((self.a_max - self.a_min) > 1e-8, self.a_max - self.a_min, 1.0)

        self._hist  = None          # rolling k-state history (features)
        self._queue = []            # remaining actions from the current chunk

    def _norm_s(self, x):
        if self.use_angular_features:
            q1, q2 = x[0], x[1]
            v = 2.0 * (x[2:] - self.vel_min) / self.v_rng - 1.0
            return np.array([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2),
                              v[0], v[1]], dtype=np.float32)
        return 2.0 * (x - self.s_min) / self.s_rng - 1.0

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
                if self.ddim_steps and self.ddim_steps > 0:
                    a_seq = self.policy.sample_ddim(
                        cond_t, num_inference_steps=self.ddim_steps).cpu().numpy()[0]
                else:
                    a_seq = self.policy.sample(cond_t, stochastic=False).cpu().numpy()[0]   # (H, nu) normalized
            a_seq = self._denorm_a(a_seq)
            self._queue = list(a_seq[: self.n_exec])

        return self._queue.pop(0)

def evaluate(hold_steps=40, angle_tol=0.20, vel_tol=1.0,
             max_steps=1200, dt_control=0.05, realtime=True, ckpt="diffusion_policy.pt",
             hybrid=True):
    controller_name = "diffusion"
    env = DoublePendulumEnv(render_mode=None, frame_skip=1)
    obs, _ = env.reset()
    key_handler = ArrowKeyForce(env.model)

    # Force the start state to the hanging-down configuration.
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    env.data.qpos[:2] = x0[:2]
    env.data.qvel[:2] = x0[2:]
    mujoco.mj_forward(env.model, env.data)
    obs = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])

    # Build the diffusion-policy controller. With hybrid=True it is wrapped in an
    # LQR catch that takes over once the swing-up arrives near the top and holds
    # the (20-Hz-unstabilizable) inverted equilibrium by re-closing the loop at
    # the integration rate -- so the pendulum stands still instead of limit-
    # cycling around upright.
    diff = DiffusionController(
        results_dir / ckpt,
        results_dir / "norm_stats.json",
        n_exec=1,
    )
    if hybrid:
        from hybrid_controller import HybridController
        ctrl = HybridController(diff, dt_sim=env.model.opt.timestep,
                                goal=np.array([np.pi, 0.0, 0.0, 0.0]))
        print("[hybrid] diffusion swing-up + LQR catch handoff enabled")
    else:
        ctrl = diff
    ctrl.reset(x0)

    dt_sim   = env.model.opt.timestep
    n_sub    = max(1, int(round(dt_control / dt_sim)))
    max_tau  = env.action_space.high[0]
    x_goal   = np.array([np.pi, 0.0, 0.0, 0.0])

    hist = {"t": [], "x": [], "u": []}
    consecutive_hold = 0
    success_step = None

    print(f"[{controller_name}] swing-up from bottom.")
    print("  Click the VIEWER window, then push the tip:")
    print("    W/Up = +Z   S/Down = -Z   A = -X   D = +X   Space = clear")
    print("    (MuJoCo reserves Left/Right arrows for history stepping -> use A/D)")
    print("  Arrow keys in THIS terminal also work while the terminal is focused.")
    print("  (Ctrl+right-drag in the viewer applies a mouse force too.)")

    # The callback MUST be passed into launch_passive: it is handed to the C++
    # Simulate object at construction. Assigning viewer.key_callback afterwards
    # is a no-op (the Handle has no such setter), which is why keys did nothing.
    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_handler.key_callback
    ) as viewer:
        # Match the env's fixed "side" camera instead of the default free view.
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        if cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        for step in range(max_steps):
            # 1. Observe the current state
            x = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])
            
            # 2. Compute the control action based on the state
            u = np.asarray(ctrl.action(x), dtype=np.float64).ravel()
            u = np.clip(u, -max_tau, max_tau)

            # 3. Log history
            hist["t"].append(step * dt_control)
            hist["x"].append(x.copy())
            hist["u"].append(u.copy())

            # 4. Check for success
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

            # 5. Apply the control action to the actuator
            substep = getattr(ctrl, "substep_action", None)
            env.data.ctrl[:] = u

            # 6. Step the physics engine forward
            for _ in range(n_sub):
                # Clear all applied forces at the start of the substep
                env.data.xfrc_applied[:] = 0.0

                # Apply the keyboard force
                env.data.xfrc_applied[key_handler.tip_id, :3] = key_handler.force

                # Apply the mouse perturbation (if active)
                if viewer.perturb.select > 0:
                    mujoco.mjv_applyPerturbForce(env.model, env.data, viewer.perturb)

                # Re-close the LQR catch at the integration rate (hybrid only) so
                # the inverted hold is regulated faster than the 20 Hz control rate.
                if substep is not None:
                    x_sub = np.concatenate([env.data.qpos[:2], env.data.qvel[:2]])
                    env.data.ctrl[:] = np.clip(substep(x_sub), -max_tau, max_tau)

                # Advance simulation by one timestep
                mujoco.mj_step(env.model, env.data)
            
            # 7. Clear the impulse force after it has been applied for one step
            key_handler.force[:] = 0.0

            # 8. Sync the viewer and wait to maintain real-time rate
            viewer.sync()

            if realtime:
                time.sleep(dt_control)

            if not viewer.is_running():
                break

    key_handler.stop()
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
    p.add_argument("--ckpt", default="diffusion_policy.pt",
                   help="checkpoint filename inside results/")
    p.add_argument("--hold-steps", type=int, default=40,
                   help="consecutive in-tolerance control steps to count as success")
    p.add_argument("--angle-tol", type=float, default=0.20)
    p.add_argument("--vel-tol", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--no-realtime", action="store_true",
                   help="run as fast as possible instead of wall-clock paced")
    p.add_argument("--no-hybrid", action="store_true",
                   help="disable the LQR catch handoff (pure diffusion policy, "
                        "which limit-cycles around the top instead of holding)")
    args = p.parse_args()

    evaluate(
        hold_steps=args.hold_steps,
        angle_tol=args.angle_tol,
        vel_tol=args.vel_tol,
        max_steps=args.max_steps,
        realtime=not args.no_realtime,
        ckpt=args.ckpt,
        hybrid=not args.no_hybrid,
    )