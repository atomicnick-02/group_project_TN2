import os
import sys

# --- libstdc++ ABI shim (must run before numpy / pyCandle / torch load) -------
# torch's bundled C++ runtime (libc10/libtorch) and pyCandle's native lib
# disagree on which libstdc++ to use; whichever loads second corrupts the
# other's codecvt/locale facets and segfaults (in torch's library init, or in
# the CANdle USB-thread setup). Forcing the system libstdc++ to preload before
# either one loads keeps a single, consistent C++ runtime in the process.
# LD_PRELOAD is read by the dynamic linker at exec time, so set it and re-exec
# ourselves once (guarded so we don't loop).
_LIBSTDCXX = "/usr/lib/libstdc++.so.6"
if os.path.exists(_LIBSTDCXX) and _LIBSTDCXX not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = ":".join(
        p for p in (_LIBSTDCXX, os.environ.get("LD_PRELOAD", "")) if p)
    os.execv(sys.executable, [sys.executable] + sys.argv)
# -----------------------------------------------------------------------------

import time
import json
import argparse
from pathlib import Path
import numpy as np


# Make the repo root importable (for `flow_matching` / `diffusion_models`)
# regardless of the directory this script is launched from.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
for _p in (_THIS, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Flow-matching policy checkpoint (drop a trained .pt + norm_stats.json here):
DEFAULT_CKPT = _THIS / "architectures" / "flow_matching" / "flow_matching_transformer.pt"
DEFAULT_STS  = _THIS / "architectures" / "flow_matching" / "norm_stats.json"


# Flow-matching-policy controller
class FlowMatchingController:
    """Closed-loop action-chunking controller backed by a FlowMatchingPolicy.

    Mirrors main_control.py:DiffusionController -- same angular-feature
    conditioning, same rolling k-state history, same n_exec action queue -- but
    generates the action chunk by integrating the learned flow ODE instead of
    running the diffusion reverse process.
    """

    def __init__(self, ckpt_path, stats_path, n_exec=2, num_steps=None, solver="euler"):
        import torch
        from flow_matching.flow_matching_policy import (
            MLP, TrajectoryTransformer, FlowMatchingPolicy,
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
        self.solver   = solver
        # num_steps: CLI override > checkpoint default. More steps = more accurate
        # ODE integration but slower; fewer = faster control loop.
        self.num_steps = num_steps if num_steps is not None else cfg.get("num_steps", 10)

        # Rebuild the exact architecture the checkpoint was trained with.
        arch       = cfg.get("arch", "transformer")
        net_kwargs = cfg.get("net_kwargs", {
            "horizon": cfg["horizon"], "action_dim": cfg["action_dim"],
            "cond_dim": cfg["cond_dim"], "hidden_dim": cfg.get("hidden_dim", 256),
        })
        net_cls    = {"mlp": MLP, "transformer": TrajectoryTransformer}[arch]
        network    = net_cls(**net_kwargs)
        print(f"[flow] loaded arch='{arch}' from {Path(ckpt_path).name} "
              f"(num_steps={self.num_steps}, solver={self.solver})")

        self.policy = FlowMatchingPolicy(
            network=network, device=device, horizon=cfg["horizon"],
            action_dim=cfg["action_dim"], num_steps=self.num_steps,
            sigma_min=cfg.get("sigma_min", 0.0),
        )
        # Inference runs on the EMA weights (what the checkpoint's model_state is).
        self.policy.ema_model.load_state_dict(ckpt["model_state"])
        self.policy.ema_model.eval()

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

    def _denorm_a(self, a):
        return (a + 1.0) * 0.5 * self.a_rng + self.a_min

    def reset(self, x0):
        x0n = self._norm_s(np.asarray(x0, dtype=np.float32))
        self._hist  = np.repeat(x0n[None], self.k, axis=0)   # (k, nx_feat)
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
                a_seq = self.policy.sample(
                    cond_t, num_steps=self.num_steps, solver=self.solver
                ).cpu().numpy()[0]                          # (H, nu) normalized
            a_seq = self._denorm_a(a_seq)
            self._queue = list(a_seq[: self.n_exec])

        return self._queue.pop(0)


# ==========================================
# 0. PARSE + VALIDATE ARGS  (no heavy imports yet)
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Physical double-pendulum flow-matching-policy control test")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help=f"path to the flow-matching checkpoint (.pt) (default: {DEFAULT_CKPT})")
    p.add_argument("--stats", type=Path, default=DEFAULT_STS,
                   help="path to norm_stats.json (default: norm_stats.json beside the checkpoint)")
    p.add_argument("--n-exec", type=int, default=1,
                   help="actions executed per generated chunk (default: 1)")
    p.add_argument("--num-steps", type=int, default=None,
                   help="ODE integration steps (default: from checkpoint)")
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler",
                   help="ODE solver for generation (default: euler)")
    p.add_argument("--max-torque", type=float, default=0.07,
                   help="per-motor torque clamp, Nm (physical hard stop)")
    p.add_argument("--dt", type=float, default=0.05, help="control period, s")
    return p.parse_args()


args = parse_args()
stats_path = args.stats if args.stats is not None else args.ckpt.parent / "norm_stats.json"
if not args.ckpt.exists():
    sys.exit(f"Error: checkpoint not found: {args.ckpt}")
if not stats_path.exists():
    sys.exit(f"Error: norm stats not found: {stats_path}")

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
# NOTE: pyCandle (and its USB/CAN native lib) must be brought up BEFORE torch is
# imported. Importing torch/CUDA first segfaults the CANdle native library, so
# the policy checkpoint is loaded in section 2, after the device is initialized.
import pyCandle
print("[1/3] Initializing Hardware...")
candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True, pyCandle.USB)
ids = candle.ping(pyCandle.CAN_BAUD_1M)

if len(ids) < 2:
    sys.exit(f"Error: Needed 2 motors, found {len(ids)}.")

for m_id in ids:
    candle.addMd80(m_id)
    candle.controlMd80SetEncoderZero(m_id)
    candle.controlMd80Mode(m_id, pyCandle.RAW_TORQUE)
    candle.controlMd80Enable(m_id, True)

candle.begin()
time.sleep(1)

motor_0 = candle.md80s[1]
motor_1 = candle.md80s[0]

# Hold zero torque while the (slow) policy checkpoint loads below.
motor_0.setTargetTorque(0.0)
motor_1.setTargetTorque(0.0)

# ==========================================
# 2. LOAD POLICY CHECKPOINT  (imports torch -> CUDA; safe now that pyCandle is up)
# ==========================================
print(f"[2/3] Loading checkpoint {args.ckpt} ...")
controller = FlowMatchingController(
    str(args.ckpt), str(stats_path),
    n_exec=args.n_exec, num_steps=args.num_steps, solver=args.solver,
)
print(f"[2/3] {type(controller).__name__} | horizon={controller.horizon} "
      f"| n_exec={controller.n_exec}")
controller.reset(np.array([
    motor_0.getPosition(),
    motor_1.getPosition(),
    motor_0.getVelocity(),
    motor_1.getVelocity(),
]))

# ==========================================
# 3. CONTROL LOOP
# ==========================================
print("[3/3] Running control loop (Ctrl-C to stop)...")
dt = args.dt
max_torque = args.max_torque

try:
    while True:
        loop_start = time.time()

        # --- A. SENSE ---
        p0 = motor_0.getPosition()
        p1 = motor_1.getPosition()
        v0 = motor_0.getVelocity()
        v1 = motor_1.getVelocity()
        x_real = np.array([p0, p1, v0, v1], dtype=float)

        # --- B. THINK ---
        torque = controller.action(x_real)
        torque = np.clip(torque, -max_torque, max_torque)
        print(f"p0: {p0:.6f}, p1: {p1:.6f}, torque1: {torque[0]:.6f}, torque2: {torque[1]:.6f}")

        # --- C. ACT ---
        motor_0.setTargetTorque(float(torque[0]))
        motor_1.setTargetTorque(float(torque[1]))

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("Emergency Stop!")

finally:
    # Always disable motors cleanly
    candle.controlMd80Enable(motor_0.getId(), False)
    candle.controlMd80Enable(motor_1.getId(), False)
    candle.end()
