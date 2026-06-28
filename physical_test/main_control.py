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


# Make the repo root importable (for `diffusion_models`) regardless of the
# directory this script is launched from.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
for _p in (_THIS, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Diffusion policy:
DEFAULT_CKPT = _THIS / "architectures" / "diffusion" / "diffusion_T5.pt"
DEFAULT_STS  = _THIS / "architectures" / "diffusion" / "norm_stats.json"
# BC policy:
# DEFAULT_CKPT = _THIS / "architectures" / "bc" / "bc_policy_3.pt"
# DEFAULT_STS  = _THIS / "architectures" / "bc" / "norm_stats.json"
# Diffusion-policy controller
class DiffusionController:


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

        # Rebuild the exact architecture the checkpoint was trained with.
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
            self.policy.model.eval()
            cond = self._hist.reshape(-1)
            if self.use_goal:
                goal_n = self._norm_s(np.array([np.pi, 0, 0, 0], dtype=np.float32))
                cond = np.concatenate([cond, goal_n])
            cond_t = self.torch.from_numpy(cond[None].astype(np.float32))
            with self.torch.no_grad():
                a_seq = self.policy.sample(cond_t, stochastic=False).cpu().numpy()[0]   # (H, nu) normalized
            a_seq = self._denorm_a(a_seq)
            self._queue = list(a_seq[: self.n_exec])

        return self._queue.pop(0)

# Behavioral-cloning controller (plain MLP baseline)
class BCController:


    def __init__(self, ckpt_path, stats_path):
        import torch
        import torch.nn as nn
        self.torch = torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        with open(stats_path) as fp:
            self.stats = json.load(fp)
        self.horizon = int(self.stats["horizon"])
        self.nu      = int(self.stats.get("nu", 2))

        # angular-feature velocity normalization + action denormalization
        self.vel_min = np.array(self.stats["vel_min"], dtype=np.float32)
        self.vel_max = np.array(self.stats["vel_max"], dtype=np.float32)
        self.v_rng   = np.where((self.vel_max - self.vel_min) > 1e-8,
                                self.vel_max - self.vel_min, 1.0)
        self.a_min = np.array(self.stats["action_min"], dtype=np.float32)
        self.a_max = np.array(self.stats["action_max"], dtype=np.float32)
        self.a_rng = np.where((self.a_max - self.a_min) > 1e-8, self.a_max - self.a_min, 1.0)

        # Mirror of behaviour_cloning/behavioral_cloning.py:BCPolicyModel. The
        # checkpoint is a bare state_dict, so this layer stack must match 1:1
        # (Linear 6->512->256->256->128->2*H, Tanh between, final Tanh).
        class BCPolicyModel(nn.Module):
            def __init__(self, horizon, action_dim):
                super().__init__()
                self.horizon, self.action_dim = horizon, action_dim
                self.layers = nn.Sequential(
                    nn.Linear(6, 512),   nn.Tanh(), nn.Dropout(0.2),
                    nn.Linear(512, 256), nn.Tanh(), nn.Dropout(0.2),
                    nn.Linear(256, 256), nn.Tanh(), nn.Dropout(0.2),
                    nn.Linear(256, 128), nn.Tanh(), nn.Dropout(0.2),
                    nn.Linear(128, action_dim * horizon), nn.Tanh(),
                )

            def forward(self, x):
                return self.layers(x).view(-1, self.horizon, self.action_dim)

        self.model = BCPolicyModel(self.horizon, self.nu).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:   # tolerate wrapped ckpts
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[bc] loaded BCPolicyModel(horizon={self.horizon}) from {Path(ckpt_path).name}")

    def _feat(self, x):
        q1, q2 = x[0], x[1]
        v = 2.0 * (x[2:] - self.vel_min) / self.v_rng - 1.0
        return np.array([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), v[0], v[1]],
                        dtype=np.float32)

    def _denorm_a(self, a):
        return (a + 1.0) * 0.5 * self.a_rng + self.a_min

    def reset(self, x0=None):
        # stateless: the MLP re-plans from the current observation every step
        pass

    def action(self, x):
        f  = self._feat(np.asarray(x, dtype=np.float32))
        ft = self.torch.from_numpy(f[None]).to(self.device)       # (1, 6) on model's device
        self.model.eval()
        with self.torch.no_grad():
            a_seq = self.model(ft).cpu().numpy()[0]               # (H, nu) normalized
        return self._denorm_a(a_seq[0])



# ==========================================
# 0. PARSE + VALIDATE ARGS  (no heavy imports yet)
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Physical double-pendulum diffusion-policy control test")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="path to the diffusion-policy checkpoint (.pt) "
                        f"(default: {DEFAULT_CKPT})")
    p.add_argument("--stats", type=Path, default=DEFAULT_STS,
                   help="path to norm_stats.json "
                        "(default: norm_stats.json beside the checkpoint)")
    p.add_argument("--n-exec", type=int, default=2,
                   help="actions executed per sampled chunk (default: 2)")
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
# Pick the controller from the checkpoint itself, so switching policies only
# requires toggling DEFAULT_CKPT/DEFAULT_STS above. A diffusion checkpoint is a
# dict with config["arch"] in {"mlp","transformer"}; a BC checkpoint is either
# config["arch"] == "bc" (wrapped) or a bare state_dict (no "config").
import torch
_peek = torch.load(str(args.ckpt), map_location="cpu")
_arch = _peek["config"].get("arch", "mlp") if isinstance(_peek, dict) and "config" in _peek else "bc"
del _peek
if _arch == "bc":
    controller = BCController(str(args.ckpt), str(stats_path))
else:
    controller = DiffusionController(str(args.ckpt), str(stats_path), n_exec= 1 )
print(f"[2/3] arch='{_arch}' -> {type(controller).__name__} | horizon={controller.horizon}")
controller.reset(np.array([
    motor_0.getPosition(),
    motor_1.getPosition(),
    motor_0.getVelocity(),
    motor_1.getVelocity(),
]))

# hist = np.zeros(shape = (controller.nx , controller.k))

# ==========================================
# 3. CONTROL LOOP
# ==========================================
print("[3/3] Running control loop (Ctrl-C to stop)...")
dt = 0.05
if 'bc' == type(controller).__name__:
    max_torque = 0.1          # Physical hard stop
else :
    max_torque = 0.07          # Physical hard stop


current_idx = 0
try:
    while True:
        loop_start = time.time()

        # --- A. SENSE ---
        p0 = motor_0.getPosition()
        p1 = motor_1.getPosition()
        v0 = motor_0.getVelocity()
        v1 = motor_1.getVelocity()
        x_real = np.array([p0, p1, v0, v1], dtype=float)
        torque = controller.action(x_real)
        print(f"p0: {p0:.6f}, p1: {p1:.6f}, torque1: {torque[0]:.6f}, torque2: {torque[1]:.6f}")
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
