"""
Conditional Flow Matching policy over action sequences.

This is the flow-matching counterpart of diffusion_models/diffusion_policy.py.
Where the Diffusion Policy learns to predict the noise epsilon and removes it via
an iterative DDPM reverse process, the Flow Matching policy learns a *velocity
field* v_theta(x, t, cond) that transports a Gaussian source distribution into
the expert-action distribution, and generates by integrating an ODE.

Conventions (rectified-flow / optimal-transport linear path):
    x_0 ~ N(0, I)          source (pure noise)          at t = 0
    x_1 = expert actions    target (data)                at t = 1
    x_t = (1 - t) * x_0 + t * x_1        straight-line interpolant
    dx_t/dt = x_1 - x_0                  constant target velocity

Training (conditional flow matching, Lipman et al. 2023 / Liu et al. 2022):
    sample t ~ U(0, 1), x_0 ~ N(0, I)
    regress  v_theta(x_t, t, cond)  ->  (x_1 - x_0)     with an MSE loss

Generation:
    start from x = x_0 ~ N(0, I) and integrate dx/dt = v_theta(x, t, cond) from
    t = 0 to t = 1 with `num_steps` Euler (or midpoint) steps. The flow is
    deterministic given x_0 -- there is no per-step noise injection -- which makes
    it naturally well behaved for stabilizing control (cf. the DiffusionPolicy
    note about ancestral noise toppling the upright hold).

The noise-prediction backbones (MLP, TrajectoryTransformer) are reused verbatim
from diffusion_models: their contract forward(x, t_normalized, cond) -> (B, H, nu)
with t in [0, 1] is exactly what a velocity network needs, so a flow-matching
model and a diffusion model with the same --arch share an identical network.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse the diffusion backbones (same forward(x, t, cond) -> (B, H, nu) contract).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from diffusion_models.diffusion_policy import MLP, TrajectoryTransformer  # noqa: E402

__all__ = ["MLP", "TrajectoryTransformer", "FlowMatchingPolicy"]


class FlowMatchingPolicy:
    """
    Conditional Flow Matching policy over action sequences (Chi-style action
    chunking, but with a flow-matching generative head instead of diffusion).

    Training samples are (cond, action_seq):
        cond       : (batch, cond_dim)   clean state history (+ goal), NOT noised
        action_seq : (batch, H, nu)      the data endpoint x_1 the flow targets

    Backbone-agnostic: pass either MLP(...) or TrajectoryTransformer(...).
    """

    def __init__(self, network: nn.Module, device, horizon: int, action_dim: int,
                 num_steps: int = 10, learning_rate: float = 1e-4,
                 ema_decay: float = 0.999, sigma_min: float = 0.0):
        self.device      = device
        self.horizon     = horizon
        self.action_dim  = action_dim
        # num_steps is the number of Euler integration steps used at GENERATION
        # time (the flow-matching analogue of the diffusion timestep count). It
        # does not affect the training objective, only the sampler.
        self.num_steps   = num_steps
        # sigma_min keeps a thin band of source noise around the data endpoint:
        # x_t = (1 - (1 - sigma_min) t) x_0 + t x_1. 0.0 = pure rectified flow.
        self.sigma_min   = sigma_min

        self.model       = network.to(device)
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn       = nn.MSELoss()
        self._lr_scheduler = None
        self._scaler       = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        self.loss_hist     = []

        # Exponential moving average of the weights -- a smoother estimate of the
        # learned velocity field than the last-step weights, which gives a more
        # CONSISTENT flow near the unstable upright equilibrium (less torque
        # jitter). Inference uses ema_model (the train script saves it as the ckpt).
        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model).to(device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def _update_ema(self):
        for pe, p in zip(self.ema_model.parameters(), self.model.parameters()):
            pe.mul_(self.ema_decay).add_(p.detach(), alpha=1.0 - self.ema_decay)
        for be, b in zip(self.ema_model.buffers(), self.model.buffers()):
            be.copy_(b)

    def _interpolate(self, x_0, x_1, t):
        """Linear conditional path x_t and its (constant) target velocity.

        t: (batch,) in [0, 1]. Broadcasts over (H, nu).
        Returns (x_t, target_velocity).
        """
        t_b   = t.view(-1, 1, 1)
        # x_t = (1 - (1 - sigma_min) t) x_0 + t x_1 ; with sigma_min=0 this is the
        # pure rectified-flow straight line (1 - t) x_0 + t x_1.
        scale = 1.0 - (1.0 - self.sigma_min) * t_b
        x_t   = scale * x_0 + t_b * x_1
        v_tgt = x_1 - (1.0 - self.sigma_min) * x_0
        return x_t, v_tgt

    def train(self, dataloader, epochs: int, on_epoch_end=None):
        """dataloader yields (cond, action_seq) batches.

        on_epoch_end: optional callback(epoch, avg_loss) invoked after each epoch
        (epoch is 1-based) -- used e.g. for periodic checkpointing.
        """
        total_steps = epochs * len(dataloader)
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        use_amp = self.device == "cuda"
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for cond, action_seq in dataloader:
                cond       = cond.to(self.device)          # (batch, cond_dim)
                x_1        = action_seq.to(self.device)     # (batch, H, nu) data endpoint
                batch      = x_1.size(0)

                x_0 = torch.randn_like(x_1)                                   # source noise
                t   = torch.rand(batch, device=self.device)                   # U(0, 1)
                x_t, v_target = self._interpolate(x_0, x_1, t)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_pred = self.model(x_t, t, cond)       # cond stays clean (not noised)
                    loss   = self.loss_fn(v_pred, v_target)

                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self._lr_scheduler.step()    # advance cosine schedule per-step
                self._update_ema()           # track EMA weights for inference

                epoch_losses.append(loss.item())
                self.loss_hist.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            lr_now   = self._lr_scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1:>3d}/{epochs}  avg_loss={avg_loss:.5f}  lr={lr_now:.2e}")

            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, avg_loss)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = None, solver: str = "euler"):
        """
        Generate one action sequence per conditioning vector by integrating the
        learned ODE dx/dt = v_theta(x, t, cond) from t=0 (noise) to t=1 (data).

        cond: (batch, cond_dim) -> returns (batch, H, nu).

        solver: "euler" (1st order, num_steps evals) or "midpoint" (2nd order,
        2*num_steps evals, lower discretization error for the same step count).
        Both are deterministic given the initial noise x_0, unlike DDPM ancestral
        sampling -- desirable for stabilizing an unstable equilibrium.
        """
        steps = num_steps if num_steps is not None else self.num_steps
        model = self.ema_model
        model.eval()
        cond  = cond.to(self.device)
        batch = cond.size(0)

        x  = torch.randn((batch, self.horizon, self.action_dim), device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch,), i * dt, device=self.device, dtype=torch.float32)
            if solver == "midpoint":
                v_half = model(x, t, cond)
                x_mid  = x + 0.5 * dt * v_half
                v      = model(x_mid, t + 0.5 * dt, cond)
            else:  # euler
                v = model(x, t, cond)
            x = x + dt * v

        return x  # (batch, H, nu) ~ data distribution

    @torch.no_grad()
    def act(self, state_history, goal=None, n_exec: int = 1, num_steps: int = None,
            solver: str = "euler"):
        """
        Convenience wrapper for closed-loop control.

        state_history : (k, nx) array/tensor of the last k observed states
        goal          : (nx,) optional goal vector (must match training config)
        n_exec        : how many leading actions of the horizon to return

        Returns the first n_exec actions as a numpy array (n_exec, nu).
        """
        sh = torch.as_tensor(np.asarray(state_history), dtype=torch.float32).reshape(1, -1)
        if goal is not None:
            g  = torch.as_tensor(np.asarray(goal), dtype=torch.float32).reshape(1, -1)
            sh = torch.cat([sh, g], dim=-1)
        action_seq = self.sample(sh, num_steps=num_steps, solver=solver)   # (1, H, nu)
        return action_seq[0, :n_exec].cpu().numpy()
