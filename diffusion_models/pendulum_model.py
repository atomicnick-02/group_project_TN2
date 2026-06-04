import numpy as np
import torch
import torch.nn as nn


# ── Noise-prediction network ────────────────────────────────────────────────────
# Define MLP BEFORE DiffusionPolicy so it can be referenced at construction time.
# The forward(a_noisy, t, cond) contract below is what any drop-in replacement
# (e.g. a future Transformer) must also satisfy, so swapping architectures later
# is a one-line change at the call site.
class MLP(nn.Module):
    """
    Flatten-everything conditional MLP.

    Diffuses over an action sequence of shape (H, nu).
    Conditioning context `cond` (clean, NOT noised) is the flattened
    k-state history plus optional goal: dim = k*nx (+ nx if goal).

    The MLP flattens the action sequence to H*nu, so it ignores the temporal
    structure of the action chunk -- exactly what a Transformer would exploit.
    """

    def __init__(self, horizon: int, action_dim: int, cond_dim: int,
                 hidden_dim: int = 256, time_dim: int = 24, cond_embed_dim: int = 64):
        super().__init__()
        self.horizon    = horizon
        self.action_dim = action_dim
        flat_action     = horizon * action_dim

        # Embed the scalar (normalized) timestep
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Embed the conditioning context (state history + optional goal)
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        # Main net: [flattened_actions || t_embed || cond_embed] -> predicted noise
        self.net = nn.Sequential(
            nn.Linear(flat_action + time_dim + cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action),
        )

    def forward(self, a_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # a_noisy: (batch, H, nu)   t: (batch,)   cond: (batch, cond_dim)
        batch    = a_noisy.size(0)
        a_flat   = a_noisy.reshape(batch, -1)              # (batch, H*nu)
        t_embed  = self.time_embed(t.unsqueeze(-1))        # (batch, time_dim)
        c_embed  = self.cond_embed(cond)                   # (batch, cond_embed_dim)
        out_flat = self.net(torch.cat([a_flat, t_embed, c_embed], dim=-1))
        return out_flat.reshape(batch, self.horizon, self.action_dim)


# ── Noise scheduler ──────────────────────────────────────────────────────────────
class Scheduler:
    """Linear beta schedule with a forward-diffusion helper."""

    def __init__(self, num_steps: int, device, start_beta: float = 0.0003, end_beta: float = 0.03):
        self.num_steps  = num_steps
        self.device     = device
        self.beta_array = torch.linspace(start_beta, end_beta, num_steps).to(device)
        self.alphas     = 1 - self.beta_array
        self.alpha_bar  = torch.cumprod(self.alphas, dim=0)   # cumulative product alpha_bar_t
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Forward process q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I).
        x_0: (batch, H, nu), t: (batch,). Returns noisy actions + the noise.
        """
        epsilon = torch.randn_like(x_0)
        # reshape to (batch, 1, 1) so it broadcasts over horizon and action dims
        sqrt_alpha_bar           = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        z_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        return z_t, epsilon


# ── Conditional Diffusion Policy ────────────────────────────────────────────────
class DiffusionPolicy:
    """
    Diffusion Policy (Chi et al. 2023) over action sequences.

    Training samples are (cond, action_seq):
        cond       : (batch, cond_dim)   clean state history (+ goal), NOT noised
        action_seq : (batch, H, nu)      the target the diffusion denoises toward
    """

    def __init__(self, scheduler: Scheduler, network: nn.Module, device, timesteps: int,
                 horizon: int, action_dim: int, learning_rate: float = 1e-3):
        self.scheduler   = scheduler
        self.device      = device
        self.timesteps   = timesteps
        self.horizon     = horizon
        self.action_dim  = action_dim
        self.model       = network.to(device)
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn     = nn.MSELoss()
        self.loss_hist   = []

    def train(self, dataloader, epochs: int):
        """dataloader yields (cond, action_seq) batches."""
        size = len(dataloader.dataset)
        for epoch in range(epochs):
            self.model.train()
            for batch, (cond, action_seq) in enumerate(dataloader):
                cond       = cond.to(self.device)          # (batch, cond_dim)
                action_seq = action_seq.to(self.device)    # (batch, H, nu)
                t = torch.randint(1, self.timesteps, (action_seq.size(0),), device=self.device)

                # Only the actions are noised; cond stays clean
                z_t, epsilon = self.scheduler.add_noise(action_seq, t)
                t_normalized = t.float() / self.timesteps
                epsilon_pred = self.model(z_t, t_normalized, cond)

                loss = self.loss_fn(epsilon_pred, epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_hist.append(loss.item())

                if batch % 10 == 0:
                    current = batch * action_seq.size(0)
                    print(f"Epoch {epoch + 1}, loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    @torch.no_grad()
    def sample(self, cond: torch.Tensor):
        """
        Reverse diffusion (DDPM ancestral sampling) of one action sequence per
        conditioning vector. cond: (batch, cond_dim) -> returns (batch, H, nu).
        """
        self.model.eval()
        cond  = cond.to(self.device)
        batch = cond.size(0)

        a_t = torch.randn((batch, self.horizon, self.action_dim), device=self.device)
        for t in reversed(range(1, self.timesteps)):
            t_normalized = torch.full((batch,), t / self.timesteps,
                                      device=self.device, dtype=torch.float32)
            beta_t  = self.scheduler.beta_array[t]
            alpha_t = self.scheduler.alpha_bar[t]

            first  = (1 / torch.sqrt(1 - beta_t)) * a_t
            second = (beta_t / (torch.sqrt(1 - beta_t) * torch.sqrt(1 - alpha_t))) \
                     * self.model(a_t, t_normalized, cond)
            a_t = first - second

            noise = torch.randn_like(a_t) if t > 1 else torch.zeros_like(a_t)
            a_t  += torch.sqrt(beta_t) * noise

        return a_t  # (batch, H, nu)

    @torch.no_grad()
    def act(self, state_history, goal=None, n_exec: int = 1):
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
        action_seq = self.sample(sh)                  # (1, H, nu)
        return action_seq[0, :n_exec].cpu().numpy()