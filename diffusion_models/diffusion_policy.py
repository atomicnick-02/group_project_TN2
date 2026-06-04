import numpy as np
import torch
import torch.nn as nn

# ── Noise-prediction network: MLP ───────────────────────────────────────────────
# Define networks BEFORE DiffusionPolicy so they can be referenced at construction
# time. The forward(a_noisy, t, cond) contract below is shared by every backbone,
# so swapping MLP <-> TrajectoryTransformer is a one-line change at the call site.
class MLP(nn.Module):
    """
    Flatten-everything conditional MLP.

    Diffuses over an action sequence of shape (H, nu).
    Conditioning context `cond` (clean, NOT noised) is the flattened
    k-state history plus optional goal: dim = k*nx (+ nx if goal).

    The MLP flattens the action sequence to H*nu, so it ignores the temporal
    structure of the action chunk -- exactly what the Transformer below exploits.
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


# ── Sinusoidal timestep embedding (shared utility) ──────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) normalized in [0, 1]. Scale up so frequencies are useful.
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args   = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # zero-pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (batch, dim)


# ── Noise-prediction network: Transformer ──────────────────────────────────────
class TrajectoryTransformer(nn.Module):
    """
    Conditional Transformer over the action chunk.

    Unlike the MLP, this treats the H action steps as a SEQUENCE of tokens and
    attends across them, preserving temporal structure. The diffusion timestep
    and the conditioning vector are prepended as two extra context tokens so
    every action token can attend to them.

    Same forward(a_noisy, t, cond) contract as MLP -> drop-in replacement.

    Token layout fed to the encoder:
        [ t_token, cond_token, a_1, a_2, ..., a_H ]   length = H + 2
    Only the H action-position outputs are projected back to nu.
    """

    def __init__(self, horizon: int, action_dim: int, cond_dim: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.0):
        super().__init__()
        self.horizon    = horizon
        self.action_dim = action_dim
        self.d_model    = d_model

        # Project a single action (nu,) to a token, and learn its position in the chunk.
        self.action_in  = nn.Linear(action_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, horizon, d_model))

        # Timestep token: sinusoidal embed -> d_model
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Conditioning token: cond vector -> d_model
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out   = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, action_dim)

        nn.init.normal_(self.pos_embed, std=0.002) # small init for positional embeddings

    def forward(self, a_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # a_noisy: (batch, H, nu)   t: (batch,)   cond: (batch, cond_dim)
        a_tok = self.action_in(a_noisy) + self.pos_embed       # (batch, H, d_model)
        t_tok = self.time_embed(t).unsqueeze(1)                # (batch, 1, d_model)
        c_tok = self.cond_embed(cond).unsqueeze(1)             # (batch, 1, d_model)

        # Prepend the two context tokens; attention is full (no mask needed).
        tokens = torch.cat([t_tok, c_tok, a_tok], dim=1)       # (batch, H+2, d_model)
        out    = self.encoder(tokens)
        out    = self.norm_out(out[:, 2:])                     # keep only action positions
        return self.action_out(out)                            # (batch, H, nu)


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

    Backbone-agnostic: pass either MLP(...) or TrajectoryTransformer(...).
    """

    def __init__(self, scheduler: Scheduler, network: nn.Module, device, timesteps: int,
                 horizon: int, action_dim: int, learning_rate: float = 1e-3):
        self.scheduler   = scheduler
        self.device      = device
        self.timesteps   = timesteps
        self.horizon     = horizon
        self.action_dim  = action_dim
        self.model       = network.to(device)
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn       = nn.MSELoss()
        self._lr_scheduler = None
        self._scaler       = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        self.loss_hist     = []

    def train(self, dataloader, epochs: int):
        """dataloader yields (cond, action_seq) batches."""
        size = len(dataloader.dataset)
        total_steps = epochs * len(dataloader)
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        use_amp = self.device == "cuda"
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for batch, (cond, action_seq) in enumerate(dataloader):
                cond       = cond.to(self.device)          # (batch, cond_dim)
                action_seq = action_seq.to(self.device)    # (batch, H, nu)
                t = torch.randint(1, self.timesteps, (action_seq.size(0),), device=self.device)

                # Only the actions are noised; cond stays clean
                z_t, epsilon = self.scheduler.add_noise(action_seq, t)
                t_normalized = t.float() / self.timesteps

                with torch.amp.autocast("cuda", enabled=use_amp):
                    epsilon_pred = self.model(z_t, t_normalized, cond)
                    loss = self.loss_fn(epsilon_pred, epsilon)

                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self._lr_scheduler.step()

                epoch_losses.append(loss.item())
                self.loss_hist.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            lr_now   = self._lr_scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1:>3d}/{epochs}  avg_loss={avg_loss:.5f}  lr={lr_now:.2e}")

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