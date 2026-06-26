import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_model import SinusoidalPosEmb, DDPMScheduler  # noqa: F401


class DiffusionScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps

        self.betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)

        self.alphas = 1.0 - self.betas

        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.tensor([1.0]), self.alphas_bar[:-1]])

        self.post_var = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        self.post_var = self.post_var.to(device)
        return self

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar = self.alphas_bar[t].view(-1, 1, 1)

        x_noisy = torch.sqrt(alpha_bar) * x + (torch.sqrt(1.0 - alpha_bar)) * noise
        return x_noisy, noise

    def denoise(self, x_pred, x_sample, t):
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_bar[t].view(-1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1)

        x_prev = (1 / torch.sqrt(alpha)) * (
            x_sample - (beta / torch.sqrt(1 - alpha_bar)) * x_pred
        )

        if t > 0:
            noise = torch.randn_like(x_pred)
            # x_prev += torch.sqrt(self.post_var[t].view(-1, 1, 1)) * noise
            # x_prev += torch.sqrt(self.betas[t].view(-1, 1, 1)) * noise

        return x_prev

    def score(self, noise_pred, t):
        alpha_bar = self.alphas_bar[t].view(-1, 1, 1)
        return -noise_pred / torch.sqrt(1.0 - alpha_bar)

    def pf_ode_drift(self, noise_pred, x_t, t):
        return 0.5 * (x_t + self.score(noise_pred, t))

    def ddim_step(self, noise_pred, x_t, t, t_prev, eta=0.0):

        alphas_bar = self.alphas_bar[t].view(-1, 1, 1)
        alphas_bar_prev = (
            self.alphas_bar[t_prev].view(-1, 1, 1)
            if t_prev >= 0
            else torch.ones_like(alphas_bar)
        )
        x0 = (x_t - torch.sqrt(1.0 - alphas_bar) * noise_pred) / torch.sqrt(alphas_bar)

        sigma = eta * torch.sqrt(
            (1.0 - alphas_bar_prev)
            / (1.0 - alphas_bar)
            * (1.0 - alphas_bar / alphas_bar_prev)
        )

        dx = torch.sqrt((1.0 - alphas_bar_prev - sigma**2).clamp(min=0.0)) * noise_pred

        x_prev = torch.sqrt(alphas_bar_prev) * x0 + dx
        if t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev


class TransformerDenoiser(nn.Module):

    def __init__(
        self,
        action_dim,
        obs_dim,
        goal_dim=0,
        horizon=8,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.goal_dim = goal_dim

        self.act_in = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        nn.init.normal_(self.pos_emb, std=0.002)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.state_emb = nn.Sequential(
            nn.Linear(obs_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        if goal_dim > 0:
            self.goal_emb = nn.Sequential(
                nn.Linear(goal_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )


        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.act_out = nn.Linear(d_model, action_dim)

    def forward(self, a_t, t, state, goal=None):

        action_token = self.act_in(a_t) + self.pos_emb
        time_token = self.time_emb(t).unsqueeze(1)
        state_token = self.state_emb(state).unsqueeze(1)
        full_token = torch.cat([time_token, action_token, state_token], dim=1)

        x = self.encoder(full_token)
        x = self.norm_out(x[:, 1:1 + self.horizon])

        return self.act_out(x)


class EMAModel:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_to(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def save_state(self):
        return {name: param.clone() for name, param in self.shadow.items()}


class DiffusionPolicy:
    def __init__(
        self,
        action_dim,
        obs_dim,
        device,
        optimizer=None,
        loss_fn=None,
        lr_scheduler=None,
        goal_dim=0,
        horizon=8,
        num_train_timesteps=100,
        d_model=256,
        n_heads=4,
        n_layers=4,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.scheduler = DiffusionScheduler(num_train_timesteps).to(device)
        self.transformer = TransformerDenoiser(
            action_dim, obs_dim, goal_dim, horizon, d_model, n_heads, n_layers
        ).to(device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        self.ema = EMAModel(self.transformer)
        self.loss_hist = []
        self.val_loss_hist = []

        self.ckpt_path = None
        self.best_loss = float("inf")

    def set_checkpoint_path(self, path):
        """Set where train() saves the best checkpoint. Call after construction."""
        self.ckpt_path = path
        ckpt_dir = os.path.dirname(path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(self, path=None, use_ema=True):
        """Save the denoiser weights (EMA by default) to `path` or self.ckpt_path."""
        path = path or self.ckpt_path
        if path is None:
            raise ValueError("No checkpoint path set. Call set_checkpoint_path() first.")
        state_dict = self.transformer.state_dict()
        if use_ema:
            # Overlay EMA parameter values onto the full state dict (keeps buffers).
            for name, param in self.ema.shadow.items():
                state_dict[name] = param.clone()
        torch.save(state_dict, path)
        return path

    def train(self, data, epochs, val_data=None):

        for epoch in range(epochs):
            self.transformer.train()
            running_train_loss = 0.0
            for states, actions in data:
                states = states.to(self.device)
                actions = actions.to(self.device)
                t = torch.randint(1, self.num_train_timesteps, (actions.size(0),), device=self.device)
                action_noisy, noise = self.scheduler.add_noise(actions, t)
                t_norm = t / self.num_train_timesteps
                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    pred = self.transformer(action_noisy, t_norm, states)
                    loss = self.loss_fn(pred, noise)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.transformer)

                running_train_loss += loss.item() * actions.size(0)

            self.lr_scheduler.step()
            epoch_loss = running_train_loss / len(data.dataset)
            self.loss_hist.append(epoch_loss)

            msg = f"epoch {epoch + 1}/{epochs} - train loss: {epoch_loss:.6f}"
            monitored = epoch_loss
            if val_data is not None:
                val_loss = self.evaluate(val_data)
                self.val_loss_hist.append(val_loss)
                msg += f" - val loss: {val_loss:.6f}"
                monitored = val_loss

            if self.ckpt_path is not None and monitored < self.best_loss:
                self.best_loss = monitored
                self.save_checkpoint(use_ema=True)
                msg += f"  (saved best -> {self.ckpt_path})"
            print(msg)

    @torch.no_grad()
    def evaluate(self, data):
        self.transformer.eval()
        running_val_loss = 0.0
        for states, actions in data:
            states = states.to(self.device)
            actions = actions.to(self.device)
            t = torch.randint(1, self.num_train_timesteps, (actions.size(0),), device=self.device)
            action_noisy, noise = self.scheduler.add_noise(actions, t)
            t_norm = t / self.num_train_timesteps
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                pred = self.transformer(action_noisy, t_norm, states)
                loss = self.loss_fn(pred, noise)
            running_val_loss += loss.item() * actions.size(0)
        return running_val_loss / len(data.dataset)


    @torch.no_grad()
    def sample(
        self,
        state,
        goal=None,
        n_samples=None,
        sampler="ddpm",
        num_steps=None,
        eta=0.0,
        ode_solver="heun",
    ):

        self.transformer.eval()
        device = next(self.transformer.parameters()).device
        N = state.shape[0] if n_samples is None else n_samples
        a = torch.randn(N, self.horizon, self.action_dim, device=device)
        T = self.scheduler.num_train_timesteps

        if sampler == "ddpm":
            for t in reversed(range(1,T)):
                # Normalize t to match training (t_norm = t / num_train_timesteps).
                t_batch = torch.full((N,), t / self.num_train_timesteps, device=device, dtype=torch.float32)
                noise_pred = self.transformer(a, t_batch, state, goal)
                a = self.scheduler.denoise(noise_pred, a, t)
            return a

        elif sampler == "ddim":
            num_steps = num_steps or T
            ts = torch.linspace(T - 1, 1, num_steps, device=device).long().tolist()
            for i, t in enumerate(ts):
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                # Normalize t to match training (t_norm = t / num_train_timesteps).
                t_batch = torch.full((N,), t / self.num_train_timesteps, device=device, dtype=torch.float32)
                noise_pred = self.transformer(a, t_batch, state)
                a = self.scheduler.ddim_step(noise_pred, a, t, t_prev, eta=eta)
            return a

        elif sampler == "ode":
            num_steps = num_steps or T
            ts = torch.linspace(T - 1, 1, num_steps, device=device).long().tolist()
            log_abar = torch.log(self.scheduler.alphas_bar)

            for i, t in enumerate(ts):
                t_batch = torch.full((N,), t, device=device, dtype=torch.float32)
                eps = self.transformer(a, t_batch, state, goal)
                drift = self.scheduler.pf_ode_drift(eps, a, t)

                if i + 1 < len(ts):
                    t_prev = ts[i + 1]
                    delta = log_abar[t_prev] - log_abar[t]
                else:
                    t_prev = -1
                    delta = -log_abar[t]

                a_euler = a + delta * drift

                if ode_solver == "heun" and t_prev > 0:
                    tp_batch = torch.full((N,), t_prev, device=device, dtype=torch.float32)
                    eps2 = self.transformer(a_euler, tp_batch, state, goal)
                    drift2 = self.scheduler.pf_ode_drift(eps2, a_euler, t_prev)
                    a = a + delta * 0.5 * (drift + drift2)
                else:
                    a = a_euler
            return a

        else:
            raise ValueError(f"unknown sampler: {sampler}")


if __name__ == "__main__":
    model = TransformerDenoiser()

    test_sample = torch.randn(2, 8, 7)
    test_time = torch.randint(0, 100, (2,))
    test_cond = torch.randn(2, 32)

    out = model(test_sample, test_time, test_cond)
    print("DiffusionTransformer output shape:", out.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {n_params:,}")
