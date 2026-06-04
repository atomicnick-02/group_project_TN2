import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ── Hyperparameters ────────────────────────────────────────────────────────────
TIMESTEPS  = 100
BATCH_SIZE = 100
EPOCHS     = 100
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ── Dataset ────────────────────────────────────────────────────────────────────

class ToyDataset(Dataset):
    """Simple 2D point dataset arranged on a noisy unit circle."""

    def __init__(self, num_samples: int = 1000):
        self.data = self._generate(num_samples)

    def _generate(self, num_samples: int) -> torch.Tensor:
        angles = torch.rand(num_samples) * 2 * np.pi
        radius = 1.0 + torch.randn(num_samples) * 0.05  # slight radial noise
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        return torch.stack([x, y], dim=1).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset    = ToyDataset(num_samples=5000)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# ── Noise Scheduler ────────────────────────────────────────────────────────────
class Scheduler:
    """Linear beta schedule with a forward-diffusion helper."""

    def __init__(self, num_steps: int, beta_start: float = 0.0003, beta_end: float = 0.03):
        self.num_steps  = num_steps
        self.beta_array = torch.linspace(beta_start, beta_end, num_steps).to(DEVICE)
        self.alphas     = 1 - self.beta_array
        self.alpha_bar  = torch.cumprod(self.alphas, dim=0)   # ᾱ_t = ∏ᵢ αᵢ
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Forward process: q(x_t | x_0) = N(√ᾱ_t · x_0, (1 − ᾱ_t) · I)
        Returns the noisy sample and the noise that was added.
        """
        epsilon                  = torch.randn_like(x_0)
        sqrt_alpha_bar           = torch.sqrt(self.alpha_bar[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        z_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        return z_t, epsilon


scheduler = Scheduler(num_steps=TIMESTEPS)


# ── Noise-prediction Network (time-conditioned MLP) ────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, time_dim: int = 24):
        super().__init__()

        # Embed the scalar timestep into a higher-dimensional representation
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Main network: [x ‖ t_embed] → predicted noise ε
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x, t_embed], dim=-1))


model     = MLP(input_dim=2, hidden_dim=128, time_dim=24).to(DEVICE)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_hist = []


# ── Training ───────────────────────────────────────────────────────────────────
def train_epoch(dataloader, model, scheduler, loss_fn, optimizer):
    """Single training epoch: predict and regress the added noise."""
    model.train()
    for x_0 in dataloader:
        x_0          = x_0.to(DEVICE)
        t            = torch.randint(1, TIMESTEPS, (x_0.size(0),), device=DEVICE)
        z_t, epsilon = scheduler.add_noise(x_0, t)

        t_normalized = t.float() / TIMESTEPS
        epsilon_pred = model(z_t, t_normalized)

        loss = loss_fn(epsilon_pred, epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())


for epoch in range(EPOCHS):
    train_epoch(dataloader, model, scheduler, loss_fn, optimizer)


# ── Sampling (reverse diffusion) ───────────────────────────────────────────────
def sample(model, num_samples: int, timesteps: int):
    """
    Ancestral sampling: iteratively denoise x_T ~ N(0,I) back to x_0
    using the DDPM update rule.
    """
    model.eval()
    save_at       = {timesteps - 1, timesteps // 2, timesteps // 5, timesteps // 10, 1}
    intermediates = []

    with torch.no_grad():
        x_t = torch.randn((num_samples, 2), device=DEVICE)

        for t in reversed(range(1, timesteps)):
            t_norm  = torch.full((num_samples,), t / timesteps, device=DEVICE, dtype=torch.float32)
            beta_t  = scheduler.beta_array[t]
            alpha_t = scheduler.alpha_bar[t]

            # DDPM mean estimate (eq. 11 in Ho et al. 2020)
            x_t = (
                (1 / torch.sqrt(1 - beta_t)) * x_t
                - (beta_t / (torch.sqrt(1 - beta_t) * torch.sqrt(1 - alpha_t))) * model(x_t, t_norm)
            )

            # Add stochastic noise for all steps except the last
            noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            x_t  += torch.sqrt(beta_t) * noise

            if t in save_at:
                intermediates.append((t, x_t.cpu().numpy()))

    return x_t.cpu().numpy(), intermediates


# ── Visualisation Utilities ────────────────────────────────────────────────────
def visualize_schedules(scheduler: Scheduler):
    """Plot the beta and alpha-bar schedules side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(scheduler.beta_array.cpu(), label='β schedule')
    ax1.set(title='Beta Schedule', xlabel='Timestep', ylabel='β')
    ax1.grid(); ax1.legend()

    ax2.plot(scheduler.alpha_bar.cpu(), color='orange', label='ᾱ schedule')
    ax2.set(title='Alpha-bar Schedule', xlabel='Timestep', ylabel='ᾱ')
    ax2.grid(); ax2.legend()

    plt.tight_layout()
    plt.savefig('alpha_beta_schedules.png')


def visualize_forward_diffusion(dataset: ToyDataset, scheduler: Scheduler, n_plots: int = 10):
    """Show how the data distribution gradually becomes isotropic Gaussian."""
    step             = TIMESTEPS // n_plots
    timesteps_to_plot = [t * step for t in range(n_plots)]

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 3, 3))
    x_0       = dataset.data[:BATCH_SIZE].to(DEVICE)

    for ax, t in zip(axes, timesteps_to_plot):
        t_tensor = torch.full((BATCH_SIZE,), t, device=DEVICE, dtype=torch.long)
        x_t, _   = scheduler.add_noise(x_0, t_tensor)
        x_t      = x_t.cpu().numpy()

        ax.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=10)
        ax.set_title(f"t={t}")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect('equal'); ax.grid()

    plt.tight_layout()
    plt.savefig('forward_diffusion_visualization.png')


# ── Run sampling and plot results ──────────────────────────────────────────────
num_samples   = 1000
samples, intermediates = sample(model, num_samples, TIMESTEPS)
original_data = dataset.data[:num_samples].cpu().numpy()

# Sampling progression across saved timesteps
plt.figure(figsize=(20, 5))
for i, (t_val, sample_at_t) in enumerate(intermediates):
    plt.subplot(1, len(intermediates), i + 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.1, s=5, color='gray', label='Original')
    plt.scatter(sample_at_t[:, 0],   sample_at_t[:, 1],  alpha=0.5, s=5, color='blue',  label='Samples')
    plt.title(f'Step {t_val}')
    plt.xlim(-3, 3); plt.ylim(-3, 3)
    plt.legend(); plt.grid()
plt.tight_layout()

# Training loss curve
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='Training Loss')
plt.title('Training Loss History')
plt.xlabel('Batch'); plt.ylabel('Loss')
plt.grid(); plt.legend()
plt.show()