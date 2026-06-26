import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """
    Maps discrete diffusion timesteps (k) to continuous embedding vectors.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    A residual MLP block with LayerNorm and Dropout for stable gradient flow.
    The skip connection allows the network to easily learn identity mappings,
    which is critical for diffusion models at low noise timesteps.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.Mish()
    
    def forward(self, x):
        return self.act(x + self.net(x))  # Skip connection


class DiffusionTrajectoryDenoiser(nn.Module):
    def __init__(self, state_dim=32, horizon=8, action_dim=7, model_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # 1. Timestep processing block
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.Mish(),  # Mish activations are standard practice for diffusion models
            nn.Linear(model_dim * 2, model_dim),
        )
        
        # Calculate total flattened feature input space:
        # (Actions per window * action channels) + state channels + timestep embedding size
        # (8 * 7) + 32 + 256 = 344 dimensions
        flattened_input_dim = (horizon * action_dim) + state_dim + model_dim
        
        # 2. Input projection to common hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(flattened_input_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
        )
        
        # 3. Stack of residual blocks for deep, stable noise prediction
        self.res_blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        
        # 4. Output projection back to action trajectory space
        self.output_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            # Outputs prediction of target shape noise: 8 * 7 = 56 elements
            nn.Linear(256, horizon * action_dim),
        )
        
    def forward(self, sample, timestep, cond):
        """
        sample:   Noisy actions tensor of shape (B, Horizon, Action_Dim)
        timestep: Diffusion scalar tracking steps tensor of shape (B,)
        cond:     Conditioning state vector tensor of shape (B, State_Dim)
        """
        batch_size = sample.size(0)
        
        # Flatten the input action sequence: (B, 8, 7) -> (B, 56)
        flat_sample = sample.view(batch_size, -1)
        
        # Project our discrete timestep array into a dense continuous context vector
        time_emb = self.time_mlp(timestep)  # Shape: (B, model_dim)
        
        # Concatenate noisy actions, environment context, and time embeddings together
        # Combined Shape: (B, 56 + 32 + 256) -> (B, 344)
        x = torch.cat([flat_sample, cond, time_emb], dim=-1)
        
        # Project to hidden dimension, pass through residual blocks, project to output
        x = self.input_proj(x)
        x = self.res_blocks(x)
        predicted_noise = self.output_proj(x)
        
        # Reshape output cleanly back to standard trajectory workspace coordinates
        # Target Output Shape: (B, Horizon, Action_Dim) -> matching our raw input data footprint
        return predicted_noise.view(batch_size, self.horizon, self.action_dim)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine variance schedule as proposed in 'Improved DDPM' (Nichol & Dhariwal, 2021).
    Distributes noise more evenly across timesteps compared to linear schedule,
    which generally gives better results for smaller models.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DDPMScheduler:
    """
    Manages the forward noise injection (variance scheduling) and 
    the reverse denoising steps for trajectory generation.
    Supports both DDPM (stochastic) and DDIM (deterministic) reverse sampling.
    """
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        self.num_train_timesteps = num_train_timesteps
        
        # 1. Define beta schedule (noise variance injected at each step)
        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        
        # 2. Derive alpha constants using the standard DDPM mathematical identity
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations required for the reverse denoising mathematics
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        """Moves all scheduler parameters to the active training device (CPU/GPU)."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, original_samples, noise, timesteps):
        """
        The Forward Pass: Mathematically mixes noise into clean trajectories.
        Formally: q(a_k | a_0)
        """
        # Gather scheduled parameters matching the current batch elements' timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        # Combine clean actions with random noise based on the time step ratio
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        The Reverse Pass (DDPM - Stochastic): Subtracts a fraction of predicted noise to refine the sample.
        Formally: p(a_{k-1} | a_k)
        """
        t = timestep
        
        # Extract scheduled coefficients for the current timestep
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Compute the mean of the cleaner preceding trajectory sample
        pred_prev_sample = (1.0 / sqrt_alpha) * (sample - (beta / sqrt_one_minus_alpha_prod) * model_output)
        
        # If we are not at the very final step (t=0), add a small variance fraction back 
        # to ensure the stochastic process remains physically diverse
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1)
            pred_prev_sample = pred_prev_sample + variance * noise
            
        return pred_prev_sample

    def ddim_step(self, model_output, timestep, sample, prev_timestep, clip_sample=True, clip_range=1.0):
        """
        Deterministic DDIM Reverse Step (Song et al., 2020).
        No stochastic noise is added, producing smoother and more consistent trajectories.
        Also supports stride-based timestep skipping for faster inference.
        
        model_output:  Predicted noise from the denoiser
        timestep:      Current diffusion timestep (t)
        sample:        Current noisy sample (a_t)
        prev_timestep: Previous diffusion timestep to jump to (t-1 or further back)
        clip_sample:   If True, clip predicted x_0 to prevent divergence (recommended)
        clip_range:    Range to clip predicted x_0 to [-clip_range, +clip_range]
        """
        # Get cumulative alpha products for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep].view(-1, 1, 1)
        
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].view(-1, 1, 1)
        else:
            # At the final step, alpha_cumprod_prev = 1.0 (no noise at t=0)
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)
        
        # Step 1: Predict the original clean sample (x_0) from the noise prediction
        # Using the rearranged forward diffusion formula: x_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
        pred_original = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # Clip predicted x_0 to a valid range to prevent divergence.
        # Without this, small noise prediction errors get amplified by 1/sqrt(α̅_t),
        # which can be catastrophic at high timesteps (e.g., 6416× at t=99 with cosine schedule).
        if clip_sample:
            pred_original = torch.clamp(pred_original, -clip_range, clip_range)
        
        # Step 2: Compute x_{t-1} deterministically using the DDIM update rule
        # x_{t-1} = sqrt(α̅_{t-1}) * x_0_pred + sqrt(1 - α̅_{t-1}) * ε_pred
        pred_prev_sample = (torch.sqrt(alpha_prod_t_prev) * pred_original + 
                           torch.sqrt(1.0 - alpha_prod_t_prev) * model_output)
        
        return pred_prev_sample




class FiLMConvBlock1D(nn.Module):
    """
    1D conv residual block with FiLM (Feature-wise Linear Modulation) conditioning.
    FiLM lets the conditioning vector (time + state) scale and shift every feature
    channel, giving the network expressive control over temporal smoothing at each
    noise level — something pure additive injection cannot do.
    """
    def __init__(self, in_channels, out_channels, cond_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.film1 = nn.Linear(cond_dim, out_channels * 2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.film2 = nn.Linear(cond_dim, out_channels * 2)

        self.act = nn.Mish()
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        residual = self.residual_proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        g1, b1 = self.film1(cond).chunk(2, dim=-1)
        x = x * g1.unsqueeze(-1) + b1.unsqueeze(-1)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        g2, b2 = self.film2(cond).chunk(2, dim=-1)
        x = x * g2.unsqueeze(-1) + b2.unsqueeze(-1)
        x = self.act(x)

        return x + residual


class TemporalUNet1D(nn.Module):
    """
    1D temporal U-Net denoiser for diffusion policy.
    Treats the action horizon as a 1D sequence and applies Conv1D over the time axis,
    so the network explicitly models temporal continuity between consecutive action steps.
    FiLM conditioning injects the combined time + state signal at every layer.

    Architecture (horizon=8, action_dim=7):
        Encoder:  (B,7,8) -> C1=32@8 -> C2=64@4 -> C3=128@2
        Bottleneck: C3=128@2
        Decoder:  C3=128@2 -> C2=64@4 [+skip] -> C1=32@8 [+skip]
        Output:   C1=32@8 -> action_dim=7@8
    """
    def __init__(self, state_dim=32, horizon=8, action_dim=7,
                 time_emb_dim=256, base_channels=32):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        cond_dim = state_dim + time_emb_dim  # 32 + 256 = 288

        C1, C2, C3 = base_channels, base_channels * 2, base_channels * 4  # 32, 64, 128

        # Encoder: 8 -> 4 -> 2
        self.enc1  = FiLMConvBlock1D(action_dim, C1, cond_dim)                        # (B, 32, 8)
        self.down1 = nn.Conv1d(C1, C2, kernel_size=3, stride=2, padding=1)            # (B, 64, 4)
        self.enc2  = FiLMConvBlock1D(C2, C2, cond_dim)                                # (B, 64, 4)
        self.down2 = nn.Conv1d(C2, C3, kernel_size=3, stride=2, padding=1)            # (B, 128, 2)

        # Bottleneck
        self.bottleneck = FiLMConvBlock1D(C3, C3, cond_dim)                           # (B, 128, 2)

        # Decoder: 2 -> 4 -> 8
        self.up2  = nn.ConvTranspose1d(C3, C2, kernel_size=4, stride=2, padding=1)   # (B, 64, 4)
        self.dec2 = FiLMConvBlock1D(C2 + C2, C2, cond_dim)                           # (B, 64, 4)

        self.up1  = nn.ConvTranspose1d(C2, C1, kernel_size=4, stride=2, padding=1)   # (B, 32, 8)
        self.dec1 = FiLMConvBlock1D(C1 + C1, C1, cond_dim)                           # (B, 32, 8)

        self.out_conv = nn.Conv1d(C1, action_dim, kernel_size=1)                      # (B, 7, 8)

    def forward(self, sample, timestep, cond):
        """
        sample:   (B, horizon, action_dim)  — noisy action trajectory
        timestep: (B,)                      — diffusion timestep indices
        cond:     (B, state_dim)            — robot state conditioning
        Returns:  (B, horizon, action_dim)  — predicted noise
        """
        time_emb = self.time_mlp(timestep)               # (B, 256)
        cond_vec = torch.cat([cond, time_emb], dim=-1)   # (B, 288)

        x = sample.permute(0, 2, 1)                      # (B, 7, 8)

        # Encoder
        skip1 = self.enc1(x, cond_vec)                   # (B, 32, 8)
        x     = self.down1(skip1)                        # (B, 64, 4)
        skip2 = self.enc2(x, cond_vec)                   # (B, 64, 4)
        x     = self.down2(skip2)                        # (B, 128, 2)

        # Bottleneck
        x = self.bottleneck(x, cond_vec)                 # (B, 128, 2)

        # Decoder
        x = self.up2(x)                                  # (B, 64, 4)
        x = torch.cat([x, skip2], dim=1)                 # (B, 128, 4)
        x = self.dec2(x, cond_vec)                       # (B, 64, 4)

        x = self.up1(x)                                  # (B, 32, 8)
        x = torch.cat([x, skip1], dim=1)                 # (B, 64, 8)
        x = self.dec1(x, cond_vec)                       # (B, 32, 8)

        x = self.out_conv(x)                             # (B, 7, 8)
        return x.permute(0, 2, 1)                        # (B, 8, 7)


if __name__ == "__main__":
    # Structural functional evaluation shape test:
    model = DiffusionTrajectoryDenoiser()
    
    test_sample = torch.randn(2, 8, 7)     # Batch of 2, 8-step noisy actions
    test_time = torch.randint(0, 100, (2,)) # Batch of 2, random noise step counts
    test_cond = torch.randn(2, 32)         # Batch of 2, 32-dimensional robot states
    
    output = model(test_sample, test_time, test_cond)
    print("Inference Shape output test verification:", output.shape) 
    # Expected Return: torch.Size([2, 8, 7])
    
    # Test DDIM sampling path
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="cosine")
    noisy = torch.randn(2, 8, 7)
    noise_pred = model(noisy, torch.tensor([50, 50]), test_cond)
    result = scheduler.ddim_step(noise_pred, 50, noisy, 40)
    print("DDIM step output shape:", result.shape)
    # Expected Return: torch.Size([2, 8, 7])

    # Test TemporalUNet1D
    unet = TemporalUNet1D()
    unet_out = unet(test_sample, test_time, test_cond)
    print("TemporalUNet1D output shape:", unet_out.shape)
    # Expected Return: torch.Size([2, 8, 7])