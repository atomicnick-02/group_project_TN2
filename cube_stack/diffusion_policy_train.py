import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
import copy
import numpy as np

from dataset_loader import ShortHorizonRoboticsDataset, ShortHorizonRoboticsDatasetThree
from diffusion_policy_model import TemporalUNet1D, DDPMScheduler
from normalization_utils import normalize_actions, NormalizationResults

# --- 1. Configuration & Hyperparameters ---
# Single dataset : DATASET = ["stack_d0"]
# Combined stack : DATASET = ["stack_d0", "stack_d1"]
# Combined three : DATASET = ["stack_three_d0", "stack_three_d1"]
DATASET = ["stack_three_d0"]

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 150
NUM_DIFFUSION_STEPS = 100
GRAD_CLIP_MAX_NORM = 1.0    # Stabilizes noisy gradients from noise prediction targets
EMA_DECAY = 0.995           # Exponential moving average decay for weight smoothing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Validate dataset list ---
STACK_DATASETS = {"stack_d0", "stack_d1"}
STACK_THREE_DATASETS = {"stack_three_d0", "stack_three_d1"}

dataset_set = set(DATASET)
if dataset_set <= STACK_DATASETS:
    family = "stack"
elif dataset_set <= STACK_THREE_DATASETS:
    family = "stack_three"
else:
    raise ValueError(
        f"Datasets must all belong to the same family. "
        f"Got: {DATASET}. "
        f"Allowed combinations: subsets of {sorted(STACK_DATASETS)} or {sorted(STACK_THREE_DATASETS)}."
    )

# Checkpoint name encodes which datasets were combined
RUN_NAME = "_".join(DATASET)
print(f"Training on: {DATASET}  (family={family}, run={RUN_NAME})")

# --- Combined normalization bounds (element-wise min/max across all datasets) ---
normalization_results = NormalizationResults()
ACTION_MIN = torch.tensor(np.min([normalization_results.norm_results[d]["ACTION_MIN"] for d in DATASET], axis=0), dtype=torch.float32).to(DEVICE)
ACTION_MAX = torch.tensor(np.max([normalization_results.norm_results[d]["ACTION_MAX"] for d in DATASET], axis=0), dtype=torch.float32).to(DEVICE)

# --- EMA (Exponential Moving Average) Helper ---
class EMAModel:
    """
    Maintains an exponential moving average of model weights during training.
    EMA smooths out training noise and typically produces better inference results
    for diffusion models, since the training target (random noise) is inherently noisy.
    """
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_to(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def save_state(self):
        return {name: param.clone() for name, param in self.shadow.items()}


# --- 2. Load and Split Dataset ---
if family == "stack":
    datasets = [ShortHorizonRoboticsDataset(f"datasets/core/{d}.hdf5", horizon=8) for d in DATASET]
    state_dim = 32
else:
    datasets = [ShortHorizonRoboticsDatasetThree(f"datasets/core/{d}.hdf5", horizon=8) for d in DATASET]
    state_dim = 48

full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total dataset samples: {len(full_dataset)}")
print(f"Train samples: {train_size} | Validation samples: {val_size}")

# --- 3. Initialize Model, Scheduler, and Optimizer ---
model = TemporalUNet1D(state_dim=state_dim, horizon=8, action_dim=7).to(DEVICE)

noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS, beta_schedule="cosine").to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
ema = EMAModel(model, decay=EMA_DECAY)

# --- 4. Core Denoising Training Loop ---
os.makedirs("./checkpoints", exist_ok=True)
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    
    for states, clean_actions in train_loader:
        states = states.to(DEVICE)
        clean_actions = clean_actions.to(DEVICE)
        batch_size = states.size(0)

        clean_actions = normalize_actions(clean_actions, ACTION_MIN, ACTION_MAX)

        noise = torch.randn_like(clean_actions).to(DEVICE)
        timesteps = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=DEVICE).long()

        noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)
        
        optimizer.zero_grad()
        noise_pred = model(noisy_actions, timesteps, states)
        loss = criterion(noise_pred, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        ema.update(model)

        running_train_loss += loss.item() * batch_size

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # --- Validation Phase (with EMA weights) ---
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    ema.apply_to(model)

    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for states, clean_actions in val_loader:
            states = states.to(DEVICE)
            clean_actions = clean_actions.to(DEVICE)
            batch_size = states.size(0)

            clean_actions = normalize_actions(clean_actions, ACTION_MIN, ACTION_MAX)

            noise = torch.randn_like(clean_actions).to(DEVICE)
            timesteps = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=DEVICE).long()

            noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)
            noise_pred = model(noisy_actions, timesteps, states)
            loss = criterion(noise_pred, noise)

            running_val_loss += loss.item() * batch_size

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Noise MSE: {epoch_train_loss:.6f} | Val Noise MSE: {epoch_val_loss:.6f} | LR: {current_lr:.2e}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), f"./checkpoints/temporal_unet_{RUN_NAME}_best.pth")
        print("  --> New best diffusion model checkpoint saved! (EMA weights)")

    # Restore non-EMA weights for continued training
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])

print("\nDiffusion Policy training cycle finished successfully!")
