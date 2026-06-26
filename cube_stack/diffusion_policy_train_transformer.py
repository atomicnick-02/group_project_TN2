import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import os
import csv
import argparse
import numpy as np

from dataset_loader import (
    ShortHorizonRoboticsDataset, ShortHorizonRoboticsDatasetThree,
    ShortHorizonRoboticsDatasetGoal, ShortHorizonRoboticsDatasetThreeGoal,
)
from diffusion_transformer_model import DiffusionPolicy
from normalization_utils import normalize_actions, NormalizationResults
from state_utils import STACK_GOAL_STATE_DIM, STACK_THREE_GOAL_STATE_DIM

parser = argparse.ArgumentParser(description="Train the transformer DiffusionPolicy.")
parser.add_argument("--goal", action="store_true",
                    help="Train the goal-conditioned model: state += final cube positions "
                         "(StackThree: 48 -> 57, Stack: 32 -> 38).")
parser.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible init / shuffling / train-val split.")
args = parser.parse_args()

# Seed RNGs so init, DataLoader shuffling, the train/val split, and the diffusion
# noise are reproducible across runs (makes goal vs non-goal a controlled comparison).
torch.manual_seed(args.seed)
np.random.seed(args.seed)
split_generator = torch.Generator().manual_seed(args.seed)

DATASET = ["stack_d0"]

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
EPOCHS = 200
NUM_DIFFUSION_STEPS = 100
HORIZON = 8
ACTION_DIM = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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

RUN_NAME = "_".join(DATASET)
print(f"Training on: {DATASET}  (family={family}, run={RUN_NAME})")

normalization_results = NormalizationResults()
ACTION_MIN = torch.tensor(np.min([normalization_results.norm_results[d]["ACTION_MIN"] for d in DATASET], axis=0), dtype=torch.float32)
ACTION_MAX = torch.tensor(np.max([normalization_results.norm_results[d]["ACTION_MAX"] for d in DATASET], axis=0), dtype=torch.float32)


class NormalizedActionDataset(Dataset):
    def __init__(self, base, action_min, action_max):
        self.base = base
        self.action_min = action_min
        self.action_max = action_max

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        states, actions = self.base[idx]
        actions = normalize_actions(actions, self.action_min, self.action_max)
        return states, actions


if family == "stack":
    loader_cls = ShortHorizonRoboticsDatasetGoal if args.goal else ShortHorizonRoboticsDataset
    state_dim = STACK_GOAL_STATE_DIM if args.goal else 32
else:
    loader_cls = ShortHorizonRoboticsDatasetThreeGoal if args.goal else ShortHorizonRoboticsDatasetThree
    state_dim = STACK_THREE_GOAL_STATE_DIM if args.goal else 48

datasets = [loader_cls(f"datasets/core/{d}.hdf5", horizon=HORIZON) for d in DATASET]

base_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
full_dataset = NormalizedActionDataset(base_dataset, ACTION_MIN, ACTION_MAX)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_generator)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total dataset samples: {len(full_dataset)}")
print(f"Train samples: {train_size} | Validation samples: {val_size}")

model = DiffusionPolicy(
    action_dim=ACTION_DIM,
    obs_dim=state_dim,
    device=DEVICE,
    horizon=HORIZON,
    num_train_timesteps=NUM_DIFFUSION_STEPS,
)

model.optimizer = optim.AdamW(model.transformer.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
model.loss_fn = nn.MSELoss()
model.lr_scheduler = CosineAnnealingLR(model.optimizer, T_max=EPOCHS, eta_min=1e-6)

os.makedirs("./logs", exist_ok=True)

tag = "with_goal" if args.goal else "no_goal"
ckpt_path = f"./checkpoints/diffusion_transformer_{tag}_{RUN_NAME}_best.pth"
model.set_checkpoint_path(ckpt_path)

model.train(train_loader, EPOCHS, val_data=val_loader)

log_path = f"./logs/diffusion_transformer_{tag}_{RUN_NAME}_loss.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for epoch, (train_loss, val_loss) in enumerate(zip(model.loss_hist, model.val_loss_hist), start=1):
        writer.writerow([epoch, train_loss, val_loss])
print(f"Saved loss history to {log_path}")

print(f"Best checkpoint (EMA) saved to {ckpt_path} (best val loss: {model.best_loss:.6f})")
print("Diffusion Policy (Transformer) training cycle finished successfully!")
