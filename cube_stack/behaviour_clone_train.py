import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os

from dataset_loader import ShortHorizonRoboticsDataset, ShortHorizonRoboticsDatasetThree
from behaviour_clone_model import BehaviorCloningBaseline

# --- 1. Configuration & Hyperparameters ---
# Single dataset : DATASET = ["stack_d0"]
# Combined stack : DATASET = ["stack_d0", "stack_d1"]
# Combined three : DATASET = ["stack_three_d0", "stack_three_d1"]
DATASET = ["stack_d0", "stack_d1"]

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
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

# --- 2. Load and Split Dataset ---
if family == "stack":
    datasets = [ShortHorizonRoboticsDataset(f"datasets/core/{d}.hdf5", horizon=8) for d in DATASET]
    input_dim = 32
else:
    datasets = [ShortHorizonRoboticsDatasetThree(f"datasets/core/{d}.hdf5", horizon=8) for d in DATASET]
    input_dim = 48

full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total dataset samples: {len(full_dataset)}")
print(f"Train samples: {train_size} | Validation samples: {val_size}")

# --- 3. Initialize Model, Loss Function, and Optimizer ---
model = BehaviorCloningBaseline(input_dim=input_dim, horizon=8, action_dim=7).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Core Training & Validation Loop ---
os.makedirs("./checkpoints", exist_ok=True)
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0

    for states, target_actions in train_loader:
        states = states.to(DEVICE)
        target_actions = target_actions.to(DEVICE)

        optimizer.zero_grad()
        predicted_actions = model(states)
        loss = criterion(predicted_actions, target_actions)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * states.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    # --- Validation Phase ---
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for states, target_actions in val_loader:
            states = states.to(DEVICE)
            target_actions = target_actions.to(DEVICE)
            predicted_actions = model(states)
            loss = criterion(predicted_actions, target_actions)
            running_val_loss += loss.item() * states.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train MSE Loss: {epoch_train_loss:.6f} | Val MSE Loss: {epoch_val_loss:.6f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), f"./checkpoints/bc_baseline_{RUN_NAME}_best.pth")
        print("  --> New best model checkpoint saved!")

print("\nTraining completed successfully!")
