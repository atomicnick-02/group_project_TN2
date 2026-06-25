"""
Train the conditional Diffusion Policy on the double-pendulum expert
trajectories produced by generate_dataset.py.

The full network spec (class name + hyperparameters) is saved into the
checkpoint's "config", so the evaluation script can rebuild the exact same
network with no manual edits.

Pipeline:
  1. Load expert_trajectories.h5  (groups traj_*, each with states (T,4), actions (T,2))
  2. Slice every trajectory into (cond, action_seq) windows:
        cond       = last k state features   -> (k*NX_FEAT,)  [+ NX_FEAT if USE_GOAL]
        action_seq = next H actions          -> (H, nu)
  3. States are converted to angular features [sin(q1),cos(q1),sin(q2),cos(q2),dq1_n,dq2_n]
     to avoid angle-wrap discontinuities. Velocities are min-max normalized to [-1,1].
     Actions are min-max normalized to [-1,1]. Stats saved for inference.
  4. Train via the DiffusionPolicy class (noise-prediction MSE).
"""

import os
import json
# import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Config (defaults; some overridable via CLI) ──────────────────────────────
H5_PATH      = "C:\\Users\\theod\\Downloads\\group_project_TN2-pendulum\\group_project_TN2-pendulum\\double_pendulum\\results/expert_trajectories.h5"
OUT_DIR      = "C:\\Users\\theod\\Downloads\\group_project_TN2-pendulum\\group_project_TN2-pendulum\\double_pendulum\\results_bc"
# CKPT_PATH    = os.path.join(OUT_DIR, "diffusion_policy.pt")
STATS_PATH   = os.path.join(OUT_DIR, "norm_stats.json")
CKPT_DIR     = os.path.join(OUT_DIR, "checkpoints")  # periodic per-epoch snapshots
MODEL_N = 3
CKPT_PATH = os.path.join(OUT_DIR, f"bc_policy_{MODEL_N}.pt")
CKPT_EVERY   = 20                                    # save a checkpoint every N epochs
LOG_F_PATH = os.path.join(OUT_DIR, f"training_log_{MODEL_N}.csv")

STATE_DIM, CONTROL_INPUT_DIM       = 4, 2          # raw state dim (from HDF5), action dim
STATE_DIM_FEAT      = 6             # feature dim: [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
#K            = 6             # observation-history length
H            = 8             # action prediction horizon
USE_GOAL     = True          # append goal features to conditioning vector
X_GOAL       = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float32)  # upright

TIMESTEPS    = 100
EPOCHS       = 400
BATCH_SIZE   = 512
LR           = 1e-4
SEED         = 42
# Real upright-hold rollouts now come from generate_tvlqr_dataset.py (--n-hold),
# which capture the deviation->corrective-torque map. The old synthetic hack just
# repeated each trajectory's final state/action, teaching the exact fixed point
# but NOT how to recover -- so it's disabled (0) in favor of the real data.
HOLD_STEPS   = 0

# MLP-specific
# MLP_HIDDEN   = 256

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device('cpu')

if(torch.cuda.is_available()):
    DEVICE = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(DEVICE)))
else:
    print("Device set to : cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
class BehavioralCloningPolicyDataset(Dataset):
    """Slices expert trajectories into (cond, action_seq) windows + normalizes."""

    def __init__(self, data_path, horizon=H, nx=STATE_DIM, nx_feat=STATE_DIM_FEAT, nu=CONTROL_INPUT_DIM,
                 use_goal=USE_GOAL, x_goal=X_GOAL, hold_steps=HOLD_STEPS):
        
        self.data_path = data_path
        self.horizon = horizon
        self.nx, self.nx_feat, self.nu = STATE_DIM, STATE_DIM_FEAT, CONTROL_INPUT_DIM
        self.use_goal        = use_goal
        self.hold_steps      = hold_steps
        self.x_goal          = np.asarray(x_goal, dtype=np.float32)

        trajs = []
        with h5py.File(self.data_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                states  = np.asarray(grp["states"],  dtype=np.float32)
                actions = np.asarray(grp["actions"], dtype=np.float32)
                trajs.append((states, actions))
        if not trajs:
            raise RuntimeError(f"No trajectories found in {self.data_path}")
        train_trajs = trajs[:int(len(trajs) * 0.9)]
        test_trajs = trajs[int(len(trajs) * 0.9):]

        all_vels    = np.concatenate([s[:, 2:] for s, _ in train_trajs], axis=0)
        all_actions = np.concatenate([a for _, a in train_trajs], axis=0)
        self.vel_min,    self.vel_max    = all_vels.min(0),    all_vels.max(0)
        self.action_min, self.action_max = all_actions.min(0), all_actions.max(0)
        self._vel_range    = np.where((self.vel_max - self.vel_min) > 1e-8,
                                      self.vel_max - self.vel_min, 1.0)
        self._action_range = np.where((self.action_max - self.action_min) > 1e-8,
                                      self.action_max - self.action_min, 1.0)

        goal_feat = self._state_to_features(self.x_goal)      # (NX_FEAT,), computed once

        self.train_samples = []
        self.test_samples = []
        for states, actions in train_trajs:
            states_f  = self._state_to_features(states)       # (T, NX_FEAT)
            actions_n = self._norm_action(actions)
            T = len(states_f)
            for i in range(T):
                # lo = i - k + 1
                # if lo < 0:
                #     pad  = np.repeat(states_f[:1], -lo, axis=0)
                #     hist = np.concatenate([pad, states_f[:i + 1]], axis=0)
                # else:
                #     hist = states_f[lo:i + 1]
                # cond = hist.reshape(-1)
                # if use_goal:
                #     cond = np.concatenate([cond, goal_feat])

                acts = actions_n[i:i + horizon]
                if len(acts) < horizon:
                    pad  = np.repeat(acts[-1:], horizon - len(acts), axis=0)
                    acts = np.concatenate([acts, pad], axis=0)
                self.train_samples.append((states_f[i].astype(np.float32).reshape(-1), acts.astype(np.float32)))
                # self.samples.append((cond.astype(np.float32), acts.astype(np.float32)))
        for states, actions in test_trajs:
            states_f  = self._state_to_features(states)       # (T, NX_FEAT)
            actions_n = self._norm_action(actions)
            T = len(states_f)
            for i in range(T):
                # lo = i - k + 1
                # if lo < 0:
                #     pad  = np.repeat(states_f[:1], -lo, axis=0)
                #     hist = np.concatenate([pad, states_f[:i + 1]], axis=0)
                # else:
                #     hist = states_f[lo:i + 1]
                # cond = hist.reshape(-1)
                # if use_goal:
                #     cond = np.concatenate([cond, goal_feat])

                acts = actions_n[i:i + horizon]
                if len(acts) < horizon:
                    pad  = np.repeat(acts[-1:], horizon - len(acts), axis=0)
                    acts = np.concatenate([acts, pad], axis=0)
                self.test_samples.append((states_f[i].astype(np.float32).reshape(-1), acts.astype(np.float32)))

            # Append synthetic holding samples using the trajectory's final
            # state and final action (the IPOPT-computed stabilizing torque).
            # This teaches the model to maintain the upright position.
            # if hold_steps > 0:
            #     final_feat = states_f[-1]                        # (NX_FEAT,)
            #     final_act  = actions_n[-1]                       # (nu,)
            #     hold_hist  = np.repeat(final_feat[None], k, axis=0)  # (k, NX_FEAT)
            #     hold_cond  = hold_hist.reshape(-1)
            #     if use_goal:
            #         hold_cond = np.concatenate([hold_cond, goal_feat])
            #     hold_acts = np.repeat(final_act[None], horizon, axis=0)  # (H, nu)
            #     for _ in range(hold_steps):
            #         self.samples.append((
            #             hold_cond.astype(np.float32),
            #             hold_acts.astype(np.float32),
            #         ))

        self.cond_dim = self.train_samples[0][0].shape[0]

    def _state_to_features(self, x):
        """(T, 4) or (4,) -> (T, NX_FEAT) or (NX_FEAT,): sin/cos angles + normed velocities."""
        single = x.ndim == 1
        if single:
            x = x[None]
        q1, q2  = x[:, 0], x[:, 1]
        v_norm  = 2.0 * (x[:, 2:] - self.vel_min) / self._vel_range - 1.0
        feat    = np.column_stack([np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), v_norm])
        return feat[0] if single else feat

    def _norm_action(self, a):  return 2.0 * (a - self.action_min) / self._action_range - 1.0
    def denorm_action(self, a): return (a + 1.0) * 0.5 * self._action_range + self.action_min

    def save_stats(self, path):
        with open(path, "w") as fp:
            json.dump({
                "vel_min":  self.vel_min.tolist(),
                "vel_max":  self.vel_max.tolist(),
                "action_min": self.action_min.tolist(),
                "action_max": self.action_max.tolist(),
                "horizon": self.horizon,
                "nx": self.nx, "nx_feat": self.nx_feat, "nu": self.nu,
                "use_goal": self.use_goal,
                "use_angular_features": True,
                "cond_dim": self.cond_dim,
            }, fp, indent=2)

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        cond, acts = self.train_samples[idx]
        return torch.from_numpy(cond), torch.from_numpy(acts)

class BCPolicyModel(nn.Module):
    def __init__(self, horizon):
        self.horizon = horizon
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 512),
            # nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.2),
            # nn.Linear(128, 64),
            # # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 2 * self.horizon),
            # nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)

        return x.view(-1, self.horizon, 2)

def train(data, model, loss_fn, optimizer, batch_size):

    # Set the network to training mode
    model.train()

    size = len(data.dataset)
    train_loss = 0
    for batch, (X, y) in enumerate(data):

        # Prepare the data
        X = X.to(DEVICE, torch.float)
        y = y.to(DEVICE, torch.float)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        # Print a summary every 10 batches
        # if batch % 10 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # print(f"loss: {loss:>7f}")
    return train_loss / batch

def evaluate(dataloader, model, loss_fn):

    # Set the network to evaluation mode
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:

            # Prepare the data
            X = X.to(DEVICE, torch.float)
            y = y.to(DEVICE, torch.float)

            # Forward pass
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    # print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


# ── Checkpoint serialization (single source of truth) ────────────────────────
def save_checkpoint(path, model, cond_dim):
    """
    Write a SELF-DESCRIBING checkpoint: store arch + exact net kwargs so eval can
    rebuild the matching class with zero manual edits. Saves the EMA weights as
    model_state -> inference uses the smoother, more consistent controller.

    Used for both the periodic per-epoch snapshots and the final checkpoint, so
    every saved file has an identical, eval-loadable format.
    """
    torch.save(model.state_dict(), path)
    # torch.save({
    #     "model_state": model.state_dict(),
    #     # "model_state_raw": policy.model.state_dict(),
    #     "config": {
    #         # "arch": arch,
    #         # "net_kwargs": net_kwargs,
    #         "timesteps": TIMESTEPS, "horizon": H, "action_dim": CONTROL_INPUT_DIM,
    #         "cond_dim": cond_dim,
    #         "nx": STATE_DIM, "nx_feat": STATE_DIM_FEAT, "use_goal": USE_GOAL,
    #         "use_angular_features": True,
    #     },
    # }, path)


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    train_loss_buffer = []
    test_loss_buffer = []
    dataset = BehavioralCloningPolicyDataset(H5_PATH, horizon=H, nx=STATE_DIM, nx_feat=STATE_DIM_FEAT, nu=CONTROL_INPUT_DIM,
                                             use_goal=USE_GOAL, x_goal=X_GOAL, hold_steps=HOLD_STEPS)
    dataset.save_stats(STATS_PATH)
    print(f"Dataset: {len(dataset)} windows | cond_dim={dataset.cond_dim} "
          f"| action_seq=({H},{CONTROL_INPUT_DIM})")
    # print(DEVICE == "cuda")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(dataset.test_samples, batch_size=BATCH_SIZE, shuffle=False,
                        drop_last=True, pin_memory=(DEVICE.type == "cuda"))
    model = BCPolicyModel(horizon=H).to(DEVICE)
    # policy = DiffusionPolicy(
    #     scheduler=scheduler, network=network, device=DEVICE,
    #     timesteps=TIMESTEPS, horizon=H, action_dim=CONTROL_INPUT_DIM, learning_rate=LR,
    # )
    optimizer = torch.optim.Adam([
                        {'params': model.parameters(), 'lr':LR}
                    ])
    loss_fn = nn.MSELoss()
    # print(f"Training arch='{args.arch}' on {DEVICE} for {args.epochs} epochs "
    #       f"({sum(p.numel() for p in network.parameters()):,} params)...")
    # if args.ckpt_every > 0:
    #     print(f"Periodic checkpoints every {args.ckpt_every} epochs -> {args.ckpt_dir}/")

    # Save a snapshot every N epochs into the checkpoints folder. Skip the final
    # epoch here -- it's written once below as the canonical --ckpt path.
    # def on_epoch_end(epoch, avg_loss):
    #     if args.ckpt_every > 0 and epoch % args.ckpt_every == 0 and epoch != args.epochs:
    #         snap = os.path.join(args.ckpt_dir, f"{args.arch}_epoch{epoch:04d}.pt")
    #         save_checkpoint(snap, policy, args.arch, net_kwargs, dataset.cond_dim)
    #         print(f"  ↳ checkpoint saved: {snap}")
    for t in tqdm(range(EPOCHS)):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loss_buffer.append(train(loader, model, loss_fn, optimizer, BATCH_SIZE))
        test_loss_buffer.append(evaluate(test_loader, model, loss_fn))
        # print("VALIDATION:", end=" ")
        # evaluate(valid_dataloader, model, loss_fn)
    # train(loader, model, nn.MSELoss(), optimizer, BATCH_SIZE)
    plt.plot(train_loss_buffer)
    plt.plot(test_loss_buffer)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    log_f = open(LOG_F_PATH,"w+")
    log_f.write('epoch,Training_Loss,Test_Loss\n')
    for i in range(EPOCHS):
        log_f.write(f"{i+1},{train_loss_buffer[i]},{test_loss_buffer[i]}\n")
    log_f.close()

    # Final checkpoint (canonical path used by eval). Same self-describing format
    # as the periodic snapshots, written via the shared helper.
    save_checkpoint(CKPT_PATH, model,  dataset.cond_dim)
    print(f"Saved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()