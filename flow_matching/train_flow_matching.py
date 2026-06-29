"""
Train the conditional Flow Matching policy on the double-pendulum expert
trajectories (the same expert_trajectories.h5 the Diffusion Policy uses).

This is the flow-matching counterpart of
double_pendulum/train_diffusion_policy.py. It deliberately reuses that script's
DiffusionPolicyDataset and train_val_test_keys so the two methods see an
IDENTICAL data representation (same angular features, same normalization, same
k-history / H-horizon windows, same trajectory-level 70/15/15 split). That makes
a flow-matching vs. diffusion comparison apples-to-apples -- the only thing that
changes is the generative head.

Architecture is selectable and self-describing in the checkpoint:
    python flow_matching/train_flow_matching.py --arch mlp
    python flow_matching/train_flow_matching.py --arch transformer

Pipeline:
  1. Load expert_trajectories.h5 (groups traj_*, each states (T,4), actions (T,2)).
  2. Slice into (cond, action_seq) windows via the shared dataset:
        cond       = last k state features (+ goal)  -> (cond_dim,)
        action_seq = next H actions (data endpoint x_1) -> (H, nu)
  3. Train via FlowMatchingPolicy (regress the OT/rectified-flow velocity field).
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the repo root importable (diffusion_models, flow_matching) and the
# double_pendulum folder importable (the shared dataset lives there).
_THIS      = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parent
_DP_DIR    = _REPO_ROOT / "double_pendulum"
for _p in (_REPO_ROOT, _DP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flow_matching.flow_matching_policy import MLP, TrajectoryTransformer, FlowMatchingPolicy
# Reuse the diffusion data pipeline verbatim so both methods train on identical data.
from train_diffusion_policy import DiffusionPolicyDataset, train_val_test_keys


# Config (defaults; some overridable via CLI). Paths are resolved against the
# repo root so the script runs correctly from any working directory.
H5_PATH    = str(_REPO_ROOT / "double_pendulum" / "results" / "expert_trajectories.h5")
OUT_DIR    = str(_THIS / "results")
CKPT_DIR   = os.path.join(OUT_DIR, "checkpoints")     # final ckpts -> checkpoints/<arch>/
LOSS_DIR   = os.path.join(OUT_DIR, "losses")          # per-epoch train/val curves + test summary
STATS_PATH = os.path.join(OUT_DIR, "norm_stats.json")

SPLIT_RATIOS = (0.7, 0.15, 0.15)   # train / val / test (trajectory-level, like diffusion)

NX, NU   = 4, 2     # raw state dim (from HDF5), action dim
NX_FEAT  = 6        # feature dim: [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
K        = 4        # observation-history length
H        = 10       # action prediction horizon
USE_GOAL = True     # append goal features to conditioning vector

NUM_STEPS = 10      # Euler integration steps used at GENERATION time
SIGMA_MIN = 0.0     # 0.0 = pure rectified flow (straight-line OT path)

EPOCHS     = 400
BATCH_SIZE = 256
LR         = 1e-4
SEED       = 42

# MLP-specific
MLP_HIDDEN = 512

# Transformer-specific
TF_D_MODEL = 128
TF_HEADS   = 4
TF_LAYERS  = 2
TF_FF      = 256
TF_DROPOUT = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Network builder (mirrors train_diffusion_policy.build_network)
def build_network(arch, cond_dim, horizon=H, action_dim=NU):
    """Returns (network, net_kwargs). net_kwargs is everything needed to rebuild
    the same architecture later -- it gets stored in the checkpoint."""
    if arch == "mlp":
        kwargs = {"horizon": horizon, "action_dim": action_dim,
                  "cond_dim": cond_dim, "hidden_dim": MLP_HIDDEN}
        return MLP(**kwargs), kwargs
    elif arch == "transformer":
        kwargs = {"horizon": horizon, "action_dim": action_dim,
                  "cond_dim": cond_dim, "d_model": TF_D_MODEL,
                  "n_heads": TF_HEADS, "n_layers": TF_LAYERS,
                  "dim_feedforward": TF_FF, "dropout": TF_DROPOUT}
        return TrajectoryTransformer(**kwargs), kwargs
    raise ValueError(f"unknown arch: {arch}")


def build_checkpoint(policy, arch, net_kwargs, cond_dim):
    """Self-describing checkpoint dict shared by periodic + final saves.

    method='flow_matching' lets a controller dispatch between diffusion and flow
    matching by inspecting the checkpoint. model_state = EMA weights (inference).
    """
    return {
        "model_state": policy.ema_model.state_dict(),
        "model_state_raw": policy.model.state_dict(),
        "config": {
            "method": "flow_matching",
            "arch": arch,
            "net_kwargs": net_kwargs,
            "num_steps": NUM_STEPS, "sigma_min": SIGMA_MIN,
            "horizon": H, "action_dim": NU,
            "cond_dim": cond_dim,
            "k": K, "nx": NX, "nx_feat": NX_FEAT, "use_goal": USE_GOAL,
            "use_angular_features": True,
        },
    }


@torch.no_grad()
def validation_loss(policy, loader, seed=0):
    """Mean flow-matching velocity-regression MSE over `loader`, mirroring the
    train objective. Uses the EMA model in eval mode. A fixed seed makes t and
    the source noise x_0 reproducible, so the val curve reflects model change
    rather than noise change across epochs.

    NOTE: this reseeds the global RNG. Call it AFTER training (e.g. test loss),
    or via eval_loss_keep_rng during training, which restores RNG state.
    """
    if loader is None or len(loader.dataset) == 0:
        return float("nan")
    model = policy.ema_model
    model.eval()
    torch.manual_seed(seed)
    total, n = 0.0, 0
    for cond, action_seq in loader:
        cond = cond.to(policy.device)
        x_1  = action_seq.to(policy.device)
        x_0  = torch.randn_like(x_1)
        t    = torch.rand(x_1.size(0), device=policy.device)
        x_t, v_target = policy._interpolate(x_0, x_1, t)
        v_pred = model(x_t, t, cond)
        total += torch.nn.functional.mse_loss(v_pred, v_target).item() * x_1.size(0)
        n += x_1.size(0)
    return total / max(n, 1)


def eval_loss_keep_rng(policy, loader, seed=0):
    """validation_loss, but transparent to the global RNG so it can be called
    DURING training without making every epoch's training noise identical."""
    cpu_state  = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return validation_loss(policy, loader, seed=seed)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def build_summary(arch, epochs, train_keys, val_keys, test_keys, history,
                  test_loss=float("nan")):
    """Run-summary dict (config + latest losses), rebuilt from `history`."""
    final_val = next((vl for _, _, vl in reversed(history) if not np.isnan(vl)),
                     float("nan"))
    _nan2none = lambda x: None if (isinstance(x, float) and np.isnan(x)) else x
    return {
        "method": "flow_matching", "arch": arch, "epochs": epochs, "seed": SEED,
        "num_steps": NUM_STEPS, "sigma_min": SIGMA_MIN,
        "split_ratios": list(SPLIT_RATIOS),
        "n_train_traj": len(train_keys), "n_val_traj": len(val_keys),
        "n_test_traj": len(test_keys),
        "epochs_completed": history[-1][0] if history else 0,
        "final_train_loss": _nan2none(history[-1][1] if history else float("nan")),
        "final_val_loss":   _nan2none(final_val),
        "test_loss":        _nan2none(test_loss),
    }


# Warm-start
def find_last_checkpoint(arch=None):
    """Path to the most recently modified flow-matching checkpoint, or None.

    Looks in the per-arch checkpoint folders (checkpoints/<arch>/) and the
    results/ root. If `arch` is given and that arch's folder holds checkpoints,
    those win; otherwise the newest across all locations is used.
    """
    arch_dir = Path(CKPT_DIR) / arch if arch else None
    if arch_dir is not None and arch_dir.exists():
        cands = list(arch_dir.glob("*.pt"))
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    cands = list(Path(CKPT_DIR).rglob("*.pt")) + list(Path(OUT_DIR).glob("flow_matching*.pt"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def load_warmstart(network, path, arch, cond_dim):
    """Initialize `network` IN PLACE from a checkpoint (weights only).

    Warm start = weights only: the caller builds a fresh optimizer, LR schedule
    and EMA from these weights. Verifies method/arch/cond_dim match so the
    state_dict loads strictly. Prefers the raw trainable weights
    (model_state_raw) -- the natural starting point for continued training --
    and falls back to the EMA weights (model_state) if that's all there is.
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt.get("config", {})
    ck_method = cfg.get("method")
    if ck_method is not None and ck_method != "flow_matching":
        raise SystemExit(f"--warmstart method mismatch: checkpoint is '{ck_method}', "
                         "not a flow_matching checkpoint")
    ck_arch, ck_cond = cfg.get("arch"), cfg.get("cond_dim")
    if ck_arch is not None and ck_arch != arch:
        raise SystemExit(f"--warmstart arch mismatch: checkpoint is '{ck_arch}', --arch is '{arch}'")
    if ck_cond is not None and ck_cond != cond_dim:
        raise SystemExit(f"--warmstart cond_dim mismatch: checkpoint {ck_cond} vs current {cond_dim}")
    state = ckpt.get("model_state_raw")
    if state is None:
        state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"--warmstart: no model weights found in {path}")
    network.load_state_dict(state)


def main():
    global NUM_STEPS, H, K, LR, SIGMA_MIN, TF_D_MODEL, TF_HEADS, TF_LAYERS, TF_FF, MLP_HIDDEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="transformer")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint output path; default checkpoints/<arch>/flow_matching_<arch>.pt")
    ap.add_argument("--val-every", type=int, default=1,
                    help="compute & display held-out validation loss every N epochs (0 = off)")
    ap.add_argument("--save-every", type=int, default=20,
                    help="save an intermediate checkpoint every N epochs (0 = off)")
    ap.add_argument("--warmstart", default="auto",
                    help="initialize weights from a checkpoint before training. "
                         "'auto' (default) = the most recent flow-matching checkpoint "
                         "(falls back to from-scratch if none exists); a path = that "
                         "checkpoint; 'none'/'off' = always train from scratch")
    # Sweep knobs (override the module defaults from the CLI)
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS,
                    help="Euler integration steps at generation time -- the main "
                         "inference-cost knob (training is unaffected)")
    ap.add_argument("--sigma-min", type=float, default=SIGMA_MIN,
                    help="residual source-noise band around the data endpoint "
                         "(0.0 = pure rectified flow)")
    ap.add_argument("--horizon", type=int, default=H, help="action prediction horizon H")
    ap.add_argument("--k", type=int, default=K, help="observation-history length")
    ap.add_argument("--lr", type=float, default=LR, help="learning rate")
    ap.add_argument("--tf-d-model", type=int, default=TF_D_MODEL,
                    help="transformer width (must be divisible by --tf-heads)")
    ap.add_argument("--tf-heads", type=int, default=TF_HEADS, help="transformer attention heads")
    ap.add_argument("--tf-layers", type=int, default=TF_LAYERS, help="transformer encoder layers")
    ap.add_argument("--tf-ff", type=int, default=TF_FF, help="transformer feed-forward dim")
    ap.add_argument("--mlp-hidden", type=int, default=MLP_HIDDEN, help="MLP hidden width")
    args = ap.parse_args()

    NUM_STEPS  = args.num_steps
    SIGMA_MIN  = args.sigma_min
    H          = args.horizon
    K          = args.k
    LR         = args.lr
    TF_D_MODEL = args.tf_d_model
    TF_HEADS   = args.tf_heads
    TF_LAYERS  = args.tf_layers
    TF_FF      = args.tf_ff
    MLP_HIDDEN = args.mlp_hidden

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    ckpt = (Path(CKPT_DIR) / args.arch / f"flow_matching_{args.arch}.pt"
            if args.ckpt is None else Path(args.ckpt))
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    # 70/15/15 trajectory-level split (shared helper -> same split as diffusion).
    train_keys, val_keys, test_keys = train_val_test_keys(H5_PATH, SPLIT_RATIOS, SEED)
    print(f"Split (seed={SEED}): {len(train_keys)} train / {len(val_keys)} val / "
          f"{len(test_keys)} test trajectories "
          f"({SPLIT_RATIOS[0]:.0%}/{SPLIT_RATIOS[1]:.0%}/{SPLIT_RATIOS[2]:.0%})")

    dataset = DiffusionPolicyDataset(H5_PATH, keys=train_keys, k=K, horizon=H)
    dataset.save_stats(STATS_PATH)
    print(f"Train dataset: {len(dataset)} windows | cond_dim={dataset.cond_dim} "
          f"| action_seq=({H},{NU})")

    # Val/test reuse the TRAIN fold's normalization stats (no held-out leakage).
    def _make_loader(keys, name):
        if not keys:
            return None
        ds = DiffusionPolicyDataset(H5_PATH, keys=keys, stats=dataset.get_stats(), k=K, horizon=H)
        print(f"{name} dataset: {len(ds)} windows")
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=(DEVICE == "cuda"))

    val_loader  = _make_loader(val_keys,  "Val ")
    test_loader = _make_loader(test_keys, "Test")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=(DEVICE == "cuda"))

    Path(LOSS_DIR).mkdir(parents=True, exist_ok=True)
    summary_path = Path(LOSS_DIR) / f"summary_{args.arch}.json"
    history = []   # rows of [epoch, train_loss, val_loss]
    def write_summary(test_loss=float("nan")):
        s = build_summary(args.arch, args.epochs, train_keys, val_keys,
                          test_keys, history, test_loss)
        with open(summary_path, "w") as fp:
            json.dump(s, fp, indent=2)
        return s
    write_summary()

    network, net_kwargs = build_network(args.arch, dataset.cond_dim, horizon=H)

    # Warm start: load weights into the freshly built network before wrapping it
    # in FlowMatchingPolicy, so the policy's EMA copy (a deepcopy made at init)
    # also starts from the checkpoint instead of random init. Default is 'auto' =
    # pick up the last previous checkpoint; 'none'/'off' forces a from-scratch run.
    warmstart_used = None
    if args.warmstart and args.warmstart.lower() not in ("none", "off"):
        if args.warmstart == "auto":
            ws = find_last_checkpoint(args.arch)
            if ws is None:
                print("Warm-start: no previous checkpoint found, training from scratch.")
        else:
            ws = Path(args.warmstart)
            if not ws.exists():
                raise SystemExit(f"--warmstart: no checkpoint found at '{ws}'")
        if ws is not None:
            load_warmstart(network, ws, args.arch, dataset.cond_dim)
            warmstart_used = str(ws)
            print(f"Warm-start: initialized '{args.arch}' weights from {ws}")

    policy = FlowMatchingPolicy(
        network=network, device=DEVICE, horizon=H, action_dim=NU,
        num_steps=NUM_STEPS, learning_rate=LR, sigma_min=SIGMA_MIN,
    )

    print(f"Training flow-matching arch='{args.arch}' on {DEVICE} for {args.epochs} "
          f"epochs ({sum(p.numel() for p in network.parameters()):,} params)...")

    def on_epoch_end(epoch, avg_loss):
        vloss = float("nan")
        if (val_loader is not None and args.val_every > 0 and
                (epoch % args.val_every == 0 or epoch == args.epochs)):
            vloss = eval_loss_keep_rng(policy, val_loader)
            print(f"            val_loss(EMA, velocity MSE)={vloss:.5f}")
        history.append([epoch, avg_loss, vloss])

        if args.save_every > 0 and epoch % args.save_every == 0 and epoch < args.epochs:
            path = ckpt.with_name(f"{ckpt.stem}_epoch{epoch:03d}{ckpt.suffix}")
            torch.save(build_checkpoint(policy, args.arch, net_kwargs, dataset.cond_dim), path)
            write_summary()
            print(f"  [checkpoint] saved {path}")

    policy.train(loader, epochs=args.epochs, on_epoch_end=on_epoch_end)

    # Held-out TEST loss: single final pass on data never seen in train/val.
    test_loss = float("nan")
    if test_loader is not None:
        test_loss = validation_loss(policy, test_loader)
        print(f"Test loss (EMA, velocity MSE): {test_loss:.5f}")

    # Persist loss curves + finalize the run summary.
    curve_path = Path(LOSS_DIR) / f"loss_history_{args.arch}.csv"
    with open(curve_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for ep, tr, vl in history:
            w.writerow([ep, f"{tr:.6f}", "" if np.isnan(vl) else f"{vl:.6f}"])

    summary = write_summary(test_loss)
    print(f"Saved loss curve to {curve_path} and summary to {summary_path}")

    # Final self-describing checkpoint (EMA weights -> smoother inference).
    torch.save(build_checkpoint(policy, args.arch, net_kwargs, dataset.cond_dim), ckpt)
    print(f"Saved checkpoint to {ckpt}")

    # Sidecar files: hyperparameters + a per-checkpoint copy of the norm stats
    # (the shared results/norm_stats.json is overwritten by each run; the copy
    # lets old models still be evaluated with the exact normalization used).
    hparams = {
        "method": "flow_matching",
        "arch": args.arch,
        "net_kwargs": net_kwargs,
        "num_steps": NUM_STEPS, "sigma_min": SIGMA_MIN,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "seed": SEED,
        "k": K, "horizon": H, "nx": NX, "nx_feat": NX_FEAT, "nu": NU,
        "use_goal": USE_GOAL,
        "use_angular_features": True,
        "cond_dim": dataset.cond_dim,
        "split_ratios": list(SPLIT_RATIOS),
        "n_train_traj": len(train_keys),
        "n_val_traj": len(val_keys),
        "n_test_traj": len(test_keys),
        "n_params": int(sum(p.numel() for p in network.parameters())),
        "warmstart": warmstart_used,
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss":   summary["final_val_loss"],
        "test_loss":        summary["test_loss"],
        "checkpoint": str(ckpt),
        "device": DEVICE,
    }
    hparams_path = ckpt.with_name(f"{ckpt.stem}_hparams.json")
    with open(hparams_path, "w") as fp:
        json.dump(hparams, fp, indent=2)
    stats_sidecar = ckpt.with_name(f"{ckpt.stem}_stats.json")
    dataset.save_stats(stats_sidecar)
    print(f"Saved hyperparameters to {hparams_path} and norm stats to {stats_sidecar}")


if __name__ == "__main__":
    main()
