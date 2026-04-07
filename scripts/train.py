#!/usr/bin/env python3
"""
Training Script for MRI-DiffRecon.

Supports:
  - Single GPU and multi-GPU (DDP) training
  - Diffusion model (score matching) and U-Net baseline
  - fastMRI knee/brain or synthetic datasets
  - Automatic config loading from YAML

Usage:
    # Single-GPU diffusion training
    python scripts/train.py --config configs/fastmri_knee_config.yaml

    # Multi-GPU training (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/fastmri_knee_config.yaml \
        --distributed

    # U-Net baseline
    python scripts/train.py --config configs/fastmri_knee_config.yaml \
        --model unet --num_cascades 12

    # Quick test with synthetic data
    python scripts/train.py --synthetic --num_steps 100 --image_size 64

    # Resume from checkpoint
    python scripts/train.py --config configs/fastmri_knee_config.yaml \
        --resume runs/knee_singlecoil_4x/checkpoints/checkpoint_latest.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MRI-DiffRecon score-based diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )

    # Data
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to fastMRI data directory (overrides config)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["fastmri_knee", "fastmri_brain", "synthetic"],
        help="Dataset type (overrides config)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no fastMRI download required)"
    )
    parser.add_argument(
        "--acceleration", type=int, default=None,
        choices=[4, 8, 16],
        help="Undersampling acceleration factor"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="diffusion",
        choices=["diffusion", "unet", "cascaded_unet"],
        help="Model architecture"
    )
    parser.add_argument(
        "--sde_type", type=str, default=None,
        choices=["vp", "ve"],
        help="SDE type (VP-SDE or VE-SDE)"
    )
    parser.add_argument(
        "--num_cascades", type=int, default=12,
        help="Number of cascades for U-Net baseline"
    )

    # Training
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Number of training steps (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )

    # Hardware
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable multi-GPU DDP training"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Training device"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable mixed precision training"
    )

    # Image
    parser.add_argument(
        "--image_size", type=int, default=None,
        help="Image size for synthetic data"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Flatten nested config for trainer
    flat_config = {}

    # Data
    data = config.get("data", {})
    flat_config["data_root"] = data.get("data_root", "")
    flat_config["dataset"] = data.get("dataset", "synthetic")
    flat_config["acceleration"] = data.get("acceleration", 4)
    flat_config["batch_size"] = data.get("batch_size", 4)
    flat_config["num_workers"] = data.get("num_workers", 4)
    flat_config["image_size"] = data.get("image_size", 320)

    # Model
    flat_config["model"] = config.get("model", {})

    # Diffusion
    diff = config.get("diffusion", {})
    flat_config["sde_type"] = diff.get("sde_type", "vp")
    flat_config["beta_min"] = diff.get("beta_min", 0.1)
    flat_config["beta_max"] = diff.get("beta_max", 20.0)
    flat_config["dc_mode"] = diff.get("dc_mode", "gradient")
    flat_config["dc_lambda"] = diff.get("dc_lambda", 1.0)
    flat_config["dc_freq"] = diff.get("dc_freq", 1)

    # Training
    train = config.get("training", {})
    flat_config["num_training_steps"] = train.get("num_training_steps", 500000)
    flat_config["learning_rate"] = train.get("learning_rate", 2e-4)
    flat_config["weight_decay"] = train.get("weight_decay", 1e-4)
    flat_config["warmup_steps"] = train.get("warmup_steps", 10000)
    flat_config["max_grad_norm"] = train.get("grad_clip", 1.0)
    flat_config["ema_decay"] = train.get("ema_decay", 0.9999)
    flat_config["use_amp"] = train.get("use_amp", True)
    flat_config["log_every"] = train.get("log_every", 100)
    flat_config["val_every"] = train.get("val_every", 5000)
    flat_config["save_every"] = train.get("save_every", 10000)
    flat_config["loss_weighting"] = train.get("loss_weighting", "likelihood")
    flat_config["distributed"] = train.get("distributed", False)

    # Output
    output = config.get("output", {})
    flat_config["output_dir"] = output.get("output_dir", "runs/default")

    return flat_config


def train_diffusion(config: dict) -> None:
    """Train the diffusion model."""
    from src.training.train_score import ScoreMatchingTrainer
    trainer = ScoreMatchingTrainer(config)
    trainer.train()


def train_unet(config: dict) -> None:
    """Train the U-Net baseline (supervised)."""
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from src.models.unet_baseline import CascadedUNet
    from src.data.fastmri_dataset import SyntheticMRIDataset, build_dataloader
    from src.evaluation.mri_metrics import ssim as compute_ssim, psnr as compute_psnr
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(config.get("device", "cuda"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = CascadedUNet(
        num_cascades=config.get("num_cascades", 12),
        base_channels=32,
    ).to(device)

    # Data
    if config.get("dataset") == "synthetic":
        train_ds = SyntheticMRIDataset(
            num_samples=config.get("num_train_samples", 1000),
            image_size=config.get("image_size", 64),
            acceleration=config.get("acceleration", 4),
        )
        val_ds = SyntheticMRIDataset(
            num_samples=config.get("num_val_samples", 100),
            image_size=config.get("image_size", 64),
            acceleration=config.get("acceleration", 4),
            seed=999,
        )
    else:
        raise ValueError("fastMRI U-Net training: set dataset to fastmri_knee or fastmri_brain")

    train_loader = build_dataloader(train_ds, config.get("batch_size", 4), num_workers=0)
    val_loader = build_dataloader(val_ds, config.get("batch_size", 4), num_workers=0, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4))
    criterion = nn.L1Loss()
    writer = SummaryWriter(output_dir / "logs")

    num_steps = config.get("num_training_steps", 100000)
    step = 0
    train_iter = iter(train_loader)

    print(f"Training U-Net for {num_steps} steps...")

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        x_gt = batch["x_gt"].to(device)
        x_zf = batch["x_zf"].to(device)
        kspace_obs = batch["kspace_obs"].to(device)
        mask = batch["mask_dc"].to(device)

        pred = model(x_zf, kspace_obs, mask)
        loss = criterion(pred, x_gt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: loss={loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)

    torch.save({"model": model.state_dict(), "step": step}, output_dir / "best_model.pt")
    print(f"U-Net saved to {output_dir}/best_model.pt")
    writer.close()


def main():
    args = parse_args()

    # Build config
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = {
            "dataset": "synthetic",
            "num_train_samples": 1000,
            "num_val_samples": 100,
            "image_size": 320,
            "batch_size": 4,
            "num_workers": 4,
            "output_dir": "runs/default",
            "sde_type": "vp",
            "beta_min": 0.1,
            "beta_max": 20.0,
            "dc_mode": "gradient",
            "dc_lambda": 1.0,
            "dc_freq": 1,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 1000,
            "num_training_steps": 10000,
            "ema_decay": 0.9999,
            "use_amp": True,
            "log_every": 100,
            "val_every": 1000,
            "save_every": 5000,
            "loss_weighting": "likelihood",
            "distributed": False,
            "device": "cuda",
            "model": {
                "model_type": "ncsn_pp",
                "base_channels": 64,
                "channel_mults": [1, 2, 4, 8],
                "num_res_blocks": 2,
                "emb_dim": 256,
                "dropout": 0.1,
            },
        }

    # Override with CLI args
    if args.synthetic or args.dataset == "synthetic":
        config["dataset"] = "synthetic"
    elif args.dataset:
        config["dataset"] = args.dataset

    if args.data_root:
        config["data_root"] = args.data_root
    if args.acceleration:
        config["acceleration"] = args.acceleration
    if args.num_steps:
        config["num_training_steps"] = args.num_steps
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.sde_type:
        config["sde_type"] = args.sde_type
    if args.distributed:
        config["distributed"] = True
    if args.no_amp:
        config["use_amp"] = False
    if args.image_size:
        config["image_size"] = args.image_size
    if args.num_cascades:
        config["num_cascades"] = args.num_cascades

    config["device"] = args.device

    print(f"Training {'diffusion' if args.model == 'diffusion' else 'U-Net'} model")
    print(f"Config: {config}")

    if args.model == "diffusion":
        train_diffusion(config)
    else:
        train_unet(config)


if __name__ == "__main__":
    main()
