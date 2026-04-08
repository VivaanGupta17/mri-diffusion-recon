"""
Score Matching Trainer for MRI Diffusion Reconstruction.

Implements denoising score matching (DSM) training for the NCSN++ score network.
Supports:
  - Continuous noise level sampling (VP-SDE and VE-SDE schedules)
  - EMA (Exponential Moving Average) of model parameters
  - Mixed precision training (torch.amp.autocast)
  - Multi-GPU distributed training (DDP)
  - Gradient clipping and warmup learning rate schedule
  - TensorBoard logging
  - Checkpoint save/resume

Denoising Score Matching Objective:
    L(θ) = E_{t, x0, ε} [ λ(t) · ‖s_θ(x_t, t, y) + ε/σ(t)‖² ]

Where:
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   (VP-SDE forward process)
    ε ~ N(0, I)
    λ(t): noise level-dependent weighting
    y: conditioning input (zero-filled aliased image)

The score network learns s_θ ≈ ∇_x log p_σ(x|y) — the gradient of the
log conditional density, which points toward cleaner images.

References:
    Song & Ermon 2019. "Generative Modeling by Estimating Gradients." NeurIPS.
    Song et al. 2021. "Score-Based Generative Modeling through SDEs." ICLR.
    Ho et al. 2020. "Denoising Diffusion Probabilistic Models." NeurIPS.
"""

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMAModel:
    """
    Exponential Moving Average of model parameters.

    EMA maintains a shadow copy of model weights that evolves as:
        θ_ema ← decay * θ_ema + (1 - decay) * θ_current

    EMA weights produce significantly better sample quality than the raw
    model weights, especially for diffusion models. Use θ_ema at inference.

    Args:
        model:  PyTorch model to track.
        decay:  EMA decay rate (typical: 0.9999 for large models).
        device: Device for shadow parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device or next(model.parameters()).device
        # Create shadow copy of parameters (detached from computation graph)
        self.shadow_params = {
            name: param.clone().detach().to(self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.num_updates = 0

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters from model."""
        self.num_updates += 1
        # Warm-up: use minimum decay early in training to avoid stale averages
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name].mul_(decay).add_(
                        param.data.to(self.device), alpha=1.0 - decay
                    )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights to model for evaluation/inference."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name].to(param.device))

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"shadow_params": self.shadow_params,
                    "num_updates": self.num_updates,
                    "decay": self.decay}, path)

    def load(self, path: Union[str, Path]) -> None:
        state = torch.load(path, map_location=self.device)
        self.shadow_params = state["shadow_params"]
        self.num_updates = state["num_updates"]
        self.decay = state["decay"]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def denoising_score_matching_loss(
    score_pred: torch.Tensor,
    noise: torch.Tensor,
    std: torch.Tensor,
    weighting: str = "likelihood",
) -> Dict[str, torch.Tensor]:
    """
    Denoising score matching loss.

    For VP-SDE, the optimal score is s*(x_t) = -noise / std.
    We weight the loss by λ(t) to balance loss across noise levels.

    Weightings:
        - 'likelihood': λ(t) = std²(t) — recovers exact ELBO
        - 'uniform':    λ(t) = 1 — uniform weighting across noise levels
        - 'snr':        λ(t) = SNR(t) — focus on high SNR (small noise) regime
        - 'truncated':  Clamp weight to avoid extreme values at t→0 and t→1

    Args:
        score_pred: Predicted score, (B, C, H, W).
        noise:      Ground truth noise, (B, C, H, W).
        std:        Noise standard deviation for each sample, (B,).
        weighting:  Loss weighting scheme.

    Returns:
        Dict with 'loss' (scalar) and 'per_sample_loss' (B,).
    """
    B = score_pred.shape[0]
    std = std.to(score_pred.device)

    # Target score: -ε / σ(t)
    target_score = -noise / std[:, None, None, None].clamp(min=1e-8)

    # Per-sample loss (mean over spatial dims and channels)
    per_sample_loss = F.mse_loss(
        score_pred, target_score, reduction="none"
    ).mean(dim=(1, 2, 3))  # (B,)

    # Loss weighting
    if weighting == "likelihood":
        weight = std**2
    elif weighting == "snr":
        snr = 1.0 / (std**2 + 1e-8)
        weight = snr / snr.mean()
    elif weighting == "truncated":
        weight = std**2
        weight = weight.clamp(0.01, 1.0)
    else:  # uniform
        weight = torch.ones(B, device=score_pred.device)

    weighted_loss = (weight * per_sample_loss).mean()

    return {
        "loss": weighted_loss,
        "per_sample_loss": per_sample_loss.detach(),
        "weight": weight.detach(),
    }


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build linear warmup + cosine decay schedule.

    Warmup: linear increase from 0 → base_lr over num_warmup_steps
    After warmup: cosine decay from base_lr → min_lr
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps],
    )


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages model checkpoints during training.

    Saves:
        - Latest checkpoint (always overwritten)
        - Best checkpoint (lowest validation loss)
        - Periodic checkpoints (every N steps, configurable)

    Also saves EMA model separately for inference.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_checkpoints: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = float("inf")
        self.checkpoint_paths: List[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ema_model: EMAModel,
        step: int,
        metrics: Dict[str, float],
        config: Dict,
    ) -> None:
        """Save checkpoint and manage checkpoint rotation."""
        state = {
            "model": model.state_dict()
            if not isinstance(model, DDP) else model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": config,
        }

        # Latest checkpoint
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(state, latest_path)

        # EMA model
        ema_path = self.output_dir / "ema_latest.pt"
        ema_model.save(ema_path)

        # Best checkpoint
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            best_path = self.output_dir / "best_model.pt"
            torch.save(state, best_path)
            ema_model.save(self.output_dir / "ema_best.pt")
            logger.info(f"New best model at step {step}: val_loss={val_loss:.4f}")

        # Periodic checkpoint
        ckpt_path = self.output_dir / f"checkpoint_step_{step:08d}.pt"
        torch.save(state, ckpt_path)
        self.checkpoint_paths.append(ckpt_path)

        # Rotate old checkpoints
        while len(self.checkpoint_paths) > self.max_checkpoints:
            old_path = self.checkpoint_paths.pop(0)
            if old_path.exists():
                old_path.unlink()

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ema_model: Optional[EMAModel] = None,
    ) -> int:
        """Load latest checkpoint. Returns step number."""
        latest_path = self.output_dir / "checkpoint_latest.pt"
        if not latest_path.exists():
            return 0

        state = torch.load(latest_path, map_location="cpu")

        if isinstance(model, DDP):
            model.module.load_state_dict(state["model"])
        else:
            model.load_state_dict(state["model"])

        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        if ema_model is not None:
            ema_path = self.output_dir / "ema_latest.pt"
            if ema_path.exists():
                ema_model.load(ema_path)

        step = state["step"]
        logger.info(f"Resumed training from step {step}")
        return step


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    diffusion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema_model: EMAModel,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """
    Single training step: forward pass, loss, backward, update.

    Args:
        model:     Score network (possibly DDP-wrapped).
        batch:     Data batch with 'x_gt', 'x_zf', 'mask_dc', 'kspace_obs'.
        diffusion: MRIDiffusionModel providing compute_loss.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        scaler:    GradScaler for mixed precision.
        ema_model: EMA model for tracking parameters.
        config:    Training config dict.
        device:    Target device.

    Returns:
        Metrics dict: loss, grad_norm, lr.
    """
    model.train()

    x_gt = batch["x_gt"].to(device, non_blocking=True)
    x_zf = batch["x_zf"].to(device, non_blocking=True)

    # Sample noise levels from the SDE schedule
    B = x_gt.shape[0]
    t = diffusion.sde.sample_t(B, device)

    # Forward diffusion: perturb x_gt → x_t
    x_t, noise = diffusion.sde.forward_sample(x_gt, t)

    # Mixed precision forward pass
    with autocast(enabled=config.get("use_amp", True)):
        # Predict score
        score_pred = model(x_t, t, x_zf)

        # Get std for loss weighting
        _, std = diffusion.sde.marginal_prob(x_gt, t)

        # Loss
        loss_dict = denoising_score_matching_loss(
            score_pred, noise, std,
            weighting=config.get("loss_weighting", "likelihood"),
        )
        loss = loss_dict["loss"]

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    # Gradient clipping
    if config.get("max_grad_norm", None):
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )
    else:
        grad_norm = torch.tensor(0.0)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # EMA update
    if isinstance(model, DDP):
        ema_model.update(model.module)
    else:
        ema_model.update(model)

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "lr": scheduler.get_last_lr()[0],
        "t_mean": t.mean().item(),
        "std_mean": std.mean().item(),
    }


# ---------------------------------------------------------------------------
# Validation Step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    diffusion,
    ema_model: EMAModel,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Compute validation loss using EMA model parameters.

    Args:
        model:       Score network.
        val_loader:  Validation dataloader.
        diffusion:   Diffusion model for loss computation.
        ema_model:   EMA parameters (used for validation, not raw model).
        device:      Device.
        max_batches: Limit validation batches for speed.

    Returns:
        Metrics dict: val_loss, val_loss_std.
    """
    # Copy EMA weights to model for validation
    original_params = {
        name: param.clone() for name, param in model.named_parameters()
    }
    ema_model.copy_to(model)
    model.eval()

    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        x_gt = batch["x_gt"].to(device)
        x_zf = batch["x_zf"].to(device)
        B = x_gt.shape[0]
        t = diffusion.sde.sample_t(B, device)
        x_t, noise = diffusion.sde.forward_sample(x_gt, t)

        with autocast(enabled=True):
            score_pred = model(x_t, t, x_zf)
            _, std = diffusion.sde.marginal_prob(x_gt, t)
            loss_dict = denoising_score_matching_loss(score_pred, noise, std)
            losses.append(loss_dict["loss"].item())

    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    return {
        "val_loss": float(torch.tensor(losses).mean()),
        "val_loss_std": float(torch.tensor(losses).std()),
    }


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class ScoreMatchingTrainer:
    """
    Complete trainer for the score-based diffusion model.

    Handles training loop, validation, checkpointing, and logging.
    Supports both single-GPU and multi-GPU (DDP) training.

    Usage:
        trainer = ScoreMatchingTrainer(config)
        trainer.train()
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main = self.local_rank == 0

        # Setup distributed training
        self.distributed = config.get("distributed", False)
        if self.distributed:
            torch.distributed.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

        self._setup_logging()
        self._build_model()
        self._build_optimizer()
        self._build_datasets()

        self.ckpt_manager = CheckpointManager(
            config["output_dir"],
            max_checkpoints=config.get("max_checkpoints", 5),
        )
        self.writer = (
            SummaryWriter(config["output_dir"])
            if self.is_main else None
        )

    def _setup_logging(self) -> None:
        """Configure Python logging."""
        level = logging.INFO if self.is_main else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _build_model(self) -> None:
        """Initialize score network, diffusion model, and EMA."""
        from src.models.score_network import build_score_network
        from src.models.diffusion_mri import MRIDiffusionModel

        score_net = build_score_network(self.config["model"]).to(self.device)

        self.diffusion = MRIDiffusionModel(
            score_network=score_net,
            sde_type=self.config.get("sde_type", "vp"),
            beta_min=self.config.get("beta_min", 0.1),
            beta_max=self.config.get("beta_max", 20.0),
            dc_mode=self.config.get("dc_mode", "gradient"),
            dc_lambda=self.config.get("dc_lambda", 1.0),
        ).to(self.device)

        if self.distributed:
            self.model = DDP(
                self.diffusion.score_network,
                device_ids=[self.local_rank],
            )
        else:
            self.model = self.diffusion.score_network

        self.ema_model = EMAModel(
            self.diffusion.score_network,
            decay=self.config.get("ema_decay", 0.9999),
            device=self.device,
        )

        if self.is_main:
            total_params = sum(
                p.numel() for p in self.diffusion.parameters() if p.requires_grad
            )
            logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

    def _build_optimizer(self) -> None:
        """Build AdamW optimizer and LR scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        num_steps = self.config.get("num_training_steps", 500000)
        warmup_steps = self.config.get("warmup_steps", 10000)

        self.scheduler = build_lr_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps,
        )

        self.scaler = GradScaler(enabled=self.config.get("use_amp", True))

    def _build_datasets(self) -> None:
        """Build train/val dataloaders."""
        from src.data.fastmri_dataset import (
            FastMRIKneeDataset,
            FastMRIBrainDataset,
            SyntheticMRIDataset,
            build_dataloader,
        )

        data_root = self.config.get("data_root", "")
        dataset_type = self.config.get("dataset", "synthetic")
        acceleration = self.config.get("acceleration", 4)

        if dataset_type == "fastmri_knee":
            train_dataset = FastMRIKneeDataset(
                data_root, split="train",
                acceleration=acceleration,
                transform_type="diffusion",
            )
            val_dataset = FastMRIKneeDataset(
                data_root, split="val",
                acceleration=acceleration,
                transform_type="diffusion",
            )
        elif dataset_type == "fastmri_brain":
            train_dataset = FastMRIBrainDataset(
                data_root, split="train",
                acceleration=acceleration,
                transform_type="diffusion",
            )
            val_dataset = FastMRIBrainDataset(
                data_root, split="val",
                acceleration=acceleration,
                transform_type="diffusion",
            )
        else:
            # Synthetic dataset for testing
            train_dataset = SyntheticMRIDataset(
                num_samples=self.config.get("num_train_samples", 1000),
                image_size=self.config.get("image_size", 64),
                acceleration=acceleration,
            )
            val_dataset = SyntheticMRIDataset(
                num_samples=self.config.get("num_val_samples", 100),
                image_size=self.config.get("image_size", 64),
                acceleration=acceleration,
                seed=999,
            )

        batch_size = self.config.get("batch_size", 4)
        num_workers = self.config.get("num_workers", 4)

        self.train_loader = build_dataloader(
            train_dataset, batch_size, num_workers,
            shuffle=True, distributed=self.distributed,
        )
        self.val_loader = build_dataloader(
            val_dataset, batch_size, num_workers,
            shuffle=False, distributed=False,
        )

        logger.info(
            f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val"
        )

    def train(self) -> None:
        """
        Main training loop.

        Runs for `num_training_steps` gradient updates.
        Logs metrics every `log_every` steps.
        Validates every `val_every` steps.
        Saves checkpoint every `save_every` steps.
        """
        config = self.config
        num_steps = config.get("num_training_steps", 500000)
        log_every = config.get("log_every", 100)
        val_every = config.get("val_every", 5000)
        save_every = config.get("save_every", 10000)

        # Resume from checkpoint if available
        start_step = self.ckpt_manager.load_latest(
            self.model, self.optimizer, self.scheduler, self.ema_model
        )

        logger.info(f"Starting training from step {start_step}")
        logger.info(f"Total steps: {num_steps}")

        step = start_step
        train_iter = iter(self.train_loader)
        metrics_accum: Dict[str, List[float]] = {
            "loss": [], "grad_norm": [], "lr": []
        }
        t0 = time.time()

        while step < num_steps:
            # Get next batch (restart iterator if exhausted)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            step_metrics = train_step(
                self.model, batch, self.diffusion,
                self.optimizer, self.scheduler, self.scaler,
                self.ema_model, config, self.device,
            )
            step += 1

            # Accumulate metrics
            for k, v in step_metrics.items():
                if k in metrics_accum:
                    metrics_accum[k].append(v)

            # Logging
            if self.is_main and step % log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = log_every / elapsed
                avg_loss = sum(metrics_accum["loss"]) / len(metrics_accum["loss"])
                avg_gnorm = sum(metrics_accum["grad_norm"]) / len(metrics_accum["grad_norm"])

                logger.info(
                    f"Step {step:7d}/{num_steps} | "
                    f"loss={avg_loss:.4f} | "
                    f"gnorm={avg_gnorm:.2f} | "
                    f"lr={step_metrics['lr']:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, step)
                    self.writer.add_scalar("train/grad_norm", avg_gnorm, step)
                    self.writer.add_scalar("train/lr", step_metrics["lr"], step)
                    self.writer.add_scalar("train/t_mean", step_metrics["t_mean"], step)

                # Reset accumulators
                metrics_accum = {"loss": [], "grad_norm": [], "lr": []}
                t0 = time.time()

            # Validation
            if self.is_main and step % val_every == 0:
                logger.info(f"Running validation at step {step}...")
                val_metrics = validate(
                    self.model, self.val_loader, self.diffusion,
                    self.ema_model, self.device,
                )
                logger.info(
                    f"Val: loss={val_metrics['val_loss']:.4f} ± "
                    f"{val_metrics['val_loss_std']:.4f}"
                )
                if self.writer:
                    self.writer.add_scalar("val/loss", val_metrics["val_loss"], step)

                # Save checkpoint at validation time
                self.ckpt_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    self.ema_model, step,
                    metrics={**step_metrics, **val_metrics},
                    config=config,
                )

            # Periodic save (without validation)
            elif self.is_main and step % save_every == 0:
                self.ckpt_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    self.ema_model, step,
                    metrics=step_metrics,
                    config=config,
                )

        logger.info("Training complete!")
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    # Minimal training test with synthetic data
    config = {
        "output_dir": "/tmp/mri_diffusion_test",
        "dataset": "synthetic",
        "num_train_samples": 32,
        "num_val_samples": 8,
        "image_size": 64,
        "acceleration": 4,
        "batch_size": 4,
        "num_workers": 0,
        "num_training_steps": 20,
        "log_every": 5,
        "val_every": 10,
        "save_every": 20,
        "sde_type": "vp",
        "learning_rate": 2e-4,
        "use_amp": False,  # Disable AMP for CPU testing
        "device": "cpu",
        "model": {
            "model_type": "ncsn_pp",
            "base_channels": 16,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "emb_dim": 64,
            "dropout": 0.0,
        },
    }

    trainer = ScoreMatchingTrainer(config)
    trainer.train()
    print("Trainer test completed successfully.")

EMA_DECAY = 0.999
