"""
MicroForge Trainer — full training loop with multi-task loss and AMP.

Handles:
    - Next-token LM loss (CrossEntropy)
    - Multi-label tool-router loss (BCEWithLogits)
    - Confidence-head regression loss (MSE)
    - Mixed-precision training via ``torch.amp``
    - Gradient accumulation and clipping
    - Cosine-warmup LR scheduling
    - Periodic validation, logging, and early stopping
    - Checkpoint save / resume
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


# ─── Training configuration ───────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All knobs for a MicroForge training run.

    Attributes:
        epochs: Maximum number of full passes over the training set.
        learning_rate: Peak learning rate (reached after warmup).
        weight_decay: AdamW weight-decay coefficient.
        warmup_steps: Steps of linear LR warmup.
        max_grad_norm: Maximum gradient L2 norm for clipping.
        gradient_accumulation_steps: Micro-batches to accumulate before
            an optimiser step.
        lm_loss_weight: Scalar multiplier for the LM cross-entropy loss.
        tool_loss_weight: Scalar multiplier (α) for the tool-router loss.
        confidence_loss_weight: Scalar multiplier (β) for confidence loss.
        val_every_steps: Run validation every N optimiser steps.
        log_every_steps: Print training metrics every N optimiser steps.
        early_stopping_patience: Validation steps without improvement
            before stopping.
        mixed_precision: Whether to use AMP (``float16``).
        checkpoint_dir: Directory for saving/loading checkpoints.
        min_lr: Minimum LR at the end of cosine decay.
    """

    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    lm_loss_weight: float = 1.0
    tool_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    val_every_steps: int = 500
    log_every_steps: int = 50
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    min_lr: float = 1e-6


# ─── Metrics tracker ──────────────────────────────────────────────────


@dataclass
class _Metrics:
    """Running statistics for a training or validation epoch.

    Attributes:
        total_loss: Sum of combined (weighted) losses.
        lm_loss: Sum of LM cross-entropy losses.
        tool_loss: Sum of tool-router losses.
        confidence_loss: Sum of confidence-head losses.
        tool_correct: Number of correctly predicted tool labels.
        tool_total: Total number of tool label predictions.
        exact_matches: Samples where the full tool-label set matched.
        sample_count: Number of samples processed.
        step_count: Number of steps accumulated.
    """

    total_loss: float = 0.0
    lm_loss: float = 0.0
    tool_loss: float = 0.0
    confidence_loss: float = 0.0
    tool_correct: int = 0
    tool_total: int = 0
    exact_matches: int = 0
    sample_count: int = 0
    step_count: int = 0

    def update(
        self,
        total: float,
        lm: float,
        tool: float,
        conf: float,
        tool_preds: torch.Tensor,
        tool_targets: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Accumulate one micro-batch of results.

        Args:
            total: Combined weighted loss (scalar).
            lm: LM loss component (scalar).
            tool: Tool-router loss component (scalar).
            conf: Confidence loss component (scalar).
            tool_preds: Raw logits from the tool router ``(B, C)``.
            tool_targets: Binary ground-truth labels ``(B, C)``.
            batch_size: Number of samples in this micro-batch.
        """
        self.total_loss += total
        self.lm_loss += lm
        self.tool_loss += tool
        self.confidence_loss += conf
        self.step_count += 1
        self.sample_count += batch_size

        # Per-label accuracy and exact-match
        with torch.no_grad():
            binary_preds = (tool_preds.detach() > 0.0).float()
            correct = (binary_preds == tool_targets).sum().item()
            self.tool_correct += int(correct)
            self.tool_total += tool_targets.numel()
            exact = (binary_preds == tool_targets).all(dim=-1).sum().item()
            self.exact_matches += int(exact)

    def summarise(self) -> Dict[str, float]:
        """Return averaged metrics as a dictionary.

        Returns:
            Dictionary with keys ``avg_total_loss``, ``avg_lm_loss``,
            ``avg_tool_loss``, ``avg_confidence_loss``,
            ``tool_accuracy``, ``exact_match_rate``.
        """
        n = max(self.step_count, 1)
        s = max(self.sample_count, 1)
        return {
            "avg_total_loss": self.total_loss / n,
            "avg_lm_loss": self.lm_loss / n,
            "avg_tool_loss": self.tool_loss / n,
            "avg_confidence_loss": self.confidence_loss / n,
            "tool_accuracy": self.tool_correct / max(self.tool_total, 1),
            "exact_match_rate": self.exact_matches / s,
        }

    def reset(self) -> None:
        """Zero every accumulator."""
        self.total_loss = 0.0
        self.lm_loss = 0.0
        self.tool_loss = 0.0
        self.confidence_loss = 0.0
        self.tool_correct = 0
        self.tool_total = 0
        self.exact_matches = 0
        self.sample_count = 0
        self.step_count = 0


# ─── Early stopping ───────────────────────────────────────────────────


class _EarlyStopping:
    """Monitors a metric and signals when to stop training.

    Args:
        patience: Number of evaluations without improvement before
            triggering a stop.
        mode: ``"max"`` to track an increasing metric (e.g. accuracy),
            ``"min"`` for a decreasing metric (e.g. loss).
    """

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, value: float) -> bool:
        """Check whether training should stop.

        Args:
            value: Latest metric value.

        Returns:
            ``True`` if patience has been exhausted.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            (value > self.best) if self.mode == "max" else (value < self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# ─── Main trainer ─────────────────────────────────────────────────────


class MicroForgeTrainer:
    """Complete training loop for a MicroForge model.

    The trainer orchestrates:
        * Multi-task loss computation (LM + tool-router + confidence).
        * Mixed-precision (AMP) forward/backward passes.
        * Gradient accumulation and norm clipping.
        * AdamW optimiser with cosine-warmup LR scheduling.
        * Periodic validation and metric logging.
        * Checkpoint saving/loading.
        * Early stopping based on tool-calling accuracy.

    The expected model output is a dictionary (or named-tuple-like object)
    with keys:

        * ``lm_logits``: ``(B, T, V)`` — next-token prediction logits.
        * ``tool_logits``: ``(B, C)`` — multi-label tool-router logits.
        * ``confidence``: ``(B, 1)`` — scalar confidence score.

    Each data-loader batch must be a dictionary with at least:

        * ``input_ids``: ``(B, T)`` — token IDs.
        * ``labels``: ``(B, T)`` — shifted LM labels (``-100`` for masked).
        * ``tool_labels``: ``(B, C)`` — binary tool labels.
        * ``confidence_target``: ``(B, 1)`` — target confidence values.

    Args:
        config: A :class:`TrainingConfig` instance.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.tool_criterion = nn.BCEWithLogitsLoss()
        self.confidence_criterion = nn.MSELoss()

        # Early stopping on tool accuracy (higher is better)
        self.early_stopping = _EarlyStopping(
            patience=self.config.early_stopping_patience, mode="max"
        )

    # ── Public entry point ─────────────────────────────────────────────

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            model: The MicroForge model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Override ``config.epochs`` if provided.
            lr: Override ``config.learning_rate`` if provided.
            device: Target device (default: CUDA if available, else CPU).

        Returns:
            Dictionary of metric histories keyed by metric name, each a
            list of per-evaluation-step values.
        """
        cfg = self.config
        epochs = epochs or cfg.epochs
        lr = lr or cfg.learning_rate
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = model.to(device)
        optimizer = self._build_optimizer(model, lr)
        total_steps = self._estimate_total_steps(train_loader, epochs)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=total_steps,
            min_lr=cfg.min_lr,
        )

        # AMP scaler (float16)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")

        # Histories
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_tool_accuracy": [],
            "val_exact_match": [],
            "lr": [],
        }

        global_step = 0
        train_metrics = _Metrics()

        logger.info(
            "Starting training — %d epochs, %d total steps, device=%s, AMP=%s",
            epochs,
            total_steps,
            device,
            cfg.mixed_precision,
        )

        for epoch in range(epochs):
            model.train()
            train_metrics.reset()

            for batch_idx, batch in enumerate(train_loader):
                batch = self._to_device(batch, device)
                batch_size = batch["input_ids"].size(0)

                # ── Forward + loss ─────────────────────────────────────
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=cfg.mixed_precision,
                ):
                    outputs = model(batch["input_ids"])
                    losses = self._compute_losses(outputs, batch)
                    total_loss = losses["total"] / cfg.gradient_accumulation_steps

                # ── Backward ───────────────────────────────────────────
                scaler.scale(total_loss).backward()

                # ── Accumulate metrics ─────────────────────────────────
                train_metrics.update(
                    total=losses["total"].item(),
                    lm=losses["lm"].item(),
                    tool=losses["tool"].item(),
                    conf=losses["confidence"].item(),
                    tool_preds=outputs["tool_logits"],
                    tool_targets=batch["tool_labels"],
                    batch_size=batch_size,
                )

                # ── Optimiser step (every accum steps) ─────────────────
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    # ── Logging ─────────────────────────────────────────
                    if global_step % cfg.log_every_steps == 0:
                        summary = train_metrics.summarise()
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "[Step %d | Epoch %d] "
                            "total=%.4f  lm=%.4f  tool=%.4f  conf=%.4f  "
                            "tool_acc=%.3f  em=%.3f  lr=%.2e",
                            global_step,
                            epoch + 1,
                            summary["avg_total_loss"],
                            summary["avg_lm_loss"],
                            summary["avg_tool_loss"],
                            summary["avg_confidence_loss"],
                            summary["tool_accuracy"],
                            summary["exact_match_rate"],
                            current_lr,
                        )
                        history["train_loss"].append(summary["avg_total_loss"])
                        history["lr"].append(current_lr)
                        train_metrics.reset()

                    # ── Validation ──────────────────────────────────────
                    if global_step % cfg.val_every_steps == 0:
                        val_summary = self._validate(model, val_loader, device)
                        history["val_loss"].append(val_summary["avg_total_loss"])
                        history["val_tool_accuracy"].append(
                            val_summary["tool_accuracy"]
                        )
                        history["val_exact_match"].append(
                            val_summary["exact_match_rate"]
                        )
                        logger.info(
                            "[Val @ Step %d] "
                            "total=%.4f  lm=%.4f  tool=%.4f  conf=%.4f  "
                            "tool_acc=%.3f  em=%.3f",
                            global_step,
                            val_summary["avg_total_loss"],
                            val_summary["avg_lm_loss"],
                            val_summary["avg_tool_loss"],
                            val_summary["avg_confidence_loss"],
                            val_summary["tool_accuracy"],
                            val_summary["exact_match_rate"],
                        )

                        # Checkpoint best model
                        self.save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            val_summary,
                        )

                        # Early stopping
                        if self.early_stopping(val_summary["tool_accuracy"]):
                            logger.info(
                                "Early stopping triggered at step %d "
                                "(patience=%d, best_tool_acc=%.4f)",
                                global_step,
                                cfg.early_stopping_patience,
                                self.early_stopping.best,
                            )
                            return history

            logger.info("Epoch %d / %d complete.", epoch + 1, epochs)

        logger.info("Training finished after %d steps.", global_step)
        return history

    # ── Loss computation ───────────────────────────────────────────────

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the multi-task loss.

        Args:
            outputs: Model outputs dict with ``lm_logits``, ``tool_logits``,
                ``confidence``.
            batch: Data batch dict with ``labels``, ``tool_labels``,
                ``confidence_target``.

        Returns:
            Dictionary with keys ``lm``, ``tool``, ``confidence``, and
            ``total`` (the weighted sum).
        """
        cfg = self.config

        # LM loss: flatten (B, T, V) → (B*T, V) vs (B*T,)
        lm_logits = outputs.get("lm_logits", outputs.get("logits"))
        B, T, V = lm_logits.shape
        lm_loss = self.lm_criterion(
            lm_logits.reshape(B * T, V),
            batch["labels"].reshape(B * T),
        )

        # Tool-router loss: multi-label binary classification
        tool_loss = self.tool_criterion(
            outputs["tool_logits"],
            batch["tool_labels"].float(),
        )

        # Confidence loss: regression
        conf_target = batch.get("confidence_target", batch.get("confidence"))
        if conf_target.dim() == 1:
            conf_target = conf_target.unsqueeze(-1)
        confidence_loss = self.confidence_criterion(
            outputs["confidence"],
            conf_target.float(),
        )

        total = (
            cfg.lm_loss_weight * lm_loss
            + cfg.tool_loss_weight * tool_loss
            + cfg.confidence_loss_weight * confidence_loss
        )

        return {
            "lm": lm_loss,
            "tool": tool_loss,
            "confidence": confidence_loss,
            "total": total,
        }

    # ── Validation loop ────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Run one full validation pass.

        Args:
            model: The model to evaluate (set to ``eval()`` internally).
            val_loader: Validation data loader.
            device: Device to run evaluation on.

        Returns:
            Summarised validation metrics dictionary.
        """
        model.eval()
        metrics = _Metrics()

        for batch in val_loader:
            batch = self._to_device(batch, device)
            batch_size = batch["input_ids"].size(0)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=self.config.mixed_precision,
            ):
                outputs = model(batch["input_ids"])
                losses = self._compute_losses(outputs, batch)

            metrics.update(
                total=losses["total"].item(),
                lm=losses["lm"].item(),
                tool=losses["tool"].item(),
                conf=losses["confidence"].item(),
                tool_preds=outputs["tool_logits"],
                tool_targets=batch["tool_labels"],
                batch_size=batch_size,
            )

        model.train()
        return metrics.summarise()

    # ── Optimizer factory ──────────────────────────────────────────────

    def _build_optimizer(
        self, model: nn.Module, lr: float
    ) -> torch.optim.AdamW:
        """Build AdamW with weight-decay exclusion for norms and biases.

        Normalization parameters (RMSNorm weights) and biases are
        excluded from weight decay following standard best practice.

        Args:
            model: The model whose parameters to optimise.
            lr: Peak learning rate.

        Returns:
            Configured ``torch.optim.AdamW`` instance.
        """
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # ── Step estimation ────────────────────────────────────────────────

    def _estimate_total_steps(
        self, train_loader: DataLoader, epochs: int
    ) -> int:
        """Estimate total optimiser steps for LR scheduling.

        Args:
            train_loader: Training data loader (used for ``len()``).
            epochs: Number of training epochs.

        Returns:
            Estimated number of optimiser steps.
        """
        batches_per_epoch = len(train_loader)
        steps_per_epoch = batches_per_epoch // self.config.gradient_accumulation_steps
        return steps_per_epoch * epochs

    # ── Checkpoint management ──────────────────────────────────────────

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
        global_step: int,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """Save a training checkpoint.

        The checkpoint contains the model state dict, optimizer state,
        scheduler state, global step, epoch, and validation metrics.

        Args:
            model: Model to checkpoint.
            optimizer: Current optimizer state.
            scheduler: Current LR scheduler state.
            global_step: Current global optimiser step.
            epoch: Current epoch number.
            metrics: Latest validation metrics.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_step_{global_step}.pt"

        checkpoint = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, ckpt_path)
        logger.info("Checkpoint saved to %s", ckpt_path)

        # Also save a "latest" symlink / copy for easy resuming
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        return ckpt_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Resume training from a saved checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to restore state into.
            scheduler: Scheduler to restore state into.
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            device: Device to map tensors to (default: CPU).

        Returns:
            The full checkpoint dictionary (includes ``global_step``,
            ``epoch``, ``metrics``, etc.).
        """
        device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            "Resumed from %s (step=%d, epoch=%d)",
            checkpoint_path,
            checkpoint["global_step"],
            checkpoint["epoch"],
        )
        return checkpoint

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_device(
        batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Move every tensor in *batch* to *device*.

        Args:
            batch: Dictionary mapping string keys to tensors.
            device: Target device.

        Returns:
            New dictionary with all tensors on *device*.
        """
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
