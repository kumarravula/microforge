"""
Cosine-Warmup and OneCycleLR Learning-Rate Schedulers for MicroForge.

Implements:
    - CosineWarmupScheduler: linear warmup followed by cosine annealing.
    - OneCycleLRWrapper: convenience wrapper around ``torch.optim.lr_scheduler.OneCycleLR``.

Both schedulers conform to PyTorch's ``_LRScheduler`` interface so they
work seamlessly with ``optimizer.step()`` / ``scheduler.step()`` loops.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR


# ─── Cosine warmup scheduler ──────────────────────────────────────────


class CosineWarmupScheduler(_LRScheduler):
    """Linear warmup + cosine annealing learning-rate scheduler.

    During the first ``warmup_steps`` optimiser steps the learning rate
    increases linearly from ``0`` to the base LR set by the optimiser.
    After warmup the LR decays following a cosine curve down to
    ``min_lr``.

    Compatible with PyTorch's ``_LRScheduler`` interface: call
    ``scheduler.step()`` once per optimiser step and use
    ``scheduler.get_last_lr()`` to inspect the current rate.

    Args:
        optimizer: Wrapped optimiser.
        warmup_steps: Number of linear-warmup steps.
        total_steps: Total training steps (warmup + cosine decay).
        min_lr: Floor learning rate at the end of cosine decay.
        last_epoch: Index of the last epoch (for resuming). ``-1`` means
            fresh start.

    Raises:
        ValueError: If ``warmup_steps >= total_steps``.

    Example::

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=1000, total_steps=50000, min_lr=1e-6,
        )
        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        # Must be set before super().__init__ which calls get_lr()
        super().__init__(optimizer, last_epoch=last_epoch)

    # ── Core schedule logic ────────────────────────────────────────────

    def get_lr(self) -> List[float]:  # noqa: D401 – required override name
        """Compute the LR for each parameter group at the current step.

        Returns:
            List of learning rates, one per param group.
        """
        step = self.last_epoch  # _LRScheduler increments before calling
        return [self._compute_lr(base_lr, step) for base_lr in self.base_lrs]

    def _compute_lr(self, base_lr: float, step: int) -> float:
        """Return the learning rate for *step* given *base_lr*.

        Args:
            base_lr: The initial LR for this parameter group.
            step: Current global step (0-indexed).

        Returns:
            Scheduled learning rate (float).
        """
        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            return base_lr * (step / max(1, self.warmup_steps))

        # Cosine annealing: base_lr → min_lr
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, decay_steps)
        progress = min(progress, 1.0)  # clamp beyond total_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * cosine_decay

    # ── Convenience ────────────────────────────────────────────────────

    def get_lr_at_step(self, step: int) -> float:
        """Preview the LR at an arbitrary step (uses the first base LR).

        This is a convenience for logging / visualisation and does *not*
        mutate any internal state.

        Args:
            step: The global optimiser step to query.

        Returns:
            Learning rate that would be applied at *step*.
        """
        return self._compute_lr(self.base_lrs[0], step)


# ─── OneCycleLR wrapper ───────────────────────────────────────────────


class OneCycleLRWrapper:
    """Convenience wrapper around :class:`torch.optim.lr_scheduler.OneCycleLR`.

    Adds optional linear warmup integration and provides a unified
    ``step()`` / ``get_last_lr()`` API identical to
    :class:`CosineWarmupScheduler`.

    Args:
        optimizer: Wrapped optimiser.
        max_lr: Peak learning rate (reached at the end of warmup / start
            of annealing).
        total_steps: Total number of optimiser steps.
        pct_start: Fraction of ``total_steps`` spent in warmup
            (default 0.1 = 10 %).
        anneal_strategy: ``"cos"`` (default) or ``"linear"``.
        div_factor: ``max_lr / div_factor`` gives the initial LR.
        final_div_factor: ``max_lr / final_div_factor`` gives the final LR.

    Example::

        scheduler = OneCycleLRWrapper(
            optimizer, max_lr=3e-4, total_steps=50000
        )
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.1,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ) -> None:
        self.optimizer = optimizer
        self._scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    # ── Public API (mirrors _LRScheduler) ──────────────────────────────

    def step(self) -> None:
        """Advance the scheduler by one optimiser step."""
        self._scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Return the last computed learning rates.

        Returns:
            List of learning rates, one per parameter group.
        """
        return self._scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """Return the scheduler state for checkpointing."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from a checkpoint.

        Args:
            state_dict: State dictionary produced by :meth:`state_dict`.
        """
        self._scheduler.load_state_dict(state_dict)
