# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Knowledge Distillation Trainer for MicroForge.
Allows distilling knowledge from a high-capacity teacher model (e.g. LLaMA/Mistral/DeepSeek/teacher checkpoints)
into the ultra-compact ~12 MB BitNet student model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from microforge.training.trainer import MicroForgeTrainer


class OfflineDistillationDataset(Dataset):
    """Dataset wrapper that pairs input samples with pre-computed teacher logits."""

    def __init__(self, base_dataset: Dataset, teacher_logits_path: Union[str, Path]) -> None:
        self.base_dataset = base_dataset
        self.teacher_logits = torch.load(teacher_logits_path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        item["teacher_logits"] = self.teacher_logits[idx]
        return item


class DistillationTrainer(MicroForgeTrainer):
    """Trainer with teacher-student Knowledge Distillation loss.

    Loss = alpha * Hard_Loss + (1 - alpha) * KD_Loss
    where KD_Loss = T^2 * KL_Div(softmax(student/T), softmax(teacher/T))
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 2.0,
        kd_alpha: float = 0.5,
        feature_matching: bool = False,
        feature_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        self.temperature = temperature
        self.kd_alpha = kd_alpha
        self.feature_matching = feature_matching
        self.feature_weight = feature_weight

    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Kullback-Leibler divergence with temperature scaling."""
        t = self.temperature
        p_student = F.log_softmax(student_logits / t, dim=-1)
        p_teacher = F.softmax(teacher_logits / t, dim=-1)

        kl_loss = F.kl_div(p_student, p_teacher, reduction="none").sum(dim=-1)
        if mask is not None:
            kl_loss = (kl_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            kl_loss = kl_loss.mean()

        return kl_loss * (t ** 2)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single forward-backward training step with KD."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        tool_labels = batch["tool_labels"].to(self.device)
        target_conf = batch["confidence"].to(self.device)

        # 1. Hard label forward pass
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=self.feature_matching,
            )
            student_logits = outputs["logits"]
            student_tool_logits = outputs["tool_logits"]
            student_conf = outputs["confidence"].squeeze(-1)

            # Standard multi-task loss
            # Shift labels for causal LM
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            hard_lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            tool_loss = F.binary_cross_entropy_with_logits(student_tool_logits, tool_labels)
            conf_loss = F.mse_loss(student_conf, target_conf)

            hard_loss = (
                self.lm_loss_weight * hard_lm_loss
                + self.tool_loss_weight * tool_loss
                + self.confidence_loss_weight * conf_loss
            )

            # 2. Distillation loss
            kd_loss = torch.tensor(0.0, device=self.device)
            teacher_logits = batch.get("teacher_logits")
            if teacher_logits is None and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_out = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_out["logits"]

            if teacher_logits is not None:
                teacher_logits = teacher_logits.to(self.device)
                shift_t_logits = teacher_logits[:, :-1, :].contiguous()
                token_mask = (shift_labels != -100).float()
                kd_loss = self.compute_kd_loss(shift_logits, shift_t_logits, token_mask)

            total_loss = self.kd_alpha * hard_loss + (1.0 - self.kd_alpha) * kd_loss

        return {
            "total_loss": total_loss,
            "lm_loss": hard_lm_loss,
            "tool_loss": tool_loss,
            "conf_loss": conf_loss,
            "kd_loss": kd_loss,
        }
