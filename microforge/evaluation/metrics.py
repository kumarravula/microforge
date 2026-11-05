# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive evaluation metrics for MicroForge and edge AI benchmarks.
Matches standard benchmarks: DroidCall, BFCL, Mobile Actions, and Micro-F1 extraction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class ExactMatchAccuracy:
    """Exact Match Accuracy for tool calling.
    
    A prediction is correct if and only if:
    1. The number of predicted calls equals ground truth calls.
    2. Each tool name matches the corresponding ground truth tool name.
    3. All argument names and values match.
    """

    def __init__(self) -> None:
        self.total = 0
        self.exact_matches = 0
        self.tool_name_matches = 0

    def update(self, preds: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> bool:
        self.total += 1
        if len(preds) != len(targets):
            return False

        name_match = True
        exact_match = True

        for p, t in zip(preds, targets):
            p_name = p.get("name", "").strip()
            t_name = t.get("name", "").strip()
            if p_name != t_name:
                name_match = False
                exact_match = False
                break

            p_args = p.get("arguments", {})
            t_args = t.get("arguments", {})
            if not self._args_match(p_args, t_args):
                exact_match = False

        if name_match:
            self.tool_name_matches += 1
        if exact_match:
            self.exact_matches += 1

        return exact_match

    @staticmethod
    def _args_match(p_args: Dict[str, Any], t_args: Dict[str, Any]) -> bool:
        if set(p_args.keys()) != set(t_args.keys()):
            return False
        for k, v in t_args.items():
            pv = p_args[k]
            if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
                if not math.isclose(float(v), float(pv), rel_tol=1e-3, abs_tol=1e-3):
                    return False
            elif str(v).strip().lower() != str(pv).strip().lower():
                return False
        return True

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {"exact_match_accuracy": 0.0, "tool_name_accuracy": 0.0}
        return {
            "exact_match_accuracy": round(self.exact_matches / self.total, 4),
            "tool_name_accuracy": round(self.tool_name_matches / self.total, 4),
            "total_samples": self.total,
        }


class ExtractionMicroF1:
    """Field-level Micro-F1 metric for structured data extraction."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred_dict: Dict[str, Any], target_dict: Dict[str, Any]) -> None:
        pred_fields = set(f"{k}:{v}".lower() for k, v in pred_dict.items())
        target_fields = set(f"{k}:{v}".lower() for k, v in target_dict.items())

        self.tp += len(pred_fields & target_fields)
        self.fp += len(pred_fields - target_fields)
        self.fn += len(target_fields - pred_fields)

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "micro_precision": round(precision, 4),
            "micro_recall": round(recall, 4),
            "micro_f1": round(f1, 4),
        }


class ConfidenceCalibration:
    """Expected Calibration Error (ECE) for calibrated confidence scores."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.confidences: List[float] = []
        self.accuracies: List[float] = []

    def update(self, confidence: float, is_correct: bool) -> None:
        self.confidences.append(float(confidence))
        self.accuracies.append(1.0 if is_correct else 0.0)

    def compute(self) -> float:
        if not self.confidences:
            return 0.0
        ece = 0.0
        n = len(self.confidences)
        for i in range(self.n_bins):
            bin_lower = i / self.n_bins
            bin_upper = (i + 1) / self.n_bins
            indices = [
                j for j, c in enumerate(self.confidences)
                if bin_lower <= c < bin_upper or (i == self.n_bins - 1 and c == 1.0)
            ]
            if indices:
                bin_acc = sum(self.accuracies[j] for j in indices) / len(indices)
                bin_conf = sum(self.confidences[j] for j in indices) / len(indices)
                ece += (len(indices) / n) * abs(bin_acc - bin_conf)
        return round(ece, 4)
