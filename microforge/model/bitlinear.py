"""
BitLinear — 1.58-bit quantisation-aware linear layer for MicroForge.

Implements the BitNet b1.58 scheme where, during the forward pass, weights
are quantised to ternary values {-1, 0, +1} and activations are quantised
to 8-bit via per-token absmax scaling.  Gradients flow through both
quantisation steps via the Straight-Through Estimator (STE).

Reference: Ma et al., "The Era of 1-bit LLMs" (2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Lightweight RMSNorm used internally before weight quantisation.

    This is a local copy so that ``bitlinear.py`` is self-contained; the
    canonical ``RMSNorm`` used elsewhere in the model lives in
    ``embedding.py``.

    Args:
        dim: Feature dimension to normalise over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


# ── Quantisation helpers (STE-compatible) ──────────────────────────────


def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Quantise weights to ternary {-1, 0, +1} using round-clip.

    The procedure follows BitNet b1.58:

    1. Compute ``gamma = mean(|w|)`` as the scaling factor.
    2. Normalise: ``w_hat = w / (gamma + eps)``.
    3. Round to nearest integer and clamp to [-1, 1].

    The ``detach`` trick implements STE so that gradients flow through as
    if quantisation were the identity function.

    Args:
        w: Full-precision weight tensor of arbitrary shape.
        eps: Small constant to avoid division by zero.

    Returns:
        Ternary weight tensor (same shape), with gradients attached via STE.
    """
    gamma = w.abs().mean()
    w_scaled = w / (gamma + eps)
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)
    # STE: forward uses quantised value, backward uses original gradient
    return (w_ternary - w_scaled).detach() + w_scaled


def activation_quant(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Per-token absmax quantisation of activations.

    Scales each token's activation vector so that the absolute maximum
    maps to the representable range ``[-Q_b, Q_b]`` where
    ``Q_b = 2^{num_bits - 1} - 1``.  After "quantising" (round) the values
    are immediately de-quantised so the tensor stays in float for the
    subsequent matmul.

    Args:
        x: Activation tensor, typically shape ``(B, T, D)`` or ``(B*T, D)``.
        num_bits: Bit-width for simulated quantisation (default 8).

    Returns:
        Simulated-quantised activation tensor (same shape, float dtype).
    """
    Q_b = 2 ** (num_bits - 1) - 1  # 127 for 8-bit
    # Per-token (last-dim) absmax
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_scaled = x * (Q_b / scale)
    x_quant = torch.clamp(torch.round(x_scaled), -Q_b, Q_b)
    # De-quantise back to float
    x_dequant = x_quant * (scale / Q_b)
    # STE
    return (x_dequant - x).detach() + x


# ── BitLinear layer ────────────────────────────────────────────────────


class BitLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with BitNet b1.58 QAT.

    During the forward pass:

    1. Activations are RMS-normalised, then absmax-quantised to 8-bit.
    2. Weights are quantised to ternary {-1, 0, +1}.
    3. The matmul is performed in float with quantised operands.
    4. Gradients flow through both quantisers via STE.

    At inference time (after ``torch.no_grad()``), the quantised weights
    can be materialised as ``int2`` for efficient deployment.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias (usually ``False`` for
            BitNet models).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weight (latent parameter that gets quantised)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMSNorm applied to activations before quantisation
        self.input_norm = RMSNorm(in_features)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming uniform initialisation (same as nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary-weight and 8-bit-activation QAT.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        # 1. Normalise + quantise activations to 8-bit
        x_norm = self.input_norm(x)
        x_quant = activation_quant(x_norm)

        # 2. Quantise weights to ternary {-1, 0, +1}
        w_quant = weight_quant(self.weight)

        # 3. Linear transform with quantised operands
        return F.linear(x_quant, w_quant, self.bias)
