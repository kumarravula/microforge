"""
Factorised Embedding and RMSNorm layers for MicroForge.

Implements ALBERT-style factorised embeddings (V×E → E×H) to reduce
parameter count when the vocabulary is large relative to the hidden size,
and the RMSNorm normalisation layer used throughout the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-centre the activations (no bias
    or mean subtraction).  This makes it cheaper and empirically equivalent
    for transformer pre-norm architectures.

    Args:
        dim: Feature dimension to normalise over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)  →  RMS computed over last axis
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalised tensor of the same shape, scaled by the learnable
            ``weight`` parameter.
        """
        # Float32 for numerical stability, then cast back.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FactorizedEmbedding(nn.Module):
    """ALBERT-style factorised embedding table.

    Instead of a single ``(vocab_size, d_model)`` embedding matrix this
    layer decomposes the lookup into two steps:

    1. ``nn.Embedding(vocab_size, embedding_factor_dim)``  — small lookup
    2. ``nn.Linear(embedding_factor_dim, d_model, bias=False)`` — up-projection

    This reduces the embedding parameter count from ``V × H`` to
    ``V × E + E × H`` where ``E ≪ H``.

    Args:
        config: A :class:`ModelConfig` instance providing ``vocab_size``,
            ``embedding_factor_dim``, and ``d_model``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Step 1: low-rank token lookup (V × E)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_factor_dim)

        # Step 2: project up to hidden size (E × H)
        self.projection = nn.Linear(config.embedding_factor_dim, config.d_model, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings and project to model dimension.

        Args:
            input_ids: Integer tensor of token IDs, shape ``(B, T)``.

        Returns:
            Embedding tensor of shape ``(B, T, d_model)``.
        """
        x = self.token_embedding(input_ids)  # (B, T, E)
        x = self.projection(x)               # (B, T, H)
        return x
