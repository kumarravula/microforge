"""
Grouped Query Attention with Rotary Position Embeddings for MicroForge.

Implements GQA (Ainslie et al., 2023) where fewer KV heads are shared
across multiple query heads, reducing KV-cache memory.  Rotary Position
Embeddings (Su et al., 2021) are applied to queries and keys for
length-extrapolatable positional information.

All linear projections use :class:`BitLinear` for 1.58-bit QAT.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear
from .config import ModelConfig


# ── RoPE utilities ─────────────────────────────────────────────────────


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cosine and sine frequency tables for RoPE.

    The frequencies are computed as:

        ``theta_i = base^{-2i / dim}``  for  ``i in [0, dim/2)``

    and then outer-producted with position indices ``[0, max_seq_len)``.

    Args:
        dim: Per-head dimension (must be even).
        max_seq_len: Maximum sequence length to pre-compute for.
        base: Base frequency (default 10 000).
        device: Target device for the tensors.

    Returns:
        Tuple ``(freqs_cos, freqs_sin)`` each of shape
        ``(max_seq_len, dim // 2)``.
    """
    assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
    # theta_i = 1 / (base^(2i/dim))  shape: (dim//2,)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # positions: (max_seq_len,)
    t = torch.arange(max_seq_len, device=device).float()
    # outer product → (max_seq_len, dim//2)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embeddings to a query or key tensor.

    The input is split into pairs along the last dimension and each pair
    is rotated by the corresponding angle.

    Args:
        x: Tensor of shape ``(B, n_heads, T, head_dim)``.
        freqs_cos: Cosine table, shape ``(T, head_dim // 2)`` or broadcastable.
        freqs_sin: Sine table, same shape as ``freqs_cos``.

    Returns:
        Rotated tensor of the same shape as *x*.
    """
    # Split last dim into pairs: (..., head_dim) → (..., head_dim//2, 2)
    head_dim = x.shape[-1]
    x_pairs = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
    x_real = x_pairs[..., 0]  # (B, n_heads, T, head_dim//2)
    x_imag = x_pairs[..., 1]

    # Reshape freq tables for broadcasting: (T, head_dim//2) → (1, 1, T, head_dim//2)
    cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    # Rotation: [cos θ, -sin θ; sin θ, cos θ] applied pair-wise
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    # Interleave back: (B, n_heads, T, head_dim//2, 2) → (B, n_heads, T, head_dim)
    out = torch.stack([out_real, out_imag], dim=-1).reshape_as(x)
    return out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads.

    Given a tensor with ``n_kv_heads`` head slots, each head is duplicated
    ``n_rep`` times so the result has ``n_kv_heads * n_rep`` heads.

    Args:
        x: Tensor of shape ``(B, n_kv_heads, T, head_dim)``.
        n_rep: Number of times to repeat each KV head.

    Returns:
        Tensor of shape ``(B, n_kv_heads * n_rep, T, head_dim)``.
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    # (B, n_kv, 1, T, D) → (B, n_kv, n_rep, T, D) → (B, n_kv*n_rep, T, D)
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)


# ── Grouped Query Attention ────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention and RoPE.

    Uses fewer KV heads than query heads (default ratio 4:1) to cut
    KV-cache memory by 4×.  All linear projections are :class:`BitLinear`
    layers for 1.58-bit quantisation-aware training.

    Args:
        config: Model configuration providing head counts and dimensions.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_query_heads = config.n_query_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_query_heads // self.n_kv_heads  # GQA repeat factor

        # Projections (BitLinear — 1.58-bit QAT)
        self.q_proj = BitLinear(config.d_model, self.n_query_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = BitLinear(self.n_query_heads * self.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Pre-compute RoPE tables (registered as buffers — not parameters)
        freqs_cos, freqs_sin = precompute_rope_freqs(
            self.head_dim, config.max_seq_len, config.rope_base
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute grouped-query attention with optional KV cache.

        Args:
            x: Input tensor, shape ``(B, T, d_model)``.
            mask: Optional boolean or float attention mask of shape
                ``(1, 1, T, T)`` or broadcastable.  Positions to **mask
                out** should be ``-inf`` (float) or ``True`` (bool).
            start_pos: Position offset for RoPE when using KV cache
                during autoregressive generation.
            kv_cache: Optional tuple ``(cached_k, cached_v)`` each of shape
                ``(B, n_kv_heads, T_cached, head_dim)``.  If provided, the
                new K/V are appended to the cache before attention is
                computed.

        Returns:
            Tuple of:
            - Output tensor of shape ``(B, T, d_model)``.
            - Updated ``(cached_k, cached_v)`` tuple (or ``None`` when
              caching is not used).
        """
        B, T, _ = x.shape

        # ── Project Q, K, V ────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_q, T, D)   k,v: (B, n_kv, T, D)

        # ── Apply RoPE ─────────────────────────────────────────────────
        freqs_cos = self.freqs_cos[start_pos : start_pos + T]
        freqs_sin = self.freqs_sin[start_pos : start_pos + T]
        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        # ── KV cache ───────────────────────────────────────────────────
        new_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_cache = (k, v)

        # ── Expand KV heads to match Q heads (GQA) ────────────────────
        k = repeat_kv(k, self.n_rep)  # (B, n_q, T_kv, D)
        v = repeat_kv(v, self.n_rep)

        # ── Scaled dot-product attention ───────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_q, T, T_kv)

        # Causal mask (only for training or first token; cache handles
        # subsequent tokens automatically because T == 1).
        if mask is None and T > 1:
            T_kv = k.shape[2]
            causal = torch.triu(
                torch.full((T, T_kv), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=T_kv - T + 1,
            )
            attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)
        elif mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Weighted sum ───────────────────────────────────────────────
        out = torch.matmul(attn_weights, v)  # (B, n_q, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)

        # ── Output projection ─────────────────────────────────────────
        out = self.o_proj(out)
        out = self.resid_dropout(out)

        return out, new_cache
