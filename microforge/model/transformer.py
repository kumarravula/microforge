"""
Transformer Block for MicroForge.

Implements a single pre-norm transformer block with:

- RMSNorm → Grouped Query Attention → residual
- RMSNorm → SwiGLU FFN → residual

All linear layers inside the attention and FFN use :class:`BitLinear` for
1.58-bit quantisation-aware training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention
from .bitlinear import BitLinear
from .config import ModelConfig
from .embedding import RMSNorm


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    The SwiGLU variant uses a gated linear unit with SiLU (swish)
    activation:

        ``output = down(SiLU(gate(x)) ⊙ up(x))``

    All projections are :class:`BitLinear` for ternary QAT.

    Args:
        d_model: Input and output feature dimension.
        d_ff: Intermediate (hidden) feature dimension.
        dropout: Dropout probability applied after the down projection.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = BitLinear(d_model, d_ff, bias=False)
        self.up = BitLinear(d_model, d_ff, bias=False)
        self.down = BitLinear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        # Gate path with SiLU activation, element-wise product with up path
        swish_gate = F.silu(self.gate(x))
        out = swish_gate * self.up(x)
        out = self.down(out)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with GQA and SwiGLU FFN.

    Architecture::

        residual ──► RMSNorm ──► GQA ──► + ──► RMSNorm ──► SwiGLU ──► + ──►
                 └────────────────────────┘ └──────────────────────────┘

    Args:
        config: Model configuration.
        layer_idx: Index of this block in the stack (for debugging /
            gradient checkpointing bookkeeping).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # Sub-layers
        self.attention = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through one transformer block.

        Args:
            x: Hidden states, shape ``(B, T, d_model)``.
            mask: Optional attention mask (see
                :meth:`GroupedQueryAttention.forward`).
            start_pos: Position offset for RoPE (autoregressive cache).
            kv_cache: Optional KV cache tuple for this layer.

        Returns:
            Tuple of:
            - Updated hidden states ``(B, T, d_model)``.
            - New KV cache tuple for this layer (or ``None``).
        """
        # ── Self-attention with residual ───────────────────────────────
        residual = x
        x = self.attn_norm(x)
        attn_out, new_cache = self.attention(x, mask=mask, start_pos=start_pos, kv_cache=kv_cache)
        x = residual + attn_out

        # ── FFN with residual ──────────────────────────────────────────
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, new_cache
