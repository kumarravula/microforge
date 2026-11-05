"""
MicroForge Model sub-package.

Exports the core building blocks and the complete model class.
"""

from .attention import GroupedQueryAttention, apply_rope, precompute_rope_freqs, repeat_kv
from .bitlinear import BitLinear, activation_quant, weight_quant
from .config import ModelConfig
from .embedding import FactorizedEmbedding, RMSNorm
from .model import MicroForgeModel
from .transformer import SwiGLUFFN, TransformerBlock

__all__ = [
    "ModelConfig",
    "MicroForgeModel",
    "BitLinear",
    "activation_quant",
    "weight_quant",
    "GroupedQueryAttention",
    "apply_rope",
    "precompute_rope_freqs",
    "repeat_kv",
    "FactorizedEmbedding",
    "RMSNorm",
    "SwiGLUFFN",
    "TransformerBlock",
]
