"""
MicroForge — Ultra-lightweight edge-AI model for tool calling and
structured extraction.

Sub-15 MB transformer with BitNet b1.58 ternary quantisation, Grouped
Query Attention, RoPE, SwiGLU, and factorised embeddings.
"""

from .model import MicroForgeModel, ModelConfig

__all__ = ["MicroForgeModel", "ModelConfig"]
__version__ = "0.1.0"
