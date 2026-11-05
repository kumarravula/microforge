"""
MicroForge Model Configuration.

Defines the ModelConfig dataclass holding all hyperparameters for the
MicroForge edge-AI model, including factory methods for standard sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the MicroForge transformer model.

    This dataclass stores every hyperparameter needed to instantiate a
    MicroForge model.  Three factory class-methods (``tiny``, ``small``,
    ``medium``) provide pre-set configurations for common model sizes.

    Attributes:
        d_model: Hidden dimension of the transformer.
        n_layers: Number of transformer blocks.
        n_query_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads (GQA).
        d_ff: Intermediate dimension of the SwiGLU FFN.
        vocab_size: Vocabulary size for the tokenizer.
        max_seq_len: Maximum sequence length (context window).
        embedding_factor_dim: Low-rank embedding factorization dimension
            (ALBERT-style V×E projection before E×H).
        head_dim: Per-head dimension, derived as ``d_model // n_query_heads``.
        dropout: Dropout probability (0.0 for inference-ready configs).
        rope_base: Base frequency for Rotary Position Embeddings.
        num_tool_classes: Maximum number of tool classes the router can
            classify (i.e. max tools in a schema).
        tie_embeddings: Whether to tie the LM-head weights to the input
            embedding projection weights.
    """

    # ── core transformer dims ──────────────────────────────────────────
    d_model: int = 512
    n_layers: int = 12
    n_query_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1536  # SwiGLU intermediate size

    # ── vocabulary & sequence ──────────────────────────────────────────
    vocab_size: int = 4096
    max_seq_len: int = 1024

    # ── factorised embedding ───────────────────────────────────────────
    embedding_factor_dim: int = 128

    # ── derived / attention ────────────────────────────────────────────
    head_dim: int = 64  # d_model // n_query_heads

    # ── regularisation ─────────────────────────────────────────────────
    dropout: float = 0.0

    # ── positional encoding ────────────────────────────────────────────
    rope_base: float = 10_000.0

    # ── task heads ─────────────────────────────────────────────────────
    num_tool_classes: int = 64
    tie_embeddings: bool = True

    # ── post-init validation ───────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.d_model % self.n_query_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_query_heads ({self.n_query_heads})"
        )
        assert self.n_query_heads % self.n_kv_heads == 0, (
            f"n_query_heads ({self.n_query_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )
        # Recompute head_dim in case the caller only changed d_model.
        object.__setattr__(self, "head_dim", self.d_model // self.n_query_heads)

    # ── factory methods ────────────────────────────────────────────────
    @classmethod
    def tiny(cls, **overrides) -> "ModelConfig":
        """6-layer model (~3-4 MB) for prototyping and unit tests."""
        defaults = dict(
            d_model=512,
            n_layers=6,
            n_query_heads=8,
            n_kv_heads=2,
            d_ff=1536,
            vocab_size=4096,
            max_seq_len=1024,
            embedding_factor_dim=128,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def small(cls, **overrides) -> "ModelConfig":
        """12-layer model (~7-8 MB) — default production size."""
        defaults = dict(
            d_model=512,
            n_layers=12,
            n_query_heads=8,
            n_kv_heads=2,
            d_ff=1536,
            vocab_size=4096,
            max_seq_len=1024,
            embedding_factor_dim=128,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def medium(cls, **overrides) -> "ModelConfig":
        """16-layer model (~10-11 MB) for higher-quality extraction."""
        defaults = dict(
            d_model=512,
            n_layers=16,
            n_query_heads=8,
            n_kv_heads=2,
            d_ff=1536,
            vocab_size=4096,
            max_seq_len=1024,
            embedding_factor_dim=128,
        )
        defaults.update(overrides)
        return cls(**defaults)
