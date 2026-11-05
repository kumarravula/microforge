"""
MicroForge — Complete Model.

Assembles the full MicroForge transformer with:

- Factorised embeddings (ALBERT-style)
- A stack of pre-norm TransformerBlocks (BitLinear + GQA + SwiGLU)
- Final RMSNorm
- Three task heads:
    1. **lm_head** — language-model logits over the vocabulary
    2. **tool_router** — multi-label classification of which tool(s) to invoke
    3. **confidence_head** — scalar confidence score in [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embedding import FactorizedEmbedding, RMSNorm
from .transformer import TransformerBlock


class MicroForgeModel(nn.Module):
    """Ultra-lightweight edge-AI transformer for tool calling and
    structured extraction.

    The model is designed to fit under 15 MB when stored with ternary
    (1.58-bit) weights.  It produces three outputs in every forward pass:

    - ``logits``: next-token log-probabilities over the vocabulary.
    - ``tool_logits``: per-tool classification scores for the router.
    - ``confidence``: a scalar ∈ [0, 1] estimating output reliability.

    Args:
        config: A :class:`ModelConfig` instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token embedding (factorised) ───────────────────────────────
        self.embedding = FactorizedEmbedding(config)

        # ── Transformer stack ──────────────────────────────────────────
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # ── Final norm ─────────────────────────────────────────────────
        self.final_norm = RMSNorm(config.d_model)

        # ── Head 1: Language-model head ────────────────────────────────
        if config.tie_embeddings:
            # Tied head — shares weights with the embedding projection.
            # We create a linear layer and will tie its weight in _tie().
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ── Head 2: Tool router ────────────────────────────────────────
        self.tool_router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.num_tool_classes),
        )

        # ── Head 3: Confidence head ───────────────────────────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Dropout ───────────────────────────────────────────────────
        self.drop = nn.Dropout(config.dropout)

        # ── Weight tying ──────────────────────────────────────────────
        if config.tie_embeddings:
            self._tie_lm_head()

        # ── Initialise weights ────────────────────────────────────────
        self.apply(self._init_weights)

    # ── Weight initialisation ──────────────────────────────────────────

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Xavier-uniform for linear layers, normal for embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tie_lm_head(self) -> None:
        """Tie the LM head weight to the embedding projection matrix.

        Because the embedding is factorised (V × E → E × H), we tie
        the LM head (H → V) to a composed view.  In practice we create
        a new parameter that projects ``d_model → vocab_size`` using the
        transpose of ``projection @ token_embedding`` conceptually.

        For simplicity and memory savings, we tie to the factorised
        projection's output weight.  The ``lm_head`` weight is set to a
        view of ``embedding.projection.weight.T @ embedding.token_embedding.weight.T``
        — but because that product would be dense, the standard approach
        is to share the *projection* weight directly when
        ``embedding_factor_dim == d_model``, or build a thin adapter.

        Here we implement the practical approach: the lm_head weight
        matrix is constructed on-the-fly in forward() when tying is
        enabled, so we delete the redundant parameter.
        """
        # Remove the standalone lm_head weight; we'll compute logits
        # via the embedding matrices directly in forward().
        # We keep the nn.Linear shell for interface consistency.
        del self.lm_head
        self.lm_head = None  # sentinel — handled in forward()

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through MicroForge.

        Args:
            input_ids: Token IDs, shape ``(B, T)``.
            mask: Optional attention mask.
            start_pos: Position offset for cached autoregressive decoding.
            kv_caches: Optional list of per-layer KV cache tuples.

        Returns:
            Dictionary with keys:

            - ``logits``: ``(B, T, vocab_size)``
            - ``tool_logits``: ``(B, num_tool_classes)``
            - ``confidence``: ``(B, 1)``
            - ``hidden_states``: ``(B, T, d_model)`` — final hidden states
        """
        B, T = input_ids.shape

        # ── Embedding ─────────────────────────────────────────────────
        h = self.embedding(input_ids)  # (B, T, d_model)
        h = self.drop(h)

        # ── Transformer layers ────────────────────────────────────────
        new_kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, mask=mask, start_pos=start_pos, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        # ── Final norm ────────────────────────────────────────────────
        h = self.final_norm(h)  # (B, T, d_model)

        # ── LM head (logits) ─────────────────────────────────────────
        if self.lm_head is not None:
            logits = self.lm_head(h)  # (B, T, V)
        else:
            # Tied embeddings: project back through the factorised
            # embedding matrices.  h @ projection.weight → (B,T,E)
            # then (B,T,E) @ token_embedding.weight.T → (B,T,V)
            h_proj = F.linear(h, self.embedding.projection.weight.t())  # (B, T, E)
            logits = F.linear(h_proj, self.embedding.token_embedding.weight)  # (B, T, V)

        # ── Tool router (pool over sequence, then classify) ───────────
        # Use the last non-padding token as the sequence representation.
        pooled = h[:, -1, :]  # (B, d_model)
        tool_logits = self.tool_router(pooled)  # (B, num_tool_classes)

        # ── Confidence head ───────────────────────────────────────────
        confidence = self.confidence_head(pooled)  # (B, 1)

        return {
            "logits": logits,
            "tool_logits": tool_logits,
            "confidence": confidence,
            "hidden_states": h,
            "kv_caches": new_kv_caches,
        }

    # ── Autoregressive generation ──────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        grammar_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive token generation with optional grammar mask.

        Args:
            input_ids: Prompt token IDs, shape ``(B, T_prompt)``.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = no change).
            top_k: Top-k filtering (0 = disabled).
            grammar_mask_fn: Optional callable
                ``(input_ids, logits) -> masked_logits`` that applies a
                grammar constraint mask (e.g. for guaranteed valid JSON).
            eos_token_id: If provided, stop generation when this token is
                produced by all sequences in the batch.

        Returns:
            Full sequence tensor ``(B, T_prompt + generated)`` including
            the prompt.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        # First pass: process the full prompt (no cache yet)
        generated = input_ids
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        for step in range(max_new_tokens):
            if kv_caches is None:
                # First step — process full prompt
                outputs = self.forward(generated, start_pos=0, kv_caches=None)
                kv_caches = outputs["kv_caches"]
                logits = outputs["logits"][:, -1, :]  # (B, V) last position
            else:
                # Subsequent steps — only feed the last token
                last_tok = generated[:, -1:]  # (B, 1)
                start_pos = generated.shape[1] - 1
                outputs = self.forward(last_tok, start_pos=start_pos, kv_caches=kv_caches)
                kv_caches = outputs["kv_caches"]
                logits = outputs["logits"][:, -1, :]  # (B, V)

            # ── Temperature scaling ────────────────────────────────────
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                continue

            # ── Grammar mask ───────────────────────────────────────────
            if grammar_mask_fn is not None:
                logits = grammar_mask_fn(generated, logits)

            # ── Top-k filtering ────────────────────────────────────────
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_vals = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth_vals, float("-inf"))

            # ── Sample ─────────────────────────────────────────────────
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # ── EOS check ─────────────────────────────────────────────
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ── Utility methods ────────────────────────────────────────────────

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of (trainable) parameters.

        Args:
            trainable_only: If ``True``, count only parameters with
                ``requires_grad=True``.

        Returns:
            Integer parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimate_size_mb(self, bits_per_param: float = 1.58) -> float:
        """Estimate model size in megabytes at the given bit-width.

        Args:
            bits_per_param: Bits per parameter (1.58 for ternary, 16 for
                fp16, 32 for fp32).

        Returns:
            Estimated size in MB.
        """
        n_params = self.count_parameters(trainable_only=False)
        return (n_params * bits_per_param) / (8 * 1024 * 1024)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "MicroForgeModel":
        """Instantiate a model from a configuration object.

        Args:
            config: A :class:`ModelConfig` instance.

        Returns:
            An initialised :class:`MicroForgeModel`.
        """
        return cls(config)
