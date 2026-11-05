# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
MicroForge: Ultra-lightweight edge-AI foundation model for tool calling and structured extraction.
Sub-8 MB transformer with BitNet b1.58 ternary quantization, GQA, RoPE, and factorized embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .inference.engine import MicroForgeEngine
from .model import MicroForgeModel, ModelConfig
from .tokenizer.builder import ToolTokenizer


class MicroForgeResponse:
    """Structured response container from MicroForge inference."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.raw = data
        self.query: str = data.get("query", "")
        self.calls: List[Dict[str, Any]] = data.get("function_calls", [])
        self.confidence: float = data.get("confidence", 0.0)
        self.latency_ms: float = data.get("latency_ms", 0.0)
        self.tokens_generated: int = data.get("tokens_generated", 0)

    @property
    def has_tool_call(self) -> bool:
        return len(self.calls) > 0

    @property
    def first_call(self) -> Optional[Dict[str, Any]]:
        return self.calls[0] if self.calls else None

    def __repr__(self) -> str:
        return f"MicroForgeResponse(calls={self.calls}, confidence={self.confidence:.2f}, latency={self.latency_ms:.1f}ms)"


class MicroForge:
    """High-level client for running MicroForge on-device."""

    def __init__(
        self,
        engine: MicroForgeEngine,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.engine = engine
        self.tools = tools or []

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "models/microforge.bin",
        tokenizer_path: Union[str, Path] = "models/tokenizer.json",
        schemas_path: Optional[Union[str, Path]] = "tools_schemas/droidcall_tools.json",
        device: Optional[str] = None,
    ) -> "MicroForge":
        """Load MicroForge model binary and return a ready-to-use agent."""
        t_path = Path(tokenizer_path)
        m_path = Path(model_path)

        tools: List[Dict[str, Any]] = []
        if schemas_path and Path(schemas_path).exists():
            with open(schemas_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tools = data.get("tools", data if isinstance(data, list) else [])

        if t_path.exists():
            tokenizer = ToolTokenizer.load(t_path)
        else:
            tokenizer = ToolTokenizer(vocab_size=4096)
            if tools:
                tokenizer.build_from_tool_schemas(tools)

        engine = MicroForgeEngine(model_path_or_model=str(m_path), tokenizer=tokenizer, device=device)
        return cls(engine=engine, tools=tools)

    def call(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_grammar: bool = True,
        max_tokens: int = 16,
    ) -> MicroForgeResponse:
        """Execute a tool-calling query and return a structured response."""
        active_tools = tools if tools is not None else self.tools
        raw_res = self.engine.run(
            query=query,
            tools=active_tools,
            use_grammar=use_grammar,
            max_tokens=max_tokens,
        )
        return MicroForgeResponse(raw_res)


__all__ = [
    "MicroForge",
    "MicroForgeResponse",
    "MicroForgeModel",
    "ModelConfig",
    "MicroForgeEngine",
    "ToolTokenizer",
]
__version__ = "0.1.0"
