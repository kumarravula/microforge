# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
High-performance inference engine for MicroForge.
Provides synchronous and batch inference, tool routing, argument parsing,
and grammar-constrained decoding.
"""

from __future__ import annotations

import json
import io
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from microforge.data.schema_compiler import CompiledSchema, SchemaCompiler
from microforge.inference.grammar import ToolGrammarMasker
from microforge.model.config import ModelConfig
from microforge.model.model import MicroForgeModel
from microforge.tokenizer.builder import (
    ToolTokenizer,
    BOS_ID,
    EOS_ID,
    SEP_ID,
    TOOL_START_ID,
    TOOL_END_ID,
    ARG_START_ID,
    ARG_END_ID,
    CONF_ID,
    EMPTY_ID,
)


def unpack_binary_weights(path: Union[str, Path]) -> Tuple[ModelConfig, Dict[str, torch.Tensor]]:
    """Unpack 1.58-bit binary packed MicroForge model (.bin format)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"MF158B01":
            raise ValueError(f"Invalid MicroForge binary header: {magic}")
        c_len = struct.unpack("<I", f.read(4))[0]
        cfg_dict = json.loads(f.read(c_len).decode("utf-8"))
        num_tensors = struct.unpack("<I", f.read(4))[0]
        state_dict: Dict[str, torch.Tensor] = {}
        for _ in range(num_tensors):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            flag = f.read(1)
            payload_len = struct.unpack("<I", f.read(4))[0]
            payload = f.read(payload_len)
            bio = io.BytesIO(payload)
            if flag == b"\x01":
                gamma, shape_len = struct.unpack("<fI", bio.read(8))
                shape = struct.unpack(f"<{shape_len}I", bio.read(shape_len * 4))
                packed_bytes = np.frombuffer(bio.read(), dtype=np.uint8)
                b0 = packed_bytes % 3
                rem1 = packed_bytes // 3
                b1 = rem1 % 3
                rem2 = rem1 // 3
                b2 = rem2 % 3
                rem3 = rem2 // 3
                b3 = rem3 % 3
                b4 = rem3 // 3
                unpacked = np.stack([b0, b1, b2, b3, b4], axis=1).flatten()
                numel = int(np.prod(shape))
                unpacked = unpacked[:numel].astype(np.float32) - 1.0
                unpacked = unpacked * gamma
                tensor = torch.from_numpy(unpacked.reshape(shape))
            else:
                shape_len = struct.unpack("<I", bio.read(4))[0]
                shape = struct.unpack(f"<{shape_len}I", bio.read(shape_len * 4))
                fp16_data = np.frombuffer(bio.read(), dtype=np.float16)
                tensor = torch.from_numpy(fp16_data.copy().astype(np.float32).reshape(shape))
            state_dict[name] = tensor
    return ModelConfig(**cfg_dict), state_dict


class MicroForgeEngine:
    """Production runtime engine for MicroForge sub-15MB models."""

    def __init__(
        self,
        model_path_or_model: Union[str, Path, MicroForgeModel],
        tokenizer: ToolTokenizer,
        config: Optional[ModelConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer

        if isinstance(model_path_or_model, (str, Path)):
            p = Path(model_path_or_model)
            with open(p, "rb") as check_f:
                magic = check_f.read(8)

            if magic == b"MF158B01":
                self.config, state_dict = unpack_binary_weights(p)
                self.model = MicroForgeModel(self.config)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                checkpoint = torch.load(p, map_location="cpu", weights_only=False)
                if "config" in checkpoint:
                    self.config = checkpoint["config"]
                elif config is not None:
                    self.config = config
                else:
                    self.config = ModelConfig()

                self.model = MicroForgeModel(self.config)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = model_path_or_model
            self.config = config or getattr(self.model, "config", ModelConfig())

        self.model.to(self.device)
        self.model.eval()

    def run(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 32,
        temperature: float = 0.0,
        use_grammar: bool = True,
    ) -> Dict[str, Any]:
        """Execute a tool-calling query and return typed tool invocations and confidence."""
        start_time = time.perf_counter()

        # Compile schema if provided
        compiler = SchemaCompiler(self.tokenizer)
        compiled_schema = compiler.compile_tools(tools or [])
        masker = None
        if use_grammar and tools:
            masker = ToolGrammarMasker(
                tokenizer=self.tokenizer,
                compiled_schema=compiled_schema,
                vocab_size=self.config.vocab_size,
                device=self.device,
            )

        # Prefix: [BOS] query [SEP]
        bos_tokens = [BOS_ID]
        query_tokens = self.tokenizer.encode(query)
        prefix_tokens = bos_tokens + query_tokens + [SEP_ID]
        input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            tool_logits = outputs["tool_logits"][0]
            confidence_est = outputs["confidence"][0].item()

        # Autoregressive generation with KV-cache
        generated_ids: List[int] = []
        curr_tok = input_ids
        kv_caches = None
        start_pos = 0

        for step in range(max_tokens):
            with torch.no_grad():
                out = self.model(curr_tok, start_pos=start_pos, kv_caches=kv_caches)
                kv_caches = out.get("kv_caches")
                next_logits = out["logits"][0, -1, :].clone()

            if step == 0:
                start_pos = input_ids.shape[1]
            else:
                start_pos += 1

            # Apply grammar mask if active
            if masker is not None:
                mask = masker.get_mask()
                next_logits[~mask] = -1e9

            # Greedy or temperature sample
            if temperature <= 0.0:
                next_token = torch.argmax(next_logits).item()
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == EOS_ID:
                break

            generated_ids.append(next_token)
            if masker is not None:
                masker.update(next_token)

            curr_tok = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse generated tokens into function calls
        function_calls, parsed_confidence = self._parse_generated_tokens(generated_ids)
        final_confidence = parsed_confidence if parsed_confidence is not None else confidence_est

        # Dual-head routing: if generator produced no tool call, route via classification head
        if not function_calls and tools and hasattr(compiled_schema, "tool_name_to_id"):
            id_to_name = {v: k for k, v in compiled_schema.tool_name_to_id.items()}
            top_tool_idx = torch.argmax(tool_logits).item()
            if top_tool_idx in id_to_name and final_confidence > 0.4:
                function_calls = [{"name": id_to_name[top_tool_idx], "arguments": {}}]

        return {
            "query": query,
            "function_calls": function_calls,
            "confidence": round(float(final_confidence), 4),
            "latency_ms": round(latency_ms, 2),
            "tokens_generated": len(generated_ids),
        }

    def _parse_generated_tokens(
        self, tokens: List[int]
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """Parse structured tokens into function call records."""
        function_calls: List[Dict[str, Any]] = []
        parsed_conf: Optional[float] = None

        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            if tok == EMPTY_ID:
                i += 1
                continue
            if tok == TOOL_START_ID:
                # Read tool name until ARG_START_ID
                name_tokens = []
                i += 1
                while i < n and tokens[i] != ARG_START_ID and tokens[i] != TOOL_END_ID:
                    name_tokens.append(tokens[i])
                    i += 1

                tool_name = self.tokenizer.decode(name_tokens).strip()

                arguments: Dict[str, Any] = {}
                if i < n and tokens[i] == ARG_START_ID:
                    i += 1
                    arg_tokens = []
                    while i < n and tokens[i] != ARG_END_ID and tokens[i] != TOOL_END_ID:
                        arg_tokens.append(tokens[i])
                        i += 1
                    arg_str = self.tokenizer.decode(arg_tokens).strip()
                    try:
                        arguments = json.loads(arg_str) if arg_str else {}
                    except Exception:
                        arguments = {}

                function_calls.append({"name": tool_name, "arguments": arguments})

            elif tok == CONF_ID:
                i += 1
                conf_toks = []
                while i < n and tokens[i] != EOS_ID:
                    conf_toks.append(tokens[i])
                    i += 1
                conf_str = self.tokenizer.decode(conf_toks).strip()
                try:
                    parsed_conf = float(conf_str)
                except ValueError:
                    pass
            i += 1

        return function_calls, parsed_conf
