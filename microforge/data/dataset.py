# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch Datasets and Collation for MicroForge tool-calling and extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from microforge.tokenizer.builder import (
    ToolTokenizer,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    TOOL_START_ID,
    TOOL_END_ID,
    ARG_START_ID,
    ARG_END_ID,
    CONF_ID,
    SEP_ID,
    EMPTY_ID,
)


class ToolCallingDataset(Dataset):
    """Dataset for tool-calling queries with multi-task supervision.

    Each example in JSONL contains:
        {
            "query": "set a timer for 10 minutes",
            "calls": [{"name": "set_timer", "arguments": {"minutes": 10}}],
            "confidence": 0.95
        }
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: ToolTokenizer,
        tool_to_id: Dict[str, int],
        num_tool_classes: int = 64,
        max_seq_len: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.tool_to_id = tool_to_id
        self.num_tool_classes = num_tool_classes
        self.max_seq_len = max_seq_len
        self.examples: List[Dict[str, Any]] = []

        path = Path(jsonl_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        query = item.get("query", "")
        calls = item.get("calls", [])
        confidence = float(item.get("confidence", 1.0))

        # 1. Multi-hot vector for tool routing
        tool_labels = torch.zeros(self.num_tool_classes, dtype=torch.float32)
        for call in calls:
            name = call.get("name", "")
            if name in self.tool_to_id:
                tid = self.tool_to_id[name]
                if tid < self.num_tool_classes:
                    tool_labels[tid] = 1.0

        # 2. Build wire tokens
        # [BOS] query [SEP] [TOOL_START] name [ARG_START] {json} [ARG_END] [TOOL_END] ... [CONF] conf [EOS]
        bos_tokens = [BOS_ID]
        query_tokens = self.tokenizer.encode(query)
        sep_tokens = [SEP_ID]

        generation_tokens: List[int] = []
        if not calls:
            generation_tokens.extend([EMPTY_ID])
        else:
            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                args_str = json.dumps(args, separators=(",", ":"))
                
                generation_tokens.append(TOOL_START_ID)
                generation_tokens.extend(self.tokenizer.encode(name))
                generation_tokens.append(ARG_START_ID)
                generation_tokens.extend(self.tokenizer.encode(args_str))
                generation_tokens.append(ARG_END_ID)
                generation_tokens.append(TOOL_END_ID)

        generation_tokens.append(CONF_ID)
        conf_str = f"{confidence:.2f}"
        generation_tokens.extend(self.tokenizer.encode(conf_str))
        generation_tokens.append(EOS_ID)

        full_tokens = bos_tokens + query_tokens + sep_tokens + generation_tokens
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]

        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Labels for next-token prediction: query tokens get -100 (loss only on completion)
        prefix_len = len(bos_tokens) + len(query_tokens) + len(sep_tokens)
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tool_labels": tool_labels,
            "confidence": torch.tensor(confidence, dtype=torch.float32),
        }


class ExtractionDataset(Dataset):
    """Dataset for structured JSON extraction tasks."""

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: ToolTokenizer,
        max_seq_len: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict[str, Any]] = []

        path = Path(jsonl_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        text = item.get("text", "")
        extracted = item.get("extracted", {})

        bos_tokens = [BOS_ID]
        text_tokens = self.tokenizer.encode(text)
        sep_tokens = [SEP_ID]

        args_str = json.dumps(extracted, separators=(",", ":"))
        gen_tokens = (
            [TOOL_START_ID]
            + self.tokenizer.encode("extract")
            + [ARG_START_ID]
            + self.tokenizer.encode(args_str)
            + [ARG_END_ID, TOOL_END_ID, EOS_ID]
        )

        full_tokens = bos_tokens + text_tokens + sep_tokens + gen_tokens
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]

        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prefix_len = len(bos_tokens) + len(text_tokens) + len(sep_tokens)
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tool_labels": torch.zeros(64, dtype=torch.float32),
            "confidence": torch.tensor(1.0, dtype=torch.float32),
        }


def collate_tool_calling(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate variable length sequences with left/right padding."""
    max_len = max(x["input_ids"].size(0) for x in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_tool_labels = []
    batch_confidences = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        if pad_len > 0:
            pad_tokens = torch.full((pad_len,), PAD_ID, dtype=torch.long)
            pad_mask = torch.zeros((pad_len,), dtype=torch.long)
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)

            input_ids = torch.cat([input_ids, pad_tokens], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_tool_labels.append(item["tool_labels"])
        batch_confidences.append(item["confidence"])

    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
        "tool_labels": torch.stack(batch_tool_labels, dim=0),
        "confidence": torch.stack(batch_confidences, dim=0),
    }
