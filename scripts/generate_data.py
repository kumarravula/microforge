# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
CLI script to generate synthetic training data and train the domain-specific BPE tokenizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microforge.data.synthetic import SyntheticDataGenerator
from microforge.tokenizer.builder import ToolTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data for MicroForge")
    parser.add_argument("--schemas", type=str, default="tools_schemas/example_tools.json", help="Path to tool schemas JSON")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for JSONL splits")
    parser.add_argument("--samples", type=int, default=3000, help="Total samples to generate")
    parser.add_argument("--train-tokenizer", action="store_true", default=True, help="Train BPE tokenizer on generated corpus")
    parser.add_argument("--tokenizer-out", type=str, default="models/tokenizer.json", help="Path to save tokenizer")
    args = parser.parse_args()

    print(f"[1/3] Generating {args.samples} synthetic training examples from {args.schemas}...")
    generator = SyntheticDataGenerator(schemas_path=args.schemas)
    splits = generator.export_splits(out_dir=args.out_dir, total_samples=args.samples)
    print(f"      Splits written to:")
    for name, path in splits.items():
        print(f"      - {name}: {path}")

    if args.train_tokenizer:
        print(f"[2/3] Training 4096-vocab domain BPE tokenizer...")
        with open(args.schemas, "r", encoding="utf-8") as f:
            schemas_data = json.load(f)
            tools_list = schemas_data.get("tools", schemas_data if isinstance(schemas_data, list) else [])

        # Collect text corpus
        corpus = []
        for p in splits.values():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    corpus.append(item.get("query", ""))

        tok = ToolTokenizer(vocab_size=4096)
        tok.train_from_corpus(corpus)
        tok.build_from_tool_schemas(tools_list)

        out_tok = Path(args.tokenizer_out)
        out_tok.parent.mkdir(parents=True, exist_ok=True)
        tok.save(out_tok)
        print(f"[3/3] Tokenizer saved to: {out_tok}")


if __name__ == "__main__":
    main()
