# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
CLI evaluation script for MicroForge vs Needle benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from microforge.evaluation.benchmarks import BenchmarkRunner, NEEDLE_3_BASELINES
from microforge.inference.engine import MicroForgeEngine
from microforge.model.config import ModelConfig
from microforge.model.model import MicroForgeModel
from microforge.tokenizer.builder import ToolTokenizer


def main() -> None:
    default_model = "models/microforge.bin" if Path("models/microforge.bin").exists() else "checkpoints/best_model.pt"
    default_test = "data/droidcall/droidcall_test.jsonl" if Path("data/droidcall/droidcall_test.jsonl").exists() else "data/test.jsonl"
    default_schemas = "tools_schemas/droidcall_tools.json" if Path("tools_schemas/droidcall_tools.json").exists() else "tools_schemas/example_tools.json"

    parser = argparse.ArgumentParser(description="Evaluate MicroForge model")
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", type=str, default=default_model, help="Model checkpoint path (.bin or .pt)")
    parser.add_argument("--test-data", type=str, default=default_test, help="Test JSONL data")
    parser.add_argument("--schemas", type=str, default=default_schemas, help="Tools schema JSON")
    parser.add_argument("--max-samples", type=int, default=100, help="Max test samples")
    args = parser.parse_args()

    print("=" * 70)
    print("  MicroForge vs Cactus Needle 3: Formal Benchmark Evaluation")
    print("=" * 70)

    with open(args.schemas, "r", encoding="utf-8") as f:
        schemas_data = json.load(f)
        tools = schemas_data.get("tools", schemas_data if isinstance(schemas_data, list) else [])

    tok_path = Path("models/tokenizer.json")
    if tok_path.exists():
        tokenizer = ToolTokenizer.load(tok_path)
    else:
        tokenizer = ToolTokenizer(vocab_size=4096)
        tokenizer.build_from_tool_schemas(tools)

    ckpt_path = Path(args.checkpoint)
    model_size_mb = 7.19
    if ckpt_path.exists():
        model_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"[+] Loaded Model Binary: {ckpt_path} ({model_size_mb:.2f} MB)")
        engine = MicroForgeEngine(model_path_or_model=str(ckpt_path), tokenizer=tokenizer)
    else:
        print("[!] Checkpoint not found, initializing un-trained model for dry-run evaluation...")
        cfg = ModelConfig()
        model = MicroForgeModel(cfg)
        engine = MicroForgeEngine(model_path_or_model=model, tokenizer=tokenizer)

    runner = BenchmarkRunner(engine=engine, tools=tools)
    print(f"[*] Running benchmark on {args.test_data} (samples={args.max_samples})...\n")
    results = runner.evaluate_tool_calling(test_jsonl_path=args.test_data, max_samples=args.max_samples)

    report = runner.generate_needle_comparison(eval_results=results, model_size_mb=model_size_mb)
    print(report)


if __name__ == "__main__":
    main()
