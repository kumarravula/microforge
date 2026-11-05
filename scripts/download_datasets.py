# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Automated downloader and preprocessor for DroidCall and Function-Calling datasets.
Converts raw datasets from Hugging Face into MicroForge's standardized wire format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import urllib.request
import urllib.error


DROIDCALL_HF_URLS = [
    "https://huggingface.co/datasets/mllmTeam/DroidCall/resolve/main/DroidCall_train.jsonl",
    "https://huggingface.co/datasets/mllmTeam/DroidCall/resolve/main/DroidCall_code_short.jsonl",
    "https://huggingface.co/datasets/mllmTeam/DroidCall/resolve/main/DroidCall_test.jsonl",
]

DROIDCALL_GH_RAW = [
    "https://raw.githubusercontent.com/UbiquitousLearning/DroidCall/main/data/api.jsonl",
]

GLAIVE_SAMPLE_URL = "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/raw/main/glaive-function-calling-v2.json"


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with User-Agent header and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        size_kb = dest_path.stat().st_size / 1024
        print(f"[+] Found cached: {dest_path.name} ({size_kb:.1f} KB)")
        return True
    print(f"[*] Downloading: {url} -> {dest_path.name}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MicroForge/0.1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        size_kb = dest_path.stat().st_size / 1024
        print(f"[+] Downloaded {dest_path.name} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"[!] Warning: Could not download {url}: {e}")
        return False


def parse_droidcall_entry(line_str: str) -> dict | None:
    """Parse a single DroidCall JSON line into MicroForge standard format."""
    try:
        data = json.loads(line_str.strip())
    except Exception:
        return None

    query = data.get("query", data.get("instruction", data.get("prompt", "")))
    calls = []

    # DroidCall native answers format
    if "answers" in data and isinstance(data["answers"], list):
        for ans in data["answers"]:
            name = ans.get("name", "")
            args = ans.get("arguments", {})
            if name:
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})

    # Handle chat messages format if answers wasn't present
    if not calls:
        messages = data.get("messages", [])
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and not query:
                    query = content.strip()
                elif role == "assistant":
                    fn_call = msg.get("function_call")
                    if fn_call:
                        name = fn_call.get("name", "")
                        raw_args = fn_call.get("arguments", "{}")
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
                    elif "{" in content:
                        match = re.search(r"\{.*\}", content, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group(0))
                                name = parsed.get("name", parsed.get("action", ""))
                                args = parsed.get("arguments", parsed.get("parameters", {}))
                                if name:
                                    calls.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
                            except Exception:
                                pass

    if query and calls:
        return {
            "query": query.strip(),
            "calls": calls,
            "confidence": 0.98,
            "tools": data.get("tools", []),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & prepare DroidCall datasets")
    parser.add_argument("--out-dir", type=str, default="data/droidcall", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MicroForge Dataset Downloader & Converter")
    print("  Downloading DroidCall Mobile Benchmark")
    print("=" * 70)

    downloaded_files = []
    for url in DROIDCALL_HF_URLS:
        fname = url.split("/")[-1]
        dest = raw_dir / fname
        if download_file(url, dest):
            downloaded_files.append(dest)

    # Process into MicroForge JSONL format
    converted_samples = []
    for fpath in downloaded_files:
        if not fpath.name.endswith(".jsonl"):
            continue
        print(f"[*] Processing {fpath.name}...")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                entry = parse_droidcall_entry(line)
                if entry and (entry["calls"] or len(entry["query"]) > 5):
                    converted_samples.append(entry)

    print(f"\n[+] Successfully parsed {len(converted_samples)} DroidCall samples!")

    if converted_samples:
        # Split 80/10/10
        n = len(converted_samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_data = converted_samples[:n_train]
        val_data = converted_samples[n_train:n_train + n_val]
        test_data = converted_samples[n_train + n_val:]

        # Extract unique tool schemas
        unique_tools = {}
        for sample in converted_samples:
            for t in sample.get("tools", []):
                name = t.get("name")
                if name and name not in unique_tools:
                    unique_tools[name] = t

        tools_list = list(unique_tools.values())
        tools_path = Path("tools_schemas/droidcall_tools.json")
        tools_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump({"tools": tools_list}, f, indent=2)
        print(f"[+] Saved {len(tools_list)} unique Android mobile tool schemas to {tools_path}")

        for split_name, s_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            target_path = out_dir / f"droidcall_{split_name}.jsonl"
            with open(target_path, "w", encoding="utf-8") as f:
                for item in s_data:
                    # Strip out tools field to keep jsonl clean
                    record = {
                        "query": item["query"],
                        "calls": item["calls"],
                        "confidence": item["confidence"],
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"  -> Saved {len(s_data)} samples to {target_path}")

        print("\n[+] DroidCall benchmark dataset is ready for training and evaluation!")


if __name__ == "__main__":
    main()
