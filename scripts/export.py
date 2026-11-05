# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Export MicroForge models to packed 1.58-bit binary format (< 12 MB).

Packing scheme:
5 ternary values {-1, 0, 1} mapped to {0, 1, 2}:
byte = v0 + 3*v1 + 9*v2 + 27*v3 + 81*v4 (max value 242 < 256, 1 byte = 5 weights = 1.6 bpw).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
import numpy as np
import torch

from microforge.model.config import ModelConfig
from microforge.model.model import MicroForgeModel


MAGIC_HEADER = b"MF158B01"  # MicroForge 1.58-bit format version 1


def pack_ternary_tensor(w: torch.Tensor) -> bytes:
    """Pack a ternary {-1, 0, 1} weight tensor into 1.6 bpw uint8 buffer."""
    # Scale and round to {-1, 0, 1}
    gamma = w.abs().mean().clamp(min=1e-8)
    w_ternary = torch.clamp(torch.round(w / gamma), -1, 1).to(torch.int8)
    # Map {-1, 0, 1} -> {0, 1, 2}
    vals = (w_ternary.flatten() + 1).detach().cpu().numpy().astype(np.uint8)

    # Pad to multiple of 5
    n = len(vals)
    pad = (5 - (n % 5)) % 5
    if pad > 0:
        vals = np.pad(vals, (0, pad), mode="constant", constant_values=1)

    vals = vals.reshape(-1, 5)
    packed = (
        vals[:, 0]
        + vals[:, 1] * 3
        + vals[:, 2] * 9
        + vals[:, 3] * 27
        + vals[:, 4] * 81
    ).astype(np.uint8)

    # Header for tensor: gamma (float32), original shape len, shape ints, packed bytes len
    shape = list(w.shape)
    hdr = struct.pack(f"<fI{len(shape)}I", float(gamma), len(shape), *shape)
    return hdr + packed.tobytes()


def export_model_binary(
    model: MicroForgeModel,
    config: ModelConfig,
    out_path: Union[str, Path],
) -> int:
    """Export the entire MicroForge model to a standalone packed binary."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    config_json = json.dumps(config.__dict__).encode("utf-8")

    with open(out_file, "wb") as f:
        # 1. Header
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        # 2. Iterate through state_dict
        state = model.state_dict()
        f.write(struct.pack("<I", len(state)))

        total_bytes = 0
        for name, tensor in state.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)

            if "weight" in name and tensor.dim() >= 2 and "norm" not in name and "embedding" not in name:
                # Pack ternary
                packed_data = pack_ternary_tensor(tensor)
                f.write(b"\x01")  # Flag 1: packed ternary
                f.write(struct.pack("<I", len(packed_data)))
                f.write(packed_data)
                total_bytes += len(packed_data)
            else:
                # Raw float16 for embeddings / norms / bias
                fp16_data = tensor.detach().cpu().to(torch.float16).numpy().tobytes()
                f.write(b"\x00")  # Flag 0: raw fp16
                shape = list(tensor.shape)
                hdr = struct.pack(f"<I{len(shape)}I", len(shape), *shape)
                payload = hdr + fp16_data
                f.write(struct.pack("<I", len(payload)))
                f.write(payload)
                total_bytes += len(payload)

    final_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[+] Model exported to: {out_file}")
    print(f"    Total File Size: {final_size_mb:.2f} MB")
    print(f"    Needle Size Comparison: Needle is 29.00 MB -> MicroForge is {final_size_mb:.2f} MB ({((29.0 - final_size_mb)/29.0)*100:.1f}% smaller!)")
    return out_file.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MicroForge model to packed ternary binary")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Input checkpoint")
    parser.add_argument("--config", type=str, default="configs/droidcall_gpu.yaml", help="Fallback config YAML")
    parser.add_argument("--out", type=str, default="models/microforge.bin", help="Output binary file")
    args = parser.parse_args()

    import yaml
    fallback_config = args.config if Path(args.config).exists() else "configs/default.yaml"
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" in ckpt and isinstance(ckpt["config"], ModelConfig):
            cfg = ckpt["config"]
        else:
            with open(fallback_config, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)["model"]
                cfg = ModelConfig(**y)
        model = MicroForgeModel(cfg)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    else:
        with open(fallback_config, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)["model"]
            cfg = ModelConfig(**y)
        model = MicroForgeModel(cfg)

    export_model_binary(model=model, config=cfg, out_path=args.out)


if __name__ == "__main__":
    main()
