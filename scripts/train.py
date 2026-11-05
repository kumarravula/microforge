# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
CLI training script for MicroForge.
Loads configuration, initializes model, runs training loop, and saves checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from microforge.data.dataset import ToolCallingDataset, collate_tool_calling
from microforge.data.schema_compiler import SchemaCompiler
from microforge.model.config import ModelConfig
from microforge.model.model import MicroForgeModel
from microforge.tokenizer.builder import ToolTokenizer
from microforge.training.trainer import MicroForgeTrainer, TrainingConfig
from microforge.training.distillation import DistillationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MicroForge model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print(" MicroForge Model Training Pipeline")
    print("=" * 60)

    # 1. Load Tools & Tokenizer
    schema_path = cfg["data"]["tools_schema"]
    with open(schema_path, "r", encoding="utf-8") as f:
        tools_data = json.load(f)
        tools = tools_data.get("tools", tools_data if isinstance(tools_data, list) else [])

    tok_path = Path("models/tokenizer.json")
    if tok_path.exists():
        tokenizer = ToolTokenizer.load(tok_path)
    else:
        print("[!] Tokenizer not found at models/tokenizer.json, building on the fly...")
        tokenizer = ToolTokenizer(vocab_size=cfg["model"]["vocab_size"])
        tokenizer.build_from_tool_schemas(tools)

    compiler = SchemaCompiler(tokenizer)
    compiled_schema = compiler.compile_tools(tools)
    tool_to_id = compiled_schema.tool_name_to_id

    # 2. Datasets & DataLoaders
    train_ds = ToolCallingDataset(
        jsonl_path=cfg["data"]["train_path"],
        tokenizer=tokenizer,
        tool_to_id=tool_to_id,
        num_tool_classes=cfg["model"]["num_tool_classes"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    val_ds = ToolCallingDataset(
        jsonl_path=cfg["data"]["val_path"],
        tokenizer=tokenizer,
        tool_to_id=tool_to_id,
        num_tool_classes=cfg["model"]["num_tool_classes"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_tool_calling,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tool_calling,
    )
    print(f"Dataset loaded: {len(train_ds)} train samples, {len(val_ds)} val samples")

    # 3. Model Architecture
    m_cfg = ModelConfig(
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_query_heads=cfg["model"]["n_query_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        d_ff=cfg["model"]["d_ff"],
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
        embedding_factor_dim=cfg["model"]["embedding_factor_dim"],
        num_tool_classes=cfg["model"]["num_tool_classes"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
    )
    model = MicroForgeModel(m_cfg)
    total_params = model.count_parameters()
    size_est = model.estimate_size_mb(bits_per_param=1.58)
    print(f"Model created:")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Target Size: {size_est:.2f} MB (< 29 MB Needle baseline)")

    # 4. Initialize Trainer
    t_cfg = cfg["training"]
    train_config = TrainingConfig(
        epochs=t_cfg.get("epochs", 3),
        learning_rate=float(t_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        warmup_steps=t_cfg.get("warmup_steps", 10),
        max_grad_norm=float(t_cfg.get("max_grad_norm", 1.0)),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 1),
        lm_loss_weight=float(t_cfg.get("lm_loss_weight", 1.0)),
        tool_loss_weight=float(t_cfg.get("tool_loss_weight", 1.0)),
        confidence_loss_weight=float(t_cfg.get("confidence_loss_weight", 0.5)),
        val_every_steps=t_cfg.get("val_every_steps", 10),
        log_every_steps=t_cfg.get("log_every_steps", 5),
        early_stopping_patience=t_cfg.get("early_stopping_patience", 10),
        mixed_precision=False if args.device == "cpu" or not torch.cuda.is_available() else t_cfg.get("mixed_precision", False),
        checkpoint_dir="checkpoints",
    )
    trainer = MicroForgeTrainer(config=train_config)

    # 5. Train
    print("[*] Starting training...")
    target_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=target_device,
    )

    # Save best_model.pt
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": m_cfg}, best_path)
    print(f"[+] Training completed successfully! Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
