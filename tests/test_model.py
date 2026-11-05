# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive test suite for MicroForge modules.
"""

import sys
import unittest
import torch

from microforge.model.config import ModelConfig
from microforge.model.bitlinear import BitLinear
from microforge.model.attention import GroupedQueryAttention
from microforge.model.embedding import FactorizedEmbedding
from microforge.model.model import MicroForgeModel
from microforge.evaluation.metrics import ExactMatchAccuracy, ExtractionMicroF1
from microforge.data.synthetic import SyntheticDataGenerator
from scripts.export import pack_ternary_tensor


class TestMicroForge(unittest.TestCase):

    def test_bitlinear_ste(self):
        """Test BitLinear forward pass and STE gradient flow."""
        layer = BitLinear(in_features=64, out_features=128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 128))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)

    def test_factorized_embedding(self):
        """Test ALBERT-style factorized embedding parameter reduction."""
        cfg = ModelConfig.tiny()
        embed = FactorizedEmbedding(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 16))
        out = embed(idx)
        self.assertEqual(out.shape, (2, 16, cfg.d_model))

    def test_gqa_attention(self):
        """Test Grouped Query Attention with RoPE."""
        cfg = ModelConfig.tiny()
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out, cache = attn(x)
        self.assertEqual(out.shape, (2, 8, cfg.d_model))

    def test_model_forward(self):
        """Test MicroForge full forward pass with multi-task heads."""
        cfg = ModelConfig.tiny()
        model = MicroForgeModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        outputs = model(input_ids)

        self.assertIn("logits", outputs)
        self.assertIn("tool_logits", outputs)
        self.assertIn("confidence", outputs)

        self.assertEqual(outputs["logits"].shape, (2, 16, cfg.vocab_size))
        self.assertEqual(outputs["tool_logits"].shape, (2, cfg.num_tool_classes))
        self.assertEqual(outputs["confidence"].shape, (2, 1))

    def test_exact_match_metric(self):
        """Test tool exact match accuracy metric."""
        metric = ExactMatchAccuracy()
        preds = [{"name": "set_timer", "arguments": {"minutes": 10}}]
        targets = [{"name": "set_timer", "arguments": {"minutes": 10}}]
        self.assertTrue(metric.update(preds, targets))

        res = metric.compute()
        self.assertEqual(res["exact_match_accuracy"], 1.0)

    def test_synthetic_generator(self):
        """Test synthetic dataset generation."""
        gen = SyntheticDataGenerator()
        data = gen.generate_dataset(n_samples=20)
        self.assertEqual(len(data), 20)
        for item in data:
            self.assertIn("query", item)
            self.assertIn("calls", item)
            self.assertIn("confidence", item)

    def test_ternary_packing(self):
        """Test 1.6 bpw ternary weight packing."""
        w = torch.randn(64, 64)
        packed = pack_ternary_tensor(w)
        self.assertGreater(len(packed), 0)
        # Expected packed size: header (~20 bytes) + (64*64 / 5) ~ 820 bytes.
        self.assertLess(len(packed), 1000)


if __name__ == "__main__":
    unittest.main()
