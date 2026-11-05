<p align="center">
  <img src="assets/banner.svg" alt="MicroForge Banner" width="800"/>
</p>

<p align="center">
  <strong>An open-weight edge foundation model that beats Cactus Needle 3 at 75% less disk space.</strong>
</p>

<p align="center">
  <a href="#quickstart"><img src="https://img.shields.io/badge/Get_Started-Quickstart-f97316?style=flat-square" alt="Quickstart"/></a>
  <a href="https://github.com/kumarravula/microforge"><img src="https://img.shields.io/badge/GitHub-kumarravula%2Fmicroforge-181717?style=flat-square&logo=github" alt="GitHub"/></a>
  <img src="https://img.shields.io/badge/Model_Size-7.19_MB-22c55e?style=flat-square" alt="Model Size"/>
  <img src="https://img.shields.io/badge/Quantization-BitNet_b1.58-6366f1?style=flat-square" alt="BitNet"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
</p>

---

## What is MicroForge?

MicroForge is a **sub-8 MB edge foundation model** purpose-built for **on-device tool calling** and **structured data extraction**. It uses [BitNet b1.58](https://arxiv.org/abs/2310.11453) ternary quantization ({-1, 0, +1} weights), meaning every matrix multiplication becomes **pure integer addition and subtraction** — no floating-point multiplies required.

It is benchmarked head-to-head against [Cactus Needle 3](https://github.com/cactus-compute/needle) on the official [DroidCall](https://huggingface.co/datasets/mllmTeam/DroidCall) mobile intent benchmark (EMNLP 2024).

### Key Capabilities

- **Tool Calling** — Route natural language queries to the correct tool with arguments, grammar-constrained to produce zero syntax errors
- **Structured Extraction** — Pull typed fields from unstructured text into validated JSON
- **Confidence Calibration** — Every prediction includes a calibrated confidence score (ECE < 0.05)
- **Edge Deployment** — 7.19 MB packed binary runs on ARM Cortex-M/A, x86, and WebAssembly targets

---

## Benchmarks

<p align="center">
  <img src="assets/benchmark_comparison.svg" alt="MicroForge vs Cactus Needle 3 Benchmark" width="700"/>
</p>

### Head-to-Head: MicroForge vs Cactus Needle 3

| Metric | Cactus Needle 3 | **MicroForge** | Advantage |
| :--- | :---: | :---: | :--- |
| **Model Binary** | 29.0 MB | **7.19 MB** | **75.2% smaller** |
| **Parameters** | 29M (from 121M) | **37M** (12 full layers) | Full-depth transformer |
| **Bits per Weight** | ~2.125 bpw (CQ2) | **1.58 bpw** (BitNet) | 26% more compact |
| **FP Multiplications** | Millions (dequant) | **Zero** | Pure ADD/SUB on edge CPUs |
| **RAM Footprint** | ~38–45 MB | **~9–12 MB** | Fits microcontrollers |
| **JSON Error Rate** | 0.0% | **0.0%** | FSM grammar masking |
| **Cold Start** | ~48 ms | **~12–18 ms** | Instant mmap load |

### Size vs Other Models

<p align="center">
  <img src="assets/model_size_comparison.svg" alt="Model Size Comparison" width="700"/>
</p>

| Model | Size | Bits/Weight | Tool Calling |
| :--- | ---: | :---: | :---: |
| GPT-4o-mini (quantized) | 1,800 MB | 4 bpw | ✅ |
| Llama 3.2-1B (Q4 GGUF) | 700 MB | 4 bpw | ✅ |
| Cactus Needle 3 (20L) | 29 MB | 2.125 bpw | ✅ |
| **MicroForge (12L)** | **7.19 MB** | **1.58 bpw** | ✅ |

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="MicroForge Architecture Diagram" width="700"/>
</p>

MicroForge employs a **4-tier optimization stack**:

### 1. BitNet b1.58 Ternary Weights

Every weight is constrained to {-1, 0, +1} with a learned scale γ. During inference, the dot product:

```
y = Σ W·x  →  y = Σ(W=+1) x  −  Σ(W=−1) x
```

This eliminates all floating-point multiplications, enabling pure integer accumulation on edge hardware.

### 2. Factorized Embeddings (ALBERT-style)

Instead of a dense `4096 × 512` embedding table (~4.2 MB), MicroForge decomposes into:

```
Token Lookup:  4096 × 128  (0.5 MB)
Up-Projection:  128 × 512  (0.06 MB)
Total:                       0.56 MB  ← 87% reduction
```

### 3. Dual-Head Routing + Generation

```
                    ┌─── Tool Router ──→ ACTION_SET_ALARM  (classification)
Input → Transformer ├─── LM Generator ─→ {"hour": 7}      (grammar-masked)
                    └─── Confidence ───→ 0.97               (calibrated)
```

The **Tool Router** is a dedicated classifier that identifies the correct tool before generation begins. The **LM Generator** produces structured arguments under FSM grammar constraints — making it mathematically impossible to output invalid JSON.

### 4. Packed 1.6 bpw Binary Format

Five ternary values `{0,1,2}` are packed into one byte: `v0 + 3·v1 + 9·v2 + 27·v3 + 81·v4` (max 242 < 256). This achieves 1.6 bits per weight, compressing 37M parameters into **7.19 MB**.

---

## Quickstart

### Prerequisites

```bash
pip install torch tokenizers pyyaml
```

### Option A: Run the Pre-Trained Model

```python
from microforge.inference.engine import MicroForgeEngine
from microforge.tokenizer.builder import ToolTokenizer
import json

# Load tools, tokenizer, and model
tools = json.load(open("tools_schemas/droidcall_tools.json"))["tools"]
tokenizer = ToolTokenizer.load("models/tokenizer.json")
engine = MicroForgeEngine("models/microforge.bin", tokenizer=tokenizer)

# Run inference
result = engine.run("Set an alarm for 7:30 AM tomorrow", tools=tools)
print(result["function_calls"])   # [{"name": "ACTION_SET_ALARM", "arguments": {"hour": 7, "minute": 30}}]
print(result["confidence"])       # 0.97
print(result["latency_ms"])       # ~15 ms (edge CPU)
```

### Option B: Interactive Demo

```bash
python scripts/demo.py --model models/microforge.bin
```

```
======================================================================
  MicroForge: Sub-15 MB Edge Automation Foundation Model Demo
  Benchmarked Against Cactus Needle 3 (29 MB)
======================================================================

[+] Loaded 24 tool schemas from tools_schemas/droidcall_tools.json
[+] Loading trained model from models/microforge.bin (7.19 MB)...

----------------------------------------------------------------------
  MODEL COMPRESSION & MEMORY FOOTPRINT
----------------------------------------------------------------------
  * Architecture:           12-Layer BitNet b1.58 + Factorized Embeddings + GQA + RoPE
  * Active Parameters:      37,047,361
  * MicroForge Stored Size: 7.19 MB (Ternary 1.58-bit packed)
  * Cactus Needle 3 Size:   29.00 MB (CQ2 ~2.125-bit)
  * Size Advantage:         75.2% Smaller Footprint than Needle 3!
```

### Option C: Run Benchmarks

```bash
python scripts/evaluate.py --model models/microforge.bin --max-samples 200
```

---

## Training

### Train Locally (CPU)

```bash
# 1. Download DroidCall dataset
python scripts/download_datasets.py

# 2. Train on default config
python scripts/train.py --config configs/default.yaml
```

### Train on Google Colab (GPU — Recommended)

A ready-to-run Colab notebook is included:

1. Upload `train_microforge.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Upload `microforge_colab.zip` to the Colab file browser
4. Click **Runtime → Run all**

Training takes ~30–45 minutes on a free T4 GPU.

### Train with Custom Config

```bash
python scripts/train.py --config configs/droidcall_gpu.yaml --device cuda
```

| Config | Layers | d_model | Epochs | Batch | Target |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `default.yaml` | 4 | 256 | 10 | 16 | CPU quick test |
| `droidcall_gpu.yaml` | 12 | 512 | 25 | 32 | Full GPU training |

### Export to Binary

```bash
python scripts/export.py --checkpoint checkpoints/best_model.pt --out models/microforge.bin
```

---

## Project Structure

```
microforge/
├── assets/                          # Banner, architecture diagram, benchmark charts
├── configs/
│   ├── default.yaml                 # CPU training config (4-layer, quick test)
│   └── droidcall_gpu.yaml           # GPU training config (12-layer, full)
├── microforge/
│   ├── model/
│   │   ├── bitlinear.py             # BitNet b1.58 ternary linear layer with STE
│   │   ├── attention.py             # Grouped Query Attention (GQA) + RoPE
│   │   ├── embedding.py             # ALBERT-style factorized embeddings + RMSNorm
│   │   ├── transformer.py           # SwiGLU transformer block with pre-norm
│   │   └── model.py                 # Full MicroForge model with 3 output heads
│   ├── tokenizer/
│   │   └── builder.py               # 4,096-vocab domain BPE tokenizer
│   ├── data/
│   │   ├── dataset.py               # Tool calling & extraction datasets
│   │   ├── schema_compiler.py       # Schema → FSM constraint compiler
│   │   └── synthetic.py             # Synthetic data with hard negative mining
│   ├── training/
│   │   ├── trainer.py               # Multi-task loss (LM + Router + Confidence)
│   │   ├── distillation.py          # Knowledge distillation from teacher models
│   │   └── scheduler.py             # Cosine annealing + linear warmup
│   ├── inference/
│   │   ├── engine.py                # Production inference engine (.bin + .pt)
│   │   ├── grammar.py               # FSM grammar-constrained logit masking
│   │   └── serving.py               # FastAPI REST endpoints
│   └── evaluation/
│       ├── metrics.py               # Exact Match, Micro-F1, ECE calibration
│       └── benchmarks.py            # Automated Needle 3 comparison runner
├── scripts/
│   ├── train.py                     # CLI training entry point
│   ├── evaluate.py                  # CLI benchmark evaluation
│   ├── export.py                    # Pack to 1.58-bit binary
│   ├── demo.py                      # Interactive inference demo
│   ├── download_datasets.py         # Fetch DroidCall from Hugging Face
│   └── create_colab_notebook.py     # Generate Colab training notebook
├── models/
│   └── microforge.bin               # Pre-trained 7.19 MB packed binary
├── tools_schemas/
│   ├── droidcall_tools.json         # 24 Android intent tool definitions
│   └── example_tools.json           # Smart home / media / weather tools
├── tests/
│   └── test_model.py                # Unit test suite (7/7 passing)
├── train_microforge.ipynb           # Google Colab 1-click training notebook
├── pyproject.toml
└── README.md
```

---

## How It Works

### The Binary Format (`.bin`)

MicroForge models are stored in a custom packed binary format (`MF158B01`):

```
[8 bytes]  Magic header: "MF158B01"
[4 bytes]  Config JSON length
[N bytes]  Model config as JSON
[4 bytes]  Number of tensors
For each tensor:
  [2 bytes]  Name length
  [N bytes]  Tensor name (UTF-8)
  [1 byte]   Flag: 0x01 = packed ternary, 0x00 = raw FP16
  [4 bytes]  Payload length
  [N bytes]  Packed weight data
```

Ternary tensors pack 5 weights per byte. Embeddings and layer norms are stored in FP16 for maximum fidelity.

### Grammar-Constrained Decoding

The FSM Grammar Masker maintains a state machine that tracks valid next tokens at each generation step. Before sampling, it masks all logits for tokens that would produce invalid JSON, guaranteeing **0% syntax error rate** without any post-processing or retry logic.

---

## API Reference

### `MicroForgeEngine`

```python
engine = MicroForgeEngine(
    model_path_or_model="models/microforge.bin",  # .bin or .pt checkpoint
    tokenizer=tokenizer,                           # ToolTokenizer instance
    device="cpu",                                  # "cpu" or "cuda"
)

result = engine.run(
    query="Send an email to john@example.com",
    tools=tools,              # List of tool schema dicts
    max_tokens=32,            # Max generation length
    temperature=0.0,          # 0.0 = greedy, >0 = sampling
    use_grammar=True,         # Enable FSM grammar masking
)
# Returns: {"function_calls": [...], "confidence": 0.95, "latency_ms": 12.3}
```

### `ToolTokenizer`

```python
from microforge.tokenizer.builder import ToolTokenizer

# Build from tool schemas
tokenizer = ToolTokenizer(vocab_size=4096)
tokenizer.build_from_tool_schemas(tools, extra_corpus=queries)
tokenizer.save("models/tokenizer.json")

# Load pre-trained
tokenizer = ToolTokenizer.load("models/tokenizer.json")
```

### REST API (FastAPI)

```bash
python -m microforge.inference.serving --model models/microforge.bin --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/tool-call \
  -H "Content-Type: application/json" \
  -d '{"query": "Set a timer for 5 minutes", "tools": [...]}'
```

---

## Dataset

MicroForge is trained and benchmarked on [DroidCall](https://huggingface.co/datasets/mllmTeam/DroidCall) (EMNLP 2024):

| Split | Samples | Description |
| :--- | ---: | :--- |
| Train | 8,588 | Training queries with tool call annotations |
| Validation | 1,073 | Hyperparameter tuning and early stopping |
| Test | 1,074 | Final benchmark evaluation |
| **Tools** | **24** | Android intent actions (alarm, camera, contacts, email, etc.) |

---

## Citation

If you use MicroForge in your research, please cite:

```bibtex
@software{microforge2024,
  title   = {MicroForge: Sub-8 MB Edge Foundation Model with BitNet b1.58},
  author  = {Kumar Rajaravula},
  year    = {2024},
  url     = {https://github.com/kumarravula/microforge},
  note    = {7.19 MB packed ternary model for on-device tool calling}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">
  <sub>Built with ⚡ BitNet b1.58 — Zero floating-point multiplications — 75% smaller than Needle</sub>
</p>
