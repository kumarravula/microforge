# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Create the 1-click Google Colab training notebook for MicroForge.
"""

import json
from pathlib import Path


def make_colab_notebook(out_path: str = "train_microforge.ipynb") -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 MicroForge: Train a Sub-15 MB Model That Beats Cactus Needle\n",
                "\n",
                "> **Goal:** Train a 12-layer BitNet b1.58 edge foundation model on the official **DroidCall** mobile intent dataset on a free GPU.\n",
                "> **Target Output:** Packed binary `microforge.bin` (< 12 MB) that runs on edge CPUs with 0% syntax errors and beats Needle's 78.4% benchmark.\n",
                "\n",
                "### ⚡ Getting Started:\n",
                "1. Make sure GPU is selected: **Runtime → Change runtime type → T4 GPU**.\n",
                "2. Click the **Files (folder icon)** on the left sidebar, and drag & drop `microforge_colab.zip` into it.\n",
                "3. Click **Runtime → Run all** (or press `Ctrl + F9`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify GPU Hardware\n",
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Install dependencies\n",
                "!pip install -q torch tokenizers pyyaml"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Unpack MicroForge Code\n",
                "import os, zipfile\n",
                "from google.colab import files\n",
                "\n",
                "if not os.path.exists('microforge_colab.zip'):\n",
                "    print('Upload microforge_colab.zip now:')\n",
                "    uploaded = files.upload()\n",
                "\n",
                "with zipfile.ZipFile('microforge_colab.zip', 'r') as z:\n",
                "    z.extractall('.')\n",
                "print('[+] Extracted microforge_colab.zip successfully!')\n",
                "print('Ready directories:', os.listdir('.'))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Download DroidCall Dataset directly in Colab\n",
                "!mkdir -p data/droidcall/raw tools_schemas models checkpoints\n",
                "!wget -q -O data/droidcall/raw/DroidCall_train.jsonl https://huggingface.co/datasets/mllmTeam/DroidCall/resolve/main/DroidCall_train.jsonl\n",
                "!wget -q -O data/droidcall/raw/DroidCall_test.jsonl https://huggingface.co/datasets/mllmTeam/DroidCall/resolve/main/DroidCall_test.jsonl\n",
                "print('[+] Datasets downloaded successfully from Hugging Face!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Parse DroidCall into MicroForge Wire Format\n",
                "!python scripts/download_datasets.py\n",
                "print('[+] Processed DroidCall into train/val/test splits!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Train the 4,096-token Domain BPE Tokenizer on DroidCall\n",
                "import json\n",
                "from pathlib import Path\n",
                "from microforge.tokenizer.builder import ToolTokenizer\n",
                "\n",
                "with open('tools_schemas/droidcall_tools.json', 'r') as f:\n",
                "    tools = json.load(f)['tools']\n",
                "\n",
                "corpus = []\n",
                "with open('data/droidcall/droidcall_train.jsonl', 'r') as f:\n",
                "    for line in f:\n",
                "        corpus.append(json.loads(line)['query'])\n",
                "\n",
                "tok = ToolTokenizer(vocab_size=4096)\n",
                "tok.build_from_tool_schemas(tools, extra_corpus=corpus)\n",
                "tok.save('models/tokenizer.json')\n",
                "print(f'[+] Tokenizer trained and saved! Vocab size: {tok.actual_vocab_size}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Train MicroForge 12-Layer BitNet Model on GPU (25 Epochs)\n",
                "!python scripts/train.py --config configs/droidcall_gpu.yaml --device cuda"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. Run Benchmark Evaluation vs Needle 3 on DroidCall Test Split\n",
                "!python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-data data/droidcall/droidcall_test.jsonl --schemas tools_schemas/droidcall_tools.json --max-samples 200"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 9. Export to Packed 1.58-bit Binary (< 12 MB) and Auto-Download to Your PC\n",
                "!python scripts/export.py --checkpoint checkpoints/best_model.pt --out models/microforge.bin\n",
                "\n",
                "from google.colab import files\n",
                "files.download('models/microforge.bin')\n",
                "print('[+] Download complete! You now have the trained sub-12 MB binary ready to deploy.')"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[+] Colab notebook created: {out_path}")


def package_colab_zip(zip_path: str = "microforge_colab.zip") -> None:
    import zipfile
    base = Path(".")
    include_dirs = ["microforge", "configs", "scripts", "tools_schemas"]
    include_files = ["pyproject.toml", "README.md"]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            fp = base / f
            if fp.exists():
                z.write(fp, f)
        for d in include_dirs:
            dp = base / d
            if dp.exists():
                for p in dp.rglob("*"):
                    if "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                        z.write(p, p.relative_to(base))
    print(f"[+] Packaged {zip_path} ({Path(zip_path).stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    make_colab_notebook()
    package_colab_zip()
