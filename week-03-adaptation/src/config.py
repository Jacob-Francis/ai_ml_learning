"""
Shared constants for week 3. Every other script imports from here so that the model name,
the target layers and the folder paths are defined in exactly one place.
"""

from pathlib import Path

import torch

# The base model we adapt. SmolLM2-360M is a Llama-style decoder-only model (32 layers,
# hidden size 960). We pick it over Qwen3-0.6B because this machine has a 4GB GPU (Quadro T1000):
# 360M parameters in fp32 is ~1.4GB, which leaves room for activations. 0.6B would not.
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

# Which Linear layers get a LoRA adapter. These are the names of the four attention projections
# inside every Llama-style attention block. The original LoRA paper only adapted q and v;
# QLoRA later found that adapting ALL linear layers matches full fine-tuning better. We adapt
# all four attention projections, which is what the week-3 README asks for ("patches the
# attention projections").
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

# The instruction dataset (cleaned version of the Stanford Alpaca data, ~52k examples).
# The README says to use 1-2k examples, so data.py takes a small slice.
DATASET_ID = "yahma/alpaca-cleaned"

# Project folders, resolved relative to this file so scripts work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Use the GPU if there is one, otherwise the CPU (CPU works but is very slow for a 360M model).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
