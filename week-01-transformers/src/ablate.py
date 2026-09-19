"""
Week 1, task 4: the ablation.

Four training runs, each swapping exactly one component out of a shared baseline, so each run's
difference in final validation loss / wall-clock time can be attributed to that one swap:

  baseline  - LayerNorm, learned positional embeddings, GELU MLP (4x hidden expansion)
  rmsnorm   - baseline, but RMSNorm instead of LayerNorm            (Zhang & Sennrich, 2019)
  rope      - baseline, but RoPE instead of learned positional embeddings (Su et al., 2021)
  swiglu    - baseline, but SwiGLU instead of the GELU MLP           (Shazeer, 2020)

Everything else (model size, data, optimizer, early stopping) is held fixed between runs, so this
is a true one-factor-at-a-time ablation rather than a comparison of four unrelated architectures.

Run with: python ablate.py
"""

import json
import time
from pathlib import Path

from train import run_training, RESULTS_DIR

# Shared hyperparameters for every run - only the ablated component differs between configs below.
BASE_CONFIG = dict(
    d_model=256,
    num_layers=6,
    num_heads=8,
    block_size=128,
    positional_embedding_type='learned',
    norm_type='layernorm',
    mlp_type='gelu',
    batch_size=32,
    learning_rate=1e-3,
    train_epochs=100,           # upper bound - early stopping is expected to end most runs sooner
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    eval_iters=50,
)

# Each entry overrides exactly one field of BASE_CONFIG - that's the "one swap at a time" design.
RUNS = {
    "baseline": {},
    "rmsnorm": {"norm_type": "rmsnorm"},
    "rope": {"positional_embedding_type": "rope"},
    "swiglu": {"mlp_type": "swiglu"},
}


def main():
    results = []
    ablation_start = time.time()

    for run_name, overrides in RUNS.items():
        config = {**BASE_CONFIG, **overrides}
        result = run_training(run_name=run_name, **config)
        results.append(result)

    total_elapsed = time.time() - ablation_start

    # --- Write the machine-readable table (notebook loads this) ---
    table_json_path = RESULTS_DIR / "ablation_table.json"
    table_json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {table_json_path}")

    # --- Write the human-readable table ---
    lines = [
        "# Week 1 ablation results\n",
        "One swap at a time from a shared baseline (LayerNorm + learned positional embeddings "
        "+ GELU MLP). See `ablate.py` for exact hyperparameters.\n",
        "| run | swap | final val loss | best epoch | wall-clock (s) |",
        "|---|---|---|---|---|",
    ]
    swap_labels = {
        "baseline": "(none)",
        "rmsnorm": "LayerNorm -> RMSNorm",
        "rope": "learned pos. emb. -> RoPE",
        "swiglu": "GELU MLP -> SwiGLU",
    }
    for r in results:
        lines.append(
            f"| {r['run_name']} | {swap_labels.get(r['run_name'], '?')} | "
            f"{r['final_val_loss']:.4f} | {r['best_epoch'] + 1} | "
            f"{r['wall_clock_seconds']:.1f} |"
        )
    lines.append(f"\nTotal wall-clock across all 4 runs: {total_elapsed:.1f}s")

    table_md_path = RESULTS_DIR / "ablation_table.md"
    table_md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {table_md_path}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
