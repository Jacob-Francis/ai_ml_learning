"""
Week 2, task 5: run the full 4-config scaling sweep end to end.

Mirrors week-1's ablate.py driver pattern (loop over configs, call run_training, collect results,
write a machine-readable + human-readable table) but pulls its configs from scaling_configs.py
instead of a one-swap-at-a-time dict, and additionally records FLOPs-at-best (via flops.py) in
the output table, since that's the whole point of this week's build task.

Run with: python run_all.py
This actually trains all 4 models (no dry-run mode) - expect this to take a while (see notes.md
for the timing this was budgeted against). Each run's full results land under
results/<run_name>/ exactly as train.py's run_training describes; this script additionally writes
results/scaling_table.json and results/scaling_table.md summarizing all 4 runs together.
"""

import json
import time
from pathlib import Path

from flops import load_run
from scaling_configs import SCALING_RUNS
from train import run_training, RESULTS_DIR


def main():
    results = []
    sweep_start = time.time()

    for cfg in SCALING_RUNS:
        # cfg carries `target_n` for bookkeeping/planning purposes only - run_training doesn't
        # take that as an argument, so it's stripped out before passing the rest through.
        cfg = dict(cfg)
        target_n = cfg.pop("target_n")
        run_name = cfg["run_name"]

        print(f"\n{'=' * 70}\nStarting {run_name} (target N={target_n:,})\n{'=' * 70}")
        result = run_training(**cfg)
        result["target_n"] = target_n
        results.append(result)

    total_elapsed = time.time() - sweep_start

    # --- Pull FLOPs-at-best for every run, now that all 4 are trained ---
    for result in results:
        run_dir = RESULTS_DIR / result["run_name"]
        loaded = load_run(run_dir)
        best_step, best_val_loss, best_flops = loaded["best"]
        result["best_step"] = best_step
        result["flops_at_best"] = best_flops

    # --- Write the machine-readable table (fit_scaling_law.py and the notebook load this) ---
    table_json_path = RESULTS_DIR / "scaling_table.json"
    table_json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {table_json_path}")

    # --- Write the human-readable table ---
    lines = [
        "# Week 2 scaling-sweep results\n",
        "4 models, only d_model varied (num_layers=2, num_heads=4, block_size=128 fixed - see "
        "scaling_configs.py). FLOPs-at-best uses the step of each run's early-stopping-best "
        "checkpoint, not total steps trained (see flops.py).\n",
        "| run | target N | achieved N | final val loss | FLOPs at best | best step | wall-clock (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['run_name']} | {r['target_n']:,} | {r['n_params']:,} | "
            f"{r['final_val_loss']:.4f} | {r['flops_at_best']:.3e} | {r['best_step']:,} | "
            f"{r['wall_clock_seconds']:.1f} |"
        )
    lines.append(f"\nTotal wall-clock across all 4 runs: {total_elapsed:.1f}s "
                 f"({total_elapsed / 60:.1f} min)")

    table_md_path = RESULTS_DIR / "scaling_table.md"
    table_md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {table_md_path}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
