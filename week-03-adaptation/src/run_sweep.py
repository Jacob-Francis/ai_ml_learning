"""
Runs every training job for week 3, one after another:

    base        - untouched model's val loss (the "before" line on the plot)
    mine_r{r}   - our LoRA at rank r in {1, 2, 4, 8, 16, 64}   (README step 4: the rank sweep)
    peft_r8     - HuggingFace peft at rank 8                    (README step 3: curves overlap?)

Resumable: a run whose results/<name>/loss_history.json already exists is skipped, so if the
power cuts or you hit Ctrl+C you can just run this script again and it continues.

Run:  python src/run_sweep.py
"""

import torch

from config import RESULTS_DIR
from train import run_finetune

RANKS = [1, 2, 4, 8, 16, 64]
ALPHA = 16   # kept FIXED across ranks, as in the LoRA paper (so alpha/r shrinks as r grows)


def done(name: str) -> bool:
    return (RESULTS_DIR / name / "loss_history.json").exists()


jobs = [("base", "base", 0)]
jobs += [(f"mine_r{r}", "mine", r) for r in RANKS]
jobs += [("peft_r8", "peft", 8)]

for name, method, rank in jobs:
    if done(name):
        print(f"skipping {name} (already done)")
        continue
    try:
        run_finetune(name, method, rank, alpha=ALPHA)
    except torch.cuda.OutOfMemoryError:
        # Out of memory: retry once with gradient checkpointing (slower, much less memory).
        print(f"[{name}] CUDA out of memory - retrying with gradient checkpointing")
        torch.cuda.empty_cache()
        run_finetune(name, method, rank, alpha=ALPHA, micro_bs=1, grad_accum=16, grad_ckpt=True)

print("\nAll runs finished. Next: python src/plot_results.py && python src/generate.py")
