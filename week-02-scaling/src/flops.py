"""
Week 2, task 4: FLOPs accounting.

No changes to the training loop are needed for any of this - every number here is derivable
after the fact from what train.py already logs (config.json's n_params, loss_history.json's
(step, loss) pairs) plus the batch/block size the run used. This module is pure post-processing.

--- Deriving flops_at_step = 6*N*tokens ---

The standard estimate (used throughout the scaling-laws literature, e.g. Kaplan et al. 2020
section 2.1) is that one forward+backward pass over one training token costs ~6*N FLOPs, where N
is the model's (non-embedding, in Kaplan's version, but we use the full tied-parameter count -
see notes.md for why that distinction barely matters here) parameter count. Where the 6 comes
from:
  - A forward pass through a matmul with W parameters costs ~2*W FLOPs per token (one multiply +
    one add per weight, i.e. each parameter is used in one multiply-accumulate) - so a forward
    pass over the whole model costs ~2*N FLOPs/token.
  - Backward pass costs about 2x the forward pass (gradients w.r.t. both the layer's inputs *and*
    its weights each cost about as much as the forward multiply did) - so backward is ~4*N
    FLOPs/token.
  - Total: forward (2N) + backward (4N) = 6*N FLOPs per token of training.
This ignores attention's O(seq_length^2) term (subdominant for these small block_size=128 models
where D >> block_size isn't quite true, but it's the standard simplification the whole scaling-
laws literature makes, and keeping it lets our alpha be compared on the same basis as Kaplan's).
"""

import json
from pathlib import Path


def count_params(model):
    """Live parameter count of a constructed model - dedupes tied weights automatically, since
    `model.parameters()` yields each underlying nn.Parameter object once no matter how many
    module attributes point at it (e.g. main.py's `head.weight = token_embedding.weight`)."""
    return sum(p.numel() for p in model.parameters())


def tokens_at_step(step, batch_size, block_size):
    """How many training tokens have been consumed by the time `step` optimizer steps have run.
    Each step processes one batch of batch_size sequences, each block_size tokens long."""
    return step * batch_size * block_size


def flops_at_step(n_params, step, batch_size, block_size):
    """Estimated total training FLOPs spent by the time `step` optimizer steps have run -
    see the module docstring for the 6*N*tokens derivation."""
    return 6 * n_params * tokens_at_step(step, batch_size, block_size)


def load_run(run_dir):
    """
    Load one run's config + loss history and annotate every logged point with its FLOPs.

    Args:
        run_dir: path to results/<run_name>/ (must contain config.json and loss_history.json,
            both written by train.py's run_training).

    Returns:
        dict with:
          run_name, n_params, batch_size, block_size,
          train: list of (step, loss, flops) - every logged training step,
          val:   list of (step, loss, flops) - every logged validation point (once per epoch),
          best:  (step, loss, flops) for the *early-stopping-best* checkpoint specifically - this
                 is the single point that should go into the power-law fit, not the final/last
                 val point, because early stopping deliberately keeps training for
                 `early_stopping_patience` more epochs *after* the best point to confirm it
                 really was the best - charging that extra compute against the best loss would
                 overstate how much FLOPs it actually took to reach that loss.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "loss_history.json").read_text())

    n_params = config["n_params"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]

    def annotate(points):
        return [
            (step, loss, flops_at_step(n_params, step, batch_size, block_size))
            for step, loss in points
        ]

    train_annotated = annotate(history["train"])
    val_annotated = annotate(history["val"])

    # best_step/best_val_loss were recorded directly by train.py at the moment the checkpoint
    # was saved - reading them back out (rather than re-deriving "the val point with min loss")
    # keeps this in lockstep with whatever train.py's early-stopping logic actually decided,
    # even if that logic changes later.
    best_step = config["best_step"]
    best_val_loss = config["best_val_loss"]
    best_flops = flops_at_step(n_params, best_step, batch_size, block_size)

    return {
        "run_name": config["run_name"],
        "n_params": n_params,
        "batch_size": batch_size,
        "block_size": block_size,
        "train": train_annotated,
        "val": val_annotated,
        "best": (best_step, best_val_loss, best_flops),
    }


if __name__ == "__main__":
    # Quick manual check against whatever runs already exist under results/.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    run_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and (d / "config.json").exists())
    if not run_dirs:
        print(f"No runs found under {results_dir} yet - run run_all.py first.")
    for run_dir in run_dirs:
        r = load_run(run_dir)
        best_step, best_loss, best_flops = r["best"]
        print(f"{r['run_name']:<12} N={r['n_params']:>10,}  best_step={best_step:>6}  "
              f"best_val_loss={best_loss:.4f}  flops_at_best={best_flops:.3e}")
