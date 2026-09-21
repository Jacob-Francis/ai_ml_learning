"""
Training script for the week-2 scaling-law sweep.

This is copy-and-adapted from week-01-transformers/src/train.py: same `run_training(...)`
contract and output files (config.json, loss_history.json, best_checkpoint.pt, model_weights.pt,
loss_curve.png, sample_generation.txt, all under results/<run_name>/), so week-1's downstream
tooling patterns (ablate.py's driver loop, explore.ipynb's "discover runs from config.json"
pattern) carry over unchanged. What's different:
  - Data: TinyStories (src/data_download.py) instead of TinyShakespeare, with an optional
    character subsample (see `subsample_chars` below) - the full ~19.4MB file makes even the
    smallest model's epochs too slow to fit 4 runs in a few hours on this machine's GPU (see
    notes.md for the actual benchmark that led to this choice).
  - The TinyStories-specific `assert len(text.splitlines()) == 40000` from week-1's version
    (which checked *that* dataset's exact line count) is removed - it doesn't apply here and
    there's no equivalent invariant worth hardcoding for this dataset.
  - The model class itself is *imported*, not copied - see the sys.path hack below - so any
    week-1 bugfix automatically applies here too, and the two weeks can never drift apart on
    what "Transformer" means.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless - we only save the plot, never show() it
import matplotlib.pyplot as plt
import torch

from data_download import download_data

# --- Fragile sibling-path import hack ---
# There's no installable package here (no pyproject.toml/setup.py for either week's src/), so the
# only way to reuse week-1's Transformer class without copy-pasting it is to point sys.path at
# week-1's src/ directory directly, by relative filesystem position. This is fragile - it breaks
# if either week's directory is ever renamed or moved relative to the other - but it's acceptable
# for a two-week personal learning repo with no packaging story; a "real" project would instead
# factor the model into its own installable package that both weeks depend on.
WEEK1_SRC = Path(__file__).resolve().parent.parent.parent / "week-01-transformers" / "src"
sys.path.insert(0, str(WEEK1_SRC))
from main import Transformer as MyTransformer  # noqa: E402 (must follow the sys.path insert)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# How much of the ~19.4MB TinyStories-valid.txt to actually train on. See notes.md for the
# benchmark this is based on: at the full file size, one epoch for the largest (d_model=200)
# config took long enough that 4 runs with early stopping risked running for many hours
# unattended on this machine's GPU. Subsampling to the first 5M characters (~4.5M in the 90%
# train split) keeps each epoch to well under a minute even for the largest model, while still
# giving every model far more training characters than it has parameters - plenty to
# differentiate model capacities, which is what this sweep is actually measuring (nobody's
# claiming these tiny models are trained to a compute-optimal frontier in the Chinchilla sense).
SUBSAMPLE_CHARS = 5_000_000


def get_batch(data, block_size, batch_size, device):
    """
    Draw `batch_size` random, independent windows of length `block_size` (+1 for the target
    shift) from `data`. Identical to week-1's version - see that file for the full explanation
    of why random windows are used instead of fixed non-overlapping chunks.

    Args:
        data: 1D LongTensor of token ids for one split (train or val) only.
        block_size: length of each input sequence.
        batch_size: how many windows to draw.
        device: device to move the resulting batch to.

    Returns:
        inputs, targets: each (batch_size, block_size), targets shifted by one position.
    """
    max_start = len(data) - block_size
    start_idxs = torch.randint(0, max_start, (batch_size,))

    inputs = torch.stack([data[i:i + block_size] for i in start_idxs])
    targets = torch.stack([data[i + 1:i + 1 + block_size] for i in start_idxs])

    return inputs.to(device), targets.to(device)


def run_training(
    run_name,
    d_model=256,
    num_layers=2,
    num_heads=4,
    block_size=128,
    positional_embedding_type='learned',
    norm_type='layernorm',
    mlp_type='gelu',
    batch_size=32,
    learning_rate=1e-3,
    train_epochs=40,
    early_stopping_patience=6,
    early_stopping_min_delta=0.0,
    eval_iters=30,
    generation_length=300,
    subsample_chars=SUBSAMPLE_CHARS,
    results_dir=None,
    device=None,
):
    """
    Run one full training run and save its results under `results_dir/run_name/`.

    Same contract as week-1's run_training (see that file's docstring for the shared args).
    New/changed args for week 2:
        subsample_chars: use only the first this-many characters of the TinyStories file for
            training/val (see SUBSAMPLE_CHARS above for why). None uses the full file.

    Returns:
        dict with run_name, final_val_loss, wall_clock_seconds, best_epoch, best_step,
        checkpoint_path - best_step (new vs week-1) is what flops.py needs to compute the FLOPs
        spent to reach the best checkpoint, since early stopping trains `patience` extra epochs
        past the best point that shouldn't be charged against it.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if results_dir is None:
        results_dir = RESULTS_DIR

    run_dir = Path(results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Run '{run_name}' | device: {device} ===")

    # --- Load data: download it (if not already present) and read as raw text ---
    data_file = download_data()
    text = data_file.read_text()
    if subsample_chars is not None:
        text = text[:subsample_chars]

    # Character-level tokenizer: vocab is just every unique character in the (possibly
    # subsampled) text - same pattern as week-1, deliberately not BPE (see the top-level task
    # notes / notes.md for the reasoning: BPE's larger vocab would eat a big chunk of the
    # smallest model's tied-embedding parameter budget, contaminating N with "vocab size" more
    # than "transformer capacity").
    chars = sorted(set(text))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: ''.join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)

    # Contiguous 90/10 split (not random per-character), same reasoning as week-1: keeps
    # validation windows from ever overlapping/leaking into training windows.
    training_split = 0.9
    split_idx = int(training_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"vocab_size: {vocab_size}")
    print(f"Training tokens: {len(train_data)}, Validation tokens: {len(val_data)}")

    train_batches_per_epoch = max(1, (len(train_data) // block_size) // batch_size)

    # --- Build model, optimizer, loss ---
    model = MyTransformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        block_size=block_size,
        positional_embedding_type=positional_embedding_type,
        norm_type=norm_type,
        mlp_type=mlp_type,
    ).to(device)

    # Live parameter count - the number that actually matters for the scaling-law fit (as
    # opposed to scaling_configs.py's hand-derived pre-training estimate). `model.parameters()`
    # correctly yields the tied head/embedding weight only once, so this doesn't double-count it.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Live parameter count: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate():
        """Average loss over `eval_iters` random validation batches, without tracking gradients."""
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(eval_iters):
                inputs, targets = get_batch(val_data, block_size, batch_size, device)
                outputs = model(inputs)
                loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                total_loss += loss.item()
        model.train()
        return total_loss / eval_iters

    train_loss_history = []  # list of (step, loss)
    val_loss_history = []  # list of (step, loss)

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    best_val_loss = float('inf')
    best_epoch = None
    best_step = None
    epochs_without_improvement = 0

    step = 0
    start_time = time.time()
    for epoch in range(train_epochs):
        model.train()
        for _ in range(train_batches_per_epoch):
            inputs, targets = get_batch(train_data, block_size, batch_size, device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss_history.append((step, loss.item()))
            step += 1

        val_loss = evaluate()
        val_loss_history.append((step, val_loss))
        elapsed = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{train_epochs} | "
            f"train loss {train_loss_history[-1][1]:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"elapsed {elapsed:.1f}s"
        )

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_step = step  # step count *at* the best checkpoint - see flops.py, this is the
                               # step used for the (N, FLOPs, loss) point in the scaling-law fit,
                               # not whatever step training eventually stops at.
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_checkpoint_path)
            print(f"  -> new best val loss, checkpoint saved to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (best val loss {best_val_loss:.4f})")
                break

    wall_clock_seconds = time.time() - start_time

    # Reload the best checkpoint, same reasoning as week-1.
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Restored best checkpoint from epoch {checkpoint['epoch'] + 1}, step {checkpoint['step']} "
        f"(val loss {checkpoint['val_loss']:.4f})"
    )

    # --- Save config, including the live parameter count (n_params) - flops.py and the
    #     scaling-law fit use this, not scaling_configs.py's pre-training estimate. ---
    config = dict(
        run_name=run_name, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        block_size=block_size, positional_embedding_type=positional_embedding_type,
        norm_type=norm_type, mlp_type=mlp_type, batch_size=batch_size,
        learning_rate=learning_rate, vocab_size=vocab_size, n_params=n_params,
        subsample_chars=subsample_chars, best_step=best_step, best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # --- Save loss history + plot ---
    loss_history_path = run_dir / "loss_history.json"
    loss_history_path.write_text(json.dumps({
        "train": train_loss_history,
        "val": val_loss_history,
    }))
    print(f"Saved raw loss history to {loss_history_path}")

    train_steps, train_vals = zip(*train_loss_history)
    val_steps, val_vals = zip(*val_loss_history)

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_vals, label="train loss", alpha=0.6)
    plt.plot(val_steps, val_vals, label="val loss", marker='o')
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss (log scale)")
    plt.title(f"Training and validation loss - {run_name} (N={n_params:,})")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = run_dir / "loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    # --- Save model weights (plain state_dict, what the notebook loads) ---
    weights_path = run_dir / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    # --- Sample generated text ---
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # seed with token id 0
    generated_ids = model.generate(context, max_new_tokens=generation_length)[0].tolist()
    generated_text = decode(generated_ids)

    print(f"\n=== Sample generation ({run_name}) ===\n")
    print(generated_text)

    sample_path = run_dir / "sample_generation.txt"
    sample_path.write_text(generated_text)
    print(f"Saved sample generation to {sample_path}")

    return {
        "run_name": run_name,
        "n_params": n_params,
        "final_val_loss": best_val_loss,
        "wall_clock_seconds": wall_clock_seconds,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "checkpoint_path": str(weights_path),
    }


if __name__ == "__main__":
    # Standalone smoke-test run: default config (d_model=256 - NOT one of the 4 scaling-sweep
    # sizes, just a quick way to sanity-check this file runs end to end). The actual sweep is
    # driven by run_all.py using scaling_configs.SCALING_RUNS.
    result = run_training(run_name="smoke_test", train_epochs=2)
    print(f"\n{result}")
