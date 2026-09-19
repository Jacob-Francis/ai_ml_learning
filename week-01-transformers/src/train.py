"""
Training script for the transformer model.
This script handles the training loop, loss computation, and optimization for the transformer model
defined in `main.py`.

Script structure:
- `run_training(...)` does one full training run for a given config: builds the model, trains it
  with early stopping, saves all results (checkpoint, loss history, plot, sample text, config) to
  its own subdirectory under results/, and returns a small summary dict.
- Running this file directly (`python train.py`) does one run with the baseline config - the same
  function `ablate.py` calls four times with different configs for the week 1 ablation.
- Data is sampled with *random* windows rather than fixed non-overlapping chunks (see
  `get_batch` below) - this is more representative of what real LM training does, and avoids
  every epoch showing the model the exact same block boundaries.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless - we only save the plot, never show() it
import matplotlib.pyplot as plt
import torch
from data_download import download_data
from main import Transformer as MyTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def get_batch(data, block_size, batch_size, device):
    """
    Draw `batch_size` random, independent windows of length `block_size` (+1 for the target
    shift) from `data`.

    This replaces the old approach of reshaping the whole split into fixed, non-overlapping
    (block_size + 1)-sized chunks and shuffling *those* - here every batch samples fresh random
    start positions, so windows can overlap between batches/epochs and aren't pinned to the same
    boundaries every time. It's the same "get_batch" pattern nanoGPT uses.

    Args:
        data: 1D LongTensor of token ids for one split (train or val) only - random start
            positions are drawn within this split, so training and validation windows never
            cross the split boundary.
        block_size: length of each input sequence.
        batch_size: how many windows to draw.
        device: device to move the resulting batch to.

    Returns:
        inputs, targets: each (batch_size, block_size). targets is inputs shifted by one
        position, i.e. targets[:, t] is what should follow inputs[:, t].
    """
    # torch.randint's upper bound is exclusive, so this needs to be one past the last valid
    # start index. The last valid start index is len(data) - block_size - 1 (need block_size
    # inputs *and* one extra token for the final target) - so the exclusive bound is that + 1.
    max_start = len(data) - block_size
    start_idxs = torch.randint(0, max_start, (batch_size,))

    inputs = torch.stack([data[i:i + block_size] for i in start_idxs])
    targets = torch.stack([data[i + 1:i + 1 + block_size] for i in start_idxs])

    return inputs.to(device), targets.to(device)


def run_training(
    run_name,
    d_model=256,
    num_layers=6,
    num_heads=8,
    block_size=128,
    positional_embedding_type='learned',
    norm_type='layernorm',
    mlp_type='gelu',
    batch_size=32,
    learning_rate=1e-3,
    train_epochs=100,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    eval_iters=50,
    generation_length=500,
    results_dir=None,
    device=None,
):
    """
    Run one full training run and save its results under `results_dir/run_name/`.

    Args:
        run_name: subdirectory name under results/ for this run's outputs - also used as this
            run's label in the ablation table.
        d_model, num_layers, num_heads, block_size: model size/shape hyperparameters.
        positional_embedding_type: 'learned' or 'rope' - see main.py / mytransformer.py.
        norm_type: 'layernorm' or 'rmsnorm'.
        mlp_type: 'gelu' or 'swiglu'.
        batch_size, learning_rate: standard training hyperparameters.
        train_epochs: upper bound on epochs - early stopping is expected to end training sooner
            in most runs.
        early_stopping_patience: stop once val loss hasn't improved for this many epochs in a row.
        early_stopping_min_delta: minimum decrease in val loss that counts as "improved".
        eval_iters: number of random validation batches averaged per epoch (fixed count, not
            tied to validation split size - see the module docstring / get_batch).
        generation_length: characters to sample from the trained model at the end.
        results_dir: base results directory (defaults to the project's results/ dir).
        device: torch device (defaults to cuda if available, else cpu).

    Returns:
        dict with run_name, final_val_loss, wall_clock_seconds, best_epoch, checkpoint_path -
        the row `ablate.py` needs for the ablation table.
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
    assert len(text.splitlines()) == 40000

    # Character-level tokenizer: vocab is just every unique character in the text
    chars = sorted(set(text))  # each unique character in the text is a token
    vocab_size = len(chars)

    # Create a mapping from characters to integers and vice versa
    # stoi: string to integer, itos: integer to string
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: ''.join(itos[i] for i in ids)

    # Encode the full text as one long tensor of token ids
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split by contiguous position, not randomly - the text is one continuous sequence,
    # so a random per-character split would let validation windows overlap with (or sit
    # right next to) training windows and leak information across the split. Only the window
    # *sampling within* each split is random (see get_batch) - the split boundary itself stays
    # fixed and contiguous.
    training_split = 0.9
    split_idx = int(training_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"vocab_size: {vocab_size}")
    print(f"Training tokens: {len(train_data)}, Validation tokens: {len(val_data)}")

    # "One epoch" = this many random training batches - a fixed step count roughly matching how
    # much data one full pass over the fixed-chunking scheme used to cover, so early stopping and
    # the loss-curve x-axis stay meaningful even though batches are drawn randomly with
    # replacement rather than iterating a fixed set of chunks.
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Q: why do we use CrossEntropyLoss for this task?
    # A: at each position the model predicts a distribution over the vocabulary for the
    # next character - a multi-class classification problem - and cross-entropy is the
    # standard loss for exactly that (it's log_softmax + negative log-likelihood fused
    # into one numerically stable step).
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

    # Loss history for the loss curve: train loss is logged every step (noisy but cheap),
    # val loss only once per epoch (still an average over eval_iters batches, to keep the
    # early-stopping signal reasonably stable).
    train_loss_history = []  # list of (step, loss)
    val_loss_history = []  # list of (step, loss)

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    best_val_loss = float('inf')
    best_epoch = None
    epochs_without_improvement = 0

    # Training loop. Wall-clock timing starts here (post model construction) and stops the
    # moment the loop ends, so the 4 ablation runs are timed on the same basis - training
    # only, not data loading/model construction/plotting/IO.
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

    # Reload the best checkpoint rather than keeping whatever the last epoch produced -
    # early stopping only pays off if we actually roll back to the best point, since by
    # construction the last `early_stopping_patience` epochs were all worse than it.
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Restored best checkpoint from epoch {checkpoint['epoch'] + 1} "
        f"(val loss {checkpoint['val_loss']:.4f})"
    )

    # --- Save config, so the notebook can reconstruct this exact model without importing
    #     ablate.py or train.py ---
    config = dict(
        run_name=run_name, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        block_size=block_size, positional_embedding_type=positional_embedding_type,
        norm_type=norm_type, mlp_type=mlp_type, batch_size=batch_size,
        learning_rate=learning_rate, vocab_size=vocab_size,
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
    plt.yscale("log")  # linear scale lets a huge early spike (bad init, see note above)
                        # dwarf the y-axis and hide whether loss is still improving late in training
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss (log scale)")
    plt.title(f"Training and validation loss - {run_name}")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = run_dir / "loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    # --- Save model weights (separately from the checkpoint, as a plain state_dict - what the
    #     notebook loads) ---
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
        "final_val_loss": best_val_loss,
        "wall_clock_seconds": wall_clock_seconds,
        "best_epoch": best_epoch,
        "checkpoint_path": str(weights_path),
    }


if __name__ == "__main__":
    # Standalone run: baseline config (LayerNorm + learned positional embeddings + GELU MLP).
    # `ablate.py` calls run_training three more times with one swap each for the week 1 ablation.
    result = run_training(run_name="baseline")
    print(f"\n{result}")
