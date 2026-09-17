"""
Training script for the transformer model.
This script handles the training loop, loss computation, and optimization for the transformer model
defined in `main.py`.

Script strucutre:
- Initialize the model, optimizer, and loss function.
- Load the training data.
- Train the model for a specified number of epochs.
- Plot the loss curve and save the trained model.
- Sample some generated text from the trained model.
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
from torch.utils.data import DataLoader

# Hyper parameters
batch_size = 32
train_epochs = 10
training_split = 0.9  # fraction of the text used for training; rest is validation
block_size = 128  # maximum sequence length (context size)
learning_rate = 1e-3
generation_length = 500  # characters to sample from the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data: download it (if not already present) and read as raw text
data_file = download_data()
text = data_file.read_text()

assert len(text.splitlines()) == 40000

# Character-level tokenizer: vocab is just every unique character in the text
chars = sorted(set(text)) # each unique character in the text is a token
vocab_size = len(chars)

print('vocab size:', vocab_size)

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
# right next to) training windows and leak information across the split.
split_idx = int(training_split * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"vocab_size: {vocab_size}")
print(f"Training tokens: {len(train_data)}, Validation tokens: {len(val_data)}")

# load into dataloader: chop each split into non-overlapping windows of block_size+1
# tokens (the +1 is so each window yields both the block_size inputs and the
# block_size targets, offset by one), then batch those windows up.
trimmed_train_data = train_data[:len(train_data) - (len(train_data) % (block_size + 1))]
train_dataloader = DataLoader(trimmed_train_data.view(-1, block_size + 1), batch_size=batch_size, shuffle=True)

trimmed_val_data = val_data[:len(val_data) - (len(val_data) % (block_size + 1))]
val_dataloader = DataLoader(trimmed_val_data.view(-1, block_size + 1), batch_size=batch_size, shuffle=False)

# load in optimizer and loss function
model = MyTransformer(
    d_model=256,
    num_layers=6,
    num_heads=8,
    vocab_size=vocab_size,
    block_size=block_size,
    positional_embedding_type='learned'
    ).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Q: why do we use CrossEntropyLoss for this task?
# A: at each position the model predicts a distribution over the vocabulary for the
# next character - a multi-class classification problem - and cross-entropy is the
# standard loss for exactly that (it's log_softmax + negative log-likelihood fused
# into one numerically stable step).
loss_fn = torch.nn.CrossEntropyLoss()


def evaluate(dataloader):
    """Average loss over a dataloader, without tracking gradients."""
    model.eval()
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches


# Loss history for the loss curve: train loss is logged every step (noisy but cheap),
# val loss only once per epoch (expensive - a full pass over val_dataloader).
train_loss_history = []  # list of (step, loss)
val_loss_history = []  # list of (step, loss)

# Training loop
step = 0
start_time = time.time()
for epoch in range(train_epochs):
    model.train()
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # We remove the last element from the inputs because that's what we're
        # predicting: inputs[i] should predict targets[i] == inputs[i+1].
        # shapes: (batch_size, block_size + 1) -> (batch_size, block_size)
        inputs, targets = batch[:, :-1], batch[:, 1:]  # shift inputs and targets by one
        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        train_loss_history.append((step, loss.item()))
        step += 1

    val_loss = evaluate(val_dataloader)
    val_loss_history.append((step, val_loss))
    elapsed = time.time() - start_time
    print(
        f"epoch {epoch + 1}/{train_epochs} | "
        f"train loss {train_loss_history[-1][1]:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"elapsed {elapsed:.1f}s"
    )

# --- Save loss history + plot ---
loss_history_path = RESULTS_DIR / "loss_history.json"
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
plt.xlabel("training step")
plt.ylabel("cross-entropy loss")
plt.title("Training and validation loss")
plt.legend()
plt.tight_layout()
loss_plot_path = RESULTS_DIR / "loss_curve.png"
plt.savefig(loss_plot_path)
print(f"Saved loss curve to {loss_plot_path}")

# --- Save model weights ---
weights_path = RESULTS_DIR / "model_weights.pt"
torch.save(model.state_dict(), weights_path)
print(f"Saved model weights to {weights_path}")

# --- Sample generated text ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # seed with token id 0
generated_ids = model.generate(context, max_new_tokens=generation_length)[0].tolist()
generated_text = decode(generated_ids)

print("\n=== Sample generation ===\n")
print(generated_text)

sample_path = RESULTS_DIR / "sample_generation.txt"
sample_path.write_text(generated_text)
print(f"\nSaved sample generation to {sample_path}")
