"""
Turn the saved runs into the week-3 deliverables:

  results/rank_sweep.png       trainable params vs validation loss  (THE deliverable)
  results/mine_vs_peft.png     our LoRA vs HuggingFace peft at rank 8 (do the curves overlap?)
  results/rank_sweep_table.md  the numbers behind the plot, plus gain per doubling of rank

Run:  python src/plot_results.py      (no GPU needed)
"""

import json

import matplotlib
matplotlib.use("Agg")                      # headless: save to file, never open a window
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


def load(name):
    """Return (config, history) for one run, or (None, None) if it has not been run yet."""
    d = RESULTS_DIR / name
    if not (d / "config.json").exists():
        return None, None
    return json.loads((d / "config.json").read_text()), json.loads((d / "loss_history.json").read_text())


def smooth(y, k=9):
    """Moving average so the noisy per-step training loss is readable."""
    y = np.asarray(y, dtype=float)
    kernel = np.ones(k) / k
    padded = np.pad(y, (k // 2, k // 2), mode="edge")   # edge-pad so the output keeps its length
    return np.convolve(padded, kernel, mode="valid")


# ---------- 1. rank sweep ----------
base_cfg, _ = load("base")
runs = []
for r in [1, 2, 4, 8, 16, 64]:
    cfg, _ = load(f"mine_r{r}")
    if cfg:
        runs.append(cfg)

if runs:
    params = [c["trainable_params"] for c in runs]
    losses = [c["final_val_loss"] for c in runs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(params, losses, "o-", color="tab:blue", label="LoRA (our implementation)")
    for c in runs:                                         # label each point with its rank
        ax.annotate(f"r={c['rank']}", (c["trainable_params"], c["final_val_loss"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    if base_cfg:
        ax.axhline(base_cfg["final_val_loss"], color="gray", ls="--", label="base model (no tuning)")
    ax.set_xscale("log")                                   # ranks double, so use a log x-axis
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title("SmolLM2-360M + LoRA on 2k Alpaca examples: rank sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rank_sweep.png", dpi=150)
    plt.close(fig)

    # Table with the gain from each doubling of rank. "Where returns flatten" = where the
    # gain column gets close to zero (or to the noise level of the run-to-run seed variation).
    lines = ["| rank | trainable params | final val loss | gain vs previous row (first row: vs base) | minutes |",
             "|---|---|---|---|---|"]
    prev = base_cfg["final_val_loss"] if base_cfg else None
    for c in runs:
        gain = "" if prev is None else f"{prev - c['final_val_loss']:+.4f}"
        lines.append(f"| {c['rank']} | {c['trainable_params']:,} | {c['final_val_loss']:.4f} "
                     f"| {gain} | {c['seconds'] / 60:.1f} |")
        prev = c["final_val_loss"]
    if base_cfg:
        lines.append(f"\nBase model val loss (before tuning): {base_cfg['final_val_loss']:.4f}")
    (RESULTS_DIR / "rank_sweep_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
else:
    print("No mine_r* runs found yet - run src/run_sweep.py first.")

# ---------- 2. mine vs peft ----------
mine_cfg, mine_hist = load("mine_r8")
peft_cfg, peft_hist = load("peft_r8")
if mine_hist and peft_hist:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for hist, label, color in [(mine_hist, "ours", "tab:blue"), (peft_hist, "HF peft", "tab:orange")]:
        steps = [h["step"] for h in hist if "train_loss" in h]
        ax1.plot(steps, smooth([h["train_loss"] for h in hist if "train_loss" in h]),
                 label=label, color=color)
        vs = [(h["step"], h["val_loss"]) for h in hist if "val_loss" in h]
        ax2.plot(*zip(*vs), "o-", label=label, color=color)
    ax1.set_title("training loss (moving average)")
    ax2.set_title("validation loss")
    for ax in (ax1, ax2):
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("rank 8: our LoRA vs HuggingFace peft")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mine_vs_peft.png", dpi=150)
    plt.close(fig)
    print(f"\nfinal val loss  ours: {mine_cfg['final_val_loss']:.4f}   "
          f"peft: {peft_cfg['final_val_loss']:.4f}   "
          f"difference: {abs(mine_cfg['final_val_loss'] - peft_cfg['final_val_loss']):.4f}")
