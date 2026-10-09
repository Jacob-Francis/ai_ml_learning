"""
Fine-tune SmolLM2-360M on 2k Alpaca examples with LoRA.

Three methods, selected with --method:
    mine : our own LoRA from lora.py
    peft : the HuggingFace `peft` library's LoRA (for the "do the curves overlap?" check)
    base : no training at all - just measure the untouched model's validation loss. This is the
           "before" baseline that the rank sweep is compared against.

Examples:
    python src/train.py --method base
    python src/train.py --method mine --rank 8
    python src/train.py --method peft --rank 8

Each run writes results/<run_name>/ with config.json, loss_history.json and the adapter weights.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, MODEL_ID, RESULTS_DIR, TARGET_MODULES
from data import build_datasets, collate, shuffled_indices
from lora import (apply_lora, count_trainable, expected_lora_params,
                  lora_state_dict)


def lm_loss_sum(model, input_ids, attention_mask, labels):
    """
    Sum (not mean) of next-token cross-entropy over all non-ignored target tokens.
    Returns (loss_sum, n_tokens). We keep the SUM so the caller can divide by the true total
    token count across several micro-batches; averaging per-micro-batch would over-weight short
    batches.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Next-token prediction: the logits at position t predict the token at position t+1.
    # So drop the last logit and the first label to line them up.
    shift_logits = logits[:, :-1, :].float()       # .float() = do the softmax in fp32 for safety
    shift_labels = labels[:, 1:]
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,                         # skip prompt + padding positions
        reduction="sum",
    )
    n_tokens = (shift_labels != -100).sum()
    return loss_sum, n_tokens


@torch.no_grad()
def evaluate(model, val_enc, pad_id, batch_size=4):
    """Average per-token loss on the held-out validation set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(val_enc), batch_size):
        ids, mask, labels = collate(val_enc[i:i + batch_size], pad_id, DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            loss_sum, n_tokens = lm_loss_sum(model, ids, mask, labels)
        total_loss += loss_sum.item()
        total_tokens += n_tokens.item()
    model.train()
    return total_loss / total_tokens


def lr_at(step, total_steps, base_lr, warmup_steps):
    """Linear warm-up for a few steps, then cosine decay down to zero."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_finetune(run_name, method, rank, alpha=16, lr=2e-4, epochs=2, micro_bs=2,
                 grad_accum=8, max_len=256, n_train=2000, n_val=200, eval_every=25,
                 seed=0, grad_ckpt=False):
    """Train one model and save everything about the run under results/<run_name>/."""
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token              # padding needs SOME token id; its labels are masked
    train_enc, val_enc, _ = build_datasets(tok, n_train, n_val, max_len, seed)
    print(f"[{run_name}] train examples: {len(train_enc)}, val examples: {len(val_enc)}")

    # Seed BEFORE building the model so the random LoRA "A" matrices are reproducible.
    torch.manual_seed(seed)
    # fp32 weights. This GPU (Turing generation) has no bf16 support, so we keep fp32 weights
    # and let autocast run the heavy matmuls in fp16.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
    model.config.use_cache = False                 # KV cache is for generation, wastes memory in training

    theory_params = expected_lora_params(model, rank, TARGET_MODULES) if method != "base" else 0

    if method == "mine":
        apply_lora(model, r=rank, alpha=alpha, target_names=TARGET_MODULES)
    elif method == "peft":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(TARGET_MODULES),
                         lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    elif method == "base":
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"unknown method {method}")

    if grad_ckpt and method != "base":
        # Recompute activations in the backward pass instead of storing them: slower, but
        # saves a lot of memory. Turn this on if you hit a CUDA out-of-memory error.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = count_trainable(model)
    print(f"[{run_name}] trainable params: {trainable:,} (theory: {theory_params:,})")

    pad_id = tok.pad_token_id
    history = []
    val0 = evaluate(model, val_enc, pad_id)
    print(f"[{run_name}] step 0 val loss: {val0:.4f}")
    history.append({"step": 0, "val_loss": val0})

    start = time.time()
    if method != "base":
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = DEVICE.type == "cuda"
        # GradScaler multiplies the loss to stop tiny fp16 gradients from rounding to zero.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        examples_per_step = micro_bs * grad_accum
        steps_per_epoch = len(train_enc) // examples_per_step
        total_steps = epochs * steps_per_epoch
        warmup = max(1, int(0.05 * total_steps))
        step = 0

        for epoch in range(epochs):
            order = shuffled_indices(len(train_enc), epoch, seed)
            for s in range(steps_per_epoch):
                # Pick this optimizer step's examples and split them into micro-batches.
                chunk = order[s * examples_per_step:(s + 1) * examples_per_step]
                micro = [collate([train_enc[j] for j in chunk[k:k + micro_bs]], pad_id, DEVICE)
                         for k in range(0, len(chunk), micro_bs)]
                # Total target tokens in the whole step, so each micro-batch loss is scaled
                # to contribute its fair share (gradient accumulation).
                step_tokens = sum((m[2][:, 1:] != -100).sum().item() for m in micro)

                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step, total_steps, lr, warmup)

                step_loss = 0.0
                for ids, mask, labels in micro:
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                        loss_sum, _ = lm_loss_sum(model, ids, mask, labels)
                    loss = loss_sum / step_tokens
                    scaler.scale(loss).backward()          # gradients ADD UP across micro-batches
                    step_loss += loss.item()

                scaler.unscale_(optimizer)                 # undo the scaling before clipping
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                record = {"step": step, "train_loss": step_loss}
                if step % eval_every == 0 or step == total_steps:
                    record["val_loss"] = evaluate(model, val_enc, pad_id)
                    print(f"[{run_name}] step {step}/{total_steps} "
                          f"train {step_loss:.4f} val {record['val_loss']:.4f}")
                history.append(record)

    elapsed = time.time() - start
    final_val = next(h["val_loss"] for h in reversed(history) if "val_loss" in h)

    # --- save everything ---
    config = {
        "run_name": run_name, "method": method, "rank": rank, "alpha": alpha, "lr": lr,
        "epochs": epochs, "micro_bs": micro_bs, "grad_accum": grad_accum, "max_len": max_len,
        "n_train": len(train_enc), "n_val": len(val_enc), "seed": seed, "model": MODEL_ID,
        "target_modules": list(TARGET_MODULES), "trainable_params": trainable,
        "theory_params": theory_params, "step0_val_loss": val0, "final_val_loss": final_val,
        "seconds": elapsed,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else None,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "loss_history.json").write_text(json.dumps(history))
    if method == "mine":
        torch.save(lora_state_dict(model), out_dir / "adapter.pt")
    elif method == "peft":
        model.save_pretrained(out_dir / "peft_adapter")
    print(f"[{run_name}] done in {elapsed / 60:.1f} min, final val loss {final_val:.4f}")

    # Free GPU memory so the next run in a sweep starts clean.
    del model
    torch.cuda.empty_cache()
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mine", "peft", "base"], required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", help="save memory, slower")
    ap.add_argument("--name", default=None, help="results folder name")
    a = ap.parse_args()

    name = a.name or ("base" if a.method == "base" else f"{a.method}_r{a.rank}")
    run_finetune(name, a.method, a.rank, alpha=a.alpha, lr=a.lr, epochs=a.epochs,
                 micro_bs=a.micro_bs, grad_accum=a.grad_accum, grad_ckpt=a.grad_ckpt)
