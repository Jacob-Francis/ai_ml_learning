"""
Data prep: turn Alpaca-cleaned rows into token ids the model can train on.

Each example becomes:   [prompt tokens][response tokens][EOS]
and we only compute loss on the response part. The prompt is the question; we do not want to
reward the model for "predicting" the question, only for answering it. We do that by setting the
label of every prompt token to -100, which PyTorch's cross-entropy ignores.
"""

import random

from datasets import load_dataset

from config import DATASET_ID

# The classic Alpaca prompt format. Two versions: with and without an extra "input" field.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further "
    "context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(example: dict) -> str:
    """Build the prompt text (everything BEFORE the answer) for one dataset row."""
    if example["input"].strip():
        return PROMPT_WITH_INPUT.format(instruction=example["instruction"], input=example["input"])
    return PROMPT_NO_INPUT.format(instruction=example["instruction"])


def build_datasets(tokenizer, n_train=2000, n_val=200, max_len=256, seed=0):
    """
    Load Alpaca-cleaned, shuffle it with a fixed seed, and return:
        train_enc: list of dicts {"input_ids": [...], "labels": [...]}
        val_enc:   same format, held-out rows (never trained on)
        val_raw:   the raw text rows for the val set (used by generate.py for before/after demos)

    The fixed seed means every run in the rank sweep sees exactly the same train/val split.
    """
    raw = load_dataset(DATASET_ID, split="train")
    raw = raw.shuffle(seed=seed)

    eos_id = tokenizer.eos_token_id
    train_enc, val_enc, val_raw = [], [], []

    for row in raw:
        # Stop as soon as both splits are full.
        if len(train_enc) >= n_train and len(val_enc) >= n_val:
            break

        prompt_ids = tokenizer(format_prompt(row), add_special_tokens=False)["input_ids"]
        # Skip rows where the prompt alone fills the window - no room left for an answer.
        if len(prompt_ids) >= max_len - 8:
            continue

        # The answer, plus an EOS token so the model learns WHEN TO STOP talking.
        resp_ids = tokenizer(row["output"], add_special_tokens=False)["input_ids"] + [eos_id]

        input_ids = (prompt_ids + resp_ids)[:max_len]                 # cut to the max length
        labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]      # -100 = "ignore this token"
        example = {"input_ids": input_ids, "labels": labels}

        # Fill the training split first, then the validation split.
        if len(train_enc) < n_train:
            train_enc.append(example)
        elif len(val_enc) < n_val:
            val_enc.append(example)
            val_raw.append(dict(row))

    return train_enc, val_enc, val_raw


def collate(batch, pad_id, device):
    """
    Pad a list of examples to the same length and stack them into tensors.
    Right-padding: shorter sequences get pad tokens on the end. Padded label positions are -100
    so they never count towards the loss, and attention_mask=0 tells the model to ignore them.
    """
    import torch

    longest = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        pad = longest - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        attn.append([1] * len(ex["input_ids"]) + [0] * pad)

    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attn, device=device),
        torch.tensor(labels, device=device),
    )


def shuffled_indices(n: int, epoch: int, seed: int):
    """
    A reproducible shuffle of range(n) for a given epoch. Uses its OWN random generator so that
    it does not depend on how much the global torch RNG has been used - this is what guarantees
    that "my LoRA" and "peft" runs see the exact same batches in the exact same order.
    """
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx
