"""
README step 5: ten before/after completions.

For 10 held-out prompts (never trained on) it generates an answer from
  - BEFORE: the untouched base model
  - AFTER:  the base model + a trained LoRA adapter
and writes everything to results/before_after.md, ready to paste into the README.

Run:  python src/generate.py                 (uses the rank-8 adapter)
      python src/generate.py --rank 16
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, MODEL_ID, RESULTS_DIR, TARGET_MODULES
from data import build_datasets, format_prompt
from lora import apply_lora

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--max-new-tokens", type=int, default=120)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# The same val split every training run used (fixed seed) - so these prompts were never trained on.
_, _, val_raw = build_datasets(tok)
examples = val_raw[:args.n]

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()


def complete(prompt: str) -> str:
    """Greedy decoding (always pick the most likely token) so before/after differences come
    from the weights, not from random sampling luck."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    # Keep only the newly generated tokens (drop the prompt we fed in).
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---- BEFORE: base model ----
before = [complete(format_prompt(ex)) for ex in examples]

# ---- AFTER: patch in LoRA layers and load the trained adapter weights ----
apply_lora(model, r=args.rank, alpha=16, target_names=TARGET_MODULES)
adapter = torch.load(RESULTS_DIR / f"mine_r{args.rank}" / "adapter.pt")
missing = model.load_state_dict(adapter, strict=False)
# strict=False because the adapter file holds ONLY the lora_* tensors; every base weight is
# (correctly) reported as "missing" - but no lora_* key should be missing or unexpected.
assert not missing.unexpected_keys and not any("lora_" in k for k in missing.missing_keys)
# Adapter tensors were saved in fp32; cast the model's LoRA params to match the fp16 base.
model.half()
after = [complete(format_prompt(ex)) for ex in examples]

# ---- write the report ----
lines = [f"# Before / after: base SmolLM2-360M vs LoRA rank {args.rank}\n",
         "Greedy decoding, held-out prompts, max "
         f"{args.max_new_tokens} new tokens.\n"]
for i, (ex, b, a) in enumerate(zip(examples, before, after), 1):
    inp = f"\n**Input:** {ex['input']}" if ex["input"].strip() else ""
    lines += [f"---\n\n## {i}. {ex['instruction']}{inp}\n",
              f"**Reference answer:** {ex['output']}\n",
              f"**Before (base):** {b}\n",
              f"**After (LoRA r={args.rank}):** {a}\n"]
out_path = RESULTS_DIR / "before_after.md"
out_path.write_text("\n".join(lines))
print(f"wrote {out_path}")
