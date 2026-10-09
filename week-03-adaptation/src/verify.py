"""
README step 2: verify our LoRA on the REAL model before spending hours training.

Checks:
  1. Trainable parameter count == theory  r * (in + out) summed over patched layers.
  2. Only adapter parameters are trainable (the base model is frozen).
  3. At step 0, logits are BIT-IDENTICAL to the unpatched base model (zero-init of B).
  4. After the first backward pass: grad(A) == 0 and grad(B) != 0 (explained below).
  5. Our trainable count equals the HuggingFace peft count for the same config.

Run:  python src/verify.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, MODEL_ID, TARGET_MODULES
from lora import LoRALinear, apply_lora, count_trainable, expected_lora_params

RANK, ALPHA = 8, 16


def check(ok: bool, msg: str):
    print(("✅ " if ok else "❌ ") + msg)
    assert ok, msg


tok = AutoTokenizer.from_pretrained(MODEL_ID)
# fp32 and NO autocast here: bit-identical comparison only makes sense with identical arithmetic.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE).eval()

if tok.pad_token is None:
    tok.pad_token = tok.eos_token          # batching two prompts needs a padding token
batch = tok(["The capital of France is", "LoRA freezes the base weights and learns"],
            return_tensors="pt", padding=True).to(DEVICE)

# --- Step A: record the base model's output BEFORE patching ---
with torch.no_grad():
    base_logits = model(**batch).logits.clone()

theory = expected_lora_params(model, RANK, TARGET_MODULES)

# --- Step B: patch with our LoRA ---
n_layers = apply_lora(model, r=RANK, alpha=ALPHA, target_names=TARGET_MODULES)
print(f"patched {n_layers} linear layers with rank {RANK}")

# Check 1: parameter count equals theory.
actual = count_trainable(model)
check(actual == theory, f"trainable params {actual:,} == theory {theory:,}")

# Check 2: every trainable parameter is an adapter weight; every base weight is frozen.
bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
check(len(bad) == 0, "all trainable params are lora_A / lora_B (base model is frozen)")
check(sum(isinstance(m, LoRALinear) for m in model.modules()) == n_layers,
      "number of LoRALinear modules == number of patched layers")

# Check 3: bit-identical outputs at step 0.
with torch.no_grad():
    lora_logits = model(**batch).logits
check(torch.equal(base_logits, lora_logits),
      "step-0 logits are bit-identical to the base model (B is zero)")

# Check 4: gradient pattern on the first backward pass.
# Why grad(A) is zero: A's gradient is multiplied by B, and B is all zeros.
# Why grad(B) is not: B's gradient is multiplied by A x, which is non-zero.
# So the very first optimizer step only moves B; after that B != 0 and A starts to learn too.
model.train()
out = model(**batch).logits
out.float().logsumexp(-1).mean().backward()       # any scalar that depends on the logits will do
a_grad = max(p.grad.abs().max().item() for n, p in model.named_parameters() if "lora_A" in n)
b_grad = max(p.grad.abs().max().item() for n, p in model.named_parameters() if "lora_B" in n)
check(a_grad == 0.0, f"max |grad(A)| at step 0 is {a_grad} (expected exactly 0)")
check(b_grad > 0.0, f"max |grad(B)| at step 0 is {b_grad:.3e} (expected > 0)")

# Check 5: compare with the HuggingFace peft library (built on CPU to save GPU memory).
from peft import LoraConfig, get_peft_model
ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
ref = get_peft_model(ref, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=list(TARGET_MODULES),
                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
check(count_trainable(ref) == actual, f"peft trainable params {count_trainable(ref):,} == ours")

print("\nAll checks passed. Safe to start training.")
