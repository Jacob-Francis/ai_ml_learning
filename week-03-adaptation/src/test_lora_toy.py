"""
Quick self-test of lora.py on a tiny made-up model. Needs only torch - no downloads, no GPU.

Run:  /home/jacob/venv/bin/python src/test_lora_toy.py

If all five checks print OK, the LoRA maths is right, and you can trust verify.py to test the
same properties on the real SmolLM2 model.
"""

import torch
import torch.nn as nn

from lora import (LoRALinear, apply_lora, count_trainable, expected_lora_params,
                  merge_lora)

TARGETS = ("q_proj", "v_proj")


class ToyAttention(nn.Module):
    """Mimics a Llama attention block: separate q/k/v/o projections (k and v are narrower)."""

    def __init__(self, d=16, d_kv=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d_kv, bias=False)
        self.v_proj = nn.Linear(d, d_kv, bias=False)
        self.o_proj = nn.Linear(d_kv, d, bias=False)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x))[..., :8] + self.v_proj(x) + self.k_proj(x))


torch.manual_seed(0)
model = ToyAttention()
x = torch.randn(4, 16)
base_out = model(x)                                   # output of the untouched model
theory = expected_lora_params(model, r=4, target_names=TARGETS)

n_patched = apply_lora(model, r=4, alpha=8, target_names=TARGETS)

# Check 1: only q_proj and v_proj were patched.
assert n_patched == 2 and isinstance(model.q_proj, LoRALinear) \
    and isinstance(model.v_proj, LoRALinear) and not isinstance(model.k_proj, LoRALinear)
print("OK 1: exactly the targeted layers were patched")

# Check 2: trainable count matches r*(in+out) summed: q: 4*(16+16)=128, v: 4*(16+8)=96 -> 224.
assert count_trainable(model) == theory == 224, (count_trainable(model), theory)
print(f"OK 2: trainable params = {count_trainable(model)} = theory")

# Check 3: at step 0 the output is BIT-IDENTICAL to the base model (torch.equal, not allclose).
assert torch.equal(model(x), base_out)
print("OK 3: step-0 output is bit-identical to the base model")

# Check 4: first backward pass - B gets gradient, A gets exactly zero gradient.
# Why: dL/dA flows through B, and B is all zeros. dL/dB flows through A, which is non-zero.
model(x).sum().backward()
assert model.q_proj.lora_A.grad.abs().max() == 0
assert model.q_proj.lora_B.grad.abs().max() > 0
assert model.q_proj.base.weight.grad is None          # frozen weights get no gradient at all
print("OK 4: at step 0, grad(A) == 0 and grad(B) != 0; frozen weights have no grad")

# Check 5: after B becomes non-zero, merging the update into W gives the same output.
with torch.no_grad():
    model.q_proj.lora_B.normal_()
    model.v_proj.lora_B.normal_()
    unmerged = model(x)
merged_out = merge_lora(model)(x)
assert torch.allclose(unmerged, merged_out, atol=1e-5)
print("OK 5: merged weights reproduce the un-merged output")
