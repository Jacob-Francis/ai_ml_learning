"""
LoRA written from scratch (Hu et al., 2021, arXiv:2106.09685).

The idea in one line: instead of changing a big weight matrix W, freeze it and learn a small
low-rank correction next to it.

    h = W x  +  (alpha / r) * B (A x)

    W : (out, in)  frozen pretrained weight
    A : (r, in)    trainable, random init   -> "squeezes" the input down to r numbers
    B : (out, r)   trainable, ZERO init     -> "expands" those r numbers back to out numbers
    r : the rank, much smaller than in/out (we sweep 1..64)

Because B starts at zero, B(Ax) = 0 at step 0, so the patched model is IDENTICAL to the base
model before any training. That is the property verify.py checks.

This file only needs torch (no transformers), so it can be tested on its own - see
test_lora_toy.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps one nn.Linear and adds a trainable low-rank update beside it."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base                 # the original frozen layer, kept untouched
        self.r = r                       # rank of the update
        self.alpha = alpha               # LoRA scaling numerator (kept fixed while r changes)
        self.scaling = alpha / r         # the update is multiplied by alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # Adapter weights are always fp32, on the same device as the base weight. Keeping them
        # in fp32 means the optimizer works on full-precision numbers even if autocast runs the
        # matmuls in fp16.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=torch.float32))

        # A gets the same random init as nn.Linear's default (and as HF peft), so our runs and
        # the peft runs start from the same kind of distribution.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays all zeros (created with torch.zeros above) - this is the "zero-init" trick.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path 1: the normal frozen layer.
        frozen_out = self.base(x)
        # Path 2: the low-rank update. F.linear(x, A) computes x @ A.T  -> shape (..., r)
        # and F.linear(., B) computes . @ B.T -> shape (..., out). Using F.linear (not a plain
        # matmul) lets torch.autocast run these in fp16 on the GPU.
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        # Add them. When B == 0 the update is exactly 0.0, and x + 0.0 == x bit for bit.
        return frozen_out + update

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """
        Fold the update into the base weight: W' = W + (alpha/r) * B @ A.
        After merging there is no extra compute at inference time - this is LoRA's
        "no added latency" claim.
        """
        self.base.weight += (self.scaling * (self.lora_B @ self.lora_A)).to(self.base.weight.dtype)
        return self.base


def apply_lora(model: nn.Module, r: int, alpha: float, target_names) -> int:
    """
    Freeze every parameter of `model`, then swap each nn.Linear whose attribute name is in
    `target_names` (e.g. "q_proj") for a LoRALinear wrapping it.

    Order matters: we freeze FIRST. The new lora_A / lora_B parameters are created afterwards,
    so they are the only things left with requires_grad=True.

    Returns the number of layers that were patched.
    """
    # Step 1: freeze the whole base model.
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: find the layers to patch. We collect them into a list first because we are about
    # to modify the module tree, and you should not change a structure while iterating over it.
    to_patch = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                to_patch.append((parent, child_name, child))

    # Step 3: replace each one with a LoRA-wrapped version.
    for parent, child_name, child in to_patch:
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha))

    return len(to_patch)


def expected_lora_params(model: nn.Module, r: int, target_names) -> int:
    """
    The theoretical trainable-parameter count, computed BEFORE patching.

    Each patched Linear (in -> out) adds A (r x in) plus B (out x r) = r * (in + out) numbers.
    For a square d x d layer that is 2 * r * d, which is the formula in the paper.
    (Our model has grouped-query attention, so k_proj and v_proj are NOT square - hence the
    general in + out form.)
    """
    total = 0
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                total += r * (child.in_features + child.out_features)
    return total


def count_trainable(model: nn.Module) -> int:
    """Number of parameters that will actually receive gradient updates."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors (a few MB), not the 360M frozen base weights."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if "lora_" in name}


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear in the model with its merged plain nn.Linear (in place)."""
    to_merge = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, LoRALinear):
                to_merge.append((parent, child_name, child))
    for parent, child_name, child in to_merge:
        setattr(parent, child_name, child.merged_linear())
    return model
