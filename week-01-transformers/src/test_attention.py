"""
1. Implement multi-head attention yourself. Unit-test it against
`torch.nn.functional.scaled_dot_product_attention` — assert outputs match to within `1e-5`.
"""

from torch import nn
import torch
from mytransformer import TransformerBlock, rope_cos_sin, apply_rope

myclass = TransformerBlock(d_model=16, num_heads=4)
myattention = myclass.scaled_dot_product_attention

# True attention function from PyTorch
trueattention = nn.functional.scaled_dot_product_attention

# Generate random input tensors for queries, keys, and values
batch_size = 2
seq_length = 5
d_k = 4  # dimension of keys and queries
d_v = 4  # dimension of values

k = torch.rand(batch_size, 4, seq_length, d_k)  # (batch_size, num_heads, seq_length, d_k)
q = torch.rand(batch_size, 4, seq_length, d_k)  # (batch_size, num_heads, seq_length, d_k)
v = torch.rand(batch_size, 4, seq_length, d_v)  # (batch_size, num_heads, seq_length

true_output = trueattention(q, k, v, attn_mask=None)
my_output = myattention(q, k, v, mask=None)

assert torch.allclose(true_output, my_output, atol=1e-5), "Outputs do not match!"


"""
4. RoPE sanity checks (Su et al., 2021).

RoPE rotates each query/key vector, so it should be norm-preserving (a rotation never changes a
vector's length) - a cheap regression check against a sign or pairing bug in apply_rope. It
should also give relative-position invariance: the dot product of a query rotated to position i
with a key rotated to position j should depend only on (i - j), not on i and j individually -
that's the entire point of using a multiplicative rotation instead of an additive embedding.
"""

rope_q = torch.randn(batch_size, 4, seq_length, d_k)
rope_cos, rope_sin = rope_cos_sin(seq_length, d_k, rope_q.device, rope_q.dtype)
rope_q_rotated = apply_rope(rope_q, rope_cos, rope_sin)

assert torch.allclose(rope_q.norm(dim=-1), rope_q_rotated.norm(dim=-1), atol=1e-5), (
    "RoPE should preserve vector norm (it's a rotation) - got a length change instead"
)


def _rotate_at_position(x, position, seq_len, d_k):
    """Rotate a single vector x as if it were at the given position."""
    cos, sin = rope_cos_sin(seq_len, d_k, x.device, x.dtype)
    # apply_rope already unsqueezes its cos/sin args from (seq, d_k/2) to (1, 1, seq, d_k/2) -
    # pass the (1, d_k/2) position slice straight through, don't unsqueeze it again here.
    return apply_rope(x, cos[position:position + 1], sin[position:position + 1])


_q_content = torch.randn(1, 1, 1, d_k)
_k_content = torch.randn(1, 1, 1, d_k)
_dots = [
    (_rotate_at_position(_q_content, i, 50, d_k) * _rotate_at_position(_k_content, j, 50, d_k)).sum().item()
    for i, j in [(5, 2), (10, 7), (30, 27)]  # all pairs share the same relative offset i - j == 3
]
assert max(_dots) - min(_dots) < 1e-3, (
    f"RoPE dot products at the same relative offset should match closely, got {_dots}"
)

print("All attention + RoPE checks passed.")
