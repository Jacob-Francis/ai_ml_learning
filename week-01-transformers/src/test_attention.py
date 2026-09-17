"""
1. Implement multi-head attention yourself. Unit-test it against 
`torch.nn.functional.scaled_dot_product_attention` — assert outputs match to within `1e-5`.
"""

from torch import nn
import torch
from mytransformer import TransformerBlock

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
