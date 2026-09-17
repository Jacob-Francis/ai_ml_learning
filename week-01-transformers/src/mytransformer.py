"""
Building my own transformer using torch

Recall the structure of a transformer block:
1. Multi-head self-attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-forward network (MLP)
4. Add & Norm (residual connection + layer normalization)

Each block itself will be repeated `num_layers` times to form the full transformer model.

input: (batch_size, seq_length) - token indices
output: (batch_size, seq_length, vocab_size) - logits for each token in the vocabulary

Within each attention head, we will compute the attention scores using the scaled dot-product 
attention mechanism. The output of each head will be concatenated and projected back to the 
original embedding dimension. 

Attention structure:
1. Compute queries, keys, and values using linear projections of the input embeddings.
2. Compute attention scores by taking the dot product of queries and keys, scaling by the square
    root of the embedding dimension, and applying a softmax to obtain attention weights.
3. Multiply the attention weights by the values to obtain the output of the attention mechanism.
"""

# TransformerBlock
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, positional_embedding_type=None):
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.num_heads = num_heads # number of attention heads in the multi-head attention
        self.d_k = d_model // num_heads # dimension of queries and keys
        self.d_v = d_model // num_heads # dimension of values

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        for i in range(num_heads):
            setattr(self, f'q_linear_{i}', nn.Linear(d_model, self.d_k, bias=False))
            setattr(self, f'k_linear_{i}', nn.Linear(d_model, self.d_k, bias=False))
            setattr(self, f'v_linear_{i}', nn.Linear(d_model, self.d_v, bias=False)) 

        self.concat_linear = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)  # pre-attention norm
        self.ln2 = nn.LayerNorm(d_model)  # pre-FFN norm; separate params from ln1

        # Feed forward bits
        self.FF1 = nn.Linear(d_model, d_model, bias=True)
        self.relu = nn.ReLU()
        self.FF2 = nn.Linear(d_model, d_model, bias=True)

        if positional_embedding_type == 'rope':
            # To Do
            pass

    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, d_model)
        batch_size, seq_length, d_model = x.size()

        # Multi-head self-attention (Pre-LN: normalize before the sublayer,
        # then add the *un-normalized* x back as the residual)
        x_attained = self.multi_head_attention(self.ln1(x), mask=mask)
        x = x + x_attained  # residual connection

        # Feed-forward network (MLP), Pre-LN
        x_ff = self.FF1(self.ln2(x))
        x_ff = self.relu(x_ff)
        x_ff = self.FF2(x_ff)
        x = x + x_ff  # residual connection

        return x

    def multi_head_attention(self, x, mask=None):
        """
        Compute the scaled dot-product attention, with multi head parallelization, an the full multi
        head attention mechanism, with ocncatenation. 
        The input x is projected to queries, keys, and values, and the attention scores 
        are computed using the scaled dot-product attention mechanism.

        Args: 
            x : Input tensor of shape (batch_size, seq_length, d_model)
        Intermidates:
            q: Queries of shape (batch_size, num_heads, seq_length, d_k)
            k: Keys of shape (batch_size, num_heads, seq_length, d_k)
            v: Values of shape (batch_size, num_heads, seq_length, d_v)
            mask: Optional mask of shape (batch_size, 1, seq_length, seq_length)

        Returns:
            output: The result of the attention mechanism.
            attention_weights: The attention weights.
        """
        batch_size, seq_length, d_model = x.size()

        # linearly project the input x to queries, keys, and values
        q = torch.zeros(batch_size, seq_length, self.num_heads, self.d_k, device=x.device, dtype=x.dtype)
        k = torch.zeros(batch_size, seq_length, self.num_heads, self.d_k, device=x.device, dtype=x.dtype)
        v = torch.zeros(batch_size, seq_length, self.num_heads, self.d_v, device=x.device, dtype=x.dtype)
        for i in range(self.num_heads):
            q[:, :, i, :] = getattr(self, f'q_linear_{i}')(x) # shape: (batch_size, seq_length, num_heads, d_k)
            k[:, :, i, :] = getattr(self, f'k_linear_{i}')(x) # shape: (batch_size, seq_length, num_heads, d_k)
            v[:, :, i, :] = getattr(self, f'v_linear_{i}')(x) # shape: (batch_size, seq_length, num_heads, d_v)

        # move heads next to batch so matmul batches over (batch, heads) independently
        q = q.transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)
        k = k.transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)
        v = v.transpose(1, 2) # (batch_size, num_heads, seq_length, d_v)

        output = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # concat the heads and project back to the original embedding dimension
        # transpose heads back next to seq_length before flattening, so each position's
        # heads end up contiguous in memory (a plain .view without this would silently
        # scramble the data instead of concatenating heads per position)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.concat_linear(output)

        return output

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute the scaled dot-product attention using PyTorch's built-in function.
        This is used for testing against the custom implementation.

        Args:
            q: Queries of shape (batch_size, num_heads, seq_length, d_k)
            k: Keys of shape (batch_size, num_heads, seq_length, d_k)
            v: Values of shape (batch_size, num_heads, seq_length, d_v)
            mask: Optional mask of shape (batch_size, 1, seq_length, seq_length)
        """

        # attain; dot product over d_k between every query/key pair, batched per (batch, head)
        # shape: (batch_size, num_heads, seq_length, seq_length)
        q_dot_k = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            q_dot_k = q_dot_k.masked_fill(mask == 0, float('-inf'))

        # softmax to get attention weights: shape (batch_size, num_heads, seq_length, seq_length)
        attention_weights = torch.softmax(q_dot_k, dim=-1)

        # apply values: shape (batch_size, num_heads, seq_length, d_v)
        output = torch.matmul(attention_weights, v)

        return output