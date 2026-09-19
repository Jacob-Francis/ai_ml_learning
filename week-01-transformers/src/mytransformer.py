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

Ablation switches (week 1, task 4): every block can be built with a choice of
- norm_type: 'layernorm' (original Transformer) or 'rmsnorm' (Zhang & Sennrich 2019)
- positional_embedding_type: 'learned' (a lookup table added in main.py) or 'rope'
  (Su et al. 2021 - rotates q/k inside attention instead, see apply_rope below)
- mlp_type: 'gelu' (a standard 4x-widened MLP) or 'swiglu' (Shazeer 2020)
"""

# TransformerBlock
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Ordinary LayerNorm centers *and* rescales its input (subtracts the mean, divides by the
    standard deviation, then applies a learned scale and shift). RMSNorm's finding is that the
    re-centering step doesn't actually matter for why LayerNorm helps training - just rescaling
    by the root-mean-square is enough. So this only divides by RMS(x) and applies a learned
    per-channel scale; there's no mean-subtraction and no bias term at all.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps  # small constant so we never divide by ~0 on an all-zero input
        # learned per-channel scale, initialised to 1 so RMSNorm starts as a no-op (before
        # training nudges it), same spirit as LayerNorm's default weight=1, bias=0 init
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (..., d_model). Mean of squares over the last dim, then sqrt -> the "RMS" of x.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def make_norm(d_model, norm_type):
    """Factory so block-level and final norms are built the same way from one switch."""
    if norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    if norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm_type {norm_type!r}, expected 'layernorm' or 'rmsnorm'")


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward variant (Shazeer, 2020, "GLU Variants Improve Transformer").

    A standard transformer MLP is Linear -> activation -> Linear. GLU variants instead compute
    *two* linear projections of the input - a "gate" and a "value" (here called `up`, since it's
    the one that gets gated rather than the one that does the gating) - multiply them together
    elementwise, then project back down. The gate passes through an activation first (here SiLU,
    i.e. swish, hence "Swi"-GLU) so the gate values are a smooth 0..~1-ish mask that lets the
    network learn to suppress or pass through different channels of `up(x)` per input.

    This needs three weight matrices instead of two, so to keep the parameter count comparable
    to a plain 4x-widened MLP, `d_ff` here is conventionally shrunk to about 2/3 of `4 * d_model`
    (see swiglu_hidden_dim below) rather than also being 4x. All three linears drop the bias
    term - standard practice for GLU-variant FFNs, since the gate already provides a learned
    per-channel offset-like effect.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)  # decides how much of `up` to let through
        self.up = nn.Linear(d_model, d_ff, bias=False)  # the value being gated
        self.down = nn.Linear(d_ff, d_model, bias=False)  # projects back to d_model
        self.silu = nn.SiLU()  # swish(x) = x * sigmoid(x)

    def forward(self, x):
        return self.down(self.silu(self.gate(x)) * self.up(x))


def swiglu_hidden_dim(d_model, multiple_of=32):
    """
    Shazeer scales SwiGLU's hidden dim down to ~2/3 of the usual 4x expansion, so that adding a
    third weight matrix (gate, on top of up/down) doesn't give SwiGLU more trainable parameters
    than the plain GELU MLP it's being compared against - otherwise a better ablation result
    could just be "more parameters" rather than "a better MLP design".

    (4 * d_model) * (2/3) simplifies to (8/3) * d_model - unrounded, that's an exact parameter
    match. We then round up to a multiple of 32 purely so matmuls hit clean tile-friendly shapes
    - a minor efficiency nicety, not required for correctness - but it does cost exactness: at
    d_model=256 this rounds 682.67 up to 704, giving SwiGLU ~3% more parameters than the GELU
    baseline (540,672 vs 524,288). Worth keeping in mind when reading the ablation table - a
    sliver of any swiglu improvement could be the extra capacity rather than the architecture.
    """
    raw = int(8 * d_model / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


def rope_cos_sin(seq_length, d_k, device, dtype):
    """
    Precompute the cos/sin tables used by RoPE (Su et al., 2021), one pair of values per
    (position, dimension-pair).

    RoPE's idea: instead of adding a positional vector to the token embedding (the 'learned'
    approach), rotate each query/key vector by an angle that depends on its position. Nearby
    positions get rotated by similar angles; a query dotted with a key naturally then reflects
    their *relative* position (a rotation composed with the inverse of another rotation - see
    notes.md - leaves only the angle difference, i.e. the relative offset).

    d_k dimensions are treated as d_k/2 independent 2D planes; each plane i gets its own rotation
    frequency, geometrically spaced between 1 and 1/10000 as i goes from 0 to d_k/2 (identical
    frequency scheme to the original sinusoidal position encoding in Vaswani et al.). Low-index
    planes rotate fast (sensitive to short-range position), high-index planes rotate slowly
    (sensitive to long-range position).

    Returns:
        cos, sin: each (seq_length, d_k/2) - angle cosine/sine for every (position, plane) pair.
    """
    # one angular frequency per dimension-pair (plane), d_k/2 of them total
    i = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (i / d_k))  # (d_k/2,)

    positions = torch.arange(seq_length, device=device, dtype=torch.float32)  # (seq_length,)

    # outer product: angle for every (position, plane) combination
    freqs = torch.outer(positions, inv_freq)  # (seq_length, d_k/2)

    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """
    Rotate a query/key tensor by the RoPE angles in `cos`/`sin`.

    x: (batch_size, num_heads, seq_length, d_k). We treat consecutive dimension pairs
    (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ... as 2D vectors and rotate each one by its
    own angle - this is the original RoFormer pairing scheme (adjacent dims), not the
    "rotate-half" variant popularised later by LLaMA, since Su et al. 2021 is the paper this
    ablation is implementing.

    Standard 2D rotation by angle theta:
        x1' = x1 * cos(theta) - x2 * sin(theta)
        x2' = x1 * sin(theta) + x2 * cos(theta)
    """
    x1 = x[..., 0::2]  # even dims of each pair: (batch, num_heads, seq_length, d_k/2)
    x2 = x[..., 1::2]  # odd dims of each pair:  (batch, num_heads, seq_length, d_k/2)

    # broadcast cos/sin (seq_length, d_k/2) over the batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, d_k/2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, d_k/2)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # interleave the rotated pairs back into the original (..., d_k) layout:
    # stacking on a new last axis then flattening gives [rot1[0], rot2[0], rot1[1], rot2[1], ...]
    # which is exactly the original (even, odd) interleaving x1/x2 were split out of.
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(start_dim=-2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, positional_embedding_type='learned',
                 norm_type='layernorm', mlp_type='gelu'):
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.num_heads = num_heads # number of attention heads in the multi-head attention
        self.d_k = d_model // num_heads # dimension of queries and keys
        self.d_v = d_model // num_heads # dimension of values
        self.positional_embedding_type = positional_embedding_type  # 'learned' or 'rope'

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        if positional_embedding_type == 'rope':
            # RoPE pairs up adjacent dimensions (2i, 2i+1) within each head - needs an even
            # head dim to pair up cleanly, silently wrong (drops the last dim) otherwise.
            assert self.d_k % 2 == 0, f"RoPE needs an even head dim (d_k), got {self.d_k}"

        for i in range(num_heads):
            setattr(self, f'q_linear_{i}', nn.Linear(d_model, self.d_k, bias=False))
            setattr(self, f'k_linear_{i}', nn.Linear(d_model, self.d_k, bias=False))
            setattr(self, f'v_linear_{i}', nn.Linear(d_model, self.d_v, bias=False))

        self.concat_linear = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.norm1 = make_norm(d_model, norm_type)  # pre-attention norm
        self.norm2 = make_norm(d_model, norm_type)  # pre-FFN norm; separate params from norm1

        # Feed forward bits. 'gelu' is a standard transformer MLP widened to 4x d_model in the
        # hidden layer (Vaswani et al.'s d_ff = 4 * d_model) before projecting back down;
        # 'swiglu' is the Shazeer 2020 gated variant, sized to roughly match gelu's param count
        # (see swiglu_hidden_dim). Either way `self.mlp` is a single callable so forward() below
        # doesn't need to know which variant it is.
        if mlp_type == 'gelu':
            d_ff = 4 * d_model
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=True),
                nn.GELU(),
                nn.Linear(d_ff, d_model, bias=True),
            )
        elif mlp_type == 'swiglu':
            self.mlp = SwiGLU(d_model, swiglu_hidden_dim(d_model))
        else:
            raise ValueError(f"unknown mlp_type {mlp_type!r}, expected 'gelu' or 'swiglu'")

    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, d_model)
        batch_size, seq_length, d_model = x.size()

        # Multi-head self-attention (Pre-LN: normalize before the sublayer,
        # then add the *un-normalized* x back as the residual)
        x_attained = self.multi_head_attention(self.norm1(x), mask=mask)
        x = x + x_attained  # residual connection

        # Feed-forward network (MLP), Pre-LN
        x_ff = self.mlp(self.norm2(x))
        x = x + x_ff  # residual connection

        return x

    def multi_head_attention(self, x, mask=None):
        """
        Compute the scaled dot-product attention, with multi head parallelization, and the full multi
        head attention mechanism, with concatenation.
        The input x is projected to queries, keys, and values, and the attention scores
        are computed using the scaled dot-product attention mechanism.

        Args:
            x : Input tensor of shape (batch_size, seq_length, d_model)
        Intermediates:
            q: Queries of shape (batch_size, num_heads, seq_length, d_k)
            k: Keys of shape (batch_size, num_heads, seq_length, d_k)
            v: Values of shape (batch_size, num_heads, seq_length, d_v)
            mask: Optional mask of shape (batch_size, 1, seq_length, seq_length)

        Returns:
            output: The result of the attention mechanism, projected back to d_model.
            (attention weights aren't returned directly - they're cached as a side effect on
            self.last_attention_weights inside scaled_dot_product_attention, for inspection
            e.g. in the notebook's attention-heatmap cells.)
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

        # RoPE: rotate q and k (never v - v carries content, not "where am I") by an angle that
        # depends on their position, before the dot product. This is the ablation swap for the
        # learned positional embedding added in main.py - with 'rope', that embedding table
        # isn't built at all, and positional information is injected here instead.
        if self.positional_embedding_type == 'rope':
            cos, sin = rope_cos_sin(seq_length, self.d_k, x.device, x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

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
        Hand-rolled scaled dot-product attention (the point of task 1) - this is the
        implementation, not a call to PyTorch's built-in. test_attention.py checks this
        method's output against PyTorch's actual built-in, torch.nn.functional
        .scaled_dot_product_attention, to within 1e-5.

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

        # Cached for inspection (e.g. plotting attention heatmaps in a notebook) -
        # purely a side effect, doesn't affect the forward pass or its output.
        self.last_attention_weights = attention_weights.detach()

        # apply values: shape (batch_size, num_heads, seq_length, d_v)
        output = torch.matmul(attention_weights, v)

        return output
