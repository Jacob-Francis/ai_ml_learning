"""
**Build: a decoder-only transformer from scratch**

PyTorch, no `nn.Transformer`, no HuggingFace. Character-level, trained on TinyShakespeare (~1MB, trains in minutes on a T4).

1. Implement multi-head attention yourself. Unit-test it against `torch.nn.functional.scaled_dot_product_attention` — assert outputs match to within `1e-5`.
2. Build the full block: pre-norm, residuals, MLP, causal mask, weight tying.
3. Train to a sensible loss, sample some text.
4. **Ablate.** Swap LayerNorm → RMSNorm, learned positional embeddings → RoPE, GELU MLP → SwiGLU. Four training runs, one table of final validation loss and wall-clock time.

Ship: `model.py`, `train.py`, `test_attention.py`, a loss curve, an ablation table, and some generated text.

Karpathy's nanoGPT is the reference if you get stuck, but write yours first and diff afterwards.
"""

import torch
from torch import nn

from mytransformer import TransformerBlock as MyTransformerBlock

# model class

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, vocab_size, block_size, positional_embedding_type='learned'):

        super().__init__()
        self.d_model = d_model # embedding dimension
        self.num_layers = num_layers # number of transformer blocks
        self.num_heads = num_heads # number of attention heads in the multi-head attention
        self.vocab_size = vocab_size # size of the vocabulary (number of unique tokens)
        self.block_size = block_size # maximum sequence length (context size)
        self.positional_embedding_type = positional_embedding_type

        # Embedding layers: from Claude
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # nn.Embedding's default init has std=1.0. That's fine for a standalone embedding,
        # but this embedding is weight-tied to the output head below, so std=1.0 also
        # becomes the output projection's weight scale - about 28x larger than a Linear's
        # usual default init (~1/sqrt(d_model)). That blew up init logits and made the
        # first-step loss ~159 instead of the ~ln(vocab_size) a random model should give.
        # GPT-2 sidesteps this the same way: rescale the tied embedding to a small std.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # RoPE is task 4's ablation swap for this learned embedding - it rotates q/k
        # inside attention instead of adding a lookup table here, so it isn't a drop-in
        # yet (TransformerBlock's rope branch is still a stub). Fail loudly rather than
        # silently training a model with no positional information.
        if positional_embedding_type == 'rope':
            raise NotImplementedError(
                "RoPE isn't implemented yet (task 4 ablation) - use positional_embedding_type='learned'"
            )

        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([MyTransformerBlock(d_model, num_heads) for _ in range(num_layers)])

        # Final layer normalization and output projection
        self.ln_f = nn.LayerNorm(d_model)
        # bias=False: head.weight is tied to token_embedding.weight below, but a bias
        # has no embedding-side counterpart - it'd just be a free per-token logit offset
        # unrelated to the tying. Standard practice (GPT-2/nanoGPT) drops it on a tied head.
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection instead of
        # learning a second (vocab_size, d_model)-shaped matrix. Shapes already line up -
        # nn.Embedding.weight is (vocab_size, d_model) and nn.Linear.weight is
        # (out_features, in_features) = (vocab_size, d_model) - so no transpose needed.
        self.head.weight = self.token_embedding.weight

        # softmax is applied in the loss function (F.cross_entropy fuses log_softmax + NLL
        # in one numerically stable step), not here - see train.py. Softmax is only applied
        # explicitly later, at sampling/generation time, when an actual probability
        # distribution is needed to draw from.

    def forward(self, x, mask=None):
        """
        Forward pass through the transformer model.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing token indices.
            mask: Optional causal attention mask of shape (seq_length, seq_length). If not
                given, one is built automatically below (every block shares the same mask).

        Architecture:
        1. Token and positional embeddings are summed to form the input embeddings.
        2. The input embeddings are passed through a series of transformer blocks, each consisting of
           multi-head self-attention, layer normalization, and a feed-forward network.
           Each block is pre-norm with residual connections (see TransformerBlock.forward), and
           every block attends under the same causal mask so no position can see the future.
        3. The output of the last transformer block is normalized and projected to the vocabulary size
           to produce logits for each token in the vocabulary.

        Returns:
            logits: Output tensor of shape (batch_size, seq_length, vocab_size) containing logits for each token in the vocabulary.
        """

        # Notes:  we employ residual connections
        # around each of the sub-layers, followed by layer normalization. We also modify the self-attention
        # sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
        # masking, combined with fact that the output embeddings are offset by one position, ensures that the
        # predictions for position i can depend only on the known outputs at positions less than i.

        batch_size, seq_length = x.size()

        # 1. Token and (learned) positional embeddings summed to form the input embeddings.
        #    RoPE would replace this addition entirely and instead rotate q/k inside
        #    attention - see the NotImplementedError note in __init__.
        positions = torch.arange(seq_length, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Causal mask: position i may only attend to positions <= i. Lower-triangular
        # matrix of ones; TransformerBlock's attention fills scores with -inf wherever
        # mask == 0 (the upper triangle) before the softmax, so future positions get
        # zero attention weight.
        if mask is None:
            mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).bool()

        # 2. The input embeddings are passed through a series of transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # 3. The output of the last transformer block is normalized and projected
        #    to the vocabulary size to produce logits (no softmax - see the note in
        #    __init__ about where softmax is actually applied).
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressively sample max_new_tokens new tokens and append them to idx.

        Args:
            idx: (batch_size, seq_length) starting token ids to condition on.
            max_new_tokens: how many new tokens to sample.
            temperature: scales logits before softmax; >1 flattens the distribution
                (more random), <1 sharpens it (more greedy).

        Returns:
            (batch_size, seq_length + max_new_tokens) token ids.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # position_embedding only has block_size rows, so the model can't see
            # further back than that - crop the context to its last block_size tokens.
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only need next-token logits
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        self.train(was_training)
        return idx