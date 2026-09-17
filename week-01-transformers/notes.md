# Attention Is All You Need

Its expensive to process sequences, which was traditionally done through recurrent neural networks (RNNs) and convolutional neural networks (CNNs). The Transformer model, introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017, revolutionized sequence processing by relying entirely on attention mechanisms,. That is, it does not use recurrence or convolution. Instead, it uses self-attention to weigh the importance of different parts of the input sequence when making predictions.
They may still be autoregressive 

They use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel

Since the model is not recurrent, positional information is attached through Sin cos functions. The perodicity of the sin cos functions allows the model to learn relative positions, which is important for understanding the structure of sequences and go beyound the active set of the training data in legnth.

definition:
- transduction models : Models that map input sequences to output sequences, but not through an inductive function. They are often used in tasks like machine translation, where the input and output sequences can have different lengths and structures. They do not learn a global mapping for unseen future data points outside the active set
-  residual connection: A technique used in deep learning where the output of a layer is added to the output of a previous layer, allowing the model to learn residual functions and mitigate the vanishing gradient problem.
- Causal / Look-Ahead Mask: A technique used in sequence modeling to prevent the model from attending to future tokens during training. It ensures that the prediction for a given token only depends on the tokens that come before it in the sequence, maintaining the autoregressive property of the model.
- Scaled Dot-Product Attention: A mechanism used in the Transformer model to compute attention scores between query, key, and value vectors. It involves taking the dot product of the query and key vectors, scaling the result by the square root of the dimension of the key vectors, and applying a softmax function to obtain attention weights. These weights are then used to compute a weighted sum of the value vectors, allowing the model to focus on relevant parts of the input sequence.
- BLEU score: A metric used to evaluate the quality of machine-generated translations by comparing them to one or more reference translations. It measures the overlap of n-grams between the generated and reference translations, with higher scores indicating better translation quality. BLEU scores are commonly used in natural language processing tasks, particularly in machine translation. standing for Bilingual Evaluation Understudy.


# ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING

Experimental results on various long text classification benchmark
datasets show that the enhanced transformer with rotary position embedding, namely RoFormer, can give better
performance compared to baseline alternatives and thus demonstrates the efficacy of the proposed RoPE.

In the next section, we show that our proposed RoPE
is related to this intuition from the sinusoidal function perspective. However, instead of directly adding the position
to the context representation, RoPE proposes to incorporate the relative position information by multiplying with the
sinusoidal functions.

In contrast to the additive nature of position embedding method adopted in the previous works, i.e., Equations (3)
to (10), our approach is multiplicative. Moreover, RoPE naturally incorporates relative position information through
rotation matrix product instead of altering terms in the expanded formulation of additive position encoding when applied
with self-attention.

The inner product decays with distance between the seuquence. I.e. less relevant tokens become less relevant as the distance between them increases. This is a desirable property for many tasks, such as language modeling and machine translation, where the importance of tokens often decreases with distance.




# Problems; 
- Q: Why the $1/\sqrt{d_k}$ scaling, in terms of the variance of the dot product of two random vectors
A: The scaling is necessary to resccale the variane and to prevent the softmax function from saturating, which can lead to vanishing gradients and hinder learning. By scaling the dot product by the square root of the dimension of the key vectors, we ensure that the attention scores remain in a reasonable range, allowing for more effective learning and better convergence during training.

- Q: The arithmetic and memory complexity of attention in sequence length, and where FlashAttention's speedup actually comes from (it is not fewer FLOPs)



- Q: RoPE as a rotation acting on 2D subspaces of the query/key vectors, and why relative position falls out of the inner product

A: Rope is multiplicative verse previous additive methods. It incorporates relative position information through rotation matrix product instead of altering terms in the expanded formulation of additive position encoding when applied with self-attention. The inner product decays with distance between the sequence, meaning less relevant tokens become less relevant as the distance between them increases. This is a desirable property for many tasks, such as language modeling and machine translation, where the importance of tokens often decreases with distance.

# Building a decorder only Transformer

Artichture for a decoder-only transformer is similar to the original transformer architecture, but it only includes the decoder part of the model. The decoder consists of multiple layers of self-attention and feed-forward neural networks, with residual connections and layer normalization applied at each layer. The input to the decoder is typically a sequence of tokens, which are first converted into embeddings and then processed through the self-attention mechanism to capture dependencies between tokens. The output of the decoder is a sequence of predictions, which can be used for tasks such as language modeling or text generation.