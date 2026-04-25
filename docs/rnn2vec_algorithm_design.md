# RNN2Vec Algorithm Design

**Author:** Donald (Performance Specialist)
**Date:** 2026-04-21
**Status:** Planning document — no code changes

## Overview

RNN2Vec replaces the skip-gram training component of Tree2Vec with a recurrent neural network that processes tree walk token sequences directly. Walk generation (TreeWalker, WalkConfig) is **reused unchanged** — the RNN replaces only `SkipGramTrainer`. This document covers the algorithm design, training objectives, pure-Python implementation plan, and performance feasibility.

---

## 1. Input Representation

### Vocabulary Building

Reuse the existing vocabulary discovery from walk sequences (same as `SkipGramTrainer._build_vocab`):

```
Token types from walks.py:
  VAR, VAR_1, VAR_2, ...        (variables, with optional identity)
  CONST:<symnum>                 (constants)
  FUNC:<symnum>/<arity>          (function symbols)
  LIT:+FUNC:<symnum>/<arity>    (literal wrappers)
  CLAUSE:<n>                     (clause-level tokens)
  SIGN:+, SIGN:-                 (sign markers when skip_predicate_wrapper=True)
  PATHLEN:<n>                    (path length markers)
```

For the vampire.in domain, vocabulary is typically 6-15 tokens. Larger problems (e.g., lattice theory) may reach 30-60 tokens.

### Learnable Token Embeddings

**Embedding matrix E**: `vocab_size x input_dim`

- `input_dim` is the size of learnable per-token vectors fed into the RNN.
- Distinct from `hidden_dim` (RNN internal state) and `embedding_dim` (final output).
- **Default: `input_dim = 32`** — sufficient for small vocabularies; larger adds parameters without benefit.

Initialization: uniform random in `[-0.5/input_dim, +0.5/input_dim]` (same scale as current skip-gram).

### OOV Handling

For tokens not in vocabulary during online updates:
1. **Extend vocab**: Add new row to E initialized to the mean of all existing embeddings (mirrors `SkipGramTrainer._extend_vocab`).
2. **UNK fallback**: Map unknown tokens to a shared `<UNK>` embedding (simpler, no vocab growth).

**Recommendation:** Extend vocab (option 1) — consistent with existing Tree2Vec online behavior and avoids information loss.

---

## 2. RNN Cell Options

### Vanilla RNN

```
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
```

| Property | Value |
|----------|-------|
| Parameters per layer | H^2 + input_dim*H + H |
| Gradient behavior | Severe vanishing/exploding gradients |
| Memory | Lowest |
| Compute per step | O(H^2 + input_dim*H) |

**Not recommended.** Gradient issues make it unsuitable for sequences longer than ~10 tokens. Many clause walks exceed this (depth-first walks of complex terms).

### GRU (Gated Recurrent Unit)

```
z_t = sigmoid(W_z @ [h_{t-1}, x_t])     # update gate
r_t = sigmoid(W_r @ [h_{t-1}, x_t])     # reset gate
h_candidate = tanh(W_h @ [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate
```

| Property | Value |
|----------|-------|
| Parameters per layer | 3 * (H^2 + input_dim*H + H) |
| Gradient behavior | Good (gated) |
| Memory | Moderate (no cell state) |
| Compute per step | O(3 * (H^2 + input_dim*H)) |

**Recommended default.** Best parameter-efficiency-to-quality ratio. ~33% fewer parameters than LSTM. For small vocabularies and short sequences (typical walk length 5-30 tokens), GRU matches LSTM quality with less compute.

### LSTM (Long Short-Term Memory)

```
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   # forget gate
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   # input gate
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)      # candidate
c_t = f_t * c_{t-1} + i_t * g_t              # cell update
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    # output gate
h_t = o_t * tanh(c_t)
```

| Property | Value |
|----------|-------|
| Parameters per layer | 4 * (H^2 + input_dim*H + H) |
| Gradient behavior | Best (cell state highway) |
| Memory | Highest (h_t and c_t) |
| Compute per step | O(4 * (H^2 + input_dim*H)) |

**Use when:** Sequences are long (>30 tokens), or when capturing very long-range dependencies matters (e.g., relating root functor to deep leaf variables).

### Default Recommendation

**GRU with `hidden_dim=64`** as the default. Rationale:
- Walk sequences are typically 5-30 tokens — within GRU's effective range.
- 33% fewer parameters than LSTM: critical for pure-Python performance.
- Empirically matches LSTM on short sequences in NLP benchmarks.
- The `rnn_type` config parameter allows switching to LSTM if needed.

---

## 3. Training Objectives

### Option A: Autoencoder (Sequence Reconstruction)

**Encoder:** RNN processes walk sequence → final hidden state h_T
**Decoder:** Second RNN reconstructs input sequence from h_T

```
Loss = sum_t CrossEntropy(decoded_t, input_t)
```

| Aspect | Assessment |
|--------|------------|
| Quality | Good general-purpose embeddings |
| Compute | 2x RNN cost (encoder + decoder) |
| Implementation | Complex (teacher forcing, decoder loop) |
| Online learning | Straightforward (reconstruct new walks) |

### Option B: Next-Token Prediction (Language Model)

**Forward:** RNN predicts next token at each step.

```
Loss = sum_t CrossEntropy(W_out @ h_t, token_{t+1})
```

| Aspect | Assessment |
|--------|------------|
| Quality | Captures sequential structure well |
| Compute | 1x RNN + output projection |
| Implementation | Simple |
| Online learning | Natural (predict on new walks) |

### Option C: Contrastive Learning

**Positive pairs:** Walks from the same clause/term.
**Negative pairs:** Walks from different clauses.

```
Loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

| Aspect | Assessment |
|--------|------------|
| Quality | Best for clause **selection** (directly optimizes similarity structure) |
| Compute | 1x RNN + pairwise similarity |
| Implementation | Moderate (sampling strategy matters) |
| Online learning | Natural (new clauses provide new pairs) |

### Recommendation: Contrastive with Next-Token Auxiliary

**Primary: Contrastive loss.** This directly optimizes the property we need for clause selection — structurally similar clauses should have similar embeddings, dissimilar ones should be far apart. The existing `EmbeddingEnhancedSelection` relies on cosine distance for diversity scoring, which aligns perfectly with contrastive training objectives.

**Auxiliary: Next-token prediction (weight 0.1-0.3).** This regularizes the hidden states to capture sequential structure, improving the representation even when contrastive pairs are scarce (early in search).

Combined loss:
```
L = L_contrastive + λ * L_next_token    (λ = 0.2 default)
```

**Why not autoencoder:** The 2x compute cost is prohibitive in pure Python, and reconstruction quality doesn't directly help selection.

---

## 4. Pure Python RNN Implementation Plan

### 4.1 GRU Cell Math (Default)

All operations use `list[list[float]]` for weight matrices and `list[float]` for vectors.

```python
def gru_cell(x_t: list[float], h_prev: list[float],
             W_z: list[list[float]], U_z: list[list[float]], b_z: list[float],
             W_r: list[list[float]], U_r: list[list[float]], b_r: list[float],
             W_h: list[list[float]], U_h: list[list[float]], b_h: list[float],
             ) -> list[float]:
    # Update gate: z = σ(W_z @ x + U_z @ h + b_z)
    z = sigmoid_vec(add_vec(add_vec(matvec(W_z, x_t), matvec(U_z, h_prev)), b_z))
    # Reset gate: r = σ(W_r @ x + U_r @ h + b_r)
    r = sigmoid_vec(add_vec(add_vec(matvec(W_r, x_t), matvec(U_r, h_prev)), b_r))
    # Candidate: h_cand = tanh(W_h @ x + U_h @ (r * h) + b_h)
    rh = elementwise_mul(r, h_prev)
    h_cand = tanh_vec(add_vec(add_vec(matvec(W_h, x_t), matvec(U_h, rh)), b_h))
    # Output: h_new = (1-z)*h + z*h_cand
    h_new = add_vec(elementwise_mul(one_minus(z), h_prev),
                    elementwise_mul(z, h_cand))
    return h_new
```

### 4.2 LSTM Cell Math (Alternative)

```python
def lstm_cell(x_t, h_prev, c_prev,
              W_f, U_f, b_f,  # forget gate
              W_i, U_i, b_i,  # input gate
              W_g, U_g, b_g,  # candidate
              W_o, U_o, b_o,  # output gate
              ) -> tuple[list[float], list[float]]:
    f = sigmoid_vec(add_vec(add_vec(matvec(W_f, x_t), matvec(U_f, h_prev)), b_f))
    i = sigmoid_vec(add_vec(add_vec(matvec(W_i, x_t), matvec(U_i, h_prev)), b_i))
    g = tanh_vec(add_vec(add_vec(matvec(W_g, x_t), matvec(U_g, h_prev)), b_g))
    o = sigmoid_vec(add_vec(add_vec(matvec(W_o, x_t), matvec(U_o, h_prev)), b_o))
    c_new = add_vec(elementwise_mul(f, c_prev), elementwise_mul(i, g))
    h_new = elementwise_mul(o, tanh_vec(c_new))
    return h_new, c_new
```

### 4.3 Forward Pass Loop

```python
def rnn_forward(sequence: list[int], embeddings: list[list[float]],
                cell_fn, cell_params) -> list[list[float]]:
    """Process a token sequence, returning all hidden states."""
    h = [0.0] * hidden_dim  # zero init
    c = [0.0] * hidden_dim  # LSTM only
    hidden_states = []

    for token_id in sequence:
        x_t = embeddings[token_id]  # lookup: O(1)
        if is_lstm:
            h, c = cell_fn(x_t, h, c, *cell_params)
        else:
            h = cell_fn(x_t, h, *cell_params)
        hidden_states.append(list(h))  # copy for BPTT

    return hidden_states
```

### 4.4 Backpropagation Through Time (BPTT)

Full BPTT with optional truncation for long sequences:

```python
def bptt(hidden_states, targets, loss_fn, cell_params, lr,
         max_bptt_steps=30):
    """Truncated BPTT: backpropagate gradients through time."""
    T = len(hidden_states)
    # Compute output loss gradient at each timestep
    d_h = [None] * T

    # For contrastive loss: gradient only at final state
    # For next-token: gradient at each step
    d_h[T-1] = loss_gradient_at(T-1)

    # Backpropagate through time (truncated)
    for t in range(T-1, max(T-1-max_bptt_steps, -1), -1):
        # Compute gate gradients
        # Update weight gradients
        # Propagate d_h[t] -> d_h[t-1] through cell Jacobian
        ...

    # Apply accumulated gradients with learning rate
    apply_gradients(cell_params, lr)
```

**Truncation:** Default `max_bptt_steps=30`. Walks rarely exceed this, so truncation seldom activates. When it does, it bounds computation without significant quality loss.

### 4.5 Primitive Operations

All matrix/vector operations implemented as pure Python loops:

```python
def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector multiply: O(rows * cols)."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

def sigmoid_vec(v: list[float]) -> list[float]:
    return [1.0 / (1.0 + math.exp(-min(max(x, -6.0), 6.0))) for x in v]

def tanh_vec(v: list[float]) -> list[float]:
    return [math.tanh(x) for x in v]

def elementwise_mul(a: list[float], b: list[float]) -> list[float]:
    return [a[i] * b[i] for i in range(len(a))]

def add_vec(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(len(a))]
```

---

## 5. Sequence Composition Strategies

After the RNN processes a walk sequence, we must reduce the sequence of hidden states `[h_1, h_2, ..., h_T]` into a single fixed-size embedding vector.

### 5.1 Last Hidden State

```
embedding = h_T
```

- **Pros:** Simplest, standard in seq2seq. No extra parameters.
- **Cons:** Recency bias — later tokens dominate. For depth-first walks, this means leaf nodes have disproportionate influence.
- **Dim:** `hidden_dim`

### 5.2 Mean Pooling

```
embedding = (1/T) * Σ_t h_t
```

- **Pros:** Equal weighting, robust to sequence length variation.
- **Cons:** Dilutes signal from structurally important positions (root functor, predicate symbol).
- **Dim:** `hidden_dim`

### 5.3 Attention Pooling (Recommended)

```
score_t = w^T @ tanh(W_a @ h_t + b_a)   # attention score
alpha_t = softmax(score_t)                # attention weight
embedding = Σ_t alpha_t * h_t             # weighted sum
```

- **Pros:** Learns which positions matter for selection quality. Can focus on root structure or leaf patterns adaptively.
- **Cons:** Additional parameters (W_a: hidden_dim x hidden_dim, w: hidden_dim, b_a: hidden_dim).
- **Dim:** `hidden_dim`
- **Extra cost:** O(T * H^2) for attention computation — marginal vs. RNN cost.

### 5.4 Recommendation

**Default: Mean pooling.** Rationale:
- Zero extra parameters, simplest implementation.
- Attention pooling adds O(H^2) parameters that need sufficient training data to learn well.
- For the typical 6-15 token vocabulary, mean pooling captures the distribution of token-level hidden states effectively.
- Config parameter allows switching to `"last"`, `"mean"`, or `"attention"`.

This mirrors Tree2Vec's composition parameter (`"mean"`, `"weighted_depth"`, `"root_concat"`).

---

## 6. Online Learning Compatibility

The existing `Tree2Vec.update_online()` → `SkipGramTrainer.update_online()` pattern must be preserved.

### 6.1 Mini-Batch SGD Updates

```python
def update_online(self, walks, learning_rate=None):
    lr = learning_rate or self.config.min_learning_rate

    # Convert walks to ID sequences (extending vocab if needed)
    id_sequences = self._encode_walks(walks)

    # For contrastive: pair walks from same clause as positives
    positive_pairs = self._make_contrastive_pairs(id_sequences)

    # Mini-batch gradient update
    for batch in chunks(positive_pairs, batch_size=32):
        loss = self._contrastive_loss(batch)
        self._backward_and_step(loss, lr)

    return {"pairs_trained": len(positive_pairs), "loss": avg_loss}
```

### 6.2 OOV Handling

Mirrors existing `SkipGramTrainer` approach:
1. New tokens get a new row in embedding matrix E (mean-initialized).
2. RNN weights are NOT extended (they operate on fixed `input_dim` vectors).
3. Only the embedding lookup table grows.

This is cleaner than skip-gram OOV handling because the RNN weight matrices don't depend on vocabulary size.

### 6.3 Vocab Extension

```python
def _extend_vocab(self, token: str) -> int:
    tid = self._vocab_size
    self._token_to_id[token] = tid
    self._id_to_token.append(token)
    self._vocab_size += 1

    # Mean-initialized embedding
    mean = [sum(row[i] for row in self._token_embeddings) / len(self._token_embeddings)
            for i in range(self.config.input_dim)]
    self._token_embeddings.append(mean)
    return tid
```

### 6.4 Model Version Bumping

After online updates, call `provider.bump_model_version()` to lazily invalidate cached embeddings — identical to Tree2Vec provider pattern.

---

## 7. Performance Feasibility

### 7.1 Skip-Gram Cost (Baseline)

Per training pair:
- 1 dot product: O(D) where D = embedding_dim
- K negative samples, each: O(D) dot + O(D) gradient update
- Total per pair: O((K+1) * D)

With defaults (D=64, K=5): ~384 float ops per pair.

For a typical vampire.in corpus:
- ~200 walks, ~15 tokens each = 3,000 tokens
- ~5 context pairs per token = 15,000 pairs
- 5 epochs = 75,000 pairs
- **Total: ~28.8M float ops**
- **Measured: ~50-200ms in pure Python** (from existing benchmarks)

### 7.2 RNN Cost (GRU, hidden_dim=64, input_dim=32)

Per timestep:
- 3 gate computations: each requires 2 matvec + 1 add + 1 sigmoid/tanh
  - W_z @ x_t: 64 * 32 = 2,048 multiplications
  - U_z @ h: 64 * 64 = 4,096 multiplications
  - Same for W_r, U_r, W_h, U_h
- Total per step: 3 * (2,048 + 4,096) = **18,432 float ops**

Per walk (avg 15 tokens): 15 * 18,432 = **276,480 float ops**

For the same corpus (200 walks, 5 epochs):
- Forward: 200 * 5 * 276,480 = **276.5M float ops**
- BPTT (roughly 2-3x forward): ~550-830M float ops
- **Total: ~830M - 1.1B float ops**

### 7.3 Pure Python Overhead

Python loop overhead per float op: ~100-200ns (vs ~1ns for C/numpy).

| Metric | Skip-Gram | RNN (GRU-64) | Ratio |
|--------|-----------|--------------|-------|
| Float ops | 28.8M | ~1B | 35x |
| Python overhead factor | ~100x | ~100x | same |
| Estimated wall time | 100-200ms | 3-7 seconds | 30-35x |
| With numpy (vectorized) | 20-40ms | 200-500ms | 10-12x |

### 7.4 Feasibility Assessment

**Pure Python (no numpy):**
- Training: 3-7 seconds for a typical vampire.in corpus. Acceptable for initial training before search begins.
- Online update (1 mini-batch of ~32 walks): ~50-150ms. Acceptable if update interval is set appropriately (every 50-100 given clauses vs. every 10 for skip-gram).
- Embedding generation (1 forward pass, 15 tokens): ~1-3ms. Cached after first computation — acceptable.

**With numpy (optional speedup):**
- All matvec operations vectorized: ~10x speedup → training in 300-700ms.
- Online updates: ~5-15ms per batch.
- Make numpy an optional dependency with graceful fallback.

### 7.5 Mitigation Strategies

1. **Reduce hidden_dim:** 64 → 32 cuts per-step cost by 4x (H^2 scaling). Quality tradeoff is minimal for small vocabularies.
2. **Shorter walks:** Set `max_walk_length=20` to bound per-walk cost.
3. **Fewer random walks:** Reduce `num_random_walks` from 10 to 3-5.
4. **Truncated BPTT:** Limit backpropagation to 20-30 steps.
5. **Batch processing with numpy:** Vectorize matvec over entire walk at once.

---

## 8. Embedding Dimensionality Pipeline

```
Token ID (int)
    ↓ lookup in E: vocab_size x input_dim
Token Embedding (input_dim)
    ↓ RNN cell (GRU/LSTM)
Hidden State (hidden_dim)
    ↓ Composition (mean/last/attention)
Composed Vector (hidden_dim)
    ↓ Linear projection W_proj: hidden_dim → embedding_dim
Final Embedding (embedding_dim)
    ↓ Optional L2 normalization
Normalized Embedding (embedding_dim)
```

### Dimension Choices

| Dimension | Default | Range | Rationale |
|-----------|---------|-------|-----------|
| `input_dim` | 32 | 16-128 | Token embedding size. Small vocab needs small dim. |
| `hidden_dim` | 64 | 32-256 | RNN internal state. H^2 scaling makes this the cost driver. |
| `embedding_dim` | 64 | 32-512 | Final output. Must match what selection/diversity scoring expects. |

### Projection Layer

When `hidden_dim != embedding_dim`, a linear projection maps the composed hidden state to the target dimensionality:

```python
W_proj: list[list[float]]  # embedding_dim x hidden_dim
b_proj: list[float]         # embedding_dim

def project(composed: list[float]) -> list[float]:
    return add_vec(matvec(W_proj, composed), b_proj)
```

When `hidden_dim == embedding_dim`, the projection is identity (no extra parameters).

### Alignment with Tree2Vec

The default `embedding_dim=64` matches `SkipGramConfig.embedding_dim=64`, ensuring that RNN2Vec embeddings are drop-in compatible with the existing selection infrastructure. The `embedding_dim` field in `RNNEmbeddingConfig` overrides this if needed.

---

## 9. Configuration Dataclass

```python
@dataclass(frozen=True, slots=True)
class RNNEmbeddingConfig:
    rnn_type: str = "gru"            # "gru", "lstm", "rnn"
    input_dim: int = 32              # token embedding size
    hidden_dim: int = 64             # RNN hidden state size
    embedding_dim: int = 64          # final output dimension
    num_layers: int = 1              # stacked RNN layers
    bidirectional: bool = False      # bidirectional RNN
    dropout: float = 0.0             # dropout between layers (>1 layer only)
    composition: str = "mean"        # "last", "mean", "attention"
    normalize: bool = True           # L2-normalize output
    learning_rate: float = 0.01      # initial learning rate
    min_learning_rate: float = 0.0001
    num_epochs: int = 5
    contrastive_temperature: float = 0.1
    next_token_weight: float = 0.2   # auxiliary loss weight
    max_bptt_steps: int = 30
    seed: int = 42
    online_vocab_extension: bool = True
```

---

## 10. Summary of Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| RNN cell | GRU | Best parameter-efficiency for short sequences |
| Training objective | Contrastive + next-token auxiliary | Directly optimizes selection-relevant similarity |
| Composition | Mean pooling | Zero extra parameters, robust |
| hidden_dim | 64 | Matches skip-gram embedding_dim; H^2 cost manageable |
| input_dim | 32 | Sufficient for small vocabularies |
| Online learning | Mini-batch SGD with vocab extension | Mirrors existing SkipGramTrainer pattern |
| Pure Python feasibility | Viable with caveats | 3-7s training, 1-3ms per embedding (cached). numpy optional for 10x speedup |
