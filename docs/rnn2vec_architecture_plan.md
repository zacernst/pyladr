# RNN2Vec Architecture Plan

## Overview

RNN2Vec replaces the skip-gram training component of Tree2Vec with a recurrent neural network that processes tree walk sequences directly. Instead of learning token co-occurrence embeddings (skip-gram), the RNN learns to encode entire walk sequences into fixed-dimensional vectors, capturing sequential and compositional structure.

**Key principle:** RNN replaces skip-gram only. Walk generation (`TreeWalker`, `WalkConfig`) is reused unchanged.

---

## 1. Directory Structure: `pyladr/ml/rnn2vec/`

```
pyladr/ml/rnn2vec/
├── __init__.py              # Public API re-exports (mirrors tree2vec/__init__.py)
├── algorithm.py             # RNN2Vec orchestrator (parallel to tree2vec/algorithm.py)
├── encoder.py               # RNN encoder: vocab→embedding→RNN→composition
├── provider.py              # RNN2VecEmbeddingProvider (EmbeddingProvider protocol)
├── formula_processor.py     # Corpus processing + augmentation (reuses tree2vec patterns)
└── tokenizer.py             # Shared token→int ID vocabulary builder
```

**Files NOT needed (reused from tree2vec):**
- `walks.py` — `TreeWalker`, `WalkConfig`, `WalkType`, `_node_token` reused directly
- `vampire_parser.py` — `VampireCorpus`, `parse_vampire_file` reused directly

---

## 2. Configuration: `RNNEmbeddingConfig`

```python
@dataclass(frozen=True, slots=True)
class RNNEmbeddingConfig:
    """Configuration for the RNN encoder.

    Attributes:
        rnn_type: RNN variant — "lstm", "gru", or "elman".
        hidden_dim: RNN hidden state dimensionality.
        num_layers: Number of stacked RNN layers.
        bidirectional: Use bidirectional RNN (doubles effective hidden dim
            before projection to embedding_dim).
        embedding_dim: Final clause/term embedding dimensionality.
        input_dim: Learnable token embedding dimensionality (input to RNN).
        dropout: Dropout between RNN layers (0.0 = none). Only active
            during training; ignored when num_layers == 1.
        normalize: L2-normalize output embeddings.
        composition: How to derive fixed-size vector from RNN outputs.
            "last_hidden" — final hidden state (default).
            "mean_pool" — mean of all output timesteps.
            "attention_pool" — learned attention over timesteps.
    """
    rnn_type: str = "gru"
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    embedding_dim: int = 64
    input_dim: int = 32
    dropout: float = 0.0
    normalize: bool = True
    composition: str = "last_hidden"
```

**Design rationale:**
- `input_dim` separate from `hidden_dim` allows a small token embedding lookup (vocabulary is tiny: ~6-20 tokens) fed into a wider RNN.
- `embedding_dim` may differ from `hidden_dim` — a linear projection layer maps from `hidden_dim` (or `2 * hidden_dim` if bidirectional) to `embedding_dim`.
- Default `gru` over `lstm`: fewer parameters, faster training on small vocabularies. User can switch to `lstm` for longer walks.

---

## 3. RNN2Vec Class Interface (`algorithm.py`)

```python
@dataclass(frozen=True, slots=True)
class RNN2VecConfig:
    """Top-level RNN2Vec configuration."""
    walk_config: WalkConfig = WalkConfig()           # REUSED from tree2vec
    rnn_config: RNNEmbeddingConfig = RNNEmbeddingConfig()
    training_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    normalize: bool = True

class RNN2Vec:
    """Unsupervised RNN-based embedding generator for logical formula trees.

    Replaces skip-gram with an RNN encoder that processes tree walk
    sequences to produce fixed-dimensional embeddings.

    The training objective is a contrastive loss: walks from the same
    clause/term should produce similar embeddings, walks from different
    clauses should produce dissimilar embeddings.
    """

    def __init__(self, config: RNN2VecConfig | None = None) -> None: ...

    @property
    def embedding_dim(self) -> int:
        """Effective output embedding dimension."""

    @property
    def trained(self) -> bool: ...

    @property
    def vocab_size(self) -> int: ...

    # ── Training ──────────────────────────────────────────────────

    def train(
        self,
        clauses: Sequence[Clause],
        progress_fn: Callable[[int, int, float, float], None] | None = None,
    ) -> dict[str, float]:
        """Train RNN embeddings from clauses.

        1. Generate tree walks via TreeWalker (reused from tree2vec).
        2. Build/extend token vocabulary.
        3. Train RNN encoder with contrastive objective.

        Returns training statistics dict.
        """

    def train_from_terms(self, terms: Sequence[Term]) -> dict[str, float]:
        """Train directly from terms (without clause wrapper)."""

    # ── Online update ─────────────────────────────────────────────

    def update_online(
        self,
        clauses: Sequence[Clause],
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Incremental update from recently kept clauses.

        Runs a few gradient steps on walks from new clauses.
        New tokens get mean-initialized embeddings in the lookup table.
        No-op if not trained yet.
        """

    # ── Embedding generation ──────────────────────────────────────

    def embed_term(self, term: Term) -> list[float] | None:
        """Embed a single term.

        Generates walks, encodes each via RNN, averages the resulting
        walk embeddings. Returns None if not trained.
        """

    def embed_clause(self, clause: Clause) -> list[float] | None:
        """Embed a clause.

        Generates walks for each literal, encodes each via RNN,
        applies sign scaling and averaging (matching Tree2Vec behavior).
        Returns None if not trained.
        """

    def embed_clauses_batch(
        self, clauses: Sequence[Clause],
    ) -> list[list[float] | None]:
        """Batch embedding for multiple clauses.

        Collects all walks, pads sequences, runs RNN in a single
        batched forward pass, then groups results by clause.
        """

    # ── Similarity ────────────────────────────────────────────────

    def similarity(self, term_a: Term, term_b: Term) -> float | None:
        """Cosine similarity between two term embeddings."""

    def clause_similarity(self, clause_a: Clause, clause_b: Clause) -> float | None:
        """Cosine similarity between two clause embeddings."""

    # ── Serialization ─────────────────────────────────────────────

    SAVE_FORMAT_VERSION: int = 1

    def save(self, path: str | Path) -> None:
        """Serialize model to disk.

        Saves: config, vocabulary, RNN state_dict, projection weights.
        Format: directory with config.json + model.pt (torch state dict).
        """

    @classmethod
    def load(cls, path: str | Path) -> RNN2Vec:
        """Load a previously saved model."""

    # ── Token-level access ────────────────────────────────────────

    def get_token_embedding(self, token: str) -> list[float] | None:
        """Get the learned lookup embedding for a specific token."""

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float: ...
```

**Differences from Tree2Vec:**
- `embed_clauses_batch` can do genuine batched RNN forward passes (padded sequences) instead of looping.
- `train` uses contrastive loss instead of skip-gram pairs.
- Online update runs gradient steps on the full RNN, not just embedding lookup update.
- Serialization uses torch `state_dict` rather than raw JSON lists.

---

## 4. Walk Sequence Reuse Strategy

The existing `TreeWalker` from `pyladr/ml/tree2vec/walks.py` is reused **as-is**:

```
Clause → TreeWalker.walks_from_clause() → list[list[str]]
                                             ↓
                                     tree2vec: skip-gram training
                                     rnn2vec:  tokenize → RNN encoding
```

**Walk generation is identical.** The RNN receives the same token sequences that skip-gram would. The difference is downstream:
- Skip-gram: learns per-token embeddings from (center, context) pairs
- RNN: encodes the full sequence into a single vector

**Configuration sharing:** `RNN2VecConfig.walk_config` uses the same `WalkConfig` dataclass. Walk types (DFS, BFS, RANDOM, PATH), position/depth encoding, var identity — all controlled identically.

---

## 5. Token Encoding Strategy (`tokenizer.py`)

```
Token string → vocab lookup → int ID → nn.Embedding → RNN input
```

### Vocabulary Building

```python
@dataclass(frozen=True, slots=True)
class TokenVocab:
    """Bidirectional token↔ID mapping with special tokens."""
    PAD_ID: int = 0    # padding for batched sequences
    UNK_ID: int = 1    # unknown/OOV tokens

    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def from_walks(cls, walks: Sequence[Sequence[str]]) -> TokenVocab:
        """Build vocabulary from walk corpus, sorted by frequency."""

    def encode_walk(self, walk: list[str]) -> list[int]:
        """Convert token sequence to int ID sequence."""

    def extend(self, token: str) -> int:
        """Add new OOV token, returns its ID."""

    @property
    def size(self) -> int: ...
```

### Learnable Embeddings

```python
# In encoder.py
self.token_embeddings = nn.Embedding(
    num_embeddings=vocab.size,
    embedding_dim=config.input_dim,
    padding_idx=TokenVocab.PAD_ID,
)
```

- Vocabulary is tiny (typically 6-20 tokens for vampire.in domains).
- `input_dim` (default 32) feeds into the RNN. This is distinct from `embedding_dim` (the final output).
- PAD token embedding is zero (frozen) for correct batch padding.

### Flow

```
"FUNC:3/2" → vocab["FUNC:3/2"] → 5 → nn.Embedding[5] → [0.12, -0.34, ...] (input_dim)
                                                              ↓
                                                         RNN timestep
```

---

## 6. Composition Strategies

Three strategies for converting RNN output sequences to a fixed-size vector:

### `last_hidden` (default)
```python
# Use final hidden state from last RNN layer
_, h_n = rnn(embedded_sequence)  # h_n: (num_layers * dirs, batch, hidden_dim)
output = h_n[-1]                 # last layer: (batch, hidden_dim)
# If bidirectional: h_n[-2:] → cat → (batch, 2*hidden_dim)
output = projection(output)      # → (batch, embedding_dim)
```

**Pros:** Simple, captures entire sequence context. **Cons:** May lose early-walk information on long sequences.

### `mean_pool`
```python
# Mean of all output timesteps, ignoring padding
output, _ = rnn(embedded_sequence)  # (batch, seq_len, hidden_dim*dirs)
# Mask out padding positions, then mean
lengths = (input_ids != PAD_ID).sum(dim=1)
output = output.sum(dim=1) / lengths.unsqueeze(1)
output = projection(output)         # → (batch, embedding_dim)
```

**Pros:** All timesteps contribute equally, robust to sequence length. **Cons:** May dilute signal from structurally important tokens (root).

### `attention_pool`
```python
# Learned attention weights over timesteps
output, _ = rnn(embedded_sequence)       # (batch, seq_len, hidden_dim*dirs)
attn_scores = self.attn_linear(output)   # (batch, seq_len, 1)
attn_weights = softmax(attn_scores, dim=1, mask=padding_mask)
output = (attn_weights * output).sum(dim=1)
output = projection(output)              # → (batch, embedding_dim)
```

**Pros:** Learns which positions matter most (e.g., root tokens). **Cons:** Adds learnable parameters, needs more training data.

### Recommendation
Default to `last_hidden` for GRU (its hidden state already summarizes the sequence). Switch to `attention_pool` for bidirectional models where both directions carry information.

---

## 7. RNN2VecEmbeddingProvider (`provider.py`)

Mirrors `Tree2VecEmbeddingProvider` exactly, implementing the `EmbeddingProvider` protocol:

```python
@dataclass(frozen=True, slots=True)
class RNN2VecProviderConfig:
    """Configuration for RNN2VecEmbeddingProvider."""
    rnn2vec_config: RNN2VecConfig = field(default_factory=RNN2VecConfig)
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    cache_max_entries: int = 100_000
    enable_cache: bool = True


class RNN2VecEmbeddingProvider:
    """EmbeddingProvider backed by RNN2Vec embeddings.

    Thread-safe: concurrent get_embedding() calls are safe.
    Cache uses structural hashing (clause_structural_hash from
    pyladr.ml.embeddings.cache) for α-equivalent deduplication.
    Version-aware cache invalidation on online model updates.
    """

    __slots__ = (
        "_rnn2vec", "_config", "_cache", "_cache_lock",
        "_stats", "_model_version",
    )

    def __init__(self, rnn2vec: RNN2Vec, config: RNN2VecProviderConfig | None = None): ...

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_vampire_file(
        cls,
        filepath: str,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Parse file → augment corpus → train RNN2Vec → wrap in provider."""

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Load pre-trained RNN2Vec model from disk."""

    @classmethod
    def from_corpus(
        cls,
        corpus: VampireCorpus,
        config: RNN2VecProviderConfig | None = None,
    ) -> RNN2VecEmbeddingProvider:
        """Train on a pre-parsed corpus."""

    # ── EmbeddingProvider protocol ───────────────────────────────

    @property
    def embedding_dim(self) -> int: ...

    def get_embedding(self, clause: Clause) -> list[float] | None:
        """Cached, version-aware embedding retrieval."""

    def get_embeddings_batch(self, clauses: list[Clause]) -> list[list[float] | None]:
        """Batch retrieval. Cache misses computed in a single RNN batch pass."""

    # ── Cache management ─────────────────────────────────────────

    def bump_model_version(self) -> int:
        """Increment version after online update, lazily invalidating cache."""

    def invalidate_all(self) -> int:
        """Drop all cached embeddings."""
```

### Thread-safety model (identical to Tree2Vec provider):
- `_cache_lock: threading.Lock` protects the `OrderedDict` cache.
- RNN model is read-only after training; only the cache needs sync.
- `bump_model_version()` increments a counter; stale entries treated as misses.

### Cache key: `clause_structural_hash(clause)` from `pyladr.ml.embeddings.cache` — same as Tree2Vec.

### Batch optimization:
Unlike Tree2Vec (which loops), `get_embeddings_batch` can:
1. Check cache for all clauses.
2. Collect cache misses.
3. Run a single padded RNN forward pass for all misses.
4. Insert results into cache.

---

## 8. Save/Load Serialization Format

### Directory format (not single file):
```
rnn2vec_model/
├── config.json        # RNN2VecConfig + WalkConfig + RNNEmbeddingConfig
├── vocab.json         # token_to_id, id_to_token mappings
└── model.pt           # torch state_dict (token embeddings + RNN + projection)
```

**Why directory, not single JSON?** RNN weights are float32 tensors — JSON is impractical. Tree2Vec uses JSON because skip-gram embeddings are small list-of-lists. RNN state_dict uses `torch.save`.

### Config JSON structure:
```json
{
    "format_version": 1,
    "walk_config": { "walk_types": [...], "num_random_walks": 10, ... },
    "rnn_config": {
        "rnn_type": "gru",
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": false,
        "embedding_dim": 64,
        "input_dim": 32,
        "dropout": 0.0,
        "normalize": true,
        "composition": "last_hidden"
    },
    "training_epochs": 10,
    "normalize": true
}
```

### Factory methods:
- `from_vampire_file(filepath, config)` — parse + augment + train + wrap
- `from_saved_model(model_path, config)` — load model directory + wrap
- `from_corpus(corpus, config)` — train on pre-parsed VampireCorpus

---

## 9. Shared Utilities Between tree2vec and rnn2vec

### Already shared (import directly):
| Utility | Location | Used by |
|---------|----------|---------|
| `TreeWalker`, `WalkConfig`, `WalkType` | `tree2vec/walks.py` | Both: walk generation |
| `_node_token`, `_literal_token` | `tree2vec/walks.py` | Both: token creation |
| `VampireCorpus`, `parse_vampire_file` | `tree2vec/vampire_parser.py` | Both: input parsing |
| `AugmentationConfig`, `_rename_variables` | `tree2vec/formula_processor.py` | Both: data augmentation |
| `clause_structural_hash` | `ml/embeddings/cache.py` | Both: cache keys |

### New shared module: `tokenizer.py`
Extract the token→ID vocabulary management into `rnn2vec/tokenizer.py`. Tree2Vec's skip-gram has its own vocab (embedded in `SkipGramTrainer`), but the RNN needs a standalone vocabulary with:
- PAD/UNK special tokens
- `encode_walk(walk) → list[int]` for batching
- `extend(token)` for online vocab growth

This could later be shared back to tree2vec if desired, but initially lives in `rnn2vec/` to avoid modifying tree2vec.

### Augmentation reuse:
`rnn2vec/formula_processor.py` can directly import and call `process_vampire_corpus`-style logic from `tree2vec/formula_processor.py`, or more cleanly, import the helper functions (`_rename_variables`, `_rename_term_vars`) and the `AugmentationConfig` dataclass. The corpus processing pattern is identical — only the final `tree2vec.train(clauses)` call changes to `rnn2vec.train(clauses)`.

---

## 10. Integration Points

### 10.1 `pyladr/search/options.py`

Add RNN2Vec bounds to `_NUMERIC_BOUNDS`:

```python
# RNN2Vec
("rnn2vec_weight", 0.0, None, "RNN2Vec embedding selection ratio weight"),
("rnn2vec_embedding_dim", 1, 4096, "RNN2Vec embedding dimension"),
("rnn2vec_cache_max_entries", 1, 10_000_000, "RNN2Vec cache size"),
("rnn2vec_online_lr", 0.0001, 1.0, "RNN2Vec online learning rate"),
("rnn2vec_online_update_interval", 1, 10_000, "RNN2Vec online update interval"),
("rnn2vec_online_batch_size", 1, 10_000, "RNN2Vec online batch size"),
("rnn2vec_online_max_updates", 0, None, "RNN2Vec max online updates (0=unlimited)"),
("rnn2vec_hidden_dim", 1, 4096, "RNN2Vec hidden dimension"),
```

### 10.2 `pyladr/search/given_clause.py` — `SearchOptions`

Add options mirroring the tree2vec pattern:

```python
# RNN2Vec embeddings
rnn2vec_embeddings: bool = False
rnn2vec_weight: float = 0.0          # selection ratio weight (0 = disabled)
rnn2vec_rnn_type: str = "gru"        # lstm, gru, elman
rnn2vec_hidden_dim: int = 128
rnn2vec_embedding_dim: int = 64
rnn2vec_input_dim: int = 32
rnn2vec_num_layers: int = 1
rnn2vec_bidirectional: bool = False
rnn2vec_composition: str = "last_hidden"  # last_hidden, mean_pool, attention_pool
rnn2vec_cache_max_entries: int = 10_000
rnn2vec_include_position: bool = False
rnn2vec_include_depth: bool = False
rnn2vec_include_var_identity: bool = False
rnn2vec_skip_predicate: bool = True
rnn2vec_online_learning: bool = False
rnn2vec_online_update_interval: int = 20
rnn2vec_online_batch_size: int = 10
rnn2vec_online_lr: float = 0.001
rnn2vec_online_max_updates: int = 0
rnn2vec_model_path: str = ""          # pre-trained model directory
rnn2vec_goal_proximity: bool = False
rnn2vec_goal_proximity_weight: float = 0.3
```

### 10.3 `pyladr/search/selection.py`

Add `SelectionOrder.RNN2VEC` enum value (parallel to `TREE2VEC`).

### 10.4 `pyladr/search/given_clause.py` — `GivenClauseSearch`

Follow the exact same initialization and integration pattern as tree2vec:

```python
# In __init__:
self._rnn2vec_provider: object | None = None
self._rnn2vec_embeddings: dict[int, list[float]] = {}

# In selection rule setup:
if opts.rnn2vec_weight > 0:
    rules.append(SelectionRule("R2V", SelectionOrder.RNN2VEC, part=opts.rnn2vec_weight))

# In _initialize_providers (called when initial clauses are available):
if opts.rnn2vec_embeddings:
    # Build config, create provider (from_vampire_file or from_saved_model)
    # Pre-embed initial SOS clauses
    # Set up online learning if enabled
    # Set up goal proximity if enabled
```

### 10.5 CLI (`pyladr/apps/cli_common.py`)

Add LADR directives:
- `set(rnn2vec_embeddings).` / `clear(rnn2vec_embeddings).`
- `assign(rnn2vec_weight, 2).`
- `assign(rnn2vec_hidden_dim, 256).`
- etc.

---

## Training Objective: Contrastive Loss

The RNN uses a contrastive training objective (not skip-gram):

1. **Positive pairs:** Two different walks from the **same** clause/term.
2. **Negative pairs:** Walks from **different** clauses/terms.
3. **Loss:** For positive pair (w_i, w_j) from same clause and negative walk w_k from different clause:
   ```
   L = -log(exp(sim(f(w_i), f(w_j)) / τ) / Σ_k exp(sim(f(w_i), f(w_k)) / τ))
   ```
   where `f` is the RNN encoder and `τ` is a temperature parameter.

This is more natural for RNNs than skip-gram because:
- Skip-gram learns per-token embeddings; RNN learns sequence→vector encoding.
- Contrastive loss directly optimizes the task we care about: similar clauses → similar embeddings.

---

## Dependency Analysis

### Required:
- `torch` — for `nn.RNN`/`nn.GRU`/`nn.LSTM`, `nn.Embedding`, `nn.Linear`, optimizers

### Optional/guarded:
- Like the GNN provider, all torch imports should be guarded with try/except.
- RNN2Vec should be opt-in only (disabled by default, `rnn2vec_embeddings: bool = False`).
- When torch is not installed, importing `pyladr.ml.rnn2vec` should succeed but constructing an `RNN2Vec` instance should raise a clear error.

### No new external dependencies beyond torch (already optional in the project).
