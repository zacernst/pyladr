# Online Learning Guide

PyLADR supports real-time contrastive learning during proof search. When enabled, the system learns from inference outcomes as they occur — improving clause selection quality over the course of a single search run.

## Quick Start

```bash
# Install ML dependencies
pip install pyladr[ml]

# Run with online learning enabled
uv run pyprover9 -f problem.in --online-learning
```

The `--online-learning` flag activates the full pipeline: experience collection, contrastive model updates, and embedding-guided clause selection.

## How It Works

Online learning adds a feedback loop to the given-clause search algorithm:

```
┌──────────────────────────────────────────────────────┐
│                  Search Loop                         │
│                                                      │
│  Select given ──► Generate inferences ──► Process    │
│       ▲                                      │       │
│       │                                      ▼       │
│  ML-enhanced     ◄── Hot-swap ◄── Train ◄── Record  │
│  selection           weights      model     outcomes │
│                                                      │
└──────────────────────────────────────────────────────┘
```

1. **Experience collection**: Each inference outcome (kept, subsumed, tautology, weight-limited) is recorded to an experience buffer.
2. **Contrastive training**: When enough examples accumulate, the system samples contrastive pairs (productive vs. unproductive outcomes) and performs gradient steps on an InfoNCE + triplet loss.
3. **Model hot-swapping**: Updated GNN weights are atomically swapped into the embedding provider. The embedding cache is invalidated so new embeddings reflect the improved model.
4. **Improved selection**: The embedding-enhanced clause selector blends traditional weight/age scoring with ML-based diversity and proof-potential signals, gradually increasing the ML weight as the model improves.

## Architecture

### Core Components

| Component | Module | Role |
|-----------|--------|------|
| `OnlineLearningManager` | `pyladr.ml.online_learning` | Coordinates training loop, experience buffer, model versioning, A/B testing |
| `OnlineSearchIntegration` | `pyladr.search.online_integration` | Bridges the search loop with online learning via event hooks |
| `GNNEmbeddingProvider` | `pyladr.ml.embedding_provider` | Thread-safe GNN inference with model hot-swapping |
| `EmbeddingEnhancedSelection` | `pyladr.search.ml_selection` | Blends traditional and ML-based clause selection |
| `OnlineInfoNCELoss` | `pyladr.ml.training.online_losses` | Contrastive loss with in-batch negative mining |

### Experience Buffer

The experience buffer is a bounded circular buffer that stores `InferenceOutcome` records. Each outcome captures:

- **given_clause**: The clause selected from the set of support
- **partner_clause**: The other clause in binary inferences (if applicable)
- **child_clause**: The inferred clause
- **outcome**: What happened (`KEPT`, `SUBSUMED`, `TAUTOLOGY`, `WEIGHT_LIMIT`, `PROOF`)

Outcomes are classified as **productive** (`KEPT`, `PROOF`) or **unproductive** (`SUBSUMED`, `TAUTOLOGY`, `WEIGHT_LIMIT`). Training samples contrastive pairs from these two pools.

### Contrastive Loss

The loss function operates on paired examples from the experience buffer:

- **OnlineInfoNCELoss**: Primary loss. Uses in-batch negative mining to turn B pairs into B×(B-1) comparisons. Supports temperature annealing for adaptive discrimination.
- **OnlineTripletLoss**: Margin-based fallback for very small batches (<4 examples).
- **CombinedOnlineLoss**: Default blend — 70% InfoNCE + 30% triplet.

### Model Hot-Swapping

After a training step, updated weights are pushed to the embedding provider atomically:

- A readers–writer lock ensures no forward pass is in flight during the swap
- Multiple search threads can compute embeddings concurrently
- The embedding cache is invalidated so stale embeddings are never served
- Model version numbers allow A/B testing between the old and new model

## Configuration Reference

### OnlineLearningConfig

Controls the training loop. All parameters have sensible defaults.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master switch for online learning |
| `update_interval` | `int` | `200` | New examples between model updates |
| `min_examples_for_update` | `int` | `50` | Minimum examples before first update |
| `buffer_capacity` | `int` | `5000` | Maximum examples in experience buffer |
| `batch_size` | `int` | `32` | Batch size for gradient steps |
| `learning_rate` | `float` | `5e-5` | Learning rate (deliberately low to prevent catastrophic forgetting) |
| `gradient_steps_per_update` | `int` | `5` | Gradient steps per update cycle |
| `momentum` | `float` | `0.995` | EMA momentum for weight averaging (higher = more stable) |
| `rollback_threshold` | `float` | `0.1` | Performance drop fraction that triggers rollback |
| `ab_test_window` | `int` | `100` | Selections tracked for A/B comparison |
| `ab_test_significance` | `float` | `0.05` | Minimum win-rate difference to keep update |
| `temperature` | `float` | `0.07` | Temperature for contrastive loss |
| `max_updates` | `int` | `0` | Maximum updates per search (0 = unlimited) |

### OnlineIntegrationConfig

Controls how online learning integrates with the search loop.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master switch for integration |
| `collect_experiences` | `bool` | `True` | Record inference outcomes |
| `trigger_updates` | `bool` | `True` | Trigger model updates during search |
| `track_proof_progress` | `bool` | `True` | Track progress signals toward proof |
| `invalidate_cache_on_update` | `bool` | `True` | Clear embedding cache after model update |
| `adaptive_ml_weight` | `bool` | `True` | Dynamically adjust ML selection weight |
| `initial_ml_weight` | `float` | `0.1` | Starting ML selection weight |
| `max_ml_weight` | `float` | `0.5` | Maximum ML weight during adaptation |
| `ml_weight_increase_rate` | `float` | `0.05` | Weight increase per successful update |
| `ml_weight_decrease_rate` | `float` | `0.1` | Weight decrease per failed update/rollback |
| `min_given_before_ml` | `int` | `50` | Minimum given clauses before ML activates |
| `log_integration_events` | `bool` | `False` | Log integration events for debugging |

### MLSelectionConfig

Controls the ML-enhanced clause selection strategy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable ML-enhanced selection |
| `ml_weight` | `float` | `0.3` | Blending weight (0 = pure traditional, 1 = pure ML) |
| `diversity_weight` | `float` | `0.5` | Weight of diversity component in ML score |
| `proof_potential_weight` | `float` | `0.5` | Weight of proof-potential component |
| `diversity_window` | `int` | `20` | Recent given clauses tracked for diversity |
| `min_sos_for_ml` | `int` | `10` | Minimum SOS size before ML selection activates |
| `fallback_on_error` | `bool` | `True` | Fall back to traditional selection on ML error |
| `log_selections` | `bool` | `False` | Log ML vs traditional selection decisions |

## Subsumption-Based Learning Signals

Beyond the standard experience collection from inference outcomes, online learning can exploit **subsumption events** as high-quality positive feedback signals. Both forward and backward subsumption indicate that a clause has particularly useful structural properties — it is general enough to logically entail other clauses.

### Back-Subsumption Learning

When a newly kept clause **back-subsumes** an existing clause (the new clause is more general than one already in usable/SOS), this is strong evidence that the subsuming clause has a powerful structure worth reinforcing.

```bash
uv run pyprover9 -f problem.in --online-learning --learn-from-back-subsumption
```

**How it works:** When clause C back-subsumes clause D, the system records C as a double-weighted positive experience in the online learning buffer. This teaches the GNN to prefer clauses with similar structural patterns in future selections.

### Forward-Subsumption Learning

When an existing clause **forward-subsumes** a newly generated clause (the existing clause is more general than the new one), this confirms that the existing clause was a good choice — it is still contributing to search efficiency by pruning redundant inferences.

```bash
uv run pyprover9 -f problem.in --online-learning --learn-from-forward-subsumption
```

**How it works:** When existing clause C forward-subsumes new clause D (causing D to be discarded), the system records C as a double-weighted positive experience. This reinforces the GNN's preference for general, subsumption-capable clauses.

### Using Both Together

For maximum learning signal, enable both subsumption feedback channels:

```bash
uv run pyprover9 -f problem.in --online-learning \
    --learn-from-back-subsumption \
    --learn-from-forward-subsumption
```

### Comparison of Subsumption Signals

| Property | Back-Subsumption | Forward-Subsumption |
|----------|-----------------|---------------------|
| **Direction** | New clause subsumes existing | Existing clause subsumes new |
| **Subsuming clause** | Recently kept | Already in usable/SOS |
| **Subsumed clause** | Removed from usable/SOS | Discarded before keeping |
| **Signal meaning** | New clause is powerfully general | Existing clause still contributes |
| **Frequency** | Less frequent | More frequent |
| **SearchOptions flag** | `learn_from_back_subsumption` | `learn_from_forward_subsumption` |
| **CLI flag** | `--learn-from-back-subsumption` | `--learn-from-forward-subsumption` |

### Why Subsumption Is a Strong Signal

In the given-clause algorithm, most inference outcomes are weakly informative — a clause being kept or deleted by weight limit says little about its structural quality. Subsumption events are different:

1. **Logical entailment**: Subsumption means one clause logically entails another (up to variable renaming). This is a semantic property, not just a syntactic one.
2. **Search pruning**: Subsuming clauses directly reduce search space by eliminating redundant clauses.
3. **Generality indicator**: A clause that subsumes others tends to be more general and thus more likely to participate in short proofs.

The double-weighting in the experience buffer ensures these rare but high-quality signals are not drowned out by the more frequent kept/deleted outcomes.

## When to Use Online Learning

Online learning is most beneficial for:

- **Hard problems** that require many given clauses (>500) — the model has time to learn
- **Problems with patterns** in productive inferences (equational reasoning, lattice theory)
- **Exploration-heavy searches** where diversity in clause selection matters

It may not help for:

- **Easy problems** solved in <100 given clauses — training doesn't have time to kick in
- **Problems requiring very specific inference chains** where traditional heuristics already work well
- **Memory-constrained environments** — the GNN and embedding cache add overhead

## Troubleshooting

### "ML dependencies not available"

Install the ML optional dependencies:

```bash
pip install pyladr[ml]
# or
pip install torch>=2.0.0 torch-geometric>=2.4.0
```

### "Failed to initialize online learning"

Check that:
1. PyTorch is correctly installed for your platform
2. `torch_geometric` is compatible with your PyTorch version
3. Sufficient memory is available (the GNN uses ~50MB baseline)

### No improvement with online learning

This is expected for easy problems. The system needs at least `min_examples_for_update` (default 50) outcomes before the first training step, and `update_interval` (default 200) outcomes between subsequent updates. For short searches, the model never gets a chance to learn.

Try with harder problems or lower thresholds:

```python
from pyladr.ml.online_learning import OnlineLearningConfig

config = OnlineLearningConfig(
    update_interval=50,
    min_examples_for_update=20,
)
```

### High memory usage

Reduce the experience buffer and embedding cache sizes:

```python
from pyladr.ml.online_learning import OnlineLearningConfig
from pyladr.ml.embedding_provider import EmbeddingProviderConfig

learning_config = OnlineLearningConfig(buffer_capacity=1000)
provider_config = EmbeddingProviderConfig(cache_max_entries=10_000)
```

### Debugging

Enable verbose logging to see integration events:

```python
import logging
logging.getLogger("pyladr.ml").setLevel(logging.DEBUG)
logging.getLogger("pyladr.search.online_integration").setLevel(logging.DEBUG)
```

Or use the `log_integration_events=True` config flag, which is automatically set when running without `--quiet`.

## Programmatic Usage

For fine-grained control beyond the CLI flag:

```python
from pyladr.ml.embedding_provider import create_embedding_provider
from pyladr.ml.online_learning import OnlineLearningConfig
from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig
from pyladr.search.online_integration import (
    OnlineIntegrationConfig,
    OnlineSearchIntegration,
)

# Create embedding provider
provider = create_embedding_provider(symbol_table=symbol_table)

# Configure ML-enhanced selection
selection = EmbeddingEnhancedSelection(
    embedding_provider=provider,
    ml_config=MLSelectionConfig(enabled=True, ml_weight=0.2),
)

# Configure online learning integration
integration = OnlineSearchIntegration.create(
    embedding_provider=provider,
    config=OnlineIntegrationConfig(
        enabled=True,
        adaptive_ml_weight=True,
        initial_ml_weight=0.1,
        max_ml_weight=0.5,
        min_given_before_ml=10,
    ),
)

# Create and run search
engine = integration.create_search(
    options=opts,
    selection=selection,
    symbol_table=symbol_table,
)
result = engine.run(usable=usable, sos=sos)
```
