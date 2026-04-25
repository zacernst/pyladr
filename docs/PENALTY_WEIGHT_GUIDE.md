# Penalty Weight System Guide

PyLADR provides a penalty-based clause management system that improves search
efficiency by deprioritizing overly general, repetitive, or structurally
redundant clauses. This guide covers all three subsystems and how they work
together.

## Overview

The penalty weight system consists of three composable components:

| Component | Purpose | Default |
|-----------|---------|---------|
| **Penalty Propagation** | Inherit generality penalties from parent clauses | Disabled |
| **Repetition Penalty** | Penalize clauses with repeated subformula patterns | Disabled |
| **Penalty Weight Adjustment** | Translate penalties into weight increases | Disabled |

All components are **disabled by default** to preserve C Prover9 compatibility.
When enabled, they work together: penalty propagation and repetition penalty
compute penalty scores, and penalty weight adjustment uses those scores to
inflate clause weights, pushing low-quality clauses further down the selection
queue.

### When to Use Penalty Weights

Penalty weights are most effective for:

- **Problems with clause bloat** — where the SOS grows rapidly with many
  similar, overly general clauses
- **Equational problems** — where paramodulation generates structurally
  redundant variants
- **Problems that exhaust resources** — where standard weight-based selection
  fails to find a proof within limits

They are less useful for:

- **Small problems** — where the search space is manageable without heuristics
- **Problems that already prove quickly** — the overhead may slow a fast proof
- **Highly constrained problems** — where most generated clauses are specific
  and useful

---

## Penalty Propagation

Penalty propagation tracks how "general" a clause is and passes that
information from parent to child clauses. The intuition: if a parent clause
was overly general, its children are likely overly general too.

### How It Works

1. Each clause receives an **own penalty** based on its generality (ratio of
   variables to total symbols, with extra weight for single-literal
   all-variable patterns like `P(x,y,z)`)
2. The penalty is **inherited** from parent clauses, decaying by a configurable
   factor per generation
3. Own and inherited penalties are **combined** into a final score

### CLI Options

```bash
uv run pyprover9 -f problem.in \
    --penalty-propagation \
    --penalty-propagation-mode additive \
    --penalty-propagation-decay 0.5 \
    --penalty-propagation-threshold 5.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--penalty-propagation` | off | Enable penalty propagation |
| `--penalty-propagation-mode` | `additive` | How to combine own + inherited: `additive`, `multiplicative`, or `max` |
| `--penalty-propagation-decay` | `0.5` | Decay factor per generation (0.0–1.0) |
| `--penalty-propagation-threshold` | `5.0` | Minimum parent penalty to trigger propagation |

### Combination Modes

- **Additive** (`additive`): `combined = own + decay * parent_penalty`.
  Straightforward accumulation. Good general-purpose default.
- **Multiplicative** (`multiplicative`): `combined = own * (1 + decay * parent_penalty)`.
  Amplifies penalties for clauses that are themselves already general.
- **Max** (`max`): `combined = max(own, decay * parent_penalty)`.
  Conservative — only the dominant penalty source matters.

### Tuning Guide

| Parameter | Lower values | Higher values |
|-----------|-------------|---------------|
| **Decay** (0.0–1.0) | Less inheritance, penalties fade fast | More inheritance, penalties persist across generations |
| **Threshold** (> 0) | More clauses affected | Only very general clauses affected |

**Recommended starting point:** defaults (`additive`, decay 0.5, threshold 5.0)

---

## Repetition Penalty

The repetition penalty detects and penalizes clauses containing repeated
subformula patterns. Clauses with structural redundancy (e.g., `f(g(x), g(x))`)
are often less useful for proof search.

### How It Works

Two matching phases:

1. **Exact matching** (Phase 1): Detects identical `Term` objects. Uses
   Python's frozen dataclass hash/equality for O(1) matching.
2. **Variable-normalized matching** (Phase 2, opt-in): Treats
   `f(x,x)` and `f(y,y)` as the same pattern by normalizing variables
   to a canonical DFS ordering before comparison.

The penalty formula: `base_penalty * sum(count - 1)` for each repeated pattern,
capped at `max_penalty`.

### CLI Options

```bash
uv run pyprover9 -f problem.in \
    --repetition-penalty \
    --repetition-penalty-weight 2.0 \
    --repetition-penalty-min-size 2 \
    --repetition-penalty-max 15.0 \
    --repetition-penalty-normalize
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repetition-penalty` | off | Enable repetition penalty |
| `--repetition-penalty-weight` | `2.0` | Penalty per extra occurrence of a repeated subformula |
| `--repetition-penalty-min-size` | `2` | Minimum subterm size (symbol count) to consider |
| `--repetition-penalty-max` | `15.0` | Maximum total penalty per clause |
| `--repetition-penalty-normalize` | off | Enable variable-agnostic matching (Phase 2) |

### Tuning Guide

| Parameter | Lower values | Higher values |
|-----------|-------------|---------------|
| **Weight** | Mild deprioritization | Aggressive deprioritization |
| **Min size** (>= 1) | Match small repeated patterns | Only match large repeated structures |
| **Max** | Tight cap on penalty | Allow heavily repeated clauses to be penalized more |

**Recommended starting point:** defaults (weight 2.0, min-size 2, max 15.0, no normalization)

Enable `--repetition-penalty-normalize` when your problem domain uses many
variable-symmetric patterns.

---

## Penalty Weight Adjustment

Penalty weight adjustment is the final stage: it converts penalty scores
(from propagation and/or repetition analysis) into clause weight increases.
This directly affects selection order — heavier clauses are selected later.

### How It Works

When a clause's combined penalty exceeds a threshold, its weight is adjusted
according to one of three modes:

| Mode | Formula | Use case |
|------|---------|----------|
| **Linear** | `adjusted = base + multiplier * penalty` | Predictable, proportional increase |
| **Exponential** | `adjusted = base * multiplier^(penalty/threshold)` | Aggressive suppression of high-penalty clauses |
| **Step** | `adjusted = base * multiplier` | Binary: either normal weight or boosted |

All adjusted weights are capped at `max_adjusted_weight` to prevent unbounded
inflation.

### CLI Options

```bash
uv run pyprover9 -f problem.in \
    --penalty-weight \
    --penalty-weight-mode exponential \
    --penalty-weight-threshold 5.0 \
    --penalty-weight-multiplier 2.0 \
    --penalty-weight-max 1000.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--penalty-weight` | off | Enable penalty weight adjustment |
| `--penalty-weight-mode` | `exponential` | Adjustment mode: `linear`, `exponential`, or `step` |
| `--penalty-weight-threshold` | `5.0` | Minimum combined penalty to trigger adjustment |
| `--penalty-weight-multiplier` | `2.0` | Weight increase factor (>= 1.0) |
| `--penalty-weight-max` | `1000.0` | Maximum adjusted clause weight |

### Mode Comparison

Given a clause with base weight 10 and penalty 15 (threshold 5.0, multiplier 2.0):

| Mode | Calculation | Result |
|------|-------------|--------|
| Linear | 10 + 2.0 * 15 = 40 | 40.0 |
| Exponential | 10 * 2.0^(15/5) = 10 * 8 | 80.0 |
| Step | 10 * 2.0 | 20.0 |

**Exponential** (default) provides the strongest differentiation between
moderately and heavily penalized clauses.

---

## Combined Usage

The three subsystems compose naturally. A typical combined configuration:

```bash
uv run pyprover9 -f problem.in \
    --penalty-propagation \
    --penalty-propagation-decay 0.5 \
    --repetition-penalty \
    --repetition-penalty-normalize \
    --penalty-weight \
    --penalty-weight-mode exponential
```

### Data Flow

```
Clause created by inference
    │
    ├─ Penalty Propagation
    │   └─ own_penalty + decay * max(parent_penalties) → combined_penalty
    │
    ├─ Repetition Penalty
    │   └─ base * sum(repetitions - 1) → repetition_penalty
    │
    └─ Total penalty = combined_penalty + repetition_penalty
        │
        └─ Penalty Weight Adjustment
            └─ If total >= threshold: adjust weight via mode formula
                │
                └─ Adjusted weight used for SOS heap ordering
```

### Configuration Presets

#### Conservative (safe for any problem)

```bash
--penalty-propagation \
--penalty-weight \
--penalty-weight-mode step \
--penalty-weight-multiplier 1.5
```

Mildly deprioritizes overly general clauses. Low risk of discarding useful
clauses.

#### Moderate (good general-purpose)

```bash
--penalty-propagation \
--repetition-penalty \
--penalty-weight \
--penalty-weight-mode exponential \
--penalty-weight-multiplier 2.0
```

Balanced approach. Effective for problems with moderate clause bloat.

#### Aggressive (for severely bloated searches)

```bash
--penalty-propagation \
--penalty-propagation-decay 0.8 \
--repetition-penalty \
--repetition-penalty-normalize \
--repetition-penalty-weight 3.0 \
--penalty-weight \
--penalty-weight-mode exponential \
--penalty-weight-multiplier 3.0 \
--penalty-weight-threshold 3.0
```

Strong suppression of general and repetitive clauses. Use when the default
search exhausts resources.

---

## Performance Characteristics

### Overhead

| Component | Cost per clause | Notes |
|-----------|----------------|-------|
| Penalty Propagation | O(1) cache lookup + O(k) parent lookup | k = number of parents (typically 1–3) |
| Repetition Penalty (exact) | O(n) | n = subterms in clause |
| Repetition Penalty (normalized) | O(n log n) | Variable normalization adds overhead |
| Penalty Weight Adjustment | O(1) | Simple arithmetic on cached penalty |

The total overhead is negligible compared to inference generation and
subsumption checking, which dominate search time.

### Memory

Penalty propagation maintains a side-structure `PenaltyCache` (dict mapping
clause IDs to penalty records). This cache:

- Grows linearly with kept clauses
- Is bounded by back-subsumption eviction (subsumed clauses are removed)
- Each record is ~80 bytes (own, inherited, combined, depth floats)

For typical problems with 10,000–100,000 kept clauses, the cache adds
< 10 MB of memory.

---

## Python API

For programmatic use:

```python
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

options = SearchOptions(
    # Penalty propagation
    penalty_propagation=True,
    penalty_propagation_mode="additive",
    penalty_propagation_decay=0.5,
    penalty_propagation_threshold=5.0,
    penalty_propagation_max_depth=3,
    penalty_propagation_max=20.0,

    # Repetition penalty
    repetition_penalty=True,
    repetition_penalty_weight=2.0,
    repetition_penalty_min_size=2,
    repetition_penalty_max=15.0,
    repetition_penalty_normalize=False,

    # Penalty weight adjustment
    penalty_weight_enabled=True,
    penalty_weight_mode="exponential",
    penalty_weight_threshold=5.0,
    penalty_weight_multiplier=2.0,
    penalty_weight_max=1000.0,
)

search = GivenClauseSearch(options=options)
result = search.run(usable=usable_clauses, sos=sos_clauses)
```

### Direct Penalty Computation

```python
from pyladr.search.penalty_weight import (
    PenaltyWeightConfig,
    PenaltyWeightMode,
    penalty_adjusted_weight,
)

config = PenaltyWeightConfig(
    enabled=True,
    threshold=5.0,
    multiplier=2.0,
    max_adjusted_weight=1000.0,
    mode=PenaltyWeightMode.EXPONENTIAL,
)

# Compute adjusted weight for a clause with base weight 10 and penalty 12
adjusted = penalty_adjusted_weight(10.0, 12.0, config)
```

---

## Troubleshooting

### Proofs that previously worked now fail

The penalty system never discards clauses — it only reorders them. If a
proof fails with penalties enabled, the search likely hit a resource limit
before reaching the proof. Try:

1. Increase `max_given` or `max_seconds`
2. Lower `--penalty-weight-multiplier` (less aggressive deprioritization)
3. Raise `--penalty-weight-threshold` (fewer clauses affected)
4. Use `--penalty-weight-mode step` (mildest adjustment)

### Search is slower with penalties enabled

The overhead should be < 5% in most cases. If you see significant slowdown:

1. Disable `--repetition-penalty-normalize` (Phase 2 is more expensive)
2. Increase `--repetition-penalty-min-size` to skip small subterms
3. Check if the penalty is effective — if it's not reducing the search,
   disable it

### Clauses with low penalty are still deprioritized

Check the threshold. Only clauses with combined penalty >= threshold are
affected. If the threshold is too low, many clauses will be adjusted.

### C Prover9 compatibility

With all penalty options disabled (the default), PyLADR's search behavior
is identical to C Prover9. Enabling penalty options adds PyLADR-specific
heuristics that may find proofs in fewer steps but may also change which
proof is found.
