# Repetition Bias Integration Mechanism Design

## Overview

This document defines how repetition bias should be integrated into the existing clause selection system to penalize structurally repetitious clauses. The design ensures compatibility with both traditional weight/age-based selection and ML-enhanced selection while maintaining backward compatibility.

## 1. Integration Architecture

### 1.1 Selection System Extension

The repetition bias integrates at the **scoring level** in both selection paths:

```python
class RepetitionBiasedSelection(EmbeddingEnhancedSelection):
    """Selection with repetition bias integrated into scoring."""

    def __init__(self,
                 repetition_detector: RepetitionDetector,
                 bias_config: RepetitionBiasConfig,
                 **kwargs):
        super().__init__(**kwargs)
        self.repetition_detector = repetition_detector
        self.bias_config = bias_config
```

**Integration Points:**
1. **Traditional Selection**: Modify weight comparison to include repetition penalty
2. **ML-Enhanced Selection**: Integrate repetition score into ML scoring blend
3. **Configuration**: Extend existing configuration system
4. **CLI**: Add command-line options for bias control

### 1.2 Backward Compatibility

**Default Behavior:**
- Repetition bias is **disabled by default** (`bias_strength = 0.0`)
- When disabled, behavior is identical to existing selection
- All repetition analysis is skipped when bias is disabled
- No performance impact when not in use

## 2. Bias Configuration System

### 2.1 Configuration Class

```python
@dataclass(frozen=True, slots=True)
class RepetitionBiasConfig:
    """Configuration for repetition bias in clause selection."""

    # Master enable/disable
    enabled: bool = False

    # Bias strength [0.0, 1.0] - how much repetition affects selection
    bias_strength: float = 0.3

    # Repetition threshold [0.0, 1.0] - minimum repetition score to trigger bias
    repetition_threshold: float = 0.2

    # Penalty curve shape ['linear', 'exponential', 'step']
    penalty_curve: str = 'exponential'

    # Maximum penalty multiplier [1.0, inf] - how much to penalize worst cases
    max_penalty_factor: float = 3.0

    # Age-based selection bias behavior ['skip', 'apply', 'light']
    age_selection_bias: str = 'light'

    # Performance limits
    max_analysis_time_ms: float = 5.0
    skip_large_clauses: bool = True
    large_clause_threshold: int = 50  # literals

    # Logging and debugging
    log_bias_decisions: bool = False
    collect_statistics: bool = True
```

### 2.2 Command-Line Integration

**New CLI Options:**
```bash
# Enable/disable repetition bias
--repetition-bias / --no-repetition-bias

# Set bias strength
--repetition-bias-strength 0.3      # [0.0, 1.0]

# Set repetition threshold
--repetition-threshold 0.2          # [0.0, 1.0]

# Penalty curve shape
--repetition-penalty-curve exponential  # linear|exponential|step

# Maximum penalty factor
--repetition-max-penalty 3.0        # [1.0, inf]

# Age selection behavior
--repetition-age-bias light         # skip|apply|light

# Performance tuning
--repetition-max-time 5.0           # milliseconds
--repetition-large-threshold 50     # literals
```

**Integration with Existing Options:**
```bash
# Combined with ML options
prover9 --ml-weight 0.4 --repetition-bias --repetition-bias-strength 0.3

# Traditional + bias
prover9 --repetition-bias --repetition-bias-strength 0.5

# Full configuration example
prover9 \
  --ml-weight 0.4 \
  --repetition-bias \
  --repetition-bias-strength 0.3 \
  --repetition-threshold 0.25 \
  --repetition-penalty-curve exponential \
  --repetition-max-penalty 4.0
```

## 3. Scoring Integration

### 3.1 Traditional Selection Integration

**Modified Weight Comparison:**
```python
def _weight_compare_with_bias(self, a: Clause, b: Clause) -> int:
    """Compare clauses by effective weight (weight + repetition penalty)."""

    effective_weight_a = self._compute_effective_weight(a)
    effective_weight_b = self._compute_effective_weight(b)

    if effective_weight_a != effective_weight_b:
        return -1 if effective_weight_a < effective_weight_b else 1

    # Tie-breaking by ID (existing behavior)
    if a.id != b.id:
        return -1 if a.id < b.id else 1
    return 0

def _compute_effective_weight(self, clause: Clause) -> float:
    """Compute effective weight including repetition penalty."""
    if not self.bias_config.enabled:
        return clause.weight

    base_weight = clause.weight

    # Get repetition analysis (cached)
    rep_score = self._get_repetition_score(clause)

    # Apply bias penalty
    penalty_factor = self._compute_penalty_factor(rep_score)

    return base_weight * penalty_factor
```

**Penalty Factor Calculation:**
```python
def _compute_penalty_factor(self, repetition_score: float) -> float:
    """Compute penalty factor from repetition score."""
    if repetition_score < self.bias_config.repetition_threshold:
        return 1.0  # No penalty

    # Normalize score above threshold
    normalized = (repetition_score - self.bias_config.repetition_threshold) / \
                (1.0 - self.bias_config.repetition_threshold)

    # Apply penalty curve
    if self.bias_config.penalty_curve == 'linear':
        penalty_weight = normalized
    elif self.bias_config.penalty_curve == 'exponential':
        penalty_weight = normalized ** 2
    else:  # 'step'
        penalty_weight = 1.0 if normalized > 0.5 else 0.0

    # Compute final penalty factor
    max_penalty = self.bias_config.max_penalty_factor
    bias_strength = self.bias_config.bias_strength

    penalty_factor = 1.0 + (max_penalty - 1.0) * penalty_weight * bias_strength

    return penalty_factor
```

### 3.2 ML-Enhanced Selection Integration

**Extended ML Score Computation:**
```python
def _compute_ml_score_with_bias(self,
                               embedding: list[float],
                               clause: Clause,
                               clause_weight: float = 0.0,
                               ml_weight: float = 0.0) -> float:
    """Compute ML score including repetition bias component."""

    # Original ML score components
    div_w = self.ml_config.diversity_weight
    pp_w = self.ml_config.proof_potential_weight
    weight_exp_w = ml_weight * 1.0

    # New: repetition bias component
    rep_bias_w = self.bias_config.bias_strength if self.bias_config.enabled else 0.0

    total_w = div_w + pp_w + weight_exp_w + rep_bias_w
    if total_w == 0:
        return 0.0

    # Compute individual components
    diversity = self._diversity_score(embedding)
    proof_potential = self._proof_potential_score(embedding)
    weight_exploration = self._weight_exploration_score(clause_weight, ml_weight)

    # New: repetition bias component (NEGATIVE - penalty)
    repetition_bias = self._repetition_bias_score(clause)

    return (div_w * diversity +
            pp_w * proof_potential +
            weight_exp_w * weight_exploration +
            rep_bias_w * repetition_bias) / total_w

def _repetition_bias_score(self, clause: Clause) -> float:
    """Compute repetition bias component (negative penalty)."""
    rep_score = self._get_repetition_score(clause)

    if rep_score < self.bias_config.repetition_threshold:
        return 0.0  # Neutral

    # Convert repetition score to negative bias (penalty)
    penalty_factor = self._compute_penalty_factor(rep_score)

    # Map penalty factor [1.0, max_penalty] to bias score [-1.0, 0.0]
    max_penalty = self.bias_config.max_penalty_factor
    normalized_penalty = (penalty_factor - 1.0) / (max_penalty - 1.0)

    return -normalized_penalty  # Negative score (penalty)
```

### 3.3 Age-Based Selection Handling

**Configurable Age Bias Behavior:**
```python
def _select_by_order_with_bias(self, sos: ClauseList, order: SelectionOrder) -> Clause | None:
    """Extended selection with repetition bias."""

    if order == SelectionOrder.AGE:
        return self._age_select_with_bias(sos)
    elif order == SelectionOrder.WEIGHT:
        return self._weight_select_with_bias(sos)
    else:
        return super()._select_by_order(sos, order)

def _age_select_with_bias(self, sos: ClauseList) -> Clause | None:
    """Age-based selection with configurable repetition bias."""

    age_bias = self.bias_config.age_selection_bias

    if age_bias == 'skip':
        # Original behavior - pure FIFO
        return sos.first

    elif age_bias == 'apply':
        # Apply full repetition bias to age selection
        candidates = list(itertools.islice(sos, 10))  # Consider first 10
        return min(candidates, key=lambda c: (c.id, self._compute_effective_weight(c)))

    else:  # 'light'
        # Light bias: skip highly repetitious first few clauses
        for i, clause in enumerate(sos):
            if i >= 5:  # Max look-ahead
                break
            rep_score = self._get_repetition_score(clause)
            if rep_score < 0.7:  # Accept if not highly repetitious
                return clause

        # Fallback to first if all are highly repetitious
        return sos.first
```

## 4. Caching and Performance

### 4.1 Repetition Score Caching

**Clause-Level Caching:**
```python
class RepetitionScoreCache:
    """Cache repetition scores to avoid re-computation."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.cache = {}  # clause_id -> (score, timestamp)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get_score(self, clause: Clause, detector: RepetitionDetector) -> float:
        """Get cached or computed repetition score."""
        now = time.time()

        # Check cache
        if clause.id in self.cache:
            score, timestamp = self.cache[clause.id]
            if now - timestamp < self.ttl_seconds:
                return score

        # Compute and cache
        try:
            score = detector.get_repetition_score(clause)
            self._add_to_cache(clause.id, score, now)
            return score
        except Exception:
            # Performance fallback - return neutral score
            return 0.0

    def _add_to_cache(self, clause_id: int, score: float, timestamp: float):
        """Add score to cache with size management."""
        if len(self.cache) >= self.max_size:
            # Simple LRU eviction
            oldest_id = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_id]

        self.cache[clause_id] = (score, timestamp)
```

### 4.2 Performance Safeguards

**Time-Based Limits:**
```python
def _get_repetition_score(self, clause: Clause) -> float:
    """Get repetition score with performance safeguards."""

    # Skip large clauses if configured
    if (self.bias_config.skip_large_clauses and
        len(clause.literals) > self.bias_config.large_clause_threshold):
        return 0.0

    # Time-limited computation
    start_time = time.perf_counter()
    try:
        score = self._score_cache.get_score(clause, self.repetition_detector)

        # Check time limit
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > self.bias_config.max_analysis_time_ms:
            logger.warning(f"Repetition analysis timeout: {elapsed_ms:.1f}ms")

        return score
    except Exception as e:
        logger.debug(f"Repetition analysis failed: {e}")
        return 0.0  # Neutral score on failure
```

## 5. Statistics and Monitoring

### 5.1 Bias Statistics

```python
@dataclass(slots=True)
class RepetitionBiasStats:
    """Statistics for repetition bias effectiveness."""

    # Selection counts
    biased_selections: int = 0
    unbiased_selections: int = 0
    bias_skipped_large: int = 0
    bias_timeouts: int = 0

    # Score distributions
    avg_repetition_score: float = 0.0
    max_repetition_score: float = 0.0
    penalty_applications: int = 0

    # Performance metrics
    avg_analysis_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    def record_biased_selection(self, rep_score: float, analysis_time: float):
        self.biased_selections += 1
        self._update_score_stats(rep_score)
        self._update_time_stats(analysis_time)

    def report(self) -> str:
        total = self.biased_selections + self.unbiased_selections
        if total == 0:
            return "repetition_bias: no selections made"

        bias_pct = 100.0 * self.biased_selections / total
        return (
            f"repetition_bias: {self.biased_selections}/{total} biased ({bias_pct:.1f}%), "
            f"penalties={self.penalty_applications}, "
            f"avg_rep_score={self.avg_repetition_score:.3f}, "
            f"avg_time={self.avg_analysis_time_ms:.2f}ms, "
            f"cache_hit_rate={self.cache_hit_rate:.1f}%"
        )
```

## 6. API Specification

### 6.1 Configuration API

```python
# Factory function for easy setup
def create_repetition_biased_selection(
    ml_config: MLSelectionConfig = None,
    bias_config: RepetitionBiasConfig = None,
    embedding_provider: EmbeddingProvider = None
) -> RepetitionBiasedSelection:
    """Create selection system with repetition bias."""

    bias_config = bias_config or RepetitionBiasConfig()
    detector = RepetitionDetector(config=RepetitionConfig()) if bias_config.enabled else None

    return RepetitionBiasedSelection(
        ml_config=ml_config or MLSelectionConfig(),
        embedding_provider=embedding_provider,
        repetition_detector=detector,
        bias_config=bias_config
    )

# Configuration from command line args
def configure_from_args(args) -> tuple[MLSelectionConfig, RepetitionBiasConfig]:
    """Extract configuration from parsed command-line arguments."""

    ml_config = MLSelectionConfig(
        enabled=args.ml_weight > 0,
        ml_weight=args.ml_weight,
        # ... other ML config
    )

    bias_config = RepetitionBiasConfig(
        enabled=getattr(args, 'repetition_bias', False),
        bias_strength=getattr(args, 'repetition_bias_strength', 0.3),
        repetition_threshold=getattr(args, 'repetition_threshold', 0.2),
        penalty_curve=getattr(args, 'repetition_penalty_curve', 'exponential'),
        max_penalty_factor=getattr(args, 'repetition_max_penalty', 3.0),
        age_selection_bias=getattr(args, 'repetition_age_bias', 'light'),
        # ... other bias config
    )

    return ml_config, bias_config
```

### 6.2 Integration with Search

```python
# Example integration with GivenClauseSearch
def create_search_with_repetition_bias(options, **kwargs):
    """Create search instance with repetition bias if configured."""

    # Extract configurations from options
    ml_config, bias_config = configure_from_args(options)

    # Create embedding provider if needed
    embedding_provider = None
    if ml_config.enabled:
        embedding_provider = create_embedding_provider(options)

    # Create biased selection system
    selection = create_repetition_biased_selection(
        ml_config=ml_config,
        bias_config=bias_config,
        embedding_provider=embedding_provider
    )

    # Create search with biased selection
    return GivenClauseSearch(
        options=options,
        selection=selection,
        **kwargs
    )
```

## 7. Usage Examples

### 7.1 Basic Usage

```python
# Enable repetition bias with default settings
args = argparse.Namespace(
    repetition_bias=True,
    repetition_bias_strength=0.3,
    ml_weight=0.0
)

search = create_search_with_repetition_bias(args)
result = search.search_until_solution(clauses)
```

### 7.2 Combined with ML

```python
# Repetition bias + ML enhancement
args = argparse.Namespace(
    ml_weight=0.4,
    repetition_bias=True,
    repetition_bias_strength=0.3,
    repetition_threshold=0.25,
    repetition_penalty_curve='exponential'
)

search = create_search_with_repetition_bias(args)
result = search.search_until_solution(clauses)

# Access statistics
selection = search.selection
print(selection.ml_stats.report())
print(selection.bias_stats.report())
```

### 7.3 Fine-Tuning Configuration

```python
# Custom configuration for specific domain
bias_config = RepetitionBiasConfig(
    enabled=True,
    bias_strength=0.5,           # Strong bias
    repetition_threshold=0.15,    # Low threshold
    penalty_curve='exponential',  # Aggressive curve
    max_penalty_factor=5.0,       # High penalty
    age_selection_bias='apply',   # Full bias on age selection
    max_analysis_time_ms=10.0,    # Allow more analysis time
)

selection = RepetitionBiasedSelection(
    bias_config=bias_config,
    repetition_detector=RepetitionDetector()
)
```

This integration design provides flexible, performant repetition bias that seamlessly integrates with both traditional and ML-enhanced clause selection while maintaining full backward compatibility.