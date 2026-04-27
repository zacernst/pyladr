# Repetition Detection Strategy Design

## Overview

This document defines a strategy for detecting structural repetition in clauses to enable biased clause selection against repetitious clauses. The goal is to identify clauses that contain complex subclauses appearing multiple times, which may indicate less promising search directions.

## 1. Defining "Complex Subclauses"

### 1.1 Complexity Thresholds

A subclause is considered "complex enough" to track for repetition if it meets these criteria:

**Primary Definition: Complex Terms**
- **Minimum depth**: Term depth ≥ 2 (e.g., `f(g(x))` but not `f(x)`)
- **Minimum symbols**: Symbol count ≥ 3 (excludes simple binary operations)
- **Non-trivial structure**: Contains at least one function symbol with arity ≥ 2, or nested function calls

**Examples:**
- `f(g(x))` ✓ (depth 2, symbols 3)
- `+(*(x,y), z)` ✓ (depth 2, symbols 4, binary ops with nesting)
- `f(x)` ✗ (depth 1, too simple)
- `x` ✗ (variable, depth 0)
- `c` ✗ (constant, depth 0)

### 1.2 Term Categories

**Trackable Complex Terms:**
1. **Nested function calls**: `f(g(x))`, `h(f(a,b), g(c))`
2. **Complex predicates**: `P(f(x,y), g(z))`
3. **Arithmetic expressions**: `+(*(x,y), *(z,w))`
4. **Logical structures**: `&(P(x), Q(f(x)))`

**Non-trackable Terms:**
1. **Simple atoms**: Variables, constants
2. **Shallow terms**: `f(x)`, `P(a)`
3. **Pure variable terms**: `f(x,y)` where all args are variables

## 2. Repetition Detection Algorithm

### 2.1 Core Algorithm: SubtermExtraction with Hashing

```python
class RepetitionDetector:
    def __init__(self,
                 min_depth: int = 2,
                 min_symbols: int = 3,
                 similarity_threshold: float = 0.8):
        self.min_depth = min_depth
        self.min_symbols = min_symbols
        self.similarity_threshold = similarity_threshold

    def detect_repetition(self, clause: Clause) -> RepetitionReport:
        """Main entry point for repetition detection."""
        complex_subterms = []

        # Extract complex subterms from all literals
        for literal in clause.literals:
            subterms = self._extract_complex_subterms(literal.atom)
            complex_subterms.extend(subterms)

        # Detect exact and structural repetitions
        exact_groups = self._group_exact_matches(complex_subterms)
        structural_groups = self._group_structural_matches(complex_subterms)

        return RepetitionReport(
            exact_repetitions=exact_groups,
            structural_repetitions=structural_groups,
            repetition_score=self._compute_repetition_score(exact_groups, structural_groups),
            total_complex_subterms=len(complex_subterms)
        )
```

### 2.2 Subterm Extraction

**Recursive Extraction with Pruning:**
```python
def _extract_complex_subterms(self, term: Term, depth: int = 0) -> list[Term]:
    """Extract all complex subterms meeting complexity thresholds."""
    subterms = []

    # Check if current term qualifies as complex
    if self._is_complex_enough(term, depth):
        subterms.append(term)

    # Recursively extract from arguments
    if term.is_complex:
        for arg in term.args:
            subterms.extend(self._extract_complex_subterms(arg, depth + 1))

    return subterms

def _is_complex_enough(self, term: Term, depth: int) -> bool:
    """Check if a term meets complexity thresholds."""
    if term.is_variable or term.is_constant:
        return False

    return (depth >= self.min_depth and
            term.symbol_count >= self.min_symbols and
            self._has_nontrivial_structure(term))
```

### 2.3 Repetition Types

**Type 1: Exact Structural Matches**
- Terms with identical structure and symbol assignments
- Use structural hashing for O(1) grouping
- Example: `f(g(x))` appears multiple times exactly

**Type 2: Isomorphic Structural Matches**
- Same structure, different variable assignments
- Normalize variables to canonical form for comparison
- Example: `f(g(x))` and `f(g(y))` are isomorphic

**Type 3: Similar Complexity Patterns**
- Terms with similar "shape" but different symbols
- Use structure signature matching
- Example: `f(g(x))` and `h(k(y))` have similar complexity

### 2.4 Complexity Analysis

**Time Complexity:** O(n log n) where n = total subterm count
- Subterm extraction: O(n) per clause
- Hashing and grouping: O(n log n)
- Structural comparison: O(n log n) with efficient hashing

**Space Complexity:** O(n) for subterm storage and hash tables

## 3. Repetition Metrics

### 3.1 Repetition Score Calculation

```python
def _compute_repetition_score(self, exact_groups: list, structural_groups: list) -> float:
    """Compute normalized repetition score [0, 1]."""

    # Base scores
    exact_score = sum(max(0, len(group) - 1) for group in exact_groups if len(group) > 1)
    structural_score = sum(max(0, len(group) - 1) * 0.7 for group in structural_groups if len(group) > 1)

    # Complexity weighting: heavier subterms contribute more to repetition
    weighted_exact = sum(
        max(0, len(group) - 1) * self._complexity_weight(group[0])
        for group in exact_groups if len(group) > 1
    )

    # Normalization: score relative to total complexity
    total_complexity = sum(self._complexity_weight(term) for group in exact_groups + structural_groups for term in group)

    if total_complexity == 0:
        return 0.0

    raw_score = (weighted_exact + structural_score) / total_complexity
    return min(1.0, raw_score)  # Cap at 1.0
```

### 3.2 Complexity Weighting

**Weight Assignment:**
- **Symbol count factor**: `log(1 + symbol_count)`
- **Depth factor**: `depth^1.5`
- **Arity factor**: `sqrt(max_arity)`
- **Combined**: `symbol_factor * depth_factor * arity_factor`

**Rationale:** More complex subterms should contribute more heavily to repetition scoring, as their repetition is more indicative of problematic structure.

## 4. Structural Similarity Detection

### 4.1 Structure Signatures

**Canonical Term Signatures:**
```python
def _compute_structure_signature(self, term: Term) -> tuple:
    """Generate canonical signature for structural comparison."""
    if term.is_variable:
        return ("VAR",)
    elif term.is_constant:
        return ("CONST", term.symnum)
    else:
        arg_sigs = tuple(self._compute_structure_signature(arg) for arg in term.args)
        return ("COMPLEX", term.symnum, term.arity, arg_sigs)
```

**Isomorphic Matching:**
```python
def _normalize_variables(self, term: Term) -> Term:
    """Normalize variable assignments for isomorphic comparison."""
    var_mapping = {}
    next_var_id = 0

    def normalize_recursive(t: Term) -> Term:
        nonlocal next_var_id
        if t.is_variable:
            if t.private_symbol not in var_mapping:
                var_mapping[t.private_symbol] = next_var_id
                next_var_id += 1
            return Term(private_symbol=var_mapping[t.private_symbol])
        elif t.is_complex:
            new_args = tuple(normalize_recursive(arg) for arg in t.args)
            return Term(private_symbol=t.private_symbol, arity=t.arity, args=new_args)
        else:
            return t  # Constants unchanged

    return normalize_recursive(term)
```

## 5. Performance Optimizations

### 5.1 Caching Strategy

**Term-Level Caching:**
- Cache complex subterm extraction results on Term objects
- Use weak references to avoid memory leaks
- Invalidate cache when term structure changes

**Clause-Level Caching:**
- Cache repetition analysis results on Clause objects
- Update cache incrementally when clauses are modified
- Expire cache after reasonable time window

### 5.2 Early Termination

**Pruning Strategies:**
1. **Size limits**: Skip analysis for clauses with > 100 subterms
2. **Complexity limits**: Skip terms with depth > 10
3. **Time limits**: Abort analysis after 10ms per clause

### 5.3 Batch Processing

**Amortized Analysis:**
- Group clauses by similarity for batch processing
- Share subterm extraction across similar clauses
- Use bloom filters for fast negative lookups

## 6. API Interface

### 6.1 Core Classes

```python
@dataclass(frozen=True, slots=True)
class RepetitionReport:
    """Results of repetition analysis for a clause."""
    exact_repetitions: list[list[Term]]
    structural_repetitions: list[list[Term]]
    repetition_score: float  # [0, 1]
    total_complex_subterms: int
    computation_time_ms: float

@dataclass(frozen=True, slots=True)
class RepetitionConfig:
    """Configuration for repetition detection."""
    enabled: bool = True
    min_depth: int = 2
    min_symbols: int = 3
    similarity_threshold: float = 0.8
    max_analysis_time_ms: float = 10.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
```

### 6.2 Integration Interface

```python
class RepetitionDetector:
    def __init__(self, config: RepetitionConfig = None):
        self.config = config or RepetitionConfig()
        self._cache = {} if config.cache_enabled else None

    def analyze_clause(self, clause: Clause) -> RepetitionReport:
        """Analyze a single clause for repetition."""

    def analyze_clause_batch(self, clauses: list[Clause]) -> list[RepetitionReport]:
        """Batch analysis for performance."""

    def get_repetition_score(self, clause: Clause) -> float:
        """Quick access to just the repetition score."""
```

## 7. Example Usage

```python
# Configuration
config = RepetitionConfig(
    enabled=True,
    min_depth=2,
    min_symbols=3,
    similarity_threshold=0.8
)

# Detection
detector = RepetitionDetector(config)
report = detector.analyze_clause(clause)

print(f"Repetition score: {report.repetition_score:.3f}")
print(f"Exact repetitions: {len(report.exact_repetitions)} groups")
print(f"Structural repetitions: {len(report.structural_repetitions)} groups")

# Integration with selection
if report.repetition_score > 0.5:
    print("High repetition detected - apply bias penalty")
```

This design provides a comprehensive, performance-oriented approach to detecting structural repetition in clauses while maintaining compatibility with the existing Term and Clause infrastructure.