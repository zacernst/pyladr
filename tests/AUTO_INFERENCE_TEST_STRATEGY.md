# Auto-Inference and Horn Problem Testing Strategy (REQ-R003)

## Overview

This document specifies the testing strategy for auto-inference cascade,
Horn problem detection, hyper-resolution activation, parser directive
handling, and cross-validation with C Prover9. It covers both already-
implemented functionality and the missing features identified in Tasks #5-#7.

## Architecture Summary

### Current State

| Component | Status | Location |
|-----------|--------|----------|
| `_analyze_problem()` | Implemented | `prover9.py:316-343` |
| `_neg_pos_depth_difference()` | Implemented | `prover9.py:346-369` |
| `_apply_settings()` | Implemented | `prover9.py:372-465` |
| `hyper_resolution` in SearchOptions | Implemented | `given_clause.py:87` |
| `hyper_resolve()` inference | Implemented | `inference/hyper_resolution.py` |
| Search loop hyper-resolution call | Implemented | `given_clause.py:413-418` |
| Parser `set()`/`clear()`/`assign()` | **Skipped** | `ladr_parser.py:215-218` |
| `set(auto)` conditional gating | **Missing** | Always runs `_apply_settings` |
| Unit deletion | **Not implemented** | Mentioned in auto_process but no-op |
| Neg-UR resolution | **Not implemented** | Referenced in C, absent in Python |

### Decision Tree (C Prover9 `auto_inference`)

```
1. If positive equality literals → set(paramodulation), set(demodulation)
2. If (!equality OR !all_units):
   a. Horn + equality      → set(hyper_resolution)
   b. Horn + !equality     → depth_diff > 0 ? hyper_resolution : binary_resolution
   c. Non-Horn             → set(binary_resolution)
3. auto_process:
   a. Horn + neg nonunits  → set(unit_deletion)
   b. Non-Horn             → set(factoring), set(unit_deletion)
```

## Test Classes

### 1. `TestProblemAnalysis` — _analyze_problem correctness

**File:** `tests/unit/test_auto_inference.py`

Tests the `(is_horn, has_equality, all_units)` classification.

| Test | Input | Expected |
|------|-------|----------|
| `test_pure_horn_no_equality` | `-P(x)\|P(f(x)). P(a).` | `(True, False, True)` |
| `test_nonhorn_disjunction` | `P(a)\|Q(a). -P(x)\|R(x).` | `(False, False, False)` |
| `test_unit_equality` | `e*x=x. x'*x=e.` | `(True, True, True)` |
| `test_nonunit_horn_equality` | `f(e,x)=x. -P(x)\|f(x,e)=x. P(a).` | `(True, True, False)` |
| `test_mixed_horn_no_eq` | `-P(x)\|-P(i(x,y))\|P(y). P(i(i(x,y),i(i(y,z),i(x,z)))).` | `(True, False, False)` |
| `test_empty_clause_list` | `(empty)` | `(True, False, True)` — vacuous |
| `test_single_negative_unit` | `-P(a).` | `(True, False, True)` |
| `test_equality_only_negative` | `-f(x)=g(x).` | `(True, False, True)` — negative eq not counted |

### 2. `TestDepthDifference` — _neg_pos_depth_difference

**File:** `tests/unit/test_auto_inference.py`

| Test | Input | Expected |
|------|-------|----------|
| `test_vampire_style_hne` | vampire.in clauses | `depth_diff > 0` (triggers hyper) |
| `test_flat_mixed_clause` | `-P(a)\|Q(a)` | `depth_diff = 0` (triggers binary) |
| `test_deep_negative` | `-P(f(f(a)))\|Q(a)` | `depth_diff > 0` |
| `test_deep_positive` | `-P(a)\|Q(f(f(a)))` | `depth_diff < 0` |
| `test_pure_positive_ignored` | `P(a). Q(b).` | `depth_diff = 0` (no mixed clauses) |
| `test_pure_negative_ignored` | `-P(a).` | `depth_diff = 0` |
| `test_multiple_mixed_clauses` | multiple mixed | sum across all mixed |

### 3. `TestAutoCascadeDecisions` — _apply_settings

**File:** `tests/unit/test_auto_inference.py`

Existing: `tests/unit/test_auto_cascade.py` covers 8 cases. Extend with:

| Test | Problem Type | Expected Inference Rules |
|------|-------------|------------------------|
| `test_hne_depth_positive` | Horn, no eq, depth_diff>0 | hyper=True, binary=False |
| `test_hne_depth_zero` | Horn, no eq, depth_diff=0 | hyper=False, binary=True |
| `test_hne_depth_negative` | Horn, no eq, depth_diff<0 | hyper=False, binary=True |
| `test_unit_equality_only` | Horn, eq, all units | para=True, hyper=False, binary=False |
| `test_nonunit_horn_eq` | Horn, eq, not all units | para=True, hyper=True |
| `test_nonhorn_eq` | Non-Horn, eq | para=True, binary=True |
| `test_nonhorn_no_eq` | Non-Horn, no eq | binary=True |
| `test_auto_process_horn_neg_nonunits` | Horn with neg nonunits | unit_deletion (when implemented) |
| `test_auto_process_nonhorn` | Non-Horn | factoring=True |
| `test_auto_limits_applied` | Any with auto | max_weight=100, sos_limit=20000 |
| `test_explicit_limits_preserved` | assign(max_weight, 50) | max_weight=50 |

### 4. `TestHyperResolutionEndToEnd` — Proof finding via hyper-resolution

**File:** `tests/integration/test_hyper_resolution_e2e.py`

These validate that hyper-resolution actually produces proofs.

| Test | Problem | Expected |
|------|---------|----------|
| `test_simple_horn_proof` | Modus ponens chain: `-P(x)\|Q(x). P(a). goal: Q(a).` | Proof found |
| `test_vampire_in_finds_proof` | vampire.in (condensed detachment) | At least one proof found |
| `test_multi_neg_literal_nucleus` | `-A\|-B\|C. A. B. goal: C.` | Proof via hyper |
| `test_hyper_produces_positive_only` | Various | All resolvents are all-positive |
| `test_hyper_with_given_trace` | Any Horn + hyper | Given clause trace appears in output |
| `test_hyper_respects_max_given` | Large problem | Stops at limit |

### 5. `TestParserDirectives` — set/clear/assign handling

**File:** `tests/unit/test_parser_directives.py`

Tests for Task #7 (parser directive handling). Currently parser skips directives.

| Test | Directive | Expected Behavior |
|------|-----------|-------------------|
| `test_set_auto_recognized` | `set(auto).` | Parsed, stored in settings |
| `test_set_print_given` | `set(print_given).` | `print_given=True` in opts |
| `test_clear_print_given` | `clear(print_given).` | `print_given=False` in opts |
| `test_assign_max_proofs` | `assign(max_proofs, -1).` | `max_proofs=-1` in opts |
| `test_assign_max_weight` | `assign(max_weight, 50).` | `max_weight=50` in opts |
| `test_set_hyper_resolution` | `set(hyper_resolution).` | `hyper_resolution=True` |
| `test_clear_binary_resolution` | `clear(binary_resolution).` | `binary_resolution=False` |
| `test_unknown_directive_warning` | `set(unknown_flag).` | Warning, no crash |
| `test_directive_order_matters` | `set(auto). clear(paramodulation).` | paramodulation=False |
| `test_auto_gating` | Without `set(auto)`, no auto-cascade | Defaults preserved |

### 6. `TestCrossValidationAutoInference` — Python vs C comparison

**File:** `tests/cross_validation/test_auto_inference_compat.py`

Requires C binary at `reference-prover9/bin/prover9`.

| Test | Problem Type | Comparison Points |
|------|-------------|-------------------|
| `test_x2_auto_inference_message` | Equational | Auto_inference output matches C |
| `test_vampire_auto_inference` | HNE | hyper_resolution selected in both |
| `test_nonhorn_auto_inference` | Non-Horn | binary_resolution in both |
| `test_horn_eq_auto_inference` | Horn+eq | para+hyper in both |
| `test_exit_codes_match` | Various | Python exit code = C exit code |
| `test_proof_found_agreement` | Solvable | Both find proof or both don't |
| `test_given_count_comparable` | Various | Python givens within 2x of C |

**Methodology:** Use `c_runner.run_c_prover9_from_string()` and compare with Python output.
Parse "Auto_inference settings:" section from both outputs and compare activated rules.

### 7. `TestAutoInferenceOutput` — Trace message validation

**File:** `tests/unit/test_auto_inference.py`

| Test | Scenario | Expected Output |
|------|----------|----------------|
| `test_auto_inference_header_printed` | Any auto run | "Auto_inference settings:" in stdout |
| `test_auto_process_header_printed` | Any auto run | "Auto_process settings:" in stdout |
| `test_paramodulation_message` | Equational | "set(paramodulation)" in output |
| `test_hyper_resolution_message` | HNE depth>0 | "set(hyper_resolution)" in output |
| `test_binary_resolution_message` | Non-Horn | "set(binary_resolution)" in output |
| `test_quiet_suppresses_auto_messages` | quiet=True | No auto messages in output |
| `test_hne_depth_diff_shown` | HNE | "depth_diff=N" in message |

## Test Input Fixtures

### New fixture files needed in `tests/fixtures/inputs/`:

```
horn_simple.in          - Simple Horn problem for hyper-resolution
horn_deep_neg.in        - HNE with depth_diff > 0 (vampire-style)
horn_flat.in            - HNE with depth_diff <= 0 (binary resolution)
horn_equality.in        - Horn + equality (both para + hyper)
nonhorn_basic.in        - Non-Horn with clear positive disjunction
nonhorn_equality.in     - Non-Horn + equality (para + binary)
set_auto_explicit.in    - Problem with set(auto) directive
directives_test.in      - Multiple set/clear/assign directives
```

## Regression Prevention Strategy

### Critical Invariants to Test

1. **Given clause trace always appears** (REQ-R002) — already covered by `test_given_clause_trace.py`
2. **Auto-inference cascade matches C** — decision tree produces same rule activation
3. **Hyper-resolution produces proofs** — Horn problems that C solves, Python also solves
4. **Parser directives affect behavior** — `set(auto)` gates auto-cascade
5. **No silent failures** — trace messages confirm which rules are active

### CI Integration

Tests should be organized into tiers:

- **Tier 1 (always run, <5s):** Unit tests for `_analyze_problem`, `_neg_pos_depth_difference`, `_apply_settings`, parser directives
- **Tier 2 (always run, <30s):** Integration tests for hyper-resolution proof finding, given clause traces
- **Tier 3 (nightly/manual, <120s):** Cross-validation with C binary, full vampire.in solve

### Markers

```python
@pytest.mark.auto_inference    # All auto-inference tests
@pytest.mark.hyper_resolution  # Hyper-resolution specific
@pytest.mark.c_compat          # Requires C binary
@pytest.mark.horn              # Horn problem tests
```

## Implementation Order

1. **Immediate (no dependencies):** Tests for `_analyze_problem`, `_neg_pos_depth_difference`, `_apply_settings` — all already implemented
2. **After Task #6:** End-to-end hyper-resolution proof tests
3. **After Task #7:** Parser directive tests
4. **After Tasks #5-#7 complete:** Full cross-validation with C Prover9

## Key Gaps Between Python and C

| Feature | C Prover9 | Python | Impact |
|---------|-----------|--------|--------|
| `set(auto)` gating | Auto only when `set(auto)` | Always runs | Different behavior without `set(auto)` |
| `neg_ur_resolution` | Implemented | Missing | Fewer inferences for Horn+eq |
| `unit_deletion` | Implemented | Flag only, no-op | Missing simplification |
| `para_lit_limit` | Set by auto for Horn+eq | Missing | No literal limit on paramodulation |
| `ordered_res` | Cleared for HNE depth<=0 | Missing | May generate more resolvents |
| Predicate elimination | Implemented | Missing | Problem may be reclassified after elimination |
