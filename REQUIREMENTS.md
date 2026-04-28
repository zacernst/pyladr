# PyLADR Requirements Specification

Formal, measurable requirements for PyLADR — the Python port of Prover9.
Derived from validation missions and cross-validation testing (last updated 2026-04-27).

**Status legend:** PASS = verified, FIXED = was broken and repaired, PENDING = not yet verified

---

## 1. C Prover9 Compatibility (REQ-C)

### REQ-C001: Process Exit Codes

Process exit codes must match C Prover9 exactly.

| Search Outcome       | ExitCode Enum          | Process Exit |
|----------------------|------------------------|--------------|
| Proof found          | MAX_PROOFS_EXIT (1)    | 0            |
| SOS exhausted        | SOS_EMPTY_EXIT (2)     | 2            |
| Given limit          | MAX_GIVEN_EXIT (3)     | 3            |
| Kept limit           | MAX_KEPT_EXIT (4)      | 4            |
| Time limit           | MAX_SECONDS_EXIT (5)   | 5            |
| Generated limit      | MAX_GENERATED_EXIT (6) | 6            |
| Fatal error          | FATAL_EXIT (7)         | 1            |

- **Status:** PASS
- **Acceptance:** All 7 exit code mappings verified via `tests/cross_validation/test_search_equivalence.py`

### REQ-C002: Auto-Inference Detection Default

`set(auto)` must be enabled by default. The auto-inference cascade analyzes problem structure and enables appropriate inference rules (paramodulation for equational, hyper-resolution for Horn).

- **Status:** PASS
- **Acceptance:** `simple_group.in` and `lattice_absorption.in` prove without explicit `set(auto)`

### REQ-C003: Equational Problem Equivalence

All problems provable by C Prover9 must also be provable by PyLADR given the same input. Statistics may differ but theorem/failure outcomes must match.

- **Status:** PASS
- **Acceptance:** Cross-validation suite in `tests/cross_validation/` passes on all reference problems

### REQ-C004: Proof Output Format

Output must include standard C Prover9 sections: Header, INPUT, PROCESS INITIAL CLAUSES, given clause trace, PROOF, STATISTICS, conclusion line, and process exit line.

- **Status:** PASS
- **Acceptance:** `tests/cross_validation/test_end_to_end.py` validates section ordering

### REQ-C005: Justification Type Compatibility

Justification enum values must match C `just.h` exactly (INPUT=0 through IVY=21). Justification chains must be traceable from empty clause back to input clauses.

- **Status:** PASS
- **Acceptance:** `tests/unit/test_justification.py`, `tests/cross_validation/test_equational_reasoning.py`

### REQ-C006: Search Loop Structure

The given-clause algorithm must follow the C `search()` structure:
`_init_clauses` -> `_process_initial_clauses` -> `while _inferences_to_make(): _make_inferences(); _limbo_process()`

- **Status:** PASS
- **Acceptance:** `tests/cross_validation/test_search_equivalence.py::TestSearchLoop`

### REQ-C007: LADR Input Syntax

Parser must accept standard LADR syntax including `formulas(sos)`, `formulas(goals)`, `formulas(usable)`, `set()`, `clear()`, `assign()` directives, operator precedence tables, and C-compatible symbol names.

- **Status:** PASS (known limitation: double-prime syntax `x''` not yet supported)
- **Acceptance:** `tests/unit/test_parsing.py`

---

## 2. ML Enhancement Architecture (REQ-ML)

### REQ-ML001: Protocol Compliance

All embedding providers must implement the `EmbeddingProvider` protocol (`get_embeddings_batch()` signature). Non-conforming providers must be rejected at construction time.

- **Status:** PASS

### REQ-ML002: Graceful Degradation

When ML components are unavailable (no torch, no model file, provider error), the system must fall back to `NoOpEmbeddingProvider` and continue with traditional clause selection. No exceptions may propagate to the search loop.

- **Status:** PASS

### REQ-ML003: Non-Intrusion

ML enhancements must not modify the core search loop, clause data structures, or justification format. All ML features must be additive and disabled by default.

- **Status:** PASS

### REQ-ML-RGP-001: RGP Selection Direction

RGP (Random Goal Proximity) selection must use argmin cosine distance to goal embeddings — lower distance = higher selection priority. Verified correct as of 2026-04-27 audit.

- **Status:** PASS
- **Acceptance:** `GoalDistanceScorer.nearest_goal_distance` returns lower values for goal-similar clauses; selection uses argmin

### REQ-ML-RGP-002: RGP Goal Embedding Normalization

Goal embeddings for RGP must be computed from deskolemized (signs forced True, constants→variables) versions of DENY-justified clauses, so comparison captures structural shape not specific constant identities.

- **Status:** PASS
- **Acceptance:** `_deskolemize_clause` forces signs positive and replaces all constants with variables before embedding

### REQ-ML-ENHANCE-001: Embedding Enhancement Scale Invariant

`GoalDirectedEmbeddingProvider._enhance_embedding` scale invariant: scale(d=0.0) = 1-weight (smallest), scale(d=1.0) = 1.0 (largest). Clauses close to the deskolemized goal get the smallest scale factor (and thus smallest norm), which `proof_potential_score` interprets as most promising.

- **Status:** PASS
- **Acceptance:** `_enhance_embedding` uses `scale = 1.0 - weight * (1.0 - distance)`, verified by unit tests

---

## 3. Quality and Testing (REQ-Q)

### REQ-Q001: Core Test Suite Zero Failures

All core unit tests must pass with zero failures.

- **Status:** PASS
- **Baseline:** 3,622 passed, 198 skipped, 3 xfailed (2026-04-27)
- **Acceptance:** `python3 -m pytest tests/ --ignore=tests/benchmarks --ignore=tests/soundness -q` reports 0 failures

### REQ-Q002: Zero Regression Tolerance

No change may introduce test regressions. All PRs must pass the full test suite before merge.

- **Status:** PASS
- **Acceptance:** CI gate enforces zero test failures on all branches

---

## 4. Performance (REQ-P)

### REQ-P001: Search Throughput Baseline

Search throughput must remain at or above 3.2 given clauses/sec on medium paramodulation problems.

- **Status:** PASS
- **Acceptance:** Measured throughput on `simple_group.in` >= 3.2 given/sec

### REQ-P002: Memory Boundedness

Memory usage must remain bounded during search. SOS displacement and clause deletion must prevent unbounded growth.

- **Status:** PENDING
- **Acceptance:** Peak RSS does not exceed 2x initial allocation on problems with >10,000 kept clauses

### REQ-PERF-BACKSUB-001: Back-Subsumption Throughput Baseline

Back-subsumption throughput baseline post-optimization: ≥12 given/sec on single-predicate equational problems (measured 16 given/sec on vampire.in with weight filter + Context/Trail reuse, April 2026).

- **Status:** PASS
- **Acceptance:** Measured throughput on `vampire.in` ≥12 given/sec with optimized back-subsumption path

---

## 5. Regression Prevention (REQ-R)

### REQ-R001: Goal Negation and Skolemization

Goal formulas must be correctly negated and skolemized. No trivial proofs that bypass actual theorem proving work.

- **Status:** PASS
- **Acceptance:** `tests/unit/test_goal_skolemization.py` (13 tests), complex problems require appropriate search effort

### REQ-R002: Given Clause Trace Display

Given clause trace must be displayed during search for debugging and progress visibility.

- **Status:** PASS
- **Acceptance:** `tests/unit/test_given_clause_trace.py` (28 tests)

### REQ-R003: Horn Problem Hyper-Resolution

Horn problems must automatically enable hyper-resolution through auto-inference cascade.

- **Status:** PASS
- **Acceptance:** `tests/unit/test_auto_inference.py` (37 tests), `vampire.in` proves with `[hyper_res]` justifications

### REQ-R004: Symbol Name Consistency

Given clause trace must preserve original function/predicate names from input.

- **Status:** PASS
- **Acceptance:** Given clause output uses user-provided symbol names, not internal IDs

### REQ-R005: Max Proofs Parameter

`assign(max_proofs, N)` must continue search after first proof and terminate after N proofs.

- **Status:** PASS
- **Acceptance:** Multi-proof search finds and reports N distinct proofs

### REQ-R006: Proof Display During Multi-Proof Search

Proofs must be printed incrementally as found, not buffered until search completion.

- **Status:** PASS
- **Acceptance:** Proof callback fires immediately on each proof discovery

### REQ-R007: Soundness — Logical Correctness

All proofs must be logically valid. No trivial proofs where extensive search is required.

- **Status:** PASS
- **Acceptance:** Cross-validation confirms equivalent proof behavior vs C Prover9

---

## 6. Integration (REQ-I)

### REQ-I001: CLI Compatibility

`pyprover9` CLI must accept `-f FILE` input syntax matching C Prover9.

- **Status:** PASS

### REQ-I002: Subprocess Invocability

PyLADR must be invocable as a subprocess with correct exit codes for scripting and CI/CD.

- **Status:** PASS

---

## Summary

| Category | Total | PASS | PENDING |
|----------|-------|------|---------|
| Compatibility (REQ-C) | 7 | 7 | 0 |
| ML Architecture (REQ-ML) | 6 | 6 | 0 |
| Quality (REQ-Q) | 2 | 2 | 0 |
| Performance (REQ-P) | 3 | 2 | 1 |
| Regression (REQ-R) | 7 | 7 | 0 |
| Integration (REQ-I) | 2 | 2 | 0 |
| **Total** | **27** | **26** | **1** |
