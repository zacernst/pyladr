# PyLADR Comprehensive Correctness Improvement Plan

## Executive Summary

This plan identifies and prioritizes correctness issues in the PyLADR codebase, provides systematic remediation strategies, and defines a phased improvement roadmap. The focus is on issues that affect **theorem proving correctness** — the soundness and completeness of the search.

**Updated 2026-04-09:** Incorporated findings from all three audit tasks (regression testing, code audit, QA testing). Total: **8 CRITICAL, 5 HIGH, 4 MEDIUM, 3 LOW** issues identified.

---

## 1. Prioritized Issue Inventory

### CRITICAL — Affects Proof Soundness/Completeness

#### C1. Oriented Equality Tracking Uses `id()` — Unsound Under GC

**Location:** `pyladr/inference/paramodulation.py:67-88`
**Issue:** `_oriented_eqs` and `_renamable_flips` use `id(atom)` (Python object identity) to track which equality atoms are oriented. Python's `id()` returns a memory address that can be **reused after garbage collection**. If an oriented atom is GC'd, a new unrelated atom may receive the same `id()`, causing it to be incorrectly treated as oriented — leading to skipped paramodulations or wrong orientations.
**Impact:** Silently incorrect inference control. Can miss valid paramodulants (incompleteness) or apply wrong orientation (unsoundness in demodulation).
**Root Cause:** Module-level mutable sets using ephemeral object identity instead of structural identity.

#### C2. Module-Level Global State Prevents Clean Re-Entrant Search

**Location:** `paramodulation.py:67-68`, `subsumption.py:36`, `ordering/termorder.py:20-21`, `substitution.py:26-27`
**Issue:** Multiple module-level global variables (`_oriented_eqs`, `_renamable_flips`, `_nonunit_subsumption_tests`, `_ordering_method`, `_next_multiplier`) are never reset between searches. Running multiple searches in the same process (e.g., for testing, or in a server context) accumulates stale state.
**Impact:** Stale oriented-eq markers from a prior search can affect a later search. Multiplier counter grows unboundedly across searches, eventually pushing variable numbers beyond `MAX_VARS` boundaries.

#### C3. Auto-Inference Detection Is Imprecise

**Location:** `pyladr/apps/prover9.py:306-328`
**Issue:** `_auto_inference()` enables paramodulation/demodulation whenever it sees **any** binary predicate, not just the `=` predicate. A clause like `P(a,b)` would trigger equality reasoning even though it's not an equation.
**Impact:** Activates paramodulation unnecessarily, generating spurious inferences. More critically, demodulation may be activated without actual equalities, leading to unexpected behavior.

#### C4. Unit Conflict Check Uses Structural Identity Only

**Location:** `pyladr/search/given_clause.py:617-648`
**Issue:** `_unit_conflict()` uses `term_ident()` (syntactic identity) to check for complementary unit clauses, but after unification/substitution, the clause variables may have been renumbered differently. Two logically complementary unit clauses `P(x)` and `-P(y)` would NOT be detected as a conflict because `x ≠ y` syntactically, even though they are trivially unifiable.
**Impact:** Misses unit conflicts that C Prover9 would catch via unification-based unit deletion. Reduces proof-finding ability.

#### C5. Back-Demodulation Doesn't Actually Rewrite

**Location:** `pyladr/search/given_clause.py:679-711`
**Issue:** When back-demodulation finds a rewritable clause, it creates a copy with `victim.literals` (the **original** unrewritten literals) plus a `BACK_DEMOD` justification, then feeds it to `_cl_process`. The actual rewriting only happens inside `_simplify()` if the demod index has been updated — but the new demodulator was just inserted, so it *should* work. However, the victim's original literals are passed through, relying on forward demodulation to do the actual rewriting. If the demod index state is not perfectly consistent at that moment, the clause passes through unchanged.
**Impact:** Potential missed rewrites. C Prover9 rewrites immediately during back-demod.

#### C6. Demodulation Step Counter Bug — Under-Simplifies Clauses

**Location:** `pyladr/inference/demodulation.py:267`
**Issue:** In `_demod_term_recursive()`, the step counter is decremented by `len(steps)` — the *total cumulative* length of the shared steps list, not the delta since the recursive call started. Each deeper recursive call subtracts more steps than it actually consumed, causing premature termination of the rewriting loop. A clause with 3 subterms, each needing 2 rewrites, would exhaust the step budget after just the first subterm.
```python
# BUG: subtracts cumulative total, not delta
remaining_steps -= len(steps)  # rough tracking
```
**Impact:** Clauses are under-simplified. Demodulation terminates too early, leaving reducible terms unreduced. This produces different (less simplified) clauses than C Prover9, affecting subsumption, clause selection, and proof discovery.

#### C7. LRPO Multiset Comparison Uses `id()` for Dedup

**Location:** `pyladr/ordering/multiset.py:46-56`
**Issue:** `_set_of_more_occurrences()` uses `id(t)` in a `seen` set to avoid processing the same term twice. But if `a` contains two *structurally identical* terms at different memory addresses (common after `apply_substitution` creates new Term objects), they both pass the `seen` check and are both added to the result list. The counting via `term_ident()` is correct, so the count is right, but the result list has duplicate entries for the same logical term. This causes `greater_multiset()` to have more `s1` elements than it should, potentially returning True when it should return False (or vice versa).
**Impact:** Wrong LRPO term ordering decisions. Affects equation orientation, demodulator classification, and paramodulation control. Silently produces incorrect results.

#### C8. Parallel Mode Produces Different Results Than Sequential

**Location:** `pyladr/parallel/inference_engine.py`, `pyladr/search/given_clause.py:367-387`
**Issue:** Parallel inference generation collects all inferences from all usable clauses, then processes them sequentially. But sequential mode interleaves inference generation with clause processing — each new inference is processed immediately before generating the next. This means in sequential mode, a newly kept clause from inference N can be forward-subsumed or back-subsume other clauses before inference N+1 is processed. In parallel mode, all inferences are generated against the same snapshot, so these interactions are lost.
**Impact:** Different clauses are kept, different proofs are found, different exit codes. Violates C equivalence guarantee. The parallel mode is functionally a different algorithm.

### HIGH — Affects Search Behavior / C Parity

#### H1. Logging Goes to `logger` But User Expects `stdout`

**Location:** `pyladr/search/given_clause.py:342-350`
**Issue:** Given clause printing uses `logger.info()` which requires Python logging configuration. If logging isn't configured (the default), no output appears. The user specifically suspects "given clauses might not be printing during search anymore." The C Prover9 writes directly to stdout.
**Impact:** User sees no search progress. No regression in correctness, but critical for usability and debugging.

#### H2. Forward Subsumption Scans All Lists Linearly

**Location:** `pyladr/search/given_clause.py:558-568`, `subsumption.py:149+`
**Issue:** `_forward_subsumed()` calls `forward_subsume_from_lists()` which iterates linearly over usable, sos, and limbo. The C implementation uses literal indexing (FPA/discrimination tree) for fast candidate retrieval. With thousands of clauses, this becomes a major performance bottleneck that can change which clauses get kept within time limits.
**Impact:** Severe performance degradation on non-trivial problems. Different kept clauses than C due to different timing = different proofs or missed proofs under `max_seconds`.

#### H3. `ClauseList.pop_first()` Is O(n)

**Location:** `pyladr/search/state.py:57-61`
**Issue:** `pop_first()` does `self._clauses.pop(0)` on a Python list, which is O(n). Called once per limbo clause processed. Combined with limbo potentially growing large, this adds up.
**Impact:** Performance degradation, especially with many kept clauses.

#### H4. `orient_equalities()` Returns Modified Clause But Caller Ignores Return

**Location:** `pyladr/search/given_clause.py:278-282`
**Issue:** In `_process_initial_clauses()`, `orient_equalities(c, ...)` is called but its return value is discarded. The function may return a NEW clause with flipped equalities. The original (unflipped) clause remains in usable/sos.
**Impact:** Equations that should be oriented right→left remain in their original left→right form for initial clauses. This affects paramodulation completeness for initial clauses.

#### H5. `orient_equalities()` Also Ignored in `_keep_clause()`

**Location:** `pyladr/search/given_clause.py:601-602`
**Issue:** Same problem as H4 — `orient_equalities(c, ...)` called but return value discarded. The clause `c` in limbo retains unoriented equalities.
**Impact:** New kept clauses with equalities are not properly oriented for demodulation/paramodulation.

### MEDIUM — Correctness Edge Cases

#### M1. Multiplier Counter Overflow

**Location:** `pyladr/core/substitution.py:26-35`
**Issue:** `_next_multiplier` increments monotonically and is never reset. `apply_substitution()` computes `c.multiplier * MAX_VARS + varnum`. With MAX_VARS=100, after 50,000 unification contexts, variable numbers reach 5,000,000+. While Python handles big ints, this creates pathologically large variable numbers in derived terms.
**Impact:** Memory bloat from large variable numbers; potential issues if any code assumes variables fit in reasonable ranges. Not a correctness bug per se, but a deviation from C behavior where the multiplier wraps.

#### M2. `match()` Doesn't Roll Back on Partial Failure

**Location:** `pyladr/core/substitution.py:407-409`
**Issue:** The `match()` function uses `all()` with a generator over argument matching. If argument 2 of 3 fails, bindings from argument 1 remain on the trail. The caller is expected to do `trail.undo_to(mark)` — and in subsumption's `_subsume_literals` it does. But other callers (e.g., demodulation matching) may not.
**Impact:** Potential stale bindings affecting subsequent match attempts within the same context.

#### M3. Tautology Count Conflated with Subsumption Count

**Location:** `pyladr/search/given_clause.py:536-538`
**Issue:** When a tautology is detected, `stats.subsumed` is incremented rather than a separate `stats.tautologies` counter. This makes statistics inaccurate compared to C Prover9 output.
**Impact:** Statistics diverge from C output. Not a correctness issue but hinders debugging/comparison.

#### M4. Proof Trace Uses DFS Instead of BFS

**Location:** `pyladr/search/given_clause.py:748-782`
**Issue:** `_trace_proof()` uses a stack (DFS) for traversal. The C implementation uses similar logic but the traversal order doesn't matter since all reachable clauses are collected and sorted by ID. This is fine — no bug here, but the sort order is by clause ID which should match C.
**Impact:** None (output is sorted). Listed for completeness.

### LOW — Code Quality / Maintainability

#### L1. `_auto_inference` Equality Detection

**Location:** `pyladr/apps/prover9.py:316-324`
**Issue:** Checks `lit.atom.arity == 2` as a proxy for "has equality". Should check via `is_eq_atom()` or symbol name. (Subsumed by C3 above, but listed separately as the fix is different.)

#### L2. No Reset for `_oriented_eqs`/`_renamable_flips`

**Location:** `pyladr/inference/paramodulation.py:67-68`
**Issue:** Unlike `_nonunit_subsumption_tests` which has `reset_subsumption_stats()`, the oriented eq sets have no reset function.

#### L3. ClauseList.remove() Silently Succeeds on Missing Clause

**Location:** `pyladr/search/state.py:45-51`
**Issue:** `remove()` returns `False` if the clause isn't found, but callers (e.g., back subsumption) don't check the return value. This masks bugs where a clause should be in a list but isn't.

---

## 2. Root Cause Analysis

| Root Cause | Issues | Systemic Pattern |
|------------|--------|-------------------|
| `id()` used for structural identity | C1, C7, L2 | Multiple places use Python `id()` (memory address) where structural term identity is needed. Fragile under GC and object duplication. |
| Module-level mutable state | C1, C2, L2 | C uses file-scope globals; Python translation kept the pattern but Python's GC makes `id()`-based tracking unsafe |
| Imprecise C→Python translation | C3, C6, H4, H5 | Functions translated from C without accounting for Python's value semantics (returned new objects vs mutating in place, shared mutable accumulators) |
| Parallel semantics divergence | C8 | Parallel mode changes algorithm semantics — interleaving of inference and processing is lost |
| Missing index structures | H2 | C uses FPA/discrimination tree indexes for subsumption; Python uses linear scan |
| Logging vs stdio mismatch | H1 | C writes to stdout; Python uses `logging` module which requires setup |
| Performance-insensitive data structures | H3 | Python list used where deque would be O(1) for left-pop |

---

## 3. Phased Improvement Roadmap

### Phase 1: Critical Correctness Fixes (Immediate)

**Goal:** Fix issues that can produce incorrect proofs or miss valid proofs.

| Task | Issue | Effort | Risk |
|------|-------|--------|------|
| 1.1 Fix oriented-eq tracking | C1 | Medium | Low — replace `id()` with structural hash |
| 1.2 Fix `_auto_inference` equality detection | C3 | Small | Low — use `is_eq_atom()` |
| 1.3 Fix orient_equalities return value handling | H4, H5 | Small | Low — use return value |
| 1.4 Add global state reset functions | C2, L2 | Small | Low — add reset functions, call before search |
| 1.5 Fix unit conflict to use unification | C4 | Medium | Medium — changes conflict detection behavior |
| 1.6 Fix demodulation step counter | C6 | Small | Low — track delta instead of total |
| 1.7 Fix multiset `id()` dedup | C7 | Small | Low — use `term_ident()` for seen set |
| 1.8 Fix or disable parallel mode | C8 | Medium | Medium — must match sequential behavior |

**Approach for 1.1:** Replace `_oriented_eqs: set[int]` with a structural approach. Options:
- **(A) Term-structural set:** Use a `set[tuple]` with a structural hash of the atom. Since Terms are frozen dataclasses, they're hashable. Store `id(atom)` in a `WeakValueDictionary` or simply use the Term itself as a key.
- **(B) Clause-attached flags:** Add an `oriented` boolean to Literal or use a separate `dict[Term, bool]` keyed by the actual Term object (which is frozen/hashable).
- **Recommended: (B)** — Use `set[Term]` directly since Terms are frozen and hashable via dataclass.

**Approach for 1.3:** In `_process_initial_clauses()` and `_keep_clause()`:
```python
# Before:
orient_equalities(c, self._symbol_table)
# After:
c = orient_equalities(c, self._symbol_table)
```
For initial clauses, also update the list entry. For keep_clause, update `c` before adding to limbo.

**Approach for 1.5:** Replace `term_ident()` with unification:
```python
ctx1, ctx2, trail = Context(), Context(), Trail()
if unify(c_lit.atom, ctx1, o_lit.atom, ctx2, trail):
    trail.undo()
    # Found unit conflict
```

**Approach for 1.6:** Track step delta instead of cumulative total:
```python
# Before each recursive call, save steps length:
steps_before = len(steps)
new_arg = _demod_term_recursive(arg, demod_index, ...)
remaining_steps -= (len(steps) - steps_before)
```

**Approach for 1.7:** Replace `id(t)` in `seen` with structural identity:
```python
# In _set_of_more_occurrences, use term_ident for dedup:
seen: list[Term] = []
for t in a:
    if any(t.term_ident(s) for s in seen):
        continue
    seen.append(t)
    # ... count and compare
```

**Approach for 1.8:** Two options:
- **(A) Disable parallel mode by default** — Set `ParallelSearchConfig.enabled = False` as default, document that parallel mode produces different (potentially non-C-equivalent) results. Quick fix.
- **(B) Fix parallel mode** — After parallel inference generation, process inferences in the same order as sequential mode. This requires careful ordering but preserves parallelism benefits.
- **Recommended: (A)** for immediate correctness, then (B) as a Phase 3 improvement.

### Phase 2: Performance & C Parity (Short-term)

**Goal:** Match C Prover9 search behavior and performance characteristics.

| Task | Issue | Effort | Risk |
|------|-------|--------|------|
| 2.1 Fix logging to stdout | H1 | Small | Low |
| 2.2 Add literal indexing for subsumption | H2 | Large | Medium — major new code |
| 2.3 Use deque for ClauseList | H3 | Small | Low |
| 2.4 Separate tautology counter | M3 | Small | Low |
| 2.5 Add multiplier reset/management | M1 | Small | Low |

**Approach for 2.1:** Configure logging in `run_prover()` to output to stdout:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
)
```
Or convert critical `logger.info()` calls to direct `print()` for C-parity.

**Approach for 2.2:** The codebase already has `Mindex` and `DiscrimWild` indexing. Extend to use `LiteralIndex` for forward/backward subsumption candidate retrieval instead of linear list scanning.

### Phase 3: Robustness & Testing (Medium-term)

| Task | Issue | Effort | Risk |
|------|-------|--------|------|
| 3.1 Match rollback audit | M2 | Medium | Medium |
| 3.2 ClauseList.remove() assertion mode | L3 | Small | Low |
| 3.3 Back-demodulation rewrite verification | C5 | Medium | Medium |
| 3.4 Add cross-validation tests for all fixes | All | Large | Low |

---

## 4. Quality Gates

### Per-Fix Gates
- [ ] Existing unit tests pass (no regressions)
- [ ] C cross-validation tests pass (`test_search_equivalence.py`, `test_c_vs_python_comprehensive.py`)
- [ ] New unit test covers the specific fix
- [ ] Exit codes match C Prover9 for all test inputs

### Phase Gates

**Phase 1 Gate:**
- All critical fixes applied
- Cross-validation shows same exit codes as C for all test inputs
- `pytest tests/unit/ tests/cross_validation/` passes 100%

**Phase 2 Gate:**
- Given clause printing visible in output
- Performance within 10x of C on benchmark problems
- Statistics output matches C format

**Phase 3 Gate:**
- No known correctness issues remaining
- Full test coverage for inference rules
- Robustness tests for edge cases (empty clauses, huge arities, deep terms)

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Oriented-eq fix changes proof discovery order | Medium | Medium | Run cross-validation before/after; accept different but valid proofs |
| Unit conflict unification introduces new proofs | Low | Low | More complete = better; verify proofs are valid |
| Logging fix breaks downstream tools parsing output | Medium | Medium | Match C output format exactly |
| Subsumption indexing introduces subtle bugs | Medium | High | Extensive cross-validation; keep linear fallback |
| orient_equalities fix changes clause ordering | Medium | Medium | This is a bug fix — new order is correct |

---

## 6. Success Metrics

1. **Exit code parity:** 100% match with C Prover9 on test suite
2. **Proof validity:** All proofs found are valid (clauses trace back to axioms)
3. **No regressions:** All existing tests continue to pass
4. **Given clause printing:** Visible in default output mode
5. **Statistics parity:** Forward_subsumed/Back_subsumed counts within 5% of C
6. **Performance:** No more than 2x slower than current on any benchmark

---

## 7. Implementation Order

```
Phase 1 (Critical - do first, sequentially):
  1.6 (C6 demod step counter - trivial fix, high impact)
  → 1.2 (C3 auto-inference fix - trivial, high impact)
  → 1.3 (H4/H5 orient_equalities return - trivial, high impact)
  → 1.7 (C7 multiset id() fix - small, fixes LRPO ordering)
  → 1.1 (C1 oriented-eq id() fix - medium, eliminates unsoundness)
  → 1.4 (C2 global state reset - small, enables clean testing)
  → 1.5 (C4 unit conflict unification - medium, improves completeness)
  → 1.8 (C8 disable parallel mode - small, restores C equivalence)

Phase 2 (Performance & User-Facing - can parallelize):
  2.1 (H1 logging fix) | 2.3 (H3 deque fix) | 2.4 (M3 tautology counter) | 2.5 (M1 multiplier)
  → 2.2 (H2 subsumption indexing - depends on stable base)

Phase 3 (Robustness - after phases 1-2 stable):
  3.1 (match rollback audit) | 3.2 (ClauseList assertions) | 3.3 (back-demod verification)
  → 3.4 (comprehensive cross-validation testing)
  → 3.5 (parallel mode fix - restore with correct interleaving semantics)
```

---

*Plan prepared by architect agent, 2026-04-09*
*Based on thorough codebase analysis of search, inference, core, and CLI modules*
