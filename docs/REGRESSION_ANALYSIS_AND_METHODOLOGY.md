# Regression Analysis & Methodology Amendments

## PyLADR 2-Line Proof Regression — Incident Report & Process Improvements

**Date:** 2026-04-08
**Severity:** Critical — prover terminates prematurely with incorrect 2-line proof
**Affected Component:** Online learning integration with search loop
**Status:** RESOLVED — fix validated, methodology amendments delivered

---

## Part 1: Root Cause Analysis

### 1.1 Incident Summary

The prover immediately terminates with a spurious 2-line proof instead of performing the expected multi-step search. The initial investigation suspected buffer refactoring changes, but diagnosis revealed the actual root cause: **missing Skolemization in goal denial**.

### 1.2 Actual Root Cause: Missing Skolemization

**File:** `pyladr/apps/prover9.py` — `_deny_goals()` function

**The Bug:** The original `_deny_goals()` only flipped literal signs but did NOT Skolemize variables:
```python
# BEFORE (buggy):
denied_lits = tuple(
    Literal(sign=not lit.sign, atom=lit.atom)  # Variables remain universal!
    for lit in goal.literals
)
```

**Correct behavior** (matching C Prover9's `process_goal_formulas()`):
```python
# AFTER (fixed):
# 1. Collect all variables in goal
# 2. Create fresh Skolem constants (c1, c2, ...)
# 3. Substitute Skolem constants for variables
# 4. THEN negate literal signs
denied_lits = tuple(
    Literal(
        sign=not lit.sign,
        atom=_substitute_term(lit.atom, var_map),  # Skolemized!
    )
    for lit in goal.literals
)
```

**Why this causes false proofs:**
- **C Prover9:** Negates goal `P(i(x,i(y,x)))` → `-P(i(c10,i(c11,c10)))` (Skolem constants — ground terms that can't unify with arbitrary terms)
- **Python (buggy):** Negates goal → `-P(i(v0,i(v1,v0)))` (universal variables that CAN unify with anything)
- **Spurious unification:** Universal `v1` binds to `n(v0)` from axioms → empty clause → false 2-line "proof"

This is a **logical soundness bug**: without Skolemization, the negated goal is too general, allowing trivial unification that produces unsound proofs.

### 1.3 Initial Misdirection: Buffer Refactoring

The initial investigation focused on buffer refactoring changes to the ML online learning system. While the buffer refactoring is **innocent** of the 2-line proof regression, the investigation revealed several **latent vulnerabilities** in the ML integration that are documented in Part 3:

- Subsumption callbacks silently fail to register (`hasattr` guard on non-existent methods)
- Experience buffer would be starved of positive signal if KEPT outcomes are removed
- `OutcomeType.SUBSUMER` depends on callbacks that don't exist

These are real risks that should be addressed even though they didn't cause this specific regression.

### 1.4 Why It Wasn't Detected

1. **No end-to-end goal denial test:** No test verifies that denied goals contain Skolem constants instead of variables
2. **No proof soundness validation:** No test checks that proofs require sufficient depth — a 2-line proof for a complex theorem should trigger an alarm
3. **Unit tests pass:** Individual components (parsing, resolution, subsumption) all work correctly — the bug lives in the integration seam between goal processing and search
4. **Cross-validation tests are optional:** C vs Python comparison tests would catch this immediately (different denied clauses → different search → different proof) but require a C binary that may not be present
5. **No CI system:** No automated enforcement of test execution on every change
6. **No proof correctness invariant:** No assertion that proof length must be ≥ expected for non-trivial problems

### 1.5 Classification

| Property | Value |
|----------|-------|
| **Type** | Logical soundness bug (unsound proofs) |
| **Severity** | Critical — produces incorrect results silently |
| **Origin** | Incomplete port from C Prover9 |
| **Scope** | All problems with universally quantified goals |
| **Detection** | End-to-end testing against C reference implementation |
| **Fix** | Add Skolemization to `_deny_goals()` — **VALIDATED** |
| **Validation** | vampire.in: spurious 3-clause proof → legitimate 10-clause proof (28s); 302 core tests pass, 87 search tests pass, zero regressions |

---

## Part 2: Testing Gap Analysis

### 2.1 What IS Tested

| Category | Coverage | Confidence |
|----------|----------|------------|
| Unit: inference rules (resolution, paramodulation) | Good | High |
| Unit: clause processing (simplify, delete, keep) | Good | High |
| Unit: search termination on proof | Good | High |
| Unit: search termination on SOS empty | Basic | Medium |
| Unit: search limit checks (max_given, max_seconds) | Minimal | Low |
| Cross-validation: C vs Python equivalence | Comprehensive (when run) | High |
| Integration: ML pipeline components | Mock-based | Medium |
| Property: unification, ordering | Good | High |

### 2.2 What is NOT Tested (Critical Gaps)

0. **Goal denial correctness (THE GAP THAT CAUSED THIS BUG)**
   - No test verifies that denied goals contain Skolem constants instead of universal variables
   - No test compares Python's denied goals against C Prover9's denied goals
   - No test checks that goal denial produces ground terms for existential variables
   - No test validates proof soundness (proof length ≥ expected minimum for problem difficulty)

1. **Search loop flow invariants**
   - No test verifies the main loop continues correctly when limits are not exceeded
   - No test verifies `_make_inferences()` returning `None` allows loop continuation
   - No test verifies `_limbo_process()` returning `None` allows loop continuation

2. **End-to-end ML search path**
   - `prover9.py:545` always creates plain `GivenClauseSearch` — the `online_learning` flag is stored in `SearchOptions` but no code in the CLI switches to `OnlineLearningSearch`
   - No test exercises the full `OnlineLearningSearch` → proof path
   - No test verifies the dynamic subclassing in `_OnlineLearningGivenClauseSearch.__new__` correctly hooks all methods

3. **Callback wiring**
   - No test verifies `set_back_subsumption_callback` exists on target class
   - No test verifies callbacks fire during `_limbo_process` back subsumption
   - No test verifies callbacks fire during `_should_delete` forward subsumption

4. **Experience buffer composition**
   - No test verifies the buffer receives both positive and negative outcomes
   - No test verifies the buffer isn't starved of positive signal
   - No test validates the ratio of outcome types during real search

5. **Search statistics regression**
   - No test establishes baseline statistics for known problems
   - No test detects when `clauses_given` drops dramatically
   - No test flags when proof length is abnormally short

6. **CI/Automation**
   - No `.github/workflows/` directory
   - No automated test execution on commit
   - Cross-validation tests require manual C binary setup

---

## Part 3: Architectural Risk Inventory

### 3.1 Dynamic Subclassing Pattern (HIGH RISK)

**Location:** `online_integration.py:1044-1121`

The `_OnlineLearningGivenClauseSearch.__new__` method creates a dynamic subclass of `GivenClauseSearch` at runtime, capturing method references via closures. This pattern has several risks:

- **Method signature drift:** If `_make_inferences`, `_keep_clause`, `_should_delete`, or `_handle_proof` change signatures, the hooks silently break
- **Slot inheritance:** `GivenClauseSearch` uses `__slots__`, making it impossible to add attributes dynamically — callbacks can't be stored
- **Double initialization:** `GivenClauseSearch.__init__` is called twice (lines 1061 and 1113) — once for the discarded initial instance and once for the hooked instance
- **No validation:** The pattern doesn't verify that hooked methods exist or have expected signatures

**Recommendation:** Replace with explicit composition or a formal plugin/hook protocol with interface validation.

### 3.2 Silent Failure Guards (HIGH RISK)

Multiple `hasattr()` and `try/except` patterns throughout the ML integration silently swallow failures:

- `online_integration.py:1116-1119` — subsumption callback registration (always fails silently)
- `online_integration.py:51-61` — `_ML_AVAILABLE` guard for import failures (no warning logged)
- `ml_selection.py:256-262` — `_ml_select` catches ALL exceptions and falls back silently when `fallback_on_error=True`
- `ml_selection.py:417-418` — `_record_embedding` catches ALL exceptions with bare `except: pass`
- Various `if not self._enabled` early returns that skip all processing

**Specific silent failure inventory:**
| Pattern | Location | Consequence |
|---------|----------|-------------|
| `hasattr()` for callbacks | `online_integration.py:1116-1119` | Subsumption learning never activates |
| `try/except ImportError` | `online_integration.py:51-61` | ML disabled with no user notification |
| `try/except ImportError` | `embedding_provider.py:40-59` | Torch failures silently degrade to no-ML |
| `except Exception: pass` | `ml_selection.py:417-418` | Embedding errors silently swallowed |
| `fallback_on_error` | `ml_selection.py:256-262` | ML crashes fall back without alerting |
| `_ML_AVAILABLE` check | `online_integration.py:935-936` | User sets `enabled=True` but ML doesn't activate |

**Recommendation:** Replace silent guards with explicit logging at WARNING level. Failed callback registration should be a loud warning, not a silent no-op. Import guards must log at WARNING when ML is unavailable.

### 3.3 ML/Search Boundary (MEDIUM RISK)

The boundary between ML components and the search core is implicit:
- ML selection (`ml_selection.py`) can alter clause selection order
- Goal-directed search (`goal_directed.py`) adds another selection modifier
- Online integration (`online_integration.py`) hooks into the search loop
- None of these modifications are validated for correctness preservation

**Specific risks:**
- `EmbeddingEnhancedSelection.select_given()` overrides base but no interface contract validates return type
- If `GivenSelection.select_given()` signature changes, subclass override silently diverges
- `GoalProximityScorer` uses a non-reentrant `threading.Lock()` — complex callback chains could deadlock

**Recommendation:** Define explicit contracts at the ML/search boundary with invariant checks.

### 3.4 Entry Point Fragmentation (MEDIUM RISK)

Multiple entry points to the search:
- `GivenClauseSearch.run()` — direct
- `OnlineLearningSearch.run()` — ML-wrapped
- `prover9.py run_prover()` — CLI (always uses plain search)
- Cross-validation helpers — testing path

Each path may exercise different code, and no test matrix covers all entry points × problem types.

**Recommendation:** Create a test matrix that exercises each entry point with a standard problem set.

### 3.5 Disconnected Monitoring Infrastructure (MEDIUM RISK)

Extensive monitoring infrastructure exists but is NOT integrated with the search loop:

| Monitor Component | File | Integrated? |
|-------------------|------|-------------|
| HealthChecker + CircuitBreaker | `monitoring/health.py` | NO |
| DiagnosticLogger | `monitoring/diagnostics.py` | NO |
| SearchAnalyzer (per-iteration stats) | `monitoring/search_analyzer.py` | NO |
| LearningMonitor (alerts) | `monitoring/learning_monitor.py` | NO |
| LearningCurves (convergence) | `monitoring/learning_curves.py` | NO |
| RegressionDetection | `monitoring/regression.py` | NO |
| MemoryMonitor | `monitoring/memory_monitor.py` | NO |
| Profiler | `monitoring/profiler.py` | NO |

**Key gap:** The `SearchAnalyzer` could detect anomalies per-iteration (rate drops, stagnation) but is never called from `GivenClauseSearch`. The `CircuitBreaker` could halt ML when it misbehaves but is never evaluated during search.

**Recommendation:** Integrate at minimum: periodic health checks during search, SearchAnalyzer per-iteration snapshots, and buffer health monitoring when online learning is active.

### 3.6 Thread Safety Risks (LOW-MEDIUM RISK)

- `TriggerPolicy._current_interval` mutated without synchronization (`online_integration.py:275-279`)
- `self._proofs` list accessed without locks in parallel context (`given_clause.py` + `online_integration.py:1105-1107`)
- `EmbeddingProvider` model lock is RLock — future refactoring to Lock would deadlock
- Cache invalidation happens outside model lock (`embedding_provider.py:293`) — potential race with concurrent reads

**Recommendation:** Audit all shared mutable state between search loop and ML components; add synchronization where needed.

### 3.7 Double Initialization in Dynamic Subclassing (LOW RISK)

`_OnlineLearningGivenClauseSearch.__new__` calls `GivenClauseSearch.__init__` twice:
- Line 1061: Initializes a temporary instance (immediately discarded)
- Line 1113: Initializes the actual hooked instance

This wastes resources and could trigger side effects twice if `__init__` has external effects (e.g., registering with a global registry).

**Recommendation:** Refactor to single initialization path.

---

## Part 4: Methodology Amendments

### Amendment 1: Mandatory End-to-End Testing Protocol

**CENTRAL RECOMMENDATION** — This regression demonstrates that passing unit tests provides NO guarantee of system correctness when integration boundaries are involved. During this incident, end-to-end testing caught **3 bugs** that unit tests missed: (1) missing Skolemization in goal denial, (2) demodulation infinite recursion, (3) cross-test state contamination.

#### Protocol Requirements:

1. **Every ML system change MUST include end-to-end test(s)** that:
   - Run a full proof search through the affected code path
   - Verify the proof is found (or search exhausts correctly)
   - Check search statistics are reasonable (not anomalously low)
   - Compare against a known-good baseline

2. **End-to-end test coverage matrix:**
   ```
   Entry Points × Problem Types × ML Configurations
   ─────────────────────────────────────────────────
   Plain GivenClauseSearch    × Simple resolution    × No ML
   OnlineLearningSearch       × Equational/paramod   × Online learning enabled
   CLI (pyprover9)            × Multi-step proofs    × ML with subsumption tracking
                              × Edge cases (trivial)  × Buffer at various capacities
   ```

3. **Baseline statistics for regression detection:**
   - Establish expected `clauses_given`, `clauses_kept`, `proof_length` for reference problems
   - Flag deviations > 10% as potential regressions
   - Auto-fail on deviations > 50%

#### Implementation:
- Create `tests/e2e/` directory for end-to-end tests
- Each test runs the full prover pipeline (parse → search → proof/exhaust)
- Tests exercise EVERY entry point, not just the primary one
- Tests MUST pass before any PR is merged
- **CRITICAL: Before concluding any work is successful, end-to-end tests MUST be run and pass** — unit test passage alone is insufficient to validate system correctness

### Amendment 2: Silent Failure Elimination

#### Requirements:

1. **Ban `hasattr()` guards for feature wiring.** If a callback is expected, assert its existence:
   ```python
   # BAD — silent failure
   if hasattr(hooked, 'set_back_subsumption_callback'):
       hooked.set_back_subsumption_callback(...)

   # GOOD — loud failure
   assert hasattr(hooked, 'set_back_subsumption_callback'), \
       "GivenClauseSearch missing callback infrastructure"
   hooked.set_back_subsumption_callback(...)

   # BEST — typed protocol
   class SupportsSubsumptionCallbacks(Protocol):
       def set_back_subsumption_callback(self, cb: Callable) -> None: ...
   ```

2. **All `try/except ImportError` blocks must log at WARNING level**
3. **All disabled-feature paths must log at DEBUG level on first invocation**
4. **No silent no-ops in event handlers** — at minimum, count skipped events for health monitoring

### Amendment 3: Change Impact Assessment for ML Modifications

#### Pre-Change Checklist:

Before modifying any ML-related component, the developer must answer:

- [ ] **Signal flow:** Does this change affect what training signal the ML system receives?
- [ ] **Positive/negative balance:** Could this change starve the model of positive examples?
- [ ] **Callback wiring:** Does this change assume callbacks exist? Are they verified?
- [ ] **Entry point coverage:** Does this change work through ALL entry points?
- [ ] **Fallback behavior:** If this change fails silently, what happens to search?
- [ ] **End-to-end test:** Is there an end-to-end test exercising this exact path?

#### Post-Change Validation:

After modifying any ML-related component:

- [ ] Run full end-to-end test suite (`tests/e2e/`)
- [ ] Verify experience buffer receives expected outcome distribution
- [ ] Check search statistics against baselines
- [ ] Run at least one problem through `OnlineLearningSearch` path
- [ ] Verify no silent failures in log output (grep for WARNING)

### Amendment 4: Search Loop Invariant Testing

#### Required Invariant Tests:

1. **Loop continuation:** Given a solvable problem requiring N > 5 given clauses, verify search runs for at least N-1 iterations before finding proof
2. **Statistics monotonicity:** `given` count increases each iteration; `kept` never decreases; `generated` never decreases
3. **SOS non-empty during active search:** If proof is found, SOS was non-empty throughout search (or exhaustion is expected)
4. **Limbo processing completeness:** After `_limbo_process`, limbo list is empty and all clauses moved to SOS
5. **Back subsumption correctness:** Clauses removed by back subsumption are genuinely subsumed

### Amendment 5: Monitoring and Early Detection

**Context:** PyLADR already has an extensive monitoring infrastructure (`pyladr/monitoring/`) including health checks, circuit breakers, search analyzers, learning monitors, and regression detection — but **none of it is integrated into the search loop**. The infrastructure is "opt-in after search completes."

#### 5A: Integrate Existing Monitoring into Search Loop

Wire the existing `SearchAnalyzer` and `HealthChecker` into `GivenClauseSearch.run()`:

```python
# In the main loop (given_clause.py:236-246):
while self._inferences_to_make():
    exit_code = self._make_inferences()
    if exit_code is not None:
        return self._make_result(exit_code)
    exit_code = self._limbo_process()
    if exit_code is not None:
        return self._make_result(exit_code)
    # NEW: Periodic health check (every N given clauses)
    if self._state.stats.given % 100 == 0:
        self._periodic_health_check()
```

#### 5B: Add Anomaly Detection Rules

1. **Suspiciously fast termination:** If `clauses_given < 3` and exit code is `MAX_PROOFS_EXIT`, log a WARNING — this may indicate a trivial proof or a regression
2. **Empty experience buffer:** If online learning is active and buffer has 0 positive outcomes after N given clauses, log a WARNING
3. **Callback registration audit:** On search start, log which callbacks are registered and which were skipped
4. **Statistics snapshot logging:** Every K given clauses (e.g., K=100), log a statistics snapshot for post-mortem analysis
5. **Generation rate collapse:** If `generated/given` ratio drops below 1.0, log WARNING (search may be stuck)
6. **Keep rate anomaly:** If `kept/generated` drops below 5%, log WARNING (over-aggressive deletion)

#### 5C: Unify Logging (Replace print with logging)

Current state: `online_integration.py` uses `print()` for progress messages (lines 585-586, 764-767) instead of the `logging` module. This means:
- Messages can't be filtered by log level
- Messages don't include timestamps or source module
- Messages aren't captured by log handlers

**Requirement:** All status output must use the `logging` module, not `print()`.

### Amendment 6: Interface Contracts for Hook Points

#### Replace dynamic subclassing with explicit protocol:

```python
class SearchEventListener(Protocol):
    """Protocol for components that observe search events."""
    def on_given_selected(self, given: Clause, selection_type: str) -> None: ...
    def on_clause_kept(self, clause: Clause) -> None: ...
    def on_clause_deleted(self, clause: Clause, reason: str) -> None: ...
    def on_back_subsumption(self, subsuming: Clause, subsumed: Clause) -> None: ...
    def on_forward_subsumption(self, subsuming: Clause, subsumed: Clause) -> None: ...
    def on_proof_found(self, proof_clause_ids: set[int]) -> None: ...
    def on_inferences_complete(self) -> None: ...
```

Benefits:
- Type checker validates interface compliance
- No `hasattr` guards needed — protocol guarantees method existence
- Multiple listeners supported (for ML, logging, monitoring simultaneously)
- Testable: mock listeners verify event firing

### Amendment 7: Porting Verification Protocol

This bug originated from an incomplete port of C Prover9 behavior. To prevent similar issues:

#### 7A: Semantic Porting Checklist

For each C function ported to Python, verify:
- [ ] **All side effects captured:** C functions often Skolemize, renumber, or orient as side effects — verify each is preserved
- [ ] **Call chain semantics:** The C caller may perform pre/post-processing that the Python port must replicate
- [ ] **Variable scope:** C's manual memory management and global state may encode invariants that need explicit Python implementation
- [ ] **Comparison test:** Run the same input through both C and Python and compare intermediate state (not just final output)

#### 7B: Critical Porting Points for Prover9

| C Function | Python Equivalent | Semantic Requirement | Status |
|------------|------------------|---------------------|--------|
| `process_goal_formulas()` | `_deny_goals()` | Negate + Skolemize | **FIXED** |
| `cl_process_simplify()` | `_simplify()` | Demod + merge + orient | Needs audit |
| `cl_process_delete()` | `_should_delete()` | Tautology + weight + fwd subsumption | Needs audit |
| `cl_process_keep()` | `_keep_clause()` | ID + index + unit conflict + demod | Needs audit |
| `limbo_process()` | `_limbo_process()` | Back subsumption + back demod + SOS append | Needs audit |
| `search()` | `run()` | Init + loop + termination | Needs audit |

#### 7C: Mandatory C vs Python Comparison

Every ported function must have a comparison test that verifies identical behavior on at least 3 representative inputs. The cross-validation infrastructure already exists — it must be **mandatory, not optional**.

### Amendment 8: Cumulative Regression Prevention Protocol

**CRITICAL METHODOLOGICAL ISSUE IDENTIFIED (2026-04-09):** The current methodology prevents individual regressions but does NOT prevent **regression cycles** where Fix A → introduces Regression B → Fix B → reintroduces Regression A. This creates infinite loops of previously-solved problems returning.

#### 8A: The Regression Cycle Problem

**Typical cycle pattern:**
1. **Day 1:** Discover hyper-resolution not working → Fix hyper-resolution
2. **Day 2:** Test reveals max_weight filtering broken by hyper-resolution fix → Fix max_weight filtering
3. **Day 3:** Test reveals hyper-resolution stopped working due to max_weight changes → Fix hyper-resolution (again)
4. **Day 4:** max_weight filtering breaks again... **CYCLE CONTINUES**

**Root cause:** Each fix is validated against the current issue in isolation, without verifying that ALL previously-fixed issues still work.

#### 8B: Cumulative Regression Prevention Requirements

**MANDATORY for every fix/change:**

1. **Cumulative Test Execution**
   - EVERY fix must run ALL regression tests from previous incidents, not just tests for the current issue
   - Create and maintain a **Cumulative Regression Suite** that contains tests for every previously-fixed regression
   - Test execution order: (1) Current issue tests, (2) ALL previous regression tests, (3) Full unit test suite

2. **Regression Test Accumulation**
   - When fixing any regression, the fix MUST include a test that prevents that specific regression from returning
   - All new regression tests get added to the Cumulative Regression Suite permanently
   - The suite grows monotonically — tests are never removed, only enhanced

3. **Fix Validation Protocol**
   ```
   For every fix/change:
   ✅ Current issue is resolved
   ✅ ALL previous regression tests still pass
   ✅ Full unit test suite still passes
   ✅ Cross-validation tests still pass
   ✅ End-to-end tests still pass
   ```

4. **Breaking Change Detection**
   - If ANY previous regression test fails during a fix, the change is **immediately rejected**
   - No exceptions — if fixing Issue B breaks the test for previously-fixed Issue A, the fix is invalid
   - Must find a solution that resolves B without breaking A, OR refactor both fixes to coexist

#### 8C: Cumulative Test Suite Structure

```
tests/regression/
├── cumulative_suite.py           # Runs ALL regression tests
├── skolemization/                 # Regression from 2026-04-09
│   ├── test_goal_denial_skolem.py
│   └── test_vampire_proof_depth.py
├── hyper_resolution/              # Regression from current incident
│   ├── test_vampire_convergence.py
│   └── test_horn_problem_auto_inference.py
├── max_weight_filtering/          # Regression from current incident
│   └── test_weight_limit_enforcement.py
└── [future regressions...]
```

#### 8D: Cross-Dependency Validation

**CRITICAL INSIGHT:** Some regression pairs have **interaction effects** where the combination of Fix A + Fix B works, but Fix A alone or Fix B alone fails.

**Requirements:**
1. **Interaction testing**: Test all combinations of major fixes, not just individual fixes
2. **Baseline preservation**: Maintain a "golden" baseline configuration that includes ALL fixes applied correctly
3. **Incremental validation**: When adding Fix N, verify it works with ALL previous fixes (Fix 1..N-1) applied
4. **Rollback capability**: Every fix must be reversible without affecting other fixes

#### 8E: Implementation Strategy

**Phase 1 (Immediate):**
- Create `tests/regression/cumulative_suite.py` that runs all existing regression tests
- Add regression tests for current hyper-resolution/max_weight issue
- Establish the "run cumulative suite on every fix" rule

**Phase 2 (Short-term):**
- Convert all previous incident fixes into regression tests and add to cumulative suite
- Implement automated "cumulative suite MUST pass" check in development workflow
- Create rollback procedures for each fix in case of interaction conflicts

**Phase 3 (Long-term):**
- Build interaction effect testing matrix for major fixes
- Implement automated baseline preservation (snapshot "all fixes working" state)
- Create regression cycle detection (alert if same test fails twice within N days)

#### 8F: Regression Cycle Early Warning System

**Indicators that a regression cycle may be forming:**
- Same component modified multiple times within short period
- Test failures that "look familiar" or similar to previous issues
- Statistics/behavior that matches previous incident signatures
- Developer comments like "this worked yesterday" or "we fixed this before"

**Automated alerts:**
- Log WARNING if same test file modified >3 times in 7 days
- Log ERROR if any test marked as "regression test" fails after previously passing
- Flag potential cycles: if Issue A test fails and Issue A was fixed <30 days ago

#### 8G: Emergency Cycle Breaking Protocol

**If a regression cycle is detected:**

1. **STOP** — cease all individual fixes immediately
2. **Consolidate** — identify ALL issues in the cycle (A, B, C...)
3. **Joint solution** — design a single integrated fix that resolves ALL issues simultaneously
4. **Comprehensive validation** — test the joint fix against ALL issues in cycle + cumulative suite
5. **Holistic commit** — commit the joint fix as a single change, not piecemeal

**Example:** If hyper-resolution ↔ max_weight form a cycle, don't alternate fixes. Instead, implement a unified search configuration that supports both hyper-resolution AND max_weight filtering simultaneously.

---

## Part 5: Implementation Priorities

### Immediate (This Sprint)

1. ~~Fix the regression (Task #4)~~ — **DONE**: Skolemization added to `_deny_goals()`
2. End-to-end regression test for 2-line proof issue — **IN PROGRESS** (Task #3)
3. Add goal denial verification test (assert Skolem constants in denied goals)
4. Add WARNING log for skipped callback registration (latent ML risk)
5. Add proof soundness invariant (warn if proof length < expected minimum)

### Short-Term (Next 2 Sprints)

6. Create `tests/e2e/` test suite with entry point × problem matrix
7. Establish baseline statistics for reference problems
8. Replace `hasattr` guards with assertions or protocol checks
9. Add search anomaly detection logging
10. Set up CI (GitHub Actions) with mandatory test execution
11. Audit all C→Python porting points per Amendment 7B table
12. Make cross-validation tests non-optional (embed C binary in CI or use pre-computed baselines)

### Medium-Term (Next Quarter)

13. Refactor `_OnlineLearningGivenClauseSearch` to use explicit `SearchEventListener` protocol
14. Implement full change impact assessment checklist as PR template
15. Add experience buffer health monitoring
16. Create automated regression detection for search statistics
17. Integrate existing monitoring infrastructure into search loop (per Appendix C roadmap)

---

## Part 6: Lessons Learned

### Lesson 1: Unit test passage ≠ system correctness
Unit tests validated individual components in isolation (parsing works, resolution works, subsumption works). The bug lived in the **integration seam** — goal denial feeds into search, and the search trusts that denied goals are correctly Skolemized. No unit test covered this trust boundary.

**Further validation:** During Task #3 (comprehensive end-to-end testing), the test engineer discovered **two additional bugs** that unit tests had missed:
- **Demodulation infinite recursion** — a system-level behavior issue only triggered by full proof search
- **Cross-test state contamination** — an integration-level problem requiring subprocess isolation

Three bugs found by end-to-end testing that unit tests could not catch. This is the strongest possible argument for Amendment 1.

**Critical principle:** Before concluding that any work is successful, end-to-end tests must be run and pass. Unit test passage alone is insufficient evidence of system correctness.

### Lesson 2: Porting requires semantic verification, not just syntactic translation
The original `_deny_goals` was a correct syntactic translation (flip literal signs), but missed the semantic requirement (Skolemize existential variables). **C code often has behavior embedded in function call chains that isn't obvious from the callee alone.** The C function `process_goal_formulas()` Skolemizes as a separate step that wasn't visible from the negation code alone.

### Lesson 3: End-to-end testing against a reference implementation is essential
This bug would have been caught instantly by any test that compared Python's denied goals against C Prover9's denied goals. The cross-validation infrastructure exists but is optional and rarely run. **Reference implementation comparison should be mandatory, not optional.**

### Lesson 4: Proof soundness needs invariant checking
A 2-line proof for a problem that C Prover9 solves in 10+ steps should trigger an alarm. **Proof length and search depth are observable invariants** that can serve as regression detectors — if they change dramatically, something is wrong.

### Lesson 5: Silent failures compound investigation cost
The ML integration's `hasattr()` guards (lines 1116-1119) are innocent of this bug, but they initially misdirected the investigation toward buffer refactoring. Silent failures create noise that obscures the actual signal. **Every silent failure that was investigated was time not spent on the real cause.**

### Lesson 6: Misdirection risk in complex systems
The initial hypothesis (buffer refactoring → ML starvation → poor selection → premature termination) was plausible, internally consistent, and wrong. The actual cause was simpler and more fundamental. **Complex systems generate plausible-but-wrong hypotheses; end-to-end tests cut through the noise.**

### Lesson 7: Every entry point needs its own tests
The CLI, the `OnlineLearningSearch` wrapper, and direct `GivenClauseSearch` usage all exercise different code paths. A bug in goal denial affects ALL entry points, but subtler bugs (like the ML integration issues uncovered during investigation) might affect only specific paths. Each entry point must be tested independently.

---

## Part 7: Meta-Analysis — Investigation Methodology Review

### 7.1 Incident Timeline

| Phase | Activity | Outcome |
|-------|----------|---------|
| **Detection** | User observes 2-line proof for complex theorem | Bug confirmed |
| **Hypothesis 1** | Buffer refactoring suspected (KEPT removed, subsumption callbacks) | Plausible but wrong |
| **Investigation** | 4 research agents analyze buffer changes, testing gaps, architecture, monitoring | Extensive latent risks found |
| **Diagnosis** | Diagnostician reproduces issue, traces proof derivation | **Root cause: missing Skolemization** |
| **Pivot** | Hypothesis updated from ML regression to porting bug | Correct diagnosis |
| **Fix** | Skolemization added to `_deny_goals()` | **DONE** — Skolem constant helpers + variable substitution |
| **Validation** | End-to-end tests confirm correct behavior | **DONE** — 302 core + 87 search tests pass, vampire.in produces 10-clause proof |

### 7.2 Investigation Efficiency Analysis

**What went well:**
- Multi-agent parallel investigation covered broad ground quickly
- Preparatory research uncovered latent ML integration risks that are independently valuable
- Diagnostician correctly identified the actual root cause through proof trace analysis
- Cross-referencing C Prover9 behavior was the decisive diagnostic step

**What could improve:**
- Initial hypothesis focused on the most recent change (buffer refactoring) rather than testing the simplest explanation first
- The investigation would have been faster if the first step was: "compare C and Python output on the same input" rather than "analyze buffer changes"
- Confirmation bias: early research focused on ML integration because the report mentioned online learning

**Recommended investigation protocol for future incidents:**

1. **Reproduce first:** Run the exact failing case and capture full output
2. **Compare against reference:** Before theorizing, compare C vs Python output on the same input — differences point directly to the bug
3. **Bisect the pipeline:** Check intermediate state (parsed input → denied goals → initial clauses → search) to localize where behavior diverges
4. **Hypothesize last:** Only after localizing the divergence, form hypotheses about root cause
5. **Test hypotheses with minimal experiments:** Don't analyze code for hours — run targeted experiments

### 7.3 Multi-Agent Investigation Patterns

This incident used 6 agents (diagnostician, test-engineer, implementer, analyst, plus 4 sub-research-agents). Observations:

**Effective patterns:**
- Diagnostician working independently to reproduce and trace the bug
- Analyst doing preparatory research in parallel while blocked
- Test engineer building coverage independently of diagnosis

**Patterns to improve:**
- Analyst's initial research focused on ML integration (wrong direction) — could have been redirected earlier if diagnostician shared preliminary findings sooner
- Two research agents (buffer refactoring + testing infrastructure) could have been one agent with broader scope
- The pivot from "ML regression" to "Skolemization bug" required re-analyzing with the correct frame

### 7.4 Process Amendments Summary

| Amendment | Type | Prevents | Effort |
|-----------|------|----------|--------|
| **1. End-to-end testing protocol** | Testing | Integration boundary bugs | Medium |
| **2. Silent failure elimination** | Code quality | Investigation misdirection | Low |
| **3. Change impact assessment** | Process | Unvalidated ML changes | Low |
| **4. Search loop invariant testing** | Testing | Loop behavior regressions | Medium |
| **5. Monitoring integration** | Observability | Undetected anomalies | High |
| **6. Interface contracts** | Architecture | Silent wiring failures | High |
| **7. Porting verification protocol** | Process | Incomplete C→Python ports | Medium |

### 7.5 Risk Assessment: Other Similar Risks

Based on this incident, the following areas share the same risk profile (porting gaps that unit tests can't catch):

1. **`_simplify()` vs C `cl_process_simplify()`**: Does Python apply the same simplification steps in the same order? Missing a step could produce different (possibly unsound) results.

2. **`_auto_cascade()` vs C auto settings**: Does Python's inference auto-detection match C's? Differences could enable/disable inference rules incorrectly.

3. **Clause ordering**: Does Python's `GivenSelection` produce the same selection order as C's `giv_select.c`? Different ordering produces different proofs but both should be sound.

4. **Weight computation**: Does Python's `default_clause_weight()` match C's `clause_weight()`? Weight differences could cause different search paths.

5. **Variable renumbering**: Does Python's `renumber_variables()` match C's behavior? Differences could affect unification.

**Recommendation:** Run the porting audit in Amendment 7B systematically. Each function pair should have a comparison test before the next development cycle.

---

## Appendix A: Files Referenced

| File | Lines | Relevance |
|------|-------|-----------|
| `pyladr/apps/prover9.py` | 249-300 | **ROOT CAUSE:** `_deny_goals()` — missing Skolemization (now fixed) |
| `pyladr/apps/prover9.py` | 545 | CLI always uses plain GivenClauseSearch |
| `pyladr/search/given_clause.py` | 177-180, 236-246, 650-714 | Search core, slots, main loop, limbo processing |
| `pyladr/search/online_integration.py` | 599-613, 660-709, 1044-1121 | Buffer change, callbacks, dynamic subclassing (latent risks) |
| `pyladr/ml/online_learning.py` | 85-94, 210, 242, 327, 553 | OutcomeType changes, positive classification (latent risks) |
| `pyladr/search/ml_selection.py` | 256-262, 417-418 | Silent exception handling in ML selection |
| `pyladr/monitoring/` | all | Extensive monitoring infrastructure — NOT integrated |
| `tests/unit/test_search.py` | 499-514 | Minimal max_given test |
| `tests/cross_validation/` | all | C binary required, often skipped — would have caught this |

## Appendix B: Risk Matrix

| Risk | Probability | Impact | Detection Difficulty | Priority |
|------|------------|--------|---------------------|----------|
| Silent callback failure | **Confirmed** | Critical | Hard (no logs) | P0 |
| Experience buffer starvation | High | Critical | Medium (inspect buffer) | P0 |
| Dynamic subclass method drift | Medium | Critical | Hard (runtime only) | P1 |
| Missing CI enforcement | **Confirmed** | High | N/A | P1 |
| Entry point test gaps | **Confirmed** | High | Medium | P1 |
| Disconnected monitoring infrastructure | **Confirmed** | High | N/A (exists but unused) | P1 |
| Broad exception catching in ML selection | Medium | High | Hard (errors masked) | P1 |
| ML import guard silent degradation | Medium | High | Hard (no warning logged) | P1 |
| Thread safety in ML/search | Low | High | Very Hard | P2 |
| Statistics regression drift | Medium | Medium | Easy (with baselines) | P2 |
| Double init in dynamic subclass | Low | Low | Easy | P3 |
| print() vs logging inconsistency | **Confirmed** | Low | Easy | P3 |

## Appendix C: Monitoring Integration Roadmap

**Phase 1 (Immediate):** Add WARNING logs for silent failures
- Callback registration skipped → WARNING
- ML import failed → WARNING
- Buffer empty after N clauses → WARNING

**Phase 2 (Short-term):** Integrate SearchAnalyzer
- Per-iteration snapshots every 100 given clauses
- Rate calculation: clauses/second, keep ratio, generation ratio
- Anomaly alerts for rate collapse

**Phase 3 (Medium-term):** Integrate HealthChecker + CircuitBreaker
- Periodic health evaluation during search
- Circuit breaker halts ML on repeated failures
- Buffer health monitoring (positive/negative ratio)

**Phase 4 (Long-term):** Full observability
- Learning regression detection during search
- Convergence monitoring for online learning
- Memory budget enforcement
- A/B comparison framework for ML vs traditional selection
