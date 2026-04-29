# Unified Multi-Agent Framework

An evidence-based system for persistent multi-agent technical collaboration, combining delegation methodology with continuous improvement protocols.

**Version:** 2.2 (Cycle-6 amendments: by-ID baseline protocol strengthened for cycle-start task, slow-test marking for external-binary tests)
**Based on:** Empirical analysis of 15+ successful missions with 100+ tasks completed, amended through 6 meta-improvement cycles

---

## 1. Core Principles

### 1.1 Evidence-Based Operation
- All protocols derived from real mission data and systematic analysis
- Methodology continuously evolves through empirical amendments
- Professional engineering standards maintained throughout
- Proportionality constraint: coordination overhead must not exceed coordination problem cost

### 1.2 Coordinator Constraints
**CRITICAL:** The coordinator NEVER writes code or modifies files under any circumstances.
- **Role:** Pure coordination, task assignment, conflict resolution
- **When agent unresponsive:** Formal task reassignment through system (never parallel implementation)
- **Core responsibility:** Continuous assessment, dynamic assignment, integration management

### 1.3 Integration-First Architecture
- Features incomplete until fully integrated and functionally accessible
- Foundation-first development: shared patterns before parallel implementation
- Architecture consistency takes priority over development velocity

---

## 2. Task Management

### 2.1 Task Taxonomy

| Category | Complexity | Duration | Delegation Value |
|----------|------------|----------|------------------|
| **Architecture** | High (10-15) | 2-6 tasks | Very High - system-wide knowledge |
| **Refactoring** | Medium-High (8-12) | 1-4 tasks | High - regression prevention |
| **Security** | Medium-High (8-12) | 1-3 tasks | Very High - specialized knowledge |
| **Performance** | Medium (6-9) | 1-2 tasks | High - profiling expertise |
| **Testing** | Medium (6-9) | 1-3 tasks | High - systematic coverage |
| **Integration** | High (10-12) | 2-4 tasks | High - cross-component view |
| **Bug Fix** | Low-Medium (4-7) | 1 task | Medium - domain dependent |

### 2.2 Complexity Scoring
Score each task (1-5 each): **Scope + Risk + Knowledge = Total (3-15)**

| Score | Classification | Approach |
|-------|---------------|----------|
| 3-5 | Simple | Individual work |
| 6-9 | Moderate | Single specialist |
| 10-12 | Complex | Specialist + coordinator oversight |
| 13-15 | Critical | Senior specialist + validation |

### 2.3 Sequential Task Building Protocol
**Key insight from Amendment Cycle 3:** Tasks that build on previous work create compounding value.
- Each task should enhance capability for subsequent tasks
- Prioritize systematic progression over independent parallel work
- Target 85-90% completion rate for optimal scope balance

### 2.4 Explicit Scope Ceilings for Refactoring Tasks

**Key insight from Cycle 5 (Task #5):** When a refactoring task's description contains the explicit prohibition "**Do NOT do a large risky refactor — only extract what is clearly self-contained**," the specialist treats it as a forcing constraint and resists scope creep. Without this language, the same specialist will be tempted to bundle moderate-value side refactors into the task, increasing risk and blurring success criteria.

**How to apply:**
- Every refactoring task description MUST specify:
  1. A **scope ceiling** ("only extract what is clearly self-contained," "touch at most N files," "do not modify shared interfaces").
  2. A **follow-up sink** ("if you find additional opportunities, create new tasks for them, do not inline").
  3. An **analysis fallback** ("if no safe extractions remain, deliver a written analysis instead").

**Cycle 5 evidence (Task #5):** The specialist initially identified 4 extraction candidates in given_clause.py: `_trace_proof` (clean), `_simplify_eq_literals` (moderate self-coupling), `_apply_hint_weight` (many external test callers), R2V subsystem (deep self-coupling). The scope ceiling prohibition kept the work focused on `_trace_proof` (-35 lines, 10 tests, zero risk) and routed the R2V subsystem observation to a new backlog task (#10) rather than attempting it mid-cycle. Without the prohibition, a plausible failure mode was: specialist attempts R2V extraction, runs over, produces a medium-risk PR that needs cycle-long validation.

**Failure mode the ceiling prevents:** "scope-creep-by-observation" — where discovering a second extraction candidate during work on a first feels like efficiency gain but is actually risk concentration.

---

## 3. Specialist Roles

### 3.1 Core Specialists

**Architecture (Christopher):** System design, patterns, integration points
- **Authority:** Can halt conflicting implementations across specialists
- **Key focus:** Foundation-first validation, cross-component consistency

**Testing (Edsger):** Coverage, quality assurance, behavioral validation
- **Authority:** Quality gatekeeper - no task accepted without passing tests
- **Dual metrics requirement:** Both test pass rate AND coverage analysis

**Performance (Donald):** Optimization, algorithmic improvements, profiling
- **Requirement:** All performance claims backed by empirical data, not estimates
- **Key deliverables:** Benchmarks, profiling reports, baseline measurements

**Security (Bruce):** Vulnerability analysis, threat modeling, input validation
- **Coordination required:** Data entry/exit, file I/O, config parsing, credentials
- **Key focus:** Preliminary validation before P1 escalation

**Dependencies (Frederick):** Package management, dependency hygiene, data workflows
- **Coordination required:** Lock files, package metadata, build system changes
- **Key focus:** Evidence-based scope assessment

### 3.2 Supplementary Specialists
- **Documentation (Edward):** API docs, architecture guides
- **UX/API (Don):** User experience, interface design, workflow validation
- **DevEx (Alan):** Developer tooling, CI/CD, workflow optimization
- **Monitoring (Edwards):** Logging, metrics, observability systems
- **Error Handling (Nancy):** Exception handling, resilience patterns
- **Compatibility (Tim):** API stability, version compatibility

### 3.3 Team Sizing Guidelines
- **Optimal:** 4-8 specialists for standard missions (20-80 tasks)
- **Hard limit:** Never exceed 12 specialists (coordination overhead exceeds benefits)
- **Hub-and-spoke:** All status through coordinator, peer-to-peer for technical details

---

## 4. Quality Gates

| Gate | When | Criteria | Enforced By |
|------|------|----------|-------------|
| **G0: Pre-Assignment** | Before start | Task scoped, dependencies resolved | Coordinator |
| **G1: Progress Check** | Mid-task (complex only) | Approach sound, tests passing | Coordinator |
| **G2: Task Completion** | Specialist reports done | All tests pass, scope verified | Testing Specialist |
| **G3: Integration** | Before merge | Cross-domain consistency | Architecture Specialist |
| **G4: Mission Complete** | All tasks done | Full suite passes, quality improved | Coordinator + Testing |

### 4.1 Validation Requirements
Every completed task MUST include:
- **Test evidence:** Pass count, new tests added, zero failures
- **Regression check:** Full test suite execution
- **Scope verification:** Changes limited to task scope
- **Empirical claims:** Performance improvements backed by benchmarks

### 4.2 Behavioral vs Structural Validation

**Key insight from Cycle 5 (Task #3):** Tests that assert an invariant *exists* (structural) are cheap to satisfy without actually fixing the problem. Tests that assert the *behavior the invariant protects* (behavioral) are the real quality gate.

| Validation style | What it asserts | Failure mode |
|------------------|-----------------|--------------|
| **Structural** | "The split exists / the config is validated / the guard is present" | Vacuously satisfiable — any change to the code shape passes it |
| **Behavioral** | "The thing the structure protects against does not happen" | Only passes when the root problem is genuinely fixed |

**How to apply (guidance for any task adding validation):**
- First ask: *what bad outcome is this validation preventing?*
- Write the test that would detect that outcome if it occurred.
- Add structural tests only to pin down implementation details the behavioral test cannot see.

**Cycle 5 evidence (Task #3):** Fixing ML training data leakage could have been validated by asserting `stats["val_loss"] in stats` (structural — satisfied by any split, including a broken one). The real validation is asserting the empirically expected pattern `train_loss < val_loss ≈ test_loss`: this only holds when the split prevents walks from a single clause straddling the boundary. A broken split would fail this check immediately. Both kinds of tests were added; the behavioral one is load-bearing.

**Related anti-pattern:** §10.1 "Testing Theater." Behavioral framing is the positive inverse — it protects against the anti-pattern by construction.

### 4.3 Ratchet-Gate Quality Patterns

**Key insight from Cycle 5 (Task #9):** When a codebase accumulates "known exceptions" to a quality rule (trust-boundary violations, xfails, skipped tests, deprecated-API uses), the path of least resistance is to add new entries to the exception list rather than fix them. A **ratchet-gate** is a test that asserts the exception list's size is at-most its current count — forcing a binary choice: fix the underlying issue, or fail the build.

**Pattern shape:**
```python
KNOWN_VIOLATIONS = {
    # ... existing entries ...
}

def test_known_violations_shrink_over_time() -> None:
    """Ratchet: this list never grows; goal is zero."""
    total = len(KNOWN_VIOLATIONS)
    assert total <= N, (
        f"Known violations increased to {total} — "
        "the fix path is the preferred path."
    )
```

**Where this applies:**
- Trust boundary violations between modules
- `xfail` / `xskip` counts in test suites
- Deprecated API usage at known call sites
- Known flaky tests pending investigation
- Any "planned to fix" list in any form

**How to apply (when adding such a list):**
1. Define the list.
2. Immediately pair it with a ratchet-gate test capped at its current count.
3. When resolving entries, lower the cap in the same commit.
4. When a new offender appears, the test blocks — forcing fix or justification in review.

**Cycle 5 evidence (Task #9):** `test_known_violations_shrink_over_time` forced the resolution of two new cross-boundary imports into a genuine architectural fix (adding `from_sos_clauses` classmethod + `build_provider_config_from_search_options` helper in rnn2vec provider) rather than admitting them to `KNOWN_VIOLATIONS_SEARCH_TO_ML`. The ratchet was subsequently tightened 19→8 in Cycle 5 after Task #8 cleaned up stale entries for deleted modules.

**Complementary anti-pattern** (added to §10.1): "Known-Exception Inflation."

### 4.4 Infrastructure Skips Must Fail Loud

**Key insight from Cycle 5 (Task #6):** A test suite with 179 silently-skipped tests can hide two categories of bugs at once:
1. The infrastructure reason the tests aren't running (in T6: a hard-coded path constant pointing to a non-existent binary).
2. The actual test failures the infrastructure was supposed to catch (in T6: 129 pre-existing C/Python equivalence mismatches).

Both were invisible until the path was fixed. The cycle's initial baseline (T1) reported "179 skipped, 46 failed" without raising an alarm, because skip-count tracking doesn't distinguish *why* tests skipped.

**Protocol:**

| Skip reason | Default behavior |
|-------------|------------------|
| **Feature unavailable** (torch not installed, optional dep missing) | Skip silently; test is genuinely unrunnable in this environment. |
| **Infrastructure misconfigured** (missing binary, bad path, stale config) | **Fail loud.** Do not silently skip. Require explicit opt-in to tolerate. |
| **Test marked flaky / pending investigation** | Track in a ratchet-gated list (§4.3); report count in baseline. |

**Cycle-start baseline report MUST surface:**
- Any module with >N skipped tests (suggested N=20) — triggers investigation.
- The reason distribution (feature-unavailable vs infra-misconfigured vs flaky vs other).
- Diff vs previous cycle's skip distribution (new skips are a smell).

**Cycle 5 evidence (Task #6):** Skip marker read "C prover9 binary not found (run 'make all' to build)" — but the binary *was* built; the harness pointed to the wrong path. The diagnostic message misdirected the reader. A cycle-start baseline that flagged "179 tests in one module skipped with single reason" would have surfaced this at cycle open, not mid-cycle.

**Related anti-pattern (§10.1): "Silent Infrastructure Skip."** A skip that is not a genuine feature-unavailable signal masquerading as one.

### 4.5 Slow-Test Marking for External-Binary Tests

**Key insight from Cycle 6 (Architecture specialist, gate-verification of T6):** Tests that run external binaries — especially theorem provers, compilers, or solvers on nontrivial inputs — have no inherent time bound. When such tests are mixed into a cross-validation suite alongside sub-second unit-level tests, a single slow test (`vampire.in`, `x2_group`) can block all subsequent gate verifications for hours. The gate-checker (who is in a time-sensitive verification loop) cannot exclude the slow test without knowing which test it is.

**Protocol:**

1. Any test that spawns an external binary on a nontrivial input MUST be marked `@pytest.mark.slow`.
2. The CI pipeline MUST define two job tiers: (a) fast gate (excludes `slow`), (b) full suite (includes `slow`). T6-class gate verifications use the fast gate.
3. When a test is added to a class that previously had no `slow` markers, the PR description MUST note the new slow test and its expected runtime.
4. The cycle-start baseline report MUST include the count of `slow`-marked tests in each module (alongside skip-count statistics).

**Why a mark rather than a timeout:** Timeouts kill the prover mid-run and can leave the test output in an indeterminate state (partial proof, no proof). The `slow` mark defers the test to a separate job where the runner can afford to wait.

**Cycle 6 evidence (T6 gate verification):** `TestHardProblems::test_vampire_in` ran for over 77 minutes while blocking parallel gate-check invocations. The prover was solving `vampire.in` (a hard equational problem); this is correct and expected behavior. The problem was that the verifier had no way to exclude it without knowing the test ID in advance. Marking it `@pytest.mark.slow` at authoring time would have allowed `pytest -k "not slow"` from the start.

**Relationship to §4.4:** §4.4 makes infrastructure skips loud. §4.5 makes slow tests predictable. Together, they ensure the cross-validation suite can serve as a fast gate.

---

## 5. Communication Protocols

### 5.1 Required Standards
**Professional Style:**
- Factual, evidence-based reporting without hyperbole
- Technical precision over emotional language
- Quantified results (tests passing, performance metrics)
- Honest assessment when no genuine work opportunities exist

**Prohibited:**
- Grandiose superlatives ("revolutionary," "unprecedented," "ultimate")
- Speculative work generation when no real needs exist
- Status messages without actionable content

### 5.2 Message Templates

**Task Completion:**
```
Task #{id} completed.
Changes: {files modified, lines changed}
Tests: {new tests added}, {total passing}, {0 failures}
Next: {claiming Task #{next_id} | proposing {description}}
```

**Coordination Request:**
```
Need coordination with {specialist role}.
Shared resource: {file or module}
Duration: {estimated time}
Proposed sequencing: {who goes first and why}
```

### 5.3 Structural Coordination
- **File Intent Registration:** Register before editing shared files
- **Task Ownership Enforcement:** MANDATORY TaskList checking before starting work
- **Unresponsive Agent Protocol:** 5-minute timeout → formal reassignment

### 5.4 Reassignment Transition Protocol

**Key insight from Cycle 5 (Task #8):** When a task is reassigned mid-cycle (timeout, verbal retasking, load rebalancing), a fresh task-assignment message to the new owner is insufficient. The original owner may still believe they hold the task, and the new owner lacks context about any partial work.

**Required notifications on reassignment:**

| Recipient | Message contents |
|-----------|------------------|
| **Original owner** | Explicit release: "Task #N has been reassigned to {new_owner}. Stop work and acknowledge." |
| **New owner** | Fresh assignment + provenance: "Task #N assigned to you. Previously held by {original_owner}; partial work: {summary or 'none'}." |
| **Coordinator log** | Transition event recorded (who→who, reason, timestamp) |

**How to apply:**
- Never rely on silent reassignment. The new owner's assignment message should cite prior ownership.
- If the task has no partial work, still say "partial work: none" — absence of mention is ambiguous.
- For verbal retasking (where the coordinator changed their mind before formalizing), retroactively notify the original owner even if no TaskList state was updated.

**Cycle 5 evidence (Task #8):** Coordinator reported Task #8 was verbally assigned to the Testing specialist first, then picked up by the Architecture specialist. From the Architecture specialist's perspective, the assignment message looked like a normal first-time assignment — no signal of reassignment, no provenance context. Testing specialist's own retrospective (Edsger, Cycle 5): "verbal assignment in prose ≠ `TaskUpdate owner=X`. The task-list state was inconsistent with team-lead's stated intent." Visibility gap caused no concrete harm this cycle (no partial work existed) but is a silent-failure class for future cycles where partial work might exist.

**Invariant to maintain:** The task list is the single source of truth for ownership. If TaskList shows no owner, the task really has no owner. Verbal assignment in conversation must be *immediately* followed by `TaskUpdate owner=X`, before any other message — otherwise a peer scanning the task list will correctly conclude the task is unclaimed and may self-claim it (→ duplicated work in the worst case).

---

## 6. Coordination Mechanisms

### 6.1 Work Distribution
1. **Automatic Assignment:** Coordinator monitors queue and agent availability
2. **Domain Optimization:** Match tasks to specialist expertise
3. **Load Balancing:** Prevent overload and idle time
4. **Dependency Tracking:** Sequence related work, prevent conflicts
5. **Architecture Consistency:** Validate coherence across all work

### 6.2 Priority Management
- **P0 Critical:** Blocking issues, security vulnerabilities, system failures
- **P1 High:** Performance bottlenecks, user experience issues
- **P2 Medium:** Technical debt, optimization opportunities
- **P3 Low:** Documentation, tooling improvements

### 6.3 Domain-Specific Coordination
**High-risk coordination required:**
- Security: Data entry/exit, credentials, input validation
- Dependencies: Lock files, build systems, package metadata
- DevEx: Shared infrastructure files, test fixtures
- Architecture: Cross-component dependencies, pattern deviations

### 6.4 Pull-First Task Claiming

**Key insight from Cycle 5 (Performance specialist retrospective, 3 observed instances):** Two valid coordination protocols can run in parallel and race:

- **Pull:** Agent completes task, queries TaskList, self-claims highest-priority available task they're qualified for, works, reports.
- **Push:** Coordinator observes completion, decides next assignment, sends explicit routing message.

Both are individually correct. They collide when agent cycle time is shorter than coordinator decision-and-send time — which is exactly when the cycle is going well.

**Protocol:**

1. **Default to pull.** After a completion, agents consult TaskList and self-claim the next task that matches their domain and meets blocking/priority rules.
2. **Push is for exceptions**, specifically:
   - (a) Overriding the natural pick (e.g., coordinator has information agent lacks).
   - (b) Declining a self-claim the agent might make (e.g., hold for cross-agent coordination).
   - (c) Providing context the task description does not capture.
   - (d) Assigning work not yet in TaskList.
3. **When push and pull collide:** if the agent has already claimed or completed the task the coordinator was about to route, the coordinator's message is **redundant, not authoritative**. Agent acknowledges the collision ("already claimed/completed") and moves on. No reconciliation round required.
4. **Separate acknowledgement from routing.** "Task N accepted" and "Task M is next" should be two messages, not one. Bundling guarantees that a fast-cycling agent will have moved past the next pick by the time the combined message lands.

**Cycle 5 evidence:** Performance specialist reported 3 collision instances in one cycle (after T2, T11, and T12). Each burned one round-trip to reconcile. Architecture specialist observed one meta-collision: the T7 assignment message for meta-improvement arrived after T7 was already completed via self-claim, with peer contributions bundled alongside — producing extension work that would have been part of the original delivery if the pull had been acknowledged as authoritative.

**Non-goal:** This is not a bid against coordinator intent. When coordinator push and agent pull disagree on the pick, the coordinator's *reasoning* (context the agent lacks) takes precedence; but if the coordinator's push arrives after the pull has already happened and was a reasonable pick, the pull stands.

---

## 7. Success Metrics

### 7.1 Mission-Level Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Task completion rate** | > 85-90% | Completed / total created |
| **Regression rate** | 0% | Test failures introduced |
| **Parallel utilization** | > 70% | Parallel work time |
| **Communication overhead** | < 20% | Coordination / total messages |
| **Assessment/delivery balance** | <20%/>80% | Analysis vs implementation time |

### 7.2 Quality Indicators
- **Integration health:** Functional vs just implemented features
- **Code duplication:** <5% duplicated code
- **Professional scope alignment:** Actual vs perceived work opportunities
- **Dual quality metrics:** Both pass rate and coverage maintained

### 7.3 Baseline Comparison: By-ID, Not By-Count

**Key insight from Cycle 5 (Testing specialist retrospective, Task #1):** A cycle-start baseline of "46 failing tests" creates a fragile non-regression bar. A new regression that brings the count from 46 to 46 (by coincidentally fixing one test and breaking another) passes count-based comparison while hiding a genuine regression.

**Protocol:**

1. **Preferred:** Cycle-start baseline is clean green. Pre-existing failures are resolved before the cycle opens. Non-regression bar is simple: no failures.
2. **If pre-existing failures must be tolerated:** Catalog them by **test ID** (module + class + test name), not by count. Store the catalog as the cycle baseline artifact. Downstream non-regression checks assert: "the set of failing tests is a subset of the baseline set" — i.e., no NEW failures, even if one was fixed while another broke.
3. **Count-based comparison** ("failures ≤ 46") is explicitly a weak signal. Use only when ID-based comparison is infeasible (e.g., parametrized tests with nondeterministic IDs), and flag the weakness in the cycle report.

**Cycle 5 evidence (Task #1):** 46-failure baseline was count-based. No regression was masked this cycle by the weakness, but the method permits it.

**Cycle 6 gap (Architecture specialist, T6 gate verification):** The T1 task was scoped to §4.4 (infrastructure-skip loudness) and produced a skip-count analysis. It did NOT produce a by-ID failure catalog (§7.3), because the T1 task template didn't require both outputs. When T6 gate verification began, the verifier had to re-establish the by-ID baseline from scratch. This is a failure of the T1 task template, not a failure of the verifier.

**T1 task template MUST include both outputs:**
1. **Skip-count statistics per §4.4:** module-by-module skip count with reason distribution.
2. **By-ID failure catalog per §7.3:** the exact test IDs of pre-existing failures, in a form usable as a set for subsequent subset-comparison.

The artifact produced by T1 should look like:
```
# Cycle-N Baseline Artifact
## Skip distribution
- module X: N skipped (all feature-unavailable)
- module Y: M skipped → INVESTIGATE (all same skip reason, may be infra-misconfigured)

## Pre-existing failures (by ID)
- tests/cross_validation/.../TestEquationalProblems::test_...[congruence]
- ...
```
Downstream gate verifications assert: `current_failures ⊆ baseline_failure_ids`.

If the baseline is clean green, the artifact is just the skip-count statistics (no by-ID section needed).

---

## 8. Meta-Improvement Protocol

### 8.1 Continuous Methodology Evolution
**Every improvement cycle concludes with systematic methodology analysis.**

**Phase 1: Results Analysis**
- Compare delivered functionality to intended goals
- Measure coordination effectiveness vs overhead
- Identify methodology blind spots

**Phase 2: Specialist Critique**
- Domain expert analysis of methodology effectiveness
- Root cause analysis of any coordination failures
- Evidence-based amendment design

**Phase 3: Framework Updates**
- Systematic amendment of this document
- Document empirical evidence for each change
- Define validation metrics for next cycle

**Phase 4: Enhanced Operation**
- Deploy updated methodology with improvements
- Monitor amendment effectiveness
- Track evolution over time

### 8.2 Amendment History Summary

**Cycle 1 (April 2026):** Task ownership enforcement, unresponsive agent protocols, structural coordination mechanisms

**Cycle 2 (April 2026):** Assessment/delivery balance (<20%/>80%), baseline measurement requirements, automated validation gates

**Cycle 3 (April 2026):** Sequential task building protocol, dual quality metrics, evidence-based scoping (85-90% completion target)

**Cycle 4 (April 2026):** Requirements tracking and validation protocol (REQUIREMENTS.md), systematic requirement compliance enforcement, reactive requirements amendment for surprising failures

**Cycle 5 (April 2026):** Framework v2.0 → v2.1. Amendments span six framework sections and derive from both Architecture-observed patterns (forcing functions that produced genuinely better outcomes) and peer retrospectives from Testing (Edsger) and Performance (Donald) specialists. New subsections: §2.4 scope ceilings for refactoring, §4.2 behavioral vs structural validation, §4.3 ratchet-gate patterns, §4.4 infrastructure skip loudness, §5.4 reassignment transition protocol, §6.4 pull-first task claiming, §7.3 baseline by-ID comparison, §9.6 numeric threshold validation at REQ-authoring time, §10.3 dead-dependency handling. Three new anti-patterns added to §10.1: Known-Exception Inflation, Silent Infrastructure Skip, Fallback-Gated-On-Fallback-Target. Cycle produced 10/12 scoped tasks plus 3 emergent (~92% completion, 0 regressions, 2581 tests green). Concrete cycle-5 evidence cited for each amendment; validation metrics defined in `cycle_6_validation_metrics.md` memory for cycle-6 effectiveness assessment.

**Cycle 6 (April 2026):** Framework v2.1 → v2.2. Two amendments from 9 validation metrics: §7.3 REVISED (T1 task template now requires both skip-count statistics and by-ID failure catalog; Cycle 6 T1 produced only count-based output, forcing gate verifier to re-establish the by-ID baseline during T6); §4.5 NEW (slow-test marking protocol for external-binary tests; `TestHardProblems::test_vampire_in` blocked gate verification for 77+ minutes in T6 with no fast-exclusion mechanism). All other 9 Cycle 5 amendments retained with supporting evidence: §4.4 directly applied (T2 fixed cross-val binary path + added ConfigurationError, unmasking 179 hidden tests); §2.4 applied cleanly to T6 (scope ceiling, follow-up sink, analysis fallback all present); §4.2 improved to ~83% behavioral tests (T2: 6 behavioral guard tests; T6: 12 behavioral subsystem tests); §6.4 pull-first worked cleanly with zero collisions; §9.6 clean cycle (no PENDING REQs created, T4 memory profiled at authoring time). Metrics 4/5 (dead-dependency handling, reassignment notices) had no qualifying events — retained tentatively; Cycle 7 should try to generate evidence. Cycle produced 8/9 scoped tasks (T9 deferred to backlog, 89% completion), 0 regressions, 2329 unit tests green + 12 new behavioral tests. Major deliverable: R2V subsystem extracted from given_clause.py (8 methods, 10 attributes, ~380 lines → r2v_subsystem.py with event interface). Validation metrics defined in `cycle_7_validation_metrics.md` for Cycle 7 effectiveness assessment.

---

## 9. Requirements Tracking and Validation

### 9.0 Initial Requirements File Creation

**If REQUIREMENTS.md does not exist in the current project:** The methodology mandates creating an initial requirements baseline by analyzing the current state of the project.

**Initial Requirements Analysis Process:**
1. **Analyze existing functionality** - Identify what the system currently does and must continue to do
2. **Discover implicit requirements** - Performance expectations, compatibility constraints, operational behaviors
3. **Document quality standards** - Testing coverage, validation criteria, acceptable failure modes
4. **Establish architectural constraints** - Dependencies, interfaces, integration requirements
5. **Create measurable criteria** - Specific, testable conditions for each requirement
6. **Baseline current state** - Document current performance, behavior, and quality metrics

**Initial REQUIREMENTS.md Template:**
```markdown
# Project Requirements Specification

Formal requirements derived from initial project analysis (YYYY-MM-DD).
Each requirement has a unique ID, current status, and measurable criteria.

## 1. Core Functionality (REQ-C)
- [REQ-C001] [Description with current behavior and acceptance criteria]

## 2. Performance Requirements (REQ-P)
- [REQ-P001] [Current performance baseline and acceptable thresholds]

## 3. Quality Requirements (REQ-Q)
- [REQ-Q001] [Current test coverage and quality standards]

## 4. Integration Requirements (REQ-I)
- [REQ-I001] [External dependencies and compatibility requirements]
```

This initial requirements file will then be amended throughout the methodology as new capabilities are added or new correctness conditions are discovered.

### 9.1 Continuous Requirements Documentation
**REQUIREMENTS.md Protocol:** All new capabilities, features, and correctness conditions must be systematically tracked.

**Requirement Capture Events:**
- **New Feature Introduction**: Any capability added to the system
- **Correctness Condition Discovery**: New requirements for maintaining system integrity
- **Architectural Constraints**: Dependencies, compatibility requirements, performance thresholds
- **Quality Standards**: Testing coverage, validation criteria, operational requirements

### 9.2 Requirements Validation Process

**After Every Major Change:**
1. **Consult REQUIREMENTS.md** before considering change complete
2. **Validate All Requirements** are still met by the modified system
3. **Identify Discrepancies** where requirements are no longer satisfied
4. **Escalate Unmet Requirements** to P1 or P0 priority immediately

**Major Change Definition:**
- Architectural modifications
- New feature deployment
- Dependency updates
- Performance optimizations
- Security enhancements
- Integration changes

### 9.3 Requirements File Structure

**REQUIREMENTS.md Format:**
```markdown
# Project Requirements

## Core Functionality Requirements
- [Requirement ID] Description with measurable criteria
- [REQ-001] System must maintain C Prover9 exit code compatibility (1-7)
- [REQ-002] ML features must degrade gracefully when torch unavailable

## Performance Requirements
- [REQ-P001] No more than 10% performance degradation vs baseline
- [REQ-P002] Thread-safe operations with <1ms locking overhead

## Quality Requirements
- [REQ-Q001] Test coverage must exceed 80% for core modules
- [REQ-Q002] Zero regression tolerance in existing functionality

## Integration Requirements
- [REQ-I001] EmbeddingProvider protocol compliance mandatory
- [REQ-I002] Cross-validation tests must pass for C compatibility
```

### 9.4 Requirement Compliance Enforcement

**Validation Workflow:**
- **Pre-Change**: Document any new requirements introduced
- **Post-Change**: Execute requirements validation checklist
- **Compliance Failure**: Immediate escalation to specialist team
- **Resolution**: Requirements discrepancy becomes blocking issue until resolved

**Team Responsibility:**
- **Architecture Specialist**: Requirements architectural consistency
- **Testing Specialist**: Requirements validation and measurement
- **All Specialists**: Requirement identification and compliance verification
- **Coordinator**: Requirements discrepancy escalation and resolution tracking

This protocol ensures systematic tracking of project correctness conditions and prevents regression through requirements drift.

### 9.5 Reactive Requirements Amendment

**Surprising Failure Protocol:** When major regressions or unexpected failures are discovered, systematic requirement amendments must prevent recurrence.

**Amendment Triggers:**
- **Integration Failures**: Components work in isolation but fail when integrated
- **End-to-End Gaps**: Features pass unit tests but don't function in complete workflows
- **Validation Blind Spots**: Testing validates components but misses critical system behaviors
- **Production vs Development Differences**: Failures that only emerge in production-like usage
- **Silent Failures**: Systems that appear to work but critical functionality is non-operational

**Reactive Amendment Workflow:**
1. **Root Cause Analysis**: Identify why existing validation missed the failure
2. **Requirements Gap Assessment**: Determine what requirement would have caught this issue
3. **REQUIREMENTS.md Amendment**: Add specific, measurable requirement to prevent recurrence
4. **Validation Enhancement**: Update testing/validation protocols to detect this class of failure
5. **Methodology Review**: Assess if framework enhancements are needed

**Requirement Categories for Amendments:**
- **REQ-INT**: Integration requirements (end-to-end functionality validation)
- **REQ-E2E**: End-to-end workflow requirements (complete user journey testing)
- **REQ-PROD**: Production readiness requirements (real-world usage scenarios)
- **REQ-SIL**: Silent failure detection requirements (functionality verification beyond unit tests)

**Example Amendment Pattern:**
```markdown
# Added due to ML integration failure discovery (Date: YYYY-MM-DD)
- [REQ-INT001] All CLI flags must have demonstrable functional effect in end-to-end workflows
- [REQ-E2E001] ML features must show measurable output when enabled (logs, statistics, behavior changes)
- [REQ-PROD001] Feature integration must be validated beyond component-level testing
```

This reactive protocol ensures that validation blind spots become systematic prevention mechanisms for future development.

### 9.6 Numeric Threshold Validation at REQ-Authoring Time

**Key insight from Cycle 5 (Performance specialist retrospective, REQ-P002):** An acceptance criterion with a numeric threshold that was never measured before checkin can be not just wrong-valued but *structurally wrong* — wrong metric shape, wrong baseline, wrong units.

**Failure pattern observed:** REQ-P002 shipped with status `PENDING` and criterion `"peak RSS ≤ 2x initial allocation on problems with >10,000 kept clauses."` When measured (Task #2), memory growth turned out to be linear in retained clauses (~5.8 KB/clause, stable across 5k/10k/14k). A linear-in-Kept process cannot hold a constant ratio to a fixed baseline as Kept grows. The threshold would have failed for any problem big enough to stress the system — exactly the regime the requirement was supposed to cover. A single measurement at authoring time would have surfaced this; the author would have seen ~5–6 KB/clause and written the right requirement (Task #11 rewrote it to `≤ 8 KB/clause`, PASS).

**Protocol:**

1. **Every acceptance criterion with a numeric threshold MUST be validated against ≥1 real measurement at REQ-authoring time.** No exceptions.
2. **Thresholds expressed as ratios MUST explicitly name the baseline they are a ratio of** (e.g., "2x initial allocation" → "2x the allocation at time of first given-clause selection" or whatever the author actually meant). Unnamed baselines invite interpretation drift.
3. **Verify the metric's shape** (linear, constant, logarithmic, bounded) before picking a threshold form. A ratio threshold is only appropriate for quantities that are actually bounded ratios of their baseline; linear-growth quantities need per-unit thresholds.
4. **REQs in status `PENDING` are a smell.** `PENDING` means "the criterion was inferred rather than measured." Flag PENDING REQs as technical debt; schedule a measurement task to promote them to a validated status.

**Cycle 5 evidence (Task #2 + Task #11):** T2's measurement exposed both a wrong-value threshold *and* a wrong-shape threshold. Without T2's explicit scoping ("verify REQ-P002"), the ill-formed REQ could have survived indefinitely — reviewers would either nod at the language or flag it as FAIL without noticing the criterion itself was malformed.

**Corollary — measurement tasks for PENDING REQs:** When authoring a REQ whose threshold cannot be validated at authoring time (e.g., system not yet implemented), *immediately* create a paired measurement task scheduled for the next relevant cycle. The REQ and its measurement task are inseparable.

---

## 10. Anti-Pattern Prevention

### 10.1 Critical Anti-Patterns
- **Competing Implementations:** Multiple agents solving same problem differently
- **Coordinator Implementation:** Coordinator writing code instead of coordinating
- **Disconnected Features:** UI components not functionally integrated
- **Testing Theater:** High test counts not verifying actual behavior
- **Speculative Work Generation:** Creating busy work without genuine needs
- **Known-Exception Inflation:** Adding entries to a "known exceptions" / "known violations" / "xfail" list as a lower-effort substitute for fixing the underlying issue. *Prevention: pair every exception list with a ratchet-gate test (§4.3).*
- **Silent Infrastructure Skip:** A test that skips due to misconfiguration (missing binary, wrong path, stale fixture) but reports as a normal feature-unavailable skip. Hides bugs that the test was supposed to catch. *Prevention: infrastructure-skip rules fail loud by default (§4.4).*
- **Fallback-Gated-On-Fallback-Target:** A test of a fallback path that is itself gated on the presence of the thing the fallback is for (e.g., `pytest.importorskip("torch")` at module scope in a test file that covers the torch-unavailable fallback). When the fallback's trigger condition holds, every test in the file silently no-ops. *Prevention: fallback tests must not be gated on the presence of what they fall back from.*

### 10.2 Detection & Resolution
- **Immediate P0 priority** for anti-pattern resolution
- **Architecture specialist override** authority to halt conflicts
- **Proportional response:** Investigation effort matches problem impact
- **Risk tolerance framework:** Accept minor duplication when architecturally constrained

### 10.3 Dead-Dependency Handling: Delete vs Repair

**Key insight from Cycle 5 (Task #8):** When a shared dependency is deleted and many tests/callsites break in response, the default reaction is to repair each callsite. For tests specifically, **wholesale removal of the test file is often cleaner than per-test repair** when the file's organizing abstraction was the deleted dependency.

**Decision heuristic:**
1. Count: what fraction of the file's tests fail due to the dead dependency?
2. Check: are the passing tests covered elsewhere in the test suite?
3. If both (many failures) AND (redundant coverage elsewhere), **delete the file.**
4. Otherwise, surgically remove the dead-dependency-coupled tests and keep the rest.

**Why this matters:** Surgical per-test repair preserves the file's original organizing metaphor (e.g., "FORTE + ML + performance optimizations") even after the metaphor has become meaningless. This leaves future readers confused about why the file exists. Wholesale deletion is honest about the dependency's death.

**Cycle 5 evidence (Task #8):** `tests/integration/test_e2e_combined_systems.py` (845 lines, 101 FORTE references across 10 classes) had 44/60 tests failing after FORTE module deletion. The 16 passing tests covered parametrized `[baseline] / [priority_sos] / [lazy_demod]` scenarios already tested in `tests/integration/test_compound_optimization.py`. Wholesale deletion removed 845 lines vs the ~200-line surgical repair alternative, AND produced a cleaner test architecture (no vestigial FORTE docstrings).

**When to prefer surgical repair:**
- Passing tests cover scenarios NOT tested elsewhere.
- File's organizing metaphor is still valid with the dead dependency removed.
- Deletion would lose test coverage that matters.

**Related cleanup:** When deleting a dependency, also audit any "known exceptions" lists for stale entries referencing the deleted modules. In Cycle 5, Task #8 purged 11 stale entries from `KNOWN_VIOLATIONS_*` referencing deleted `pyladr.ml.forte.*` and `pyladr.ml.tree2vec.*` submodules.

---

## 11. Operational Protocols

### 11.1 Mission Launch Checklist
- [ ] **Scope defined** with measurable success criteria
- [ ] **Task breakdown** with complexity scoring applied
- [ ] **Team composition** selected based on domain requirements
- [ ] **Quality baseline** established (test count, coverage)
- [ ] **Domain boundaries** documented without overlaps
- [ ] **Communication protocols** established with all agents

### 11.2 Continuous Operation
- **Self-directed agents:** Continuously assess domains, propose improvements
- **Never idle:** Always have next work identified or actively seeking
- **Evidence-based proposals:** All suggestions backed by concrete analysis
- **Integration-first:** No feature complete until demonstrably functional

### 11.3 System Health Monitoring
- **Automated assessment:** Quality, performance, security metrics
- **Adaptive response:** Learn from patterns, optimize coordination
- **Trigger thresholds:** >5% performance degradation, <90% test coverage
- **Continuous learning:** Recognize recurring issues, systematic solutions

---

## 12. Key Operational Commands

### 12.1 Team Setup
```bash
# Create persistent team
TeamCreate team_name="improvement-team" description="Multi-agent improvement"

# Spawn specialists based on mission requirements
Agent subagent_type="general-purpose" name="Christopher" # Architecture
Agent subagent_type="general-purpose" name="Edsger"     # Testing
Agent subagent_type="general-purpose" name="Donald"     # Performance
Agent subagent_type="general-purpose" name="Frederick"  # Dependencies
# Add others as needed for mission scope
```

### 12.2 Mission Management
```bash
# Status check
SendMessage to="Coordinator" message="Status report: team activity, current priorities"

# Priority adjustment
SendMessage to="Coordinator" message="Priority shift: focus on [domain]. Reallocate resources."

# Health assessment
SendMessage to="Coordinator" message="System health assessment. Report metrics and issues."
```

---

**This unified framework enables systematic, evidence-based multi-agent collaboration with continuous self-improvement. The methodology evolves through empirical amendments, ensuring effectiveness increases over time while maintaining professional engineering standards.**