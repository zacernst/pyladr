# Per-Clause Memory Overhead: PyLADR vs C Prover9

**Cycle 6 Task T4 — Donald (Performance)** · 2026-04-28
(Errata appended 2026-04-29 during T8 implementation)

## Errata (T8)

- **`given_selection: str = ""` does NOT cost ~48 B/clause.** CPython
  interns `""` as a singleton (`"" is ""` → True), so a thousand default
  Clauses share one string object. The ~48 B figure in the table below
  was an artifact of the profiler not deduplicating strings by `id()`.
  Fixed in `clause_memory_breakdown.py`; the row now reports 0.0 B.
  Consequently the "None-sentinel" idea in the reduction table saves
  effectively nothing and was dropped from T8 scope.
- With the profiler fix, synthetic per-clause total is ~1.97 KB (was
  ~2.02 KB reported originally). The overall conclusion is unchanged:
  ~2 KB/clause in the object graph + ~4 KB/clause in auxiliary index
  state accounts for the measured 5.8–6.0 KB RSS/clause.
- The two remaining near-term wins (drop `Term.container`, intern
  arity-0 rigid constants) are still valid and were executed in T8.

## T8 outcomes (2026-04-29)

### Changes applied
1. **Dropped `Term.container` field** — never assigned to non-None in
   production; two read sites in `contrastive.py` removed. Term slot
   count: 6 → 5. `sys.getsizeof(term)` dropped 80 → 72 bytes.
2. **Rigid-constant interning** — arity-0 `get_rigid_term(sn, 0)` calls
   now share a cached Term per symbol (same pattern as the existing
   `_variable_cache`). Capped at 10,000 entries, thread-safe.

### Measured impact

- **`lattice_absorption.in` (real parsed input), per-clause graph cost:**
  - Pre-T8:  949.3 B/clause
  - Post-T8: 885.3 B/clause
  - **Saving: 64.0 B/clause (6.7% reduction on parse-time graph)**
- **Synthetic best case** (constants built via `get_rigid_term`):
  interning delta alone = 190.3 B/clause when constants are shared
  across 1,000 clauses (comparing `intern=True` vs `intern=False` runs).
  Real-world savings are smaller because substitution/copy_term paths
  bypass the cache — see "Follow-up candidates" below.
- **Heap-count impact** (across all 327 parsed fixture clauses):
  only 59 unique rigid constants populate the cache. Every subsequent
  occurrence of those 59 symbols reuses the cached Term instead of
  allocating a new one — not visible in the per-clause dedup metric
  but a real reduction in total heap allocation count.

### Caveats

- The 16-byte alignment in `_real_size()` absorbs the Term slot saving
  in the profiler's rounded numbers, but CPython's `pymalloc` uses
  8-byte size classes — so the actual heap saving from dropping
  `container` likely lands between 0 and 8 B/Term depending on the
  allocator bucket.
- Per-clause RSS growth (REQ-P002) is dominated by auxiliary index
  state (~3.8 KB/clause), not the Clause graph (~2 KB/clause). Change
  1+2 reduces the graph portion; index-side wins require separate work.

### Identity-safety audit (why interning is safe here)

Interning rigid constants changes `Term` object identity: two
constants with the same symbol that used to be distinct objects are
now the same object. This is safe only because no code in the
current inference pipeline relies on distinctness.

Audit of `is`/`is not` comparisons on `Term`-typed variables across
`pyladr/`:

- `pyladr/inference/demodulation.py:426, 440, 532` — all three are
  "did this rebuild change anything" short-circuits (`new_args[i] is
  not term.args[i]`, `rewritten_term is not term`, `new_atom is not
  lit.atom`). Interning makes these strictly more accurate: a rebuild
  that produces an equal constant now correctly reports "unchanged."
- `pyladr/core/substitution.py:229` in `apply_substitute`, the
  `t is into_term` check. This is a narrow public API (exported from
  `pyladr/core/__init__.py`, unit-tested, but not called by the
  current inference pipeline — `apply_substitute_at_pos` is used
  instead). The identity compared here is on the `into_term` argument
  threaded down the recursion, not on constants fetched from the
  interning cache, so the path is safe. If a future caller starts
  passing a cached constant as `into_term` they should use the
  position-based variant (as the docstring already suggests for
  shared variables).

Hash/equality semantics are unaffected: `Term` is a frozen dataclass
with structural `eq` and `hash`, so interned and non-interned
constants that match structurally compare equal and hash identically
in both regimes.

### Follow-up candidates (NOT in T8)

Christopher flagged these for future tasks with their own gates:

- **Evaluate `Term.term_id` removal.** Currently used only by
  contrastive.py (2 reads) and FPA indexing. WeakValueDictionary
  relocation has ~64 B/entry overhead that likely outweighs the 8 B
  slot saving — needs profiling of FPA indexing cost before committing.
- **Route arity-0 rebuilds through the rigid-constant cache.**
  `substitution.py:206,282`, `term.py:321`, `formula_processor.py:207`,
  and similar sites construct `Term(private_symbol=ps)` directly
  when rebuilding constants, bypassing the cache. Migrating them to
  `get_rigid_term(-ps, 0)` or a new `_intern_rigid(ps)` helper would
  broaden the heap-count win to derived clauses. Touches substitution
  core — own task, own gate.


## Question

REQ-P002 shows PyLADR uses 5.8–6.0 KB per retained clause, whereas C Prover9
uses ~1.6 KB on the same problem. Why the 3.6× gap, and which contributors are
architecturally constrained vs. addressable?

## Method

1. **Profile Clause object graph** with a GC-aware sizer
   (`tests/benchmarks/clause_memory_breakdown.py`) that adds the 16-byte
   `PyGC_Head` missing from `sys.getsizeof()` and rounds up to CPython's
   16-byte malloc alignment.
2. **Walk the Term tree** deduplicating by `id()` so shared variable terms
   (from the `_variable_cache`) are only counted once — matching the
   running process's actual memory use.
3. **Compare to the C struct layouts** in
   `reference-prover9/ladr/{topform,literals,term}.h` to establish the
   C-side baseline per object.
4. **Subtract from measured RSS growth** (REQ-P002's 5.8–6.0 KB/clause) to
   size the residual — the auxiliary indexing / cache state that lives
   outside the Clause graph.

All numbers are for a representative synthetic unit equality clause
(`f(g(x,y), c1) = f(g(y,x), c2)`) with one `PARA` + one `DEMOD`
justification — a typical paramodulation-derived clause.

## Results

### Clause object graph: ~2.0 KB/clause

| Component                       | B/clause | % of graph |
|---------------------------------|---------:|-----------:|
| Term instances (atoms+subterms) |    672.2 |      33.3% |
| Term args tuples                |    400.0 |      19.8% |
| Justification instances         |    272.0 |      13.5% |
| Justification inner tuples      |    257.3 |      12.8% |
| Clause header (slots)           |    160.0 |       7.9% |
| Justification tuple             |     80.0 |       4.0% |
| Literal instances               |     64.0 |       3.2% |
| Literals tuple                  |     64.0 |       3.2% |
| `given_selection` empty str     |     48.0 |       2.4% |
| **Total**                       | **2,017.5** | **100%** |

Reproduce: `python3 -m tests.benchmarks.clause_memory_breakdown --synthetic -n 1000`

### Missing ~3.8 KB/clause: auxiliary search state

Measured RSS/clause in REQ-P002 is 5.8–6.0 KB. The Clause object graph above
accounts for ~2.0 KB. The remaining ~3.8 KB/clause is per-clause auxiliary
state held outside the Clause itself:

- **Discrimination tree entries** (`pyladr/indexing/discrimination_tree.py`)
  for unification, paramodulation, and demodulation. Each node is an
  `@dataclass` with a `list` of children plus a `dict` children map —
  roughly 15–30 node allocations per inserted term path.
- **FPA / feature index entries** (`pyladr/indexing/feature_index.py`).
- **Back-subsumption index entries**.
- **Demodulator index entries**.
- **Clist-equivalent container bookkeeping** (`usable`, `sos`, `process_set`).
- **Python heap/GC bookkeeping** amortized across every kept reference.

## C Prover9 baseline (from headers)

| C struct   | Fields                                                                                   | Approx bytes |
|------------|------------------------------------------------------------------------------------------|-------------:|
| `term`     | int sym, uchar arity, uchar flags, Term* args, void* container, union(uint id, void* vp) |         ~32  |
| `literals` | BOOL sign, Term atom, Literals next                                                      |         ~24  |
| `topform`  | int id, 6 pointer fields, double weight, char* compressed, Topform hint, Literals, Formula, int semantics, 7 bit-flags | ~96 |

C's term nodes are **~32 bytes** (fixed) vs PyLADR's **Term=80 + args tuple=56 + GC/alignment overhead ≈ 112–128 bytes** — a **3.5–4×** per-node inflation.

## Top 3 gap contributors

### 1. Term object overhead (~1.1 KB/clause graph + ~2× multiplier in index)

- C `struct term`: **~32 B** (no separate args container — pointer array is inline).
- PyLADR Term: **80 B slotted instance + 56 B args tuple + 32 B GC/alignment = ~112–128 B** per node.
- Plus index structures which themselves reference these Terms many times: a single 8-node term may occupy 40+ positions across four index trees.
- **Architecturally constrained:** Python object model forces separate tuple container for variadic args; each heap object incurs GC head + malloc overhead.
- **Addressable bits:**
  - Drop unused `container` field on Term (8 B × ~8 terms/clause = ~64 B/clause).
  - Drop or externalize `term_id` (rarely used per-clause; set only for FPA-indexed terms).
  - Intern rigid constants like variables are interned today (saves repeated
    `Term(private_symbol=-5)` for common constants like `e`, `0`, `1`).
  - Est. reduction: **~150–250 B/clause** without breaking the Term API.

### 2. Index structure overhead (~2.5 KB/clause — largest single item)

- PyLADR discrimination tree: `@dataclass` nodes with `list[DiscrimNode]` +
  `dict[(int,int), DiscrimNode]` children maps. Each node costs ~300+ B.
- C: `discrim_node` is a compact struct with linked-list children — ~40 B/node.
- Four parallel indexes × 15–30 nodes × ~300 B ≈ 3–5 KB per inserted term
  path, shared across clauses but with net contribution in the several-KB range.
- **Architecturally addressable but expensive to refactor.** Largest single
  reduction opportunity (est. 1–2 KB/clause) but would require rewriting
  `pyladr/indexing/` to use array/struct-of-arrays layouts. Recommend
  deferring unless memory becomes a blocking constraint.

### 3. Justification verbosity (~0.6 KB/clause)

- PyLADR: `Justification` frozen dataclass (80 B) + optional `ParaJust` (80 B) + inner tuples for `clause_ids`, `demod_steps`, `from_pos`, `into_pos`.
- C: `struct just` packs an enum tag + union payload; no inner sub-allocations for small tuples.
- **Addressable:**
  - Elide `given_selection` empty-string default (48 B/clause saved if
    stored as `None` sentinel when unset).
  - Store `demod_steps` as a single flat `array.array('i', ...)` instead of
    tuple-of-tuple-of-int (saves ~100 B/step).
  - Est. reduction: **~80–150 B/clause**.

## Reduction potential summary

| Change                                            | Est. savings | Difficulty | Risk |
|---------------------------------------------------|-------------:|-----------:|-----:|
| Drop `Term.container` field (unused post-insert)  | ~64 B/clause | Low        | Low  |
| Intern common rigid constants (like variables)    | ~100 B/clause| Low        | Low  |
| `None` sentinel for empty `given_selection`       | ~48 B/clause | Low        | Low  |
| Flatten `demod_steps` into `array.array`          | ~80 B/clause | Medium     | Low  |
| Compact discrimination-tree node layout           | ~1–2 KB/clause | High     | Med  |
| Columnar clause storage (SoA)                     | ~2–3 KB/clause | Very High| High |

**Near-term achievable without architectural change:** ~300 B/clause
(5.8 KB → ~5.5 KB/clause, ~5% reduction). Already well within REQ-P002's
8 KB/clause ceiling — **not recommended as cycle 6 priority unless another
driver surfaces.**

**Reaching C parity (1.6 KB/clause)** would require replacing Python
object-graph storage with a columnar or C-extension-backed representation —
that's a quarter-scale rewrite, not a cycle improvement.

## Recommendation

1. **Defer major rework.** REQ-P002 passes with 30% headroom; the 3.6× gap
   reflects the Python object model, not a bug or regression.
2. **Flag three trivial wins** for opportunistic pickup:
   (a) `given_selection: str | None` default;
   (b) drop unused `Term.container` (grep for live uses first);
   (c) rigid-constant interning for the top-N most common constants.
3. **Document the gap** in `REQUIREMENTS.md` as "known architectural
   characteristic — not blocking" (already done under REQ-P002 findings).

## Reproducibility

- Clause graph breakdown: `python3 -m tests.benchmarks.clause_memory_breakdown --synthetic -n 1000`
- RSS boundedness: `python3 -m tests.benchmarks.rss_boundedness tests/fixtures/inputs/<problem>.in`
- C struct layouts: `reference-prover9/ladr/{topform,literals,term}.h`
