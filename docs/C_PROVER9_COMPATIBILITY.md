# C Prover9 Compatibility Guide

PyLADR is a faithful Python reimplementation of William McCune's Prover9/LADR system. This document details the compatibility guarantees, differences, and cross-validation infrastructure.

## Compatibility Goals

PyLADR aims for **behavioral equivalence** with C Prover9:

1. **Identical input format** — Standard LADR syntax, same operator table
2. **Identical inference rules** — Resolution, paramodulation, demodulation, subsumption
3. **Matching exit codes** — Same numeric codes for proof found, SOS empty, limits, etc.
4. **Compatible output format** — Proof output, statistics, and clause formatting
5. **Cross-validated** — Automated tests compare Python and C on the same problems

## Data Structure Correspondence

| C LADR | PyLADR | Notes |
|--------|--------|-------|
| `struct term` | `Term` (frozen dataclass) | Immutable; C uses mutable linked structure |
| `struct literals` | `Literal` (frozen dataclass) | C uses linked list; Python uses tuple |
| `struct topform` | `Clause` (dataclass) | Mutable metadata (id, weight); immutable core |
| `struct context` | `Context` | Variable binding during unification |
| `struct trail` | `Trail` | Binding backtrack record |
| `struct just` | `Justification` (frozen) | Proof justification chain |
| `struct mindex` | `Mindex` | Unified index interface |
| `struct discrim` | `DiscrimWild` / `DiscrimBind` | Discrimination tree variants |
| `struct symbol_data` | `SymbolTable` | Symbol metadata and name mapping |

### Key Encoding Differences

**Term symbol encoding** — Both use the same scheme:
- `private_symbol >= 0` → variable (value = variable number)
- `private_symbol < 0` → rigid symbol (`SYMNUM = -private_symbol`)

**Literals** — C uses a singly-linked list; Python uses an immutable tuple. Order is preserved identically.

**Justifications** — C uses a linked list of `just` structs; Python uses a tuple of `Justification` dataclasses. Same information content.

## Algorithm Correspondence

### Given-Clause Algorithm

The core search loop matches C `search.c`:

```
C: search()              →  Python: GivenClauseSearch.run()
C: get_given_clause2()   →  Python: GivenSelection.select_given()
C: given_infer()         →  Python: GivenClauseSearch._given_infer()
C: cl_process()          →  Python: GivenClauseSearch._cl_process()
C: limbo_process()       →  Python: GivenClauseSearch._limbo_process()
```

### Inference Rules

| C Function | Python Function | Module |
|------------|-----------------|--------|
| `binary_resolution()` | `binary_resolve()` | `inference.resolution` |
| `all_binary_resolvents()` | `all_binary_resolvents()` | `inference.resolution` |
| `hyper_resolution()` | `hyper_resolve()` | `inference.hyper_resolution` |
| `paramodulate()` | `paramodulate()` | `inference.paramodulation` |
| `para_from_into()` | `para_from_into()` | `inference.paramodulation` |
| `orient_equalities()` | `orient_equalities()` | `inference.paramodulation` |
| `demodulate()` | `demodulate_clause()` | `inference.demodulation` |
| `back_demod()` | `back_demodulatable()` | `inference.demodulation` |
| `subsumes()` | `subsumes()` | `inference.subsumption` |
| `forward_subsumption()` | `forward_subsume()` | `inference.subsumption` |
| `back_subsume()` | `back_subsume()` | `inference.subsumption` |

### Parser

| C Function | Python Method | Module |
|------------|---------------|--------|
| `read_term()` / `sread_term()` | `LADRParser.parse_term()` | `parsing.ladr_parser` |
| `declare_standard_parse_types()` | `_STANDARD_OPS` dict | `parsing.ladr_parser` |
| `tokenize()` | `tokenize()` | `parsing.tokenizer` |

### Term Ordering

| C Function | Python Function | Module |
|------------|-----------------|--------|
| `term_greater()` | `term_greater()` | `ordering.termorder` |
| `term_order()` | `term_order()` | `ordering.termorder` |
| `kbo()` | `kbo()` | `ordering.kbo` |
| `lrpo()` | `lrpo()` | `ordering.lrpo` |

## Exit Code Mapping

PyLADR uses identical exit codes to C Prover9:

| Process Exit | Search Exit Code | Meaning |
|:---:|---|---|
| 0 | `MAX_PROOFS_EXIT` (1) | Proof found |
| 1 | `FATAL_EXIT` (7) | Fatal error |
| 2 | `SOS_EMPTY_EXIT` (2) | Search exhausted, no proof |
| 3 | `MAX_GIVEN_EXIT` (3) | Given clause limit reached |
| 4 | `MAX_KEPT_EXIT` (4) | Kept clause limit reached |
| 5 | `MAX_SECONDS_EXIT` (5) | Time limit reached |
| 6 | `MAX_GENERATED_EXIT` (6) | Generated clause limit reached |

## Cross-Validation Infrastructure

### Running Cross-Validation Tests

```bash
# Run all cross-validation tests (requires C Prover9 binary)
uv run pytest -m cross_validation

# Run specific equivalence tests
uv run pytest tests/cross_validation/test_search_equivalence.py
uv run pytest tests/cross_validation/test_equational_reasoning.py
uv run pytest tests/cross_validation/test_c_vs_python_comprehensive.py
```

### C Runner

The `tests/cross_validation/c_runner.py` module invokes the reference C Prover9 binary and parses its output:

```python
from tests.cross_validation.c_runner import run_c_prover9, ProverResult

result: ProverResult = run_c_prover9("path/to/problem.in")
assert result.theorem_proved
assert result.clauses_given == expected_given
```

**`ProverResult` fields:**
- `theorem_proved: bool`
- `search_failed: bool`
- `clauses_given: int`
- `clauses_generated: int`
- `clauses_kept: int`
- `clauses_deleted: int`
- `proof_length: int`
- `proof_clauses: list`
- `exit_code: int`
- `cpu_time: float`

### Reference Binary Location

The C Prover9 binary is at `reference-prover9/bin/prover9`. The full C source is in `reference-prover9/ladr/` and `reference-prover9/provers.src/`.

## Known Differences

### Intentional Differences (Features Not in C)

1. **Free-threading parallelism** — PyLADR can parallelize inference generation on Python 3.14+
2. **ML-guided search** — Optional machine learning components for clause selection
3. **Pydantic validation** — Runtime type checking on data structures
4. **Immutable terms** — Python terms are frozen dataclasses (C terms are mutable)

### Behavioral Notes

- **Clause ordering within proofs** — When multiple clauses have identical weight, selection order may differ due to Python dict ordering vs. C pointer ordering. This can produce different but equivalent proofs.
- **Floating-point timing** — `max_seconds` comparisons use Python `time.time()` vs. C `clock()`. Timing-dependent cutoffs may trigger at slightly different points.
- **Memory management** — C uses explicit memory pools; Python uses garbage collection. Memory-related limits (`sos_limit`) may behave differently at the margins.

## Using C Prover9 for Comparison

To validate a PyLADR result against C Prover9:

```bash
# Run both implementations on the same input
uv run pyprover9 -f problem.in > python_output.txt
./reference-prover9/bin/prover9 -f problem.in > c_output.txt

# Compare results (proof found/not found should match)
diff <(grep "THEOREM PROVED" python_output.txt) <(grep "THEOREM PROVED" c_output.txt)
```

## Auxiliary Tool Correspondence

Each C LADR auxiliary tool has a Python equivalent:

| C Tool | Python Command | Source |
|--------|---------------|--------|
| `renamer` | `pyrenamer` | `pyladr.apps.renamer` |
| `mirror-flip` | `pymirror-flip` | `pyladr.apps.mirror_flip` |
| `prooftrans` | `pyprooftrans` | `pyladr.apps.prooftrans` |
| `isofilter` | `pyisofilter` | `pyladr.apps.isofilter` |
| `isofilter0` | `pyisofilter0` | `pyladr.apps.isofilter0` |
| `isofilter2` | `pyisofilter2` | `pyladr.apps.isofilter2` |
| `clausefilter` | `pyclausefilter` | `pyladr.apps.clausefilter` |
| `clausetester` | `pyclausetester` | `pyladr.apps.clausetester` |
| `interpformat` | `pyinterpformat` | `pyladr.apps.interpformat` |
| `interpfilter` | `pyinterpfilter` | `pyladr.apps.interpfilter` |
| `attack` | `pyattack` | `pyladr.apps.attack` |
| `get_interps` | `pyget-interps` | `pyladr.apps.get_interps` |
| `get_givens` | `pyget-givens` | `pyladr.apps.get_givens` |
| `get_kept` | `pyget-kept` | `pyladr.apps.get_kept` |
| `ladr_to_tptp` | `pyladr-to-tptp` | `pyladr.apps.ladr_to_tptp` |
| `dprofiles` | `pydprofiles` | `pyladr.apps.dprofiles` |
| `sigtest` | `pysigtest` | `pyladr.apps.sigtest` |
| `latfilter` | `pylatfilter` | `pyladr.apps.latfilter` |
| `upper_covers` | `pyupper-covers` | `pyladr.apps.upper_covers` |
| `complex` | `pycomplex` | `pyladr.apps.complex` |
| `rewriter` | `pyrewriter` | `pyladr.apps.rewriter` |
| `rewriter2` | `pyrewriter2` | `pyladr.apps.rewriter2` |
