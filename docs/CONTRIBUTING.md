# Contributing to PyLADR

## Development Setup

```bash
git clone <repository-url>
cd pyladr

# Install with dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Code Quality Tools

```bash
# Run all tests
uv run pytest

# Type checking (strict mode)
uv run mypy pyladr

# Linting
uv run ruff check pyladr

# Formatting
uv run black pyladr tests
```

### Test Categories

```bash
uv run pytest tests/unit/                  # Unit tests
uv run pytest -m integration               # Integration tests
uv run pytest -m cross_validation          # C/Python equivalence
uv run pytest -m property                  # Property-based (Hypothesis)
uv run pytest -m benchmark                 # Performance benchmarks
uv run pytest -m "not slow"               # Skip slow tests
```

### Coverage

```bash
uv run pytest --cov=pyladr --cov-report=term-missing
# Minimum 80% coverage enforced
```

## Project Architecture

### Module Responsibilities

| Module | Responsibility | C Equivalent |
|--------|---------------|--------------|
| `core/` | Terms, clauses, symbols, unification | `ladr/term.c`, `ladr/topform.c`, `ladr/unify.c` |
| `inference/` | Resolution, paramodulation, demodulation, subsumption | `ladr/resolve.c`, `ladr/paramod.c`, `ladr/demod.c` |
| `search/` | Given-clause algorithm, selection, state | `provers.src/search.c`, `provers.src/giv_select.c` |
| `parsing/` | LADR format parser | `ladr/parse.c`, `ladr/tokenize.c` |
| `ordering/` | KBO, LRPO term orderings | `ladr/termorder.c`, `ladr/kbo.c`, `ladr/lrpo.c` |
| `indexing/` | Discrimination trees, feature indexing | `ladr/mindex.c`, `ladr/discrim.c` |
| `mace4/` | Finite model finder | `mace4.src/msearch.c` |
| `parallel/` | Thread pool inference generation | (no C equivalent) |
| `ml/` | Machine learning search guidance | (no C equivalent) |
| `apps/` | CLI tools | `apps.src/` |

### Design Principles

1. **Faithfulness to C** — Algorithms match the C implementation. Function and variable names reference C equivalents in docstrings.

2. **Immutability where possible** — `Term`, `Literal`, `Justification` are frozen dataclasses. This enables safe sharing across threads.

3. **Type safety** — Strict mypy checking. All public functions have type annotations.

4. **Minimal dependencies** — Only Pydantic for validation. No heavy frameworks.

### Adding a New Inference Rule

1. Create `pyladr/inference/my_rule.py`
2. Implement the rule function with signature matching existing rules
3. Add to `pyladr/inference/__init__.py` exports
4. Integrate into `GivenClauseSearch._given_infer()` in `search/given_clause.py`
5. Add a `SearchOptions` flag to enable/disable it
6. Write unit tests in `tests/unit/test_my_rule.py`
7. Add cross-validation test if C equivalent exists

### Adding a New CLI Tool

1. Create `pyladr/apps/my_tool.py` with a `main()` function
2. Add entry point in `pyproject.toml` under `[project.scripts]`
3. Write integration test in `tests/integration/`

## Coding Conventions

- **Line length:** 100 characters (enforced by Ruff)
- **Docstrings:** Reference the C function being replicated
- **Type annotations:** Required on all public functions
- **Naming:** Match C function names where reasonable (e.g., `binary_resolve` not `resolve_binary`)
- **Slots:** Use `__slots__` or `slots=True` on dataclasses for performance
- **Frozen dataclasses:** Prefer for value types (Terms, Literals, Justifications)

## Cross-Validation Testing

When modifying inference rules or search behavior, always verify against the C reference:

```bash
# Run cross-validation suite
uv run pytest -m cross_validation -v

# Test specific problem
python -c "
from tests.cross_validation.c_runner import run_c_prover9
result = run_c_prover9('examples/prover9/x2.in')
print(f'C: proved={result.theorem_proved}, given={result.clauses_given}')
"
```

The C binary is at `reference-prover9/bin/prover9`. Recompile from `reference-prover9/` if needed.

## Commit Guidelines

- Keep commits focused on a single change
- Reference the C function/module being modified when relevant
- Run `uv run pytest -m "not slow"` before committing
- Pre-commit hooks enforce formatting and linting
