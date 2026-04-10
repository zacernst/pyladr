# PyLADR Quickstart Guide

This guide gets you proving theorems with PyLADR in minutes.

## Installation

```bash
# Clone and install
git clone <repository-url>
cd pyladr
uv sync

# Verify installation
uv run pyprover9 --version
```

**Requirements:** Python 3.13+ (3.14+ recommended for free-threading parallelism)

## Your First Proof

Create a file `my_first.in`:

```
% Prove: In a group where every element is its own inverse,
% the group is commutative.

formulas(sos).
  e * x = x.           % Left identity
  x' * x = e.          % Left inverse
  (x * y) * z = x * (y * z).  % Associativity
  x * x = e.           % Every element is its own inverse
end_of_list.

formulas(goals).
  x * y = y * x.       % Commutativity (to prove)
end_of_list.
```

Run it:

```bash
uv run pyprover9 -f my_first.in
```

PyLADR will find a proof by refutation: it negates the goal, adds it to the set of support, and searches for a contradiction (the empty clause).

## Understanding the Output

A successful run produces output like:

```
============================== PROOF =================================

% Proof 1 at 0.01 seconds.

1 e * x = x.                      [assumption].
2 x' * x = e.                     [assumption].
3 (x * y) * z = x * (y * z).      [assumption].
4 x * x = e.                      [assumption].
5 a * b != b * a.                  [deny(goal)].
...
N $F.                              [binary,M,K].

============================== end of proof ==========================
```

- Clauses 1-4 are your assumptions
- Clause 5 is the negated goal (refutation method)
- `$F` is the empty clause (contradiction found = theorem proved)
- Each clause shows its justification: `[assumption]`, `[binary,M,K]`, etc.

## Input Format (LADR Syntax)

### Clause Lists

PyLADR reads standard LADR format with these sections:

```
formulas(sos).         % Set of Support — initial clauses for search
  ...
end_of_list.

formulas(goals).       % Goals — automatically negated for refutation
  ...
end_of_list.

formulas(usable).      % Usable — available for inference but not selected as given
  ...
end_of_list.

formulas(assumptions). % Alias for sos
  ...
end_of_list.
```

### Term Syntax

| Syntax | Meaning |
|--------|---------|
| `f(x,y)` | Function application |
| `x * y` | Infix binary operator (same as `*(x,y)`) |
| `x'` | Postfix unary (inverse, complement) |
| `-x` | Prefix unary (negation) |
| `x = y` | Equality |
| `x != y` | Inequality |
| `P(x)` | Predicate application |
| `$T` / `$F` | True / False |

**Variables** start with `u`-`z` (e.g., `x`, `y`, `z`, `u1`, `v2`).
**Constants and functions** start with `a`-`t` or are numeric (e.g., `a`, `f`, `g`, `0`, `e`).

### Comments

```
% This is a line comment (everything after % is ignored)
```

## Common Use Cases

### Equational Reasoning (Paramodulation)

For problems heavy on equalities, enable paramodulation and demodulation:

```bash
uv run pyprover9 -f problem.in --paramodulation --demodulation
```

### Setting Search Limits

```bash
# Stop after 100 given clauses
uv run pyprover9 -f problem.in -max_given 100

# Stop after 30 seconds
uv run pyprover9 -f problem.in -max_seconds 30

# Stop after 10000 generated clauses
uv run pyprover9 -f problem.in -max_generated 10000
```

### Quiet Mode

```bash
# Suppress search progress, show only the proof (or failure)
uv run pyprover9 -f problem.in --quiet
```

### Finite Model Finding (Mace4)

Search for a finite model (counterexample) instead of a proof:

```bash
# Find models of size 2 through 10
uv run pyprover9-mace4 -f problem.in
```

## Example Problems

PyLADR ships with example problems in `examples/`:

```bash
# Group theory: x*x=e implies commutativity
uv run pyprover9 -f examples/prover9/x2.in

# Lattice theory
uv run pyprover9 -f examples/modular_lattice.in --paramodulation --demodulation

# Robbins conjecture (famous result from 1996)
uv run pyprover9 -f examples/robbins_conjecture.in --paramodulation --demodulation -max_seconds 60
```

## Using PyLADR as a Library

PyLADR can be used programmatically from Python:

```python
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

# Parse input
parser = LADRParser()
parsed = parser.parse_input("""
formulas(sos).
  e * x = x.
  x' * x = e.
  (x * y) * z = x * (y * z).
  x * x = e.
end_of_list.

formulas(goals).
  x * y = y * x.
end_of_list.
""")

# Configure search
options = SearchOptions(
    binary_resolution=True,
    max_given=500,
    max_seconds=10.0,
)

# Run search
search = GivenClauseSearch(options=options, symbol_table=parser.symbol_table)

# Prepare clauses: negate goals for refutation
sos = list(parsed.sos) + list(parsed.goals_denied)
result = search.run(usable=list(parsed.usable), sos=sos)

# Check result
if result.proofs:
    print(f"Theorem proved! Proof length: {len(result.proofs[0].clauses)}")
else:
    print(f"Search ended: {result.exit_code.name}")
print(f"Statistics: {result.stats.given} given, {result.stats.kept} kept")
```

## Exit Codes

PyLADR uses the same exit codes as C Prover9:

| Code | Meaning |
|------|---------|
| 0 | Proof found (`MAX_PROOFS_EXIT`) |
| 1 | Fatal error |
| 2 | SOS empty — search exhausted, no proof |
| 3 | `max_given` limit reached |
| 4 | `max_kept` limit reached |
| 5 | `max_seconds` limit reached |
| 6 | `max_generated` limit reached |

## Next Steps

- **[API Reference](API_REFERENCE.md)** — Full module and class documentation
- **[C Prover9 Compatibility](C_PROVER9_COMPATIBILITY.md)** — Behavioral equivalence details
- **[Examples README](../examples/README.md)** — Walkthrough of included examples
- **[Performance Guide](PERFORMANCE.md)** — Optimization and parallelism
