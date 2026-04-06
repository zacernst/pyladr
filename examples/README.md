# PyLADR Examples

This directory contains examples demonstrating the capabilities of PyLADR, the Python implementation of Prover9/LADR automated theorem proving system.

## Directory Structure

### `prover9/` - Theorem Proving Examples

Examples for the main theorem prover (`pyprover9`):

- **`x2.in`** - Group theory: Proving commutativity from the axiom that every element is its own inverse
- **`x2.out`** - Expected proof output for x2.in
- **`x2.hints`** - Hint clauses to guide the search

**Usage:**
```bash
# Run the theorem prover on group theory example
uv run pyprover9 -f examples/prover9/x2.in

# Compare with expected output
uv run pyprover9 -f examples/prover9/x2.in > result.out
diff result.out examples/prover9/x2.out
```

### `mace4/` - Finite Model Finding Examples

Examples for the Mace4 finite model finder:

- **`group2.in`** - Finding a model of a noncommutative group
- **`rw1.in`** - Rewriting system example

**Usage:**
```bash
# Find finite models using Mace4
uv run pymace4 -f examples/mace4/group2.in

# The Python implementation integrates Mace4 into the main prover
uv run pyprover9 -f examples/mace4/group2.in --mace4
```

### `apps/` - Auxiliary Applications Examples

Examples for the 25+ auxiliary tools:

- **Lattice Examples:**
  - `lattice-sax` - Lattice axioms and properties
  - `OL.in` - Orthomodular lattice example
  - `lattice.rules` - Standard lattice rewrite rules

- **Boolean Algebra:**
  - `BA-sheffer` - Boolean algebra using Sheffer stroke
  - `distributivity` - Distributivity laws

- **Specialized Examples:**
  - `qg.in` - Quasigroup theory
  - `MOL-cand.296` - Modular ortholattice candidate
  - `interp.OL6` - Interpretation for 6-element ortholattice

**Usage Examples:**
```bash
# Variable renaming
uv run pyrenamer -f examples/apps/lattice-sax

# Clause filtering
uv run pyclausefilter -f examples/apps/BA-sheffer

# Interpretation formatting
uv run pyinterpformat -f examples/apps/interp.OL6

# Attack (substructure search)
uv run pyattack -f examples/apps/qg.in

# See full list of tools
uv run pyprover9 --help
```

## Example Categories

### 1. Group Theory
- **Commutativity proofs** from various axiom sets
- **Noncommutative group models** using Mace4
- **Identity and inverse elements**

### 2. Lattice Theory
- **Orthomodular lattices** and quantum logic
- **Boolean algebra** representations
- **Absorption and distributivity** laws

### 3. Logic and Foundations
- **Sheffer stroke** completeness
- **Quasigroup** properties
- **Rewriting systems** and term orderings

### 4. Auxiliary Tool Workflows
- **Proof transformation** and analysis
- **Model filtering** and isomorphism checking
- **Variable renaming** and clause manipulation

## Running Examples

### Basic Theorem Proving
```bash
# Simple proof search
uv run pyprover9 -f examples/prover9/x2.in

# With limits and options
uv run pyprover9 -f examples/apps/lattice-sax -max_given 100 --paramodulation

# Quiet mode (results only)
uv run pyprover9 -f examples/prover9/x2.in --quiet
```

### Model Finding
```bash
# Find finite models up to size 8
uv run pymace4 -f examples/mace4/group2.in -N 8

# Combined proof search and model finding
uv run pyprover9-mace4 -f examples/mace4/group2.in
```

### Batch Processing
```bash
# Process multiple examples
for file in examples/prover9/*.in; do
    echo "Processing $file..."
    uv run pyprover9 -f "$file" --quiet
done
```

## Creating Your Own Examples

### Input Format
PyLADR uses standard LADR syntax:

```
% Comments start with %

formulas(sos).
  % Set of Support - initial clauses for search
  f(x,e) = x.        % Right identity
  f(e,x) = x.        % Left identity
  f(f(x,y),z) = f(x,f(y,z)).  % Associativity
end_of_list.

formulas(goals).
  % Goals to prove (automatically negated)
  f(x,y) = f(y,x).   % Commutativity
end_of_list.

formulas(usable).
  % Additional clauses available for inference
end_of_list.
```

### Best Practices
1. **Start simple** - Test basic cases before complex problems
2. **Use limits** - Set `-max_given` or `-max_seconds` for experimental runs
3. **Enable relevant rules** - Use `--paramodulation` for equality-heavy problems
4. **Check output** - Compare with expected results using `diff`
5. **Profile performance** - Monitor statistics for optimization opportunities

## Compatibility

All examples are compatible with both:
- **Original C Prover9/Mace4** - Syntax and semantics preserved
- **PyLADR Python implementation** - Identical behavior and output format

Cross-validation tests ensure behavioral equivalence between implementations.

## More Examples

For additional examples and documentation, see:
- [Original Prover9 website examples](http://www.cs.unm.edu/~mccune/prover9/)
- The `tests/` directory for unit and integration test cases
- [TPTP Problem Library](http://www.tptp.org/) for standard theorem proving benchmarks