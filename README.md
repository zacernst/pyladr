# PyLADR - Python Implementation of Prover9/LADR

A complete Python 3.14+ free-threaded implementation of the Prover9/LADR automated theorem proving system.

## Overview

PyLADR provides a faithful Python implementation of the McCune Prover9 theorem prover and LADR (Logic and Data Relations) library. It supports first-order logic with equality and includes advanced automated reasoning capabilities.

## Features

### Core Theorem Proving
- **Given-clause algorithm** for systematic proof search
- **Binary resolution** with factoring
- **Paramodulation** for equational reasoning
- **Demodulation** for equation-based simplification
- **Advanced subsumption** with literal matching

### Equational Reasoning
- **AC unification** using Huet's algorithm with diophantine equation solving
- **AC canonicalization** for associative-commutative operators
- **Term ordering** (KBO, LRPO) for equation orientation
- **Discrimination trees** for efficient term indexing

### Finite Model Finding
- **Mace4 implementation** for finding finite counter-models
- **Interpretation evaluation** with complete domain operations
- **Model filtering and analysis** tools

### Modern Architecture
- **Python 3.14 free-threading** support for true parallel inference
- **Thread-safe design** with lock-free algorithms where possible
- **Pydantic validation** for robust data structures
- **Type-safe implementation** with comprehensive mypy coverage

### Auxiliary Tools
25+ utility applications matching the original C tools:
- `pyrenamer` - Variable renaming
- `pymirror-flip` - Lattice duality operations
- `pyprooftrans` - Proof transformation
- `pyattack` - Substructure search
- `pyisofilter` - Isomorphism filtering
- And many more...

## Installation

### Requirements
- Python 3.13+ (Python 3.14+ recommended for free-threading)
- [uv](https://docs.astral.sh/uv/) package manager

### Install from Source
```bash
# Clone the repository
git clone <repository-url>
cd pyladr

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Usage

### Basic Theorem Proving
```bash
# Run theorem prover on input file
uv run pyprover9 -f problem.in

# With search limits
uv run pyprover9 -f problem.in -max_given 1000 -max_seconds 30

# Enable specific inference rules
uv run pyprover9 -f problem.in --paramodulation --demodulation
```

### Input Format
PyLADR uses standard LADR syntax:
```
formulas(sos).
  % Initial clauses (Set of Support)
  f(x,e) = x.
  f(e,x) = x.
  f(f(x,y),z) = f(x,f(y,z)).
end_of_list.

formulas(goals).
  % Goals to prove (will be negated)
  f(a,a) = a -> f(b,b) = b.
end_of_list.
```

### Auxiliary Tools
```bash
# Variable renaming
uv run pyrenamer -f input.out

# Proof transformation
uv run pyprooftrans -f proof.out

# Get all available tools
uv run pyprover9 --help
```

## Development

### Setup Development Environment
```bash
# Install with dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run type checking
uv run mypy pyladr

# Run linting
uv run ruff check pyladr
```

### Testing
Comprehensive test suite with 900+ tests:
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m integration             # Integration tests only
uv run pytest -m cross_validation        # C/Python equivalence tests
uv run pytest tests/unit/                # Unit tests only
```

## Architecture

### Package Structure
```
pyladr/
├── core/           # Core data structures (Term, Clause, Literal)
├── inference/      # Inference rules (resolution, paramodulation, etc.)
├── search/         # Search algorithms (given-clause, selection)
├── indexing/       # Term indexing (discrimination trees)
├── ordering/       # Term orderings (KBO, LRPO)
├── parsing/        # LADR format parser and tokenizer
├── parallel/       # Parallel inference engine
├── mace4/          # Finite model finder
└── apps/           # Auxiliary applications
```

### Thread Safety
PyLADR is designed for Python 3.14's free-threading mode:
- **Immutable data structures** where possible
- **Lock-free algorithms** for read-heavy operations
- **Thread-local context** for inference workers
- **Sequential bottlenecks** only where necessary (indexing)

## Performance

- **Parallel inference generation** using thread pools
- **Efficient indexing** with discrimination trees
- **Optimized unification** with occur checks
- **Memory-conscious** clause management

Typical performance: 1000-10000 inferences/second depending on problem complexity.

## Compatibility

PyLADR maintains behavioral equivalence with C Prover9:
- **Identical inference rules** and search strategies
- **Compatible input/output formats**
- **Matching exit codes** and error messages
- **Cross-validated** against C implementation on standard problems

## License

GNU General Public License v2.0 - same as original Prover9/LADR.

## Acknowledgments

Based on the Prover9/LADR system by William McCune and colleagues. This Python implementation preserves the mathematical rigor and completeness of the original while adding modern language features and parallel processing capabilities.