# PyLADR Troubleshooting Guide

## Common Issues

### Search Exits Without Finding a Proof

**SOS empty (exit code 2):**
The search exhausted all clauses without finding a proof. This means either:
- The theorem is not provable from the given axioms
- The inference rules enabled are insufficient (e.g., equational problem without `--paramodulation`)
- Important clauses were discarded by weight limits

**Actions:**
```bash
# Enable more inference rules
uv run pyprover9 -f problem.in --paramodulation --demodulation

# Increase weight limit
# (in input file: set(max_weight, 50).)

# Check if the problem needs Mace4 instead (find a counterexample)
uv run pyprover9-mace4 -f problem.in
```

**Limit reached (exit codes 3–6):**
```bash
# Increase limits
uv run pyprover9 -f problem.in -max_given 5000 -max_seconds 120 -max_generated 100000
```

### Parse Errors

**`ParseError: unexpected token`:**
- Check LADR syntax: each formula ends with a period (`.`)
- Lists end with `end_of_list.`
- Variables start with `u`–`z`; constants/functions start with `a`–`t` or are numeric
- Comments start with `%`

**Example of correct syntax:**
```
formulas(sos).
  f(x,y) = f(y,x).       % Note the trailing period
  (x * y) * z = x * (y * z).
end_of_list.              % Note: period after end_of_list
```

**Common mistakes:**
```
% WRONG: missing period
formulas(sos).
  f(x,y) = f(y,x)
end_of_list.

% WRONG: missing end_of_list
formulas(sos).
  f(x,y) = f(y,x).

% WRONG: variable naming (A is a constant, not variable)
formulas(sos).
  f(A,B) = f(B,A).       % A and B are constants!
end_of_list.

% CORRECT: use lowercase u-z for variables
formulas(sos).
  f(x,y) = f(y,x).       % x and y are variables
end_of_list.
```

### Slow Performance

**Problem takes too long:**

1. **Set resource limits** to prevent runaway searches:
   ```bash
   uv run pyprover9 -f problem.in -max_given 500 -max_seconds 30
   ```

2. **Enable demodulation** for equational problems (reduces clause count):
   ```bash
   uv run pyprover9 -f problem.in --paramodulation --demodulation --back-demod
   ```

3. **Use quiet mode** to reduce I/O overhead:
   ```bash
   uv run pyprover9 -f problem.in --quiet
   ```

4. **Check problem size** — very large axiom sets produce exponential clause growth. Consider whether all axioms are needed.

### Import Errors

**`ModuleNotFoundError: No module named 'pyladr'`:**
```bash
# Ensure you're in the project directory and have installed
cd /path/to/pyladr
uv sync

# Run through uv
uv run pyprover9 -f problem.in

# Or install in development mode
pip install -e .
```

**`ModuleNotFoundError: No module named 'pydantic'`:**
```bash
uv sync  # Installs all dependencies
```

### Cross-Validation Test Failures

**`FileNotFoundError: reference-prover9/bin/prover9`:**
The C reference binary is not built. Cross-validation tests require the C Prover9 binary:

```bash
cd reference-prover9
make all
```

**Different proof found (but both valid):**
This is expected in some cases. When clauses have identical weight, selection order may differ between Python and C. The proofs are different but both valid. Cross-validation tests verify proof existence, not exact proof identity.

### Free-Threading Issues

**Parallel mode not activating:**
```python
from pyladr.threading_guide import FREE_THREADING_AVAILABLE
print(f"Free-threading available: {FREE_THREADING_AVAILABLE}")
```

If `False`, you're running standard GIL Python. Free-threading requires:
- Python 3.14+ built with `--disable-gil`
- Or the experimental free-threaded builds from python.org

The parallel engine automatically falls back to sequential execution on GIL Python with no performance penalty.

## Error Messages

| Message | Cause | Fix |
|---------|-------|-----|
| `Arity N does not match number of args M` | Term created with wrong number of arguments | Check term construction |
| `Arity N exceeds MAX_ARITY (255)` | Function with too many arguments | Simplify encoding |
| `ParseError at position N` | Syntax error in input | Check LADR syntax at indicated position |
| `FATAL_EXIT` | Internal error during search | Report as a bug with the input file |

## Getting Help

- Check existing documentation in `docs/`
- Look at example problems in `examples/`
- For bugs, include the input file and full error traceback
- For C compatibility issues, include output from both `pyprover9` and `reference-prover9/bin/prover9`
