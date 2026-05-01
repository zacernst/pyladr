"""Proper timing benchmark for PyLADR."""
import sys
import time

problem = sys.argv[1] if len(sys.argv) > 1 else 'perf_benchmark.in'
sys.argv = ['pyprover9', '-f', problem, '-q']

from pyladr.cli import main
start = time.perf_counter()
try:
    main()
except SystemExit:
    pass
elapsed = time.perf_counter() - start
# Print timing to stderr so it doesn't mix with prover output
print(f'{elapsed:.4f}', file=sys.stderr)
