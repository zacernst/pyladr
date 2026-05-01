"""Performance profiling script for PyLADR search engine."""
import cProfile
import pstats
import sys
import io
import time

def profile_problem(input_file, max_given=200):
    """Profile a single problem."""
    sys.argv = ['pyprover9', '-f', input_file, '-max_given', str(max_given), '-q']

    # Reset module state for clean run
    import importlib

    pr = cProfile.Profile()
    pr.enable()

    start = time.perf_counter()
    from pyladr.cli import main
    try:
        main()
    except SystemExit:
        pass
    elapsed = time.perf_counter() - start

    pr.disable()

    print(f"\n{'='*80}")
    print(f"Problem: {input_file}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"{'='*80}")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(60)
    print(s.getvalue())

    # Also show by tottime
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(40)
    print("\n--- Sorted by tottime ---")
    print(s2.getvalue())

if __name__ == '__main__':
    problem = sys.argv[1] if len(sys.argv) > 1 else 'tests/fixtures/inputs/bench_robbins.in'
    max_g = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    profile_problem(problem, max_g)
