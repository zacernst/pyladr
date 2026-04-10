"""Performance baseline database for regression detection.

Captures Python search engine performance metrics across benchmark problems.
Baselines are stored as frozen dataclasses and can be regenerated via:

    python -m tests.benchmarks.perf_baselines --regenerate

Each baseline records:
  - Wall-clock time (median of multiple runs)
  - Search statistics (given, generated, kept)
  - Per-phase profiling breakdown
  - Microbenchmark throughput floors

These baselines gate CI: if a commit regresses any metric beyond the allowed
tolerance, the test suite fails with a clear diagnostic.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "inputs"
BASELINES_FILE = Path(__file__).resolve().parent / "results" / "python_baselines.json"


@dataclass(frozen=True, slots=True)
class PythonBaseline:
    """Captured Python performance baseline for a single problem."""

    problem: str
    proved: bool
    wall_seconds_median: float
    wall_seconds_p95: float
    given: int
    generated: int
    kept: int
    subsumed: int
    back_subsumed: int
    # Phase timings (fraction of total)
    phase_fractions: dict[str, float] = field(default_factory=dict)
    # Timestamp of capture
    captured: str = ""


@dataclass(frozen=True, slots=True)
class MicrobenchmarkBaseline:
    """Captured throughput baseline for a microbenchmark."""

    name: str
    ops_per_second: float
    minimum_threshold: float  # Regression fails below this


@dataclass
class BaselineDatabase:
    """Collection of all performance baselines."""

    python_version: str = ""
    platform: str = ""
    captured_at: str = ""
    problem_baselines: dict[str, PythonBaseline] = field(default_factory=dict)
    microbenchmarks: dict[str, MicrobenchmarkBaseline] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        """Save baselines to JSON."""
        path = path or BASELINES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "python_version": self.python_version,
            "platform": self.platform,
            "captured_at": self.captured_at,
            "problem_baselines": {
                k: asdict(v) for k, v in self.problem_baselines.items()
            },
            "microbenchmarks": {
                k: asdict(v) for k, v in self.microbenchmarks.items()
            },
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> BaselineDatabase:
        """Load baselines from JSON. Returns empty database if file missing."""
        path = path or BASELINES_FILE
        if not path.exists():
            return cls()

        raw = json.loads(path.read_text())
        db = cls(
            python_version=raw.get("python_version", ""),
            platform=raw.get("platform", ""),
            captured_at=raw.get("captured_at", ""),
        )

        for name, d in raw.get("problem_baselines", {}).items():
            db.problem_baselines[name] = PythonBaseline(**d)

        for name, d in raw.get("microbenchmarks", {}).items():
            db.microbenchmarks[name] = MicrobenchmarkBaseline(**d)

        return db


def capture_problem_baseline(
    input_path: Path,
    *,
    runs: int = 5,
    timeout: float = 120.0,
) -> PythonBaseline | None:
    """Run a problem multiple times and capture the baseline."""
    import statistics as stats

    from tests.benchmarks.perf_profiler import profile_problem, run_problem

    name = input_path.stem
    wall_times: list[float] = []
    last_result = None
    last_profile = None

    for i in range(runs):
        try:
            t0 = time.perf_counter()
            result, profile = profile_problem(
                input_path, timeout=timeout, quiet=True,
            )
            elapsed = time.perf_counter() - t0
            wall_times.append(elapsed)
            last_result = result
            last_profile = profile
        except Exception:
            wall_times.append(float("inf"))

    if last_result is None or not wall_times:
        return None

    # Filter out infinity (failed runs)
    valid_times = [t for t in wall_times if t != float("inf")]
    if not valid_times:
        return None

    from pyladr.search.given_clause import ExitCode

    proved = last_result.exit_code == ExitCode.MAX_PROOFS_EXIT

    median_time = stats.median(valid_times)
    sorted_times = sorted(valid_times)
    p95_idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
    p95_time = sorted_times[p95_idx]

    # Extract phase fractions from the last profile
    phase_fracs: dict[str, float] = {}
    if last_profile and last_profile.total_wall_seconds > 0:
        for bname, bucket in last_profile.buckets.items():
            phase_fracs[bname] = bucket.total_seconds / last_profile.total_wall_seconds

    import datetime

    return PythonBaseline(
        problem=name,
        proved=proved,
        wall_seconds_median=round(median_time, 4),
        wall_seconds_p95=round(p95_time, 4),
        given=last_result.stats.given,
        generated=last_result.stats.generated,
        kept=last_result.stats.kept,
        subsumed=last_result.stats.subsumed,
        back_subsumed=last_result.stats.back_subsumed,
        phase_fractions=phase_fracs,
        captured=datetime.datetime.now(datetime.UTC).isoformat(),
    )


def capture_all_baselines(
    *,
    runs: int = 5,
    timeout: float = 120.0,
) -> BaselineDatabase:
    """Capture baselines for all benchmark problems."""
    import datetime
    import platform
    import sys

    db = BaselineDatabase(
        python_version=sys.version,
        platform=platform.platform(),
        captured_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    problems = sorted(INPUTS_DIR.glob("bench_*.in"))
    for input_path in problems:
        name = input_path.stem
        print(f"  Capturing baseline: {name}...", flush=True)
        bl = capture_problem_baseline(input_path, runs=runs, timeout=timeout)
        if bl is not None:
            db.problem_baselines[name] = bl
            print(
                f"    {'PROVED' if bl.proved else 'FAILED'} "
                f"median={bl.wall_seconds_median:.3f}s "
                f"given={bl.given} kept={bl.kept}"
            )
        else:
            print(f"    SKIPPED (could not run)")

    return db


def main() -> None:
    """CLI: regenerate baselines."""
    import argparse

    parser = argparse.ArgumentParser(description="Capture Python performance baselines")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Capturing Python performance baselines...")
    db = capture_all_baselines(runs=args.runs, timeout=args.timeout)

    output = Path(args.output) if args.output else None
    path = db.save(output)
    print(f"\nBaselines saved to: {path}")
    print(f"Problems captured: {len(db.problem_baselines)}")


if __name__ == "__main__":
    main()
