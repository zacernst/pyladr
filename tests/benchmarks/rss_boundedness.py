"""Peak RSS boundedness benchmark (REQ-P002).

Runs prover9 as a subprocess and polls RSS via psutil. Reports initial RSS
(right after imports/parsing), peak RSS during search, and the final Kept=
count parsed from stdout.

Usage:
    python -m tests.benchmarks.rss_boundedness tests/fixtures/inputs/<problem>.in
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class RssResult:
    problem: str
    initial_rss_mb: float
    peak_rss_mb: float
    ratio: float
    kept: int
    given: int
    generated: int
    wall_seconds: float
    exit_code: int
    settle_rss_mb: float  # RSS after parsing, before peak accumulation
    stdout_tail: str

    @property
    def kb_per_kept(self) -> float:
        """Memory growth from settle to peak, divided by kept clauses.

        This is REQ-P002's per-clause memory metric. Returns 0.0 if no
        clauses were kept (search terminated before accumulation).
        """
        if self.kept <= 0:
            return 0.0
        growth_kb = (self.peak_rss_mb - self.settle_rss_mb) * 1024
        return growth_kb / self.kept


def _parse_stats(stdout: str) -> tuple[int, int, int]:
    given = kept = generated = 0
    m = re.search(r"Given=(\d+)\. Generated=(\d+)\. Kept=(\d+)", stdout)
    if m:
        given, generated, kept = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return given, generated, kept


def measure_rss(
    input_path: Path,
    *,
    poll_interval: float = 0.05,
    settle_seconds: float = 0.5,
    timeout: float = 120.0,
) -> RssResult:
    cmd = [sys.executable, "-m", "pyladr.cli", "-f", str(input_path)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    p = psutil.Process(proc.pid)

    initial_rss = 0
    settle_rss = 0
    peak_rss = 0
    settle_captured = False

    try:
        # Retry first valid sample — psutil can race the fork and return 0
        for _ in range(20):
            try:
                rss = p.memory_info().rss
                if rss > 0:
                    initial_rss = rss
                    peak_rss = rss
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)

        while proc.poll() is None:
            elapsed = time.perf_counter() - t0
            if elapsed > timeout:
                proc.kill()
                break
            try:
                rss = p.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
                if not settle_captured and elapsed >= settle_seconds:
                    settle_rss = rss
                    settle_captured = True
            except psutil.NoSuchProcess:
                break
            time.sleep(poll_interval)
    finally:
        try:
            stdout, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()

    wall = time.perf_counter() - t0
    given, generated, kept = _parse_stats(stdout)

    if settle_rss == 0:
        settle_rss = initial_rss

    initial_mb = initial_rss / (1024 * 1024)
    settle_mb = settle_rss / (1024 * 1024)
    peak_mb = peak_rss / (1024 * 1024)
    ratio = peak_mb / settle_mb if settle_mb > 0 else float("inf")

    tail_lines = stdout.splitlines()[-30:]
    return RssResult(
        problem=input_path.name,
        initial_rss_mb=initial_mb,
        settle_rss_mb=settle_mb,
        peak_rss_mb=peak_mb,
        ratio=ratio,
        kept=kept,
        given=given,
        generated=generated,
        wall_seconds=wall,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout_tail="\n".join(tail_lines),
    )


def format_report(r: RssResult) -> str:
    kb_per_kept = r.kb_per_kept
    threshold_note = ""
    if r.kept > 0:
        # REQ-P002: ≤ 8 KB/clause per revision in Task #11
        verdict = "PASS" if kb_per_kept <= 8.0 else "FAIL"
        threshold_note = (
            f"Per-clause memory:       {kb_per_kept:8.2f} KB/clause "
            f"(REQ-P002 threshold 8.0 KB → {verdict})\n"
        )
    return (
        f"\n{'=' * 70}\n"
        f"REQ-P002 RSS BOUNDEDNESS — {r.problem}\n"
        f"{'=' * 70}\n"
        f"Initial RSS (t=0):       {r.initial_rss_mb:8.1f} MB\n"
        f"Settle RSS (post-parse): {r.settle_rss_mb:8.1f} MB\n"
        f"Peak RSS (during search):{r.peak_rss_mb:8.1f} MB\n"
        f"Ratio (peak/settle):     {r.ratio:8.2f}x\n"
        f"{threshold_note}"
        f"\n"
        f"Given:     {r.given}\n"
        f"Generated: {r.generated}\n"
        f"Kept:      {r.kept}\n"
        f"Wall:      {r.wall_seconds:.2f}s\n"
        f"Exit:      {r.exit_code}\n"
        f"{'=' * 70}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("--poll", type=float, default=0.05)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    result = measure_rss(
        Path(args.input_file),
        poll_interval=args.poll,
        settle_seconds=args.settle,
        timeout=args.timeout,
    )
    print(format_report(result))


if __name__ == "__main__":
    main()
