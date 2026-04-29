"""Smoke test: `python3 -m pyladr.apps.prover9` must invoke main.

Task #12 regression guard. Prior to the fix there was no
`if __name__ == "__main__"` block in pyladr/apps/prover9.py, so invoking
the module directly imported it and exited 0 with no output. The installed
entry point (`pyprover9` → `pyladr.cli:main`) was unaffected, which is why
this regression went unnoticed. Cross-validation harnesses that shell out
via `-m pyladr.apps.prover9` silently produced no results.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMPLE_INPUT = REPO_ROOT / "tests" / "fixtures" / "inputs" / "simple_group.in"


def test_module_entry_produces_output_and_exits_nonzero_on_sos_empty():
    """Direct `python -m pyladr.apps.prover9` must run the search, not exit silently."""
    assert SIMPLE_INPUT.exists(), f"fixture missing: {SIMPLE_INPUT}"

    proc = subprocess.run(
        [sys.executable, "-m", "pyladr.apps.prover9", "-f", str(SIMPLE_INPUT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )

    assert proc.stdout, (
        "module invocation produced no stdout — the __main__ guard is missing"
    )
    assert "SEARCH FAILED" in proc.stdout or "THEOREM PROVED" in proc.stdout, (
        f"expected search output, got:\n{proc.stdout[-400:]}"
    )
    assert proc.returncode == 2, (
        f"simple_group.in should exit 2 (sos_empty); got {proc.returncode}"
    )
