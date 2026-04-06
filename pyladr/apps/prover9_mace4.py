"""Prover9-Mace4: run Prover9 and Mace4 in parallel.

Modernized Python 3.13+ version of utilities/prover9-mace4.

Takes a Prover9 input file and runs both Prover9 and Mace4 in parallel.
If one finishes successfully first, its output is sent to stdout.

Usage:
    pyprover9-mace4 -f input_file
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile


MACE4_EXITS = {
    0: "model",
    1: "fatal error",
    2: "search exhausted (no models)",
    3: "all models (with models)",
    4: "max_seconds (with models)",
    5: "max_seconds (no models)",
    6: "max_megs (with models)",
    7: "max_megs (without models)",
    101: "process interrupt",
    102: "process crash",
}

PROVER9_EXITS = {
    0: "proof",
    1: "fatal error",
    2: "sos empty",
    3: "max_megs",
    4: "max_seconds",
    5: "max_given",
    6: "max_kept",
    7: "action",
    101: "process interrupt",
    102: "process crash",
}


def code_to_message(program: str, code: int) -> str:
    """Convert exit code to human-readable message."""
    exits = MACE4_EXITS if program == "Mace4" else PROVER9_EXITS
    return exits.get(code, "unknown")


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyprover9-mace4 command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 2 or args[0] != "-f":
        sys.stderr.write("need arguments: -f input_filename\n")
        return 1

    input_filename = args[1]
    prover9 = "prover9"
    mace4 = "mace4"

    # Create temporary files for output
    with (
        tempfile.TemporaryFile(mode="w+") as fout1,
        tempfile.TemporaryFile(mode="w+") as ferr1,
        tempfile.TemporaryFile(mode="w+") as fout2,
        tempfile.TemporaryFile(mode="w+") as ferr2,
    ):
        prover9_command = [prover9, "-f", input_filename]
        mace4_command = [mace4, "-c", "-N", "-1", "-f", input_filename]

        p1 = subprocess.Popen(
            prover9_command, stdin=None, stdout=fout1, stderr=ferr1
        )
        p2 = subprocess.Popen(
            mace4_command, stdin=None, stdout=fout2, stderr=ferr2
        )

        sys.stderr.write(f"Prover9 process ID {p1.pid}.\n")
        sys.stderr.write(f"Mace4   process ID {p2.pid}.\n")
        sys.stderr.write("Waiting for one of them to finish ... \n")

        # Wait for one to finish
        pid, status = os.waitpid(0, 0)

        # Determine the finisher; kill the non-finisher
        if pid == p1.pid:
            finisher = "Prover9"
            fout = fout1
            try:
                os.kill(p2.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            finisher = "Mace4"
            fout = fout2
            try:
                os.kill(p1.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        if not os.WIFEXITED(status):
            sys.stderr.write(f"error: {finisher} exited abnormally\n")
            return 1

        exit_code = os.WEXITSTATUS(status)

        if exit_code != 0:
            message = code_to_message(finisher, exit_code)
            sys.stderr.write(f"failure, {finisher} ended by {message}\n")
            return 1

        sys.stderr.write(
            f"{finisher} finished with success, see the standard output.\n"
        )

        # Rewind and send winning output to stdout
        fout.seek(0)
        for line in fout:
            sys.stdout.write(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
