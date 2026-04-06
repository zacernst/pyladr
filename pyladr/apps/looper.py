"""Looper: iterate Prover9/Mace4 with candidates from a stream.

Modernized Python 3.13+ version of utilities/looper.

Takes a (Prover9|Mace4) input file (the head) and a stream of candidates.
For each candidate, appends it to the head and runs the specified program.
Standard output gets any proofs/models found, plus statistics.

Usage:
    pylooper (prover9|mace4) (assumptions|goals) head < candidates
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile


def runjob(
    command_and_args: list[str],
    stdin_string: str | None = None,
    debug: bool = False,
) -> tuple[int, int, str, str]:
    """Run a subprocess and return (pid, exit_code, stdout, stderr)."""
    if debug:
        print(f"Starting job: {command_and_args}")

    result = subprocess.run(
        command_and_args,
        input=stdin_string,
        capture_output=True,
        text=True,
    )

    if debug:
        sys.stdout.write(result.stdout)
        sys.stdout.write(result.stderr)
        print(f"pid={result.returncode}, exit_code={result.returncode}")

    return (os.getpid(), result.returncode, result.stdout, result.stderr)


def goodlines(lines: list[str]) -> list[str]:
    """Filter output for proofs/models and summary lines."""
    keep: list[str] = []
    collect = False
    for line in lines:
        if re.search(r"= (PROOF|MODEL) =", line):
            collect = True
        if collect or re.search(r"Fatal|Given|Selections|CPU|Process.*exit", line):
            keep.append(line)
        if re.search(r"= end of (proof|model) =", line):
            collect = False
    return keep


def get_cpu(lines: list[str]) -> str:
    """Extract CPU time from output lines."""
    for line in lines:
        if re.match(r"User_CPU", line):
            parts = re.split(r"=|,", line)
            if len(parts) > 1:
                return parts[1].strip()
    return "???"


HELP_STRING = (
    "need 3 args: (prover9|mace4) (assumptions|goals) head < candidates\n"
)

MACE4_MODEL_EXITS = {0}
MACE4_ERROR_EXITS = {1, 101, 102}
PROVER9_PROOF_EXITS = {0}
PROVER9_ERROR_EXITS = {1, 101, 102}


def main(argv: list[str] | None = None) -> int:
    """Entry point for pylooper command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 3:
        sys.stderr.write(HELP_STRING)
        return 1

    program = args[0]
    list_type = args[1]
    head_fname = args[2]

    if program not in ("prover9", "mace4") or list_type not in (
        "assumptions",
        "goals",
    ):
        sys.stderr.write(HELP_STRING)
        return 1

    if not Path(head_fname).is_file():
        sys.stderr.write(f"head file {head_fname} not found\n")
        return 1

    debug = False

    date = time.strftime("%A, %b %d, %I:%M %p %Y", time.localtime())
    host = socket.gethostname()

    print(f"Started looper {date} on {host}.")
    print("%" * 50 + " HEAD FILE " + "%" * 9)

    with open(head_fname) as f:
        sys.stdout.writelines(f.readlines())
    sys.stdout.flush()

    print("%" * 47 + " end of head file " + "%" * 5)

    if program == "prover9":
        command_base = [program, "-f", head_fname]
    else:
        command_base = [program, "-c", "-f", head_fname]

    n = 0
    successes = 0
    failures = 0

    for cand_line in sys.stdin:
        # Strip newline and comments
        cand = re.sub(r"\n", "", cand_line)
        cand = re.sub(r" *%.*", "", cand)

        if not cand.strip():
            continue

        n += 1
        print("-" * 70)
        print(f"{cand} % Problem {n}\n")
        sys.stdout.flush()

        # Write candidate to temp file
        with NamedTemporaryFile(
            mode="w", suffix=".in", prefix="cand_", delete=False
        ) as tmp:
            tmp.write(f"formulas({list_type}).\n")
            tmp.write(f"{cand}\n")
            tmp.write("end_of_list.\n")
            tmp.write("clear(print_given).\n")
            cand_fname_tmp = tmp.name

        try:
            command_list = command_base + [cand_fname_tmp]
            _, rc, out, err = runjob(command_list, debug=debug)

            if (program == "mace4" and rc in MACE4_ERROR_EXITS) or (
                program == "prover9" and rc in PROVER9_ERROR_EXITS
            ):
                sys.stderr.write(f"program error, rc={rc}\n")
                sys.stdout.write(err)
                sys.stdout.write(out)
                failures += 1
            else:
                outlines = out.splitlines(keepends=True)
                keeplines = goodlines(outlines)
                for line in keeplines:
                    sys.stdout.write(line)
                cpu = get_cpu(keeplines)
                if program == "prover9" and rc in PROVER9_PROOF_EXITS:
                    message = "Proved"
                    successes += 1
                elif program == "mace4" and rc in MACE4_MODEL_EXITS:
                    message = "Disproved"
                    successes += 1
                else:
                    message = "Failed"
                    failures += 1
                print(f"\n{cand}  % {message} {cpu} seconds PROBLEM {n}")
            sys.stdout.flush()
        finally:
            Path(cand_fname_tmp).unlink(missing_ok=True)

    date = time.strftime("%A, %b %d, %I:%M %p %Y", time.localtime())
    print(
        f"\nFinished {date}, processed {n}, successes {successes}, "
        f"failures {failures}.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
