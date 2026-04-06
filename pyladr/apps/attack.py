"""Attack: build model databases and filter candidates against known models.

Modernized Python 3.13+ version of utilities/attack.

Takes a Mace4 input file (the head) and a stream of candidates. Builds a
list of models (M) of the candidates w.r.t. the clauses in the head.

For each candidate C:
  - If C is true in any member of M, discard it
  - Otherwise, look for a model of C and the head clauses
  - If a model is found, add it to M and discard C
  - Otherwise, C passes through to stdout

Usage:
    pyattack head interps < candidates > candidates.out
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
    command_and_args: list[str] | str,
    stdin_string: str | None = None,
    debug: bool = False,
) -> tuple[int, int, str, str]:
    """Run a subprocess and return (pid, exit_code, stdout, stderr)."""
    if debug:
        print(f"Starting job: {command_and_args}")

    shell = isinstance(command_and_args, str)
    result = subprocess.run(
        command_and_args,
        input=stdin_string,
        capture_output=True,
        text=True,
        shell=shell,
    )

    return (os.getpid(), result.returncode, result.stdout, result.stderr)


MACE4_MODEL_EXITS = {0, 3, 4, 6}
MACE4_ERROR_EXITS = {1, 102}


def main(argv: list[str] | None = None) -> int:
    """Entry point for pyattack command."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 2:
        sys.stderr.write("need 2 args: head interp\n")
        return 1

    head_fname = args[0]
    interp_fname = args[1]

    if Path(interp_fname).is_file():
        sys.stderr.write(f"file {interp_fname} already exists\n")
        return 1

    interps = ""
    checked = 0
    passed = 0
    num_interps = 0
    debug = False

    mace4 = "mace4"
    get_interps_cmd = "get_interps"
    interpfilter = "interpfilter"

    for line in sys.stdin:
        if re.match(r"\s*%", line):
            sys.stdout.write(line)
            continue

        cand = line
        checked += 1

        if checked % 100 == 0:
            sys.stderr.write(f"checked {checked}\n")
            with open(interp_fname, "w") as f:
                f.write("\nINTERMEDIATE RESULTS\n\n")
                f.write(interps)
                f.write("\nINTERMEDIATE RESULTS\n\n")

        # Write candidate to temp file
        with NamedTemporaryFile(
            mode="w", suffix=".in", prefix="cand_", delete=False
        ) as tmp:
            tmp.write(cand)
            cand_fname = tmp.name

        try:
            # Check if existing models kill this candidate
            _, rc, out, err = runjob(
                [interpfilter, cand_fname, "all_true"], stdin_string=interps
            )

            if rc != 0:  # not killed by existing model
                # Write candidate as formulas
                with open(cand_fname, "w") as f:
                    f.write("formulas(candidate).\n")
                    f.write(cand)
                    f.write("end_of_list.\n")

                _, rc, out, err = runjob([mace4, "-f", head_fname, cand_fname])

                if rc in MACE4_ERROR_EXITS:
                    sys.stderr.write(f"mace4 error {rc}\n")
                elif rc in MACE4_MODEL_EXITS:
                    # Candidate killed; extract and save interpretation
                    num_interps += 1
                    _, _, interp_out, _ = runjob(get_interps_cmd, stdin_string=out)
                    cand_stripped = cand.rstrip("\n")
                    header = f"\n% {cand_stripped} % cand {checked} killed by\n\n"
                    interps = interps + header + interp_out
                else:
                    # Candidate survives
                    passed += 1
                    cand_stripped = cand.rstrip("\n")
                    sys.stdout.write(f"{cand_stripped} % cand {checked}\n")
                    sys.stdout.flush()
        finally:
            Path(cand_fname).unlink(missing_ok=True)

    date = time.strftime("%A, %b %d, %I:%M %p %Y", time.localtime())
    host = socket.gethostname()

    # Write the interp file
    with open(interp_fname, "w") as f:
        f.write(f"% Started on {host} at {date}\n")
        f.write(f"% Here is the head file {head_fname}.\n\n")
        f.write("%" * 43 + "\n")
        with open(head_fname) as hf:
            for hline in hf:
                f.write(f"% {hline}")
        f.write("%" * 43 + "\n")
        f.write(interps)
        f.write(
            f"\n% Checked {checked}, passed {passed}, "
            f"num_interps {num_interps}.\n"
        )
        f.write(f"% Finished {date}.\n")

    sys.stdout.write(
        f"% attack {head_fname} {interp_fname} (on {host}): "
        f"checked {checked}, passed {passed}, num_interps {num_interps}.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
