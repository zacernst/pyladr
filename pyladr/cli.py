"""Command-line interface for PyLADR/Prover9.

Provides the main pyprover9 command and dispatches to auxiliary tools
via subcommands or direct entry points.
"""

from __future__ import annotations

import sys

from pyladr import __version__

AVAILABLE_TOOLS = {
    "renamer": "pyladr.apps.renamer",
    "mirror-flip": "pyladr.apps.mirror_flip",
    "perm3": "pyladr.apps.perm3",
    "prooftrans": "pyladr.apps.prooftrans",
    "looper": "pyladr.apps.looper",
    "attack": "pyladr.apps.attack",
    "get-interps": "pyladr.apps.get_interps",
    "get-givens": "pyladr.apps.get_givens",
    "get-kept": "pyladr.apps.get_kept",
    "prover9-mace4": "pyladr.apps.prover9_mace4",
    "clausefilter": "pyladr.apps.clausefilter",
    "clausetester": "pyladr.apps.clausetester",
    "interpformat": "pyladr.apps.interpformat",
    "isofilter": "pyladr.apps.isofilter",
    "isofilter0": "pyladr.apps.isofilter0",
    "isofilter2": "pyladr.apps.isofilter2",
    "interpfilter": "pyladr.apps.interpfilter",
    "dprofiles": "pyladr.apps.dprofiles",
    "sigtest": "pyladr.apps.sigtest",
    "latfilter": "pyladr.apps.latfilter",
    "olfilter": "pyladr.apps.olfilter",
    "idfilter": "pyladr.apps.idfilter",
    "upper-covers": "pyladr.apps.upper_covers",
    "miniscope": "pyladr.apps.miniscope",
    "unfast": "pyladr.apps.unfast",
    "complex": "pyladr.apps.complex",
    "ladr-to-tptp": "pyladr.apps.ladr_to_tptp",
    "rewriter": "pyladr.apps.rewriter",
    "rewriter2": "pyladr.apps.rewriter2",
    "directproof": "pyladr.apps.directproof",
    "gen-trc-defs": "pyladr.apps.gen_trc_defs",
}


def main() -> int:
    """Entry point for the pyprover9 command.

    When called with a tool name as first argument, dispatches to that tool.
    Otherwise runs the main Prover9 theorem prover.
    """
    if len(sys.argv) > 1 and sys.argv[1] in AVAILABLE_TOOLS:
        tool_name = sys.argv[1]
        module_path = AVAILABLE_TOOLS[tool_name]
        import importlib

        module = importlib.import_module(module_path)
        sys.argv = [f"py{tool_name}"] + sys.argv[2:]
        return module.main()  # type: ignore[no-any-return]

    # Not a tool dispatch — run the main prover
    from pyladr.apps.prover9 import run_prover

    return run_prover()


if __name__ == "__main__":
    sys.exit(main())
