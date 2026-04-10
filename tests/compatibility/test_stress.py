"""Stress tests for large-scale problems and long searches.

Validates that PyLADR handles resource-intensive scenarios correctly
and terminates within expected bounds.

Run with: pytest tests/compatibility/test_stress.py -v -m slow
"""

from __future__ import annotations

import time

import pytest

from tests.compatibility.conftest import run_search


# ── Large Clause Set Stress Tests ──────────────────────────────────────────


@pytest.mark.slow
class TestLargeClauseSets:
    """Test behavior with many clauses."""

    def test_many_ground_clauses(self):
        """Search handles 100+ ground clauses without crashing."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode

        # Generate P_1(a), P_2(a), ..., P_100(a)
        a = get_rigid_term(1, 0)
        clauses = []
        for i in range(2, 102):
            P_i_a = get_rigid_term(i, 1, (a,))
            clauses.append(
                Clause(literals=(Literal(sign=True, atom=P_i_a),))
            )

        result = run_search(usable=[], sos=clauses, max_given=200)
        # Should terminate (either SOS empty or max_given)
        assert result.exit_code in (
            ExitCode.SOS_EMPTY_EXIT,
            ExitCode.MAX_GIVEN_EXIT,
        )

    def test_deep_resolution_chain(self):
        """Search handles a chain requiring many resolution steps."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        # Create chain: P0(a), ~P0(x)|P1(x), ~P1(x)|P2(x), ..., ~P9(a)
        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        chain_length = 10
        base_sn = 10

        clauses = []
        # P0(a)
        P0_a = get_rigid_term(base_sn, 1, (a,))
        clauses.append(Clause(literals=(Literal(sign=True, atom=P0_a),)))

        # Chain links
        for i in range(chain_length - 1):
            Pi_x = get_rigid_term(base_sn + i, 1, (x,))
            Pi1_x = get_rigid_term(base_sn + i + 1, 1, (x,))
            clauses.append(
                Clause(
                    literals=(
                        Literal(sign=False, atom=Pi_x),
                        Literal(sign=True, atom=Pi1_x),
                    )
                )
            )

        # ~P9(a)
        Pn_a = get_rigid_term(base_sn + chain_length - 1, 1, (a,))
        clauses.append(
            Clause(literals=(Literal(sign=False, atom=Pn_a),))
        )

        result = run_search(usable=[], sos=clauses, max_given=500)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_many_variable_clauses(self):
        """Search handles clauses with many different variables."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        # P(x0,x1,x2,...,x9) and ~P(a,a,a,...,a)
        vars = [get_variable_term(i) for i in range(10)]
        a = get_rigid_term(1, 0)
        consts = [a] * 10

        P_sn = 2
        P_vars = get_rigid_term(P_sn, 10, tuple(vars))
        P_consts = get_rigid_term(P_sn, 10, tuple(consts))

        c1 = Clause(literals=(Literal(sign=True, atom=P_vars),))
        c2 = Clause(literals=(Literal(sign=False, atom=P_consts),))

        result = run_search(usable=[], sos=[c1, c2], max_given=100)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


# ── Resource Limit Stress Tests ────────────────────────────────────────────


@pytest.mark.slow
class TestResourceLimits:
    """Test that resource limits are respected under load."""

    def test_max_given_respected_under_load(self):
        """max_given terminates even with clause explosion."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        # Create a problem that generates many clauses
        x = get_variable_term(0)
        y = get_variable_term(1)
        a = get_rigid_term(1, 0)
        b = get_rigid_term(2, 0)
        P_sn, Q_sn, R_sn = 3, 4, 5

        clauses = [
            Clause(
                literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),)
            ),
            Clause(
                literals=(Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (b,))),)
            ),
            Clause(
                literals=(
                    Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
                    Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
                    Literal(sign=True, atom=get_rigid_term(R_sn, 1, (x,))),
                )
            ),
            Clause(
                literals=(
                    Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (x,))),
                    Literal(sign=True, atom=get_rigid_term(P_sn, 1, (x,))),
                    Literal(sign=True, atom=get_rigid_term(R_sn, 1, (x,))),
                )
            ),
        ]

        max_given = 20
        result = run_search(usable=[], sos=clauses, max_given=max_given)

        # Must terminate within the limit
        assert result.stats.given <= max_given + 1  # +1 for boundary

    def test_search_terminates_within_time(self):
        """Search completes within reasonable wall-clock time."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        P_a = get_rigid_term(2, 1, (a,))
        P_x = get_rigid_term(2, 1, (x,))

        c1 = Clause(literals=(Literal(sign=True, atom=P_a),))
        c2 = Clause(literals=(Literal(sign=False, atom=P_x),))

        start = time.perf_counter()
        result = run_search(usable=[], sos=[c1, c2], max_given=1000)
        elapsed = time.perf_counter() - start

        # Even with generous limits, should complete quickly
        assert elapsed < 10.0, f"Search took {elapsed:.1f}s"


# ── Repeated Execution Stability ──────────────────────────────────────────


@pytest.mark.slow
class TestRepeatedExecution:
    """Ensure stability across many repeated executions."""

    def test_100_searches_no_crash(self):
        """Run 100 searches without crashing or memory issues."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode

        a = get_rigid_term(1, 0)
        x = get_variable_term(0)
        P_a = get_rigid_term(2, 1, (a,))
        P_x = get_rigid_term(2, 1, (x,))
        Q_x = get_rigid_term(3, 1, (x,))
        Q_a = get_rigid_term(3, 1, (a,))

        c1 = Clause(literals=(Literal(sign=True, atom=P_a),))
        c2 = Clause(
            literals=(
                Literal(sign=False, atom=P_x),
                Literal(sign=True, atom=Q_x),
            )
        )
        c3 = Clause(literals=(Literal(sign=False, atom=Q_a),))

        for i in range(100):
            result = run_search(usable=[], sos=[c1, c2, c3])
            assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
                f"Failed on iteration {i}"
            )

    def test_alternating_provable_unprovable(self):
        """Alternating provable/unprovable problems don't corrupt state."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode

        a = get_rigid_term(1, 0)
        P_a = get_rigid_term(2, 1, (a,))
        Q_a = get_rigid_term(3, 1, (a,))

        provable_sos = [
            Clause(literals=(Literal(sign=True, atom=P_a),)),
            Clause(literals=(Literal(sign=False, atom=P_a),)),
        ]
        unprovable_sos = [
            Clause(literals=(Literal(sign=True, atom=P_a),)),
        ]

        for i in range(50):
            if i % 2 == 0:
                result = run_search(usable=[], sos=provable_sos)
                assert result.exit_code == ExitCode.MAX_PROOFS_EXIT, (
                    f"Provable failed on iteration {i}"
                )
            else:
                result = run_search(usable=[], sos=unprovable_sos, max_given=10)
                assert result.exit_code in (
                    ExitCode.SOS_EMPTY_EXIT,
                    ExitCode.MAX_GIVEN_EXIT,
                ), f"Unprovable unexpected code on iteration {i}"
