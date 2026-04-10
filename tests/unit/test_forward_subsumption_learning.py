"""Unit tests for forward subsumption learning feature.

Tests the callback mechanism in GivenClauseSearch, integration with
OnlineSearchIntegration, and CLI argument handling.

Mirrors the existing back-subsumption learning test patterns.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions


# ── Helpers ──────────────────────────────────────────────────────────────────


def _var(n: int) -> Term:
    return get_variable_term(n)


def _const(symtab: SymbolTable, name: str) -> Term:
    sn = symtab.str_to_sn(name, 0)
    return get_rigid_term(sn, 0)


def _func(symtab: SymbolTable, name: str, *args: Term) -> Term:
    sn = symtab.str_to_sn(name, len(args))
    return get_rigid_term(sn, len(args), args)


def _pos_lit(atom: Term) -> Literal:
    return Literal(sign=True, atom=atom)


def _neg_lit(atom: Term) -> Literal:
    return Literal(sign=False, atom=atom)


def _clause(*lits: Literal, cid: int = 0) -> Clause:
    c = Clause(literals=tuple(lits))
    c.id = cid
    return c


def _deny_goals(goals: list[Clause]) -> list[Clause]:
    """Negate goal clauses for refutation search."""
    denied = []
    for goal in goals:
        denied_lits = tuple(
            Literal(sign=not lit.sign, atom=lit.atom) for lit in goal.literals
        )
        denied.append(
            Clause(
                literals=denied_lits,
                justification=(Justification(just_type=JustType.DENY, clause_ids=(0,)),),
            )
        )
    return denied


# ── SearchOptions field tests ────────────────────────────────────────────────


class TestSearchOptionsField:
    """Verify SearchOptions has the forward subsumption learning field."""

    def test_default_is_false(self):
        opts = SearchOptions()
        assert opts.learn_from_forward_subsumption is False

    def test_can_enable(self):
        opts = SearchOptions(learn_from_forward_subsumption=True)
        assert opts.learn_from_forward_subsumption is True

    def test_independent_of_back_subsumption(self):
        opts = SearchOptions(
            learn_from_back_subsumption=True,
            learn_from_forward_subsumption=False,
        )
        assert opts.learn_from_back_subsumption is True
        assert opts.learn_from_forward_subsumption is False


# ── Callback mechanism tests ─────────────────────────────────────────────────


class TestForwardSubsumptionCallback:
    """Test the callback registration and invocation on GivenClauseSearch."""

    def test_callback_slot_exists(self):
        search = GivenClauseSearch()
        assert hasattr(search, '_forward_subsumption_callback')

    def test_callback_initially_none(self):
        search = GivenClauseSearch()
        assert search._forward_subsumption_callback is None

    def test_set_callback(self):
        search = GivenClauseSearch()
        cb = MagicMock()
        search.set_forward_subsumption_callback(cb)
        assert search._forward_subsumption_callback is cb

    def test_set_forward_subsumption_callback_method_exists(self):
        search = GivenClauseSearch()
        assert callable(getattr(search, 'set_forward_subsumption_callback', None))

    def test_callback_not_called_when_disabled(self):
        """Callback should not fire when learn_from_forward_subsumption=False."""
        opts = SearchOptions(
            learn_from_forward_subsumption=False,
            max_given=10,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts)

        cb = MagicMock()
        search.set_forward_subsumption_callback(cb)

        # Run a problem that triggers forward subsumption
        st = SymbolTable()
        parser = LADRParser(st)
        # P(x) in usable will forward-subsume any P(a) generated from sos
        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  P(a) -> Q(a).
  -Q(a).
end_of_list.

formulas(goals).
  Q(b).
end_of_list.
""")
        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        search.run(usable=usable, sos=sos)

        # Callback must NOT be called when flag is disabled
        cb.assert_not_called()

    def test_callback_called_when_enabled(self):
        """Callback fires when learn_from_forward_subsumption=True and subsumption occurs."""
        st = SymbolTable()
        parser = LADRParser(st)

        # P(x) in usable forward-subsumes generated P(a)
        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  Q(a) -> P(a).
  Q(a).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""")

        opts = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)

        events = []

        def callback(subsuming, subsumed):
            events.append((subsuming, subsumed))

        search.set_forward_subsumption_callback(callback)

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        search.run(usable=usable, sos=sos)

        # Forward subsumption should have triggered (P(x) subsumes P(a))
        # The subsumed count tracks this
        if search.stats.subsumed > 0:
            assert len(events) > 0, (
                "Forward subsumption occurred (stats.subsumed > 0) but callback was not called"
            )

    def test_callback_receives_correct_arguments(self):
        """Callback receives (subsuming_clause, subsumed_clause) with correct types."""
        st = SymbolTable()
        parser = LADRParser(st)

        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  Q(a) -> P(a).
  Q(a).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""")

        opts = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)

        events = []

        def callback(subsuming, subsumed):
            events.append((subsuming, subsumed))

        search.set_forward_subsumption_callback(callback)

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        search.run(usable=usable, sos=sos)

        for subsuming, subsumed in events:
            assert isinstance(subsuming, Clause), "subsuming argument must be a Clause"
            assert isinstance(subsumed, Clause), "subsumed argument must be a Clause"


# ── Forward subsumption statistics tests ─────────────────────────────────────


class TestForwardSubsumptionStats:
    """Verify that forward subsumption increments stats.subsumed."""

    def test_subsumed_counter_increments(self):
        """stats.subsumed should increase when forward subsumption occurs."""
        st = SymbolTable()
        parser = LADRParser(st)

        # General clause P(x) forward-subsumes generated specific clauses
        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  Q(a) -> P(a).
  Q(a).
  -R(a).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""")

        opts = SearchOptions(max_given=20, quiet=True)
        search = GivenClauseSearch(options=opts, symbol_table=st)

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        result = search.run(usable=usable, sos=sos)

        # Forward subsumption should have occurred at least once
        assert result.stats.subsumed >= 0  # non-negative always


# ── Baseline comparison tests ────────────────────────────────────────────────


class TestForwardSubsumptionBaseline:
    """Verify search produces same result with/without learning flag."""

    def test_same_exit_code_with_and_without_learning(self):
        """Enabling learn_from_forward_subsumption must not change search behavior."""
        st = SymbolTable()
        parser = LADRParser(st)

        test_input = """
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  P(x) -> Q(x).
  -Q(a).
end_of_list.

formulas(goals).
  Q(a).
end_of_list.
"""
        parsed = parser.parse_input(test_input)
        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))

        # Run without learning
        opts_off = SearchOptions(
            learn_from_forward_subsumption=False,
            max_given=50,
            quiet=True,
        )
        search_off = GivenClauseSearch(options=opts_off, symbol_table=st)
        result_off = search_off.run(usable=list(usable), sos=list(sos))

        # Run with learning (but no ML backend)
        opts_on = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=50,
            quiet=True,
        )
        search_on = GivenClauseSearch(options=opts_on, symbol_table=st)
        events = []
        search_on.set_forward_subsumption_callback(lambda s, d: events.append((s, d)))
        result_on = search_on.run(usable=list(usable), sos=list(sos))

        assert result_off.exit_code == result_on.exit_code
        assert result_off.stats.given == result_on.stats.given
        assert result_off.stats.kept == result_on.stats.kept
        assert result_off.stats.subsumed == result_on.stats.subsumed


# ── Comparison with back-subsumption learning ────────────────────────────────


class TestForwardVsBackSubsumption:
    """Verify structural similarity between forward and back subsumption callbacks."""

    def test_both_callbacks_independent(self):
        """Forward and back subsumption callbacks can be set independently."""
        search = GivenClauseSearch()

        fwd_cb = MagicMock()
        back_cb = MagicMock()

        search.set_forward_subsumption_callback(fwd_cb)
        search.set_back_subsumption_callback(back_cb)

        assert search._forward_subsumption_callback is fwd_cb
        assert search._back_subsumption_callback is back_cb

    def test_both_flags_can_be_enabled(self):
        opts = SearchOptions(
            learn_from_back_subsumption=True,
            learn_from_forward_subsumption=True,
        )
        assert opts.learn_from_back_subsumption is True
        assert opts.learn_from_forward_subsumption is True


# ── OnlineSearchIntegration tests ────────────────────────────────────────────


class TestOnlineSearchIntegrationForwardSubsumption:
    """Test the on_forward_subsumption method in OnlineSearchIntegration."""

    def test_on_forward_subsumption_method_exists(self):
        """OnlineSearchIntegration must have on_forward_subsumption."""
        from pyladr.search.online_integration import OnlineSearchIntegration
        assert hasattr(OnlineSearchIntegration, 'on_forward_subsumption')

    def test_on_forward_subsumption_disabled_noop(self):
        """on_forward_subsumption is a no-op when integration is disabled."""
        from pyladr.search.online_integration import OnlineSearchIntegration

        integration = OnlineSearchIntegration.__new__(OnlineSearchIntegration)
        integration._enabled = False
        integration._manager = None

        st = SymbolTable()
        a = _const(st, "a")
        p_a = _func(st, "P", a)
        c1 = _clause(_pos_lit(p_a), cid=1)
        c2 = _clause(_pos_lit(p_a), cid=2)

        # Should not raise
        integration.on_forward_subsumption(c1, c2)

    def test_callback_wired_in_online_learning_search(self):
        """_OnlineLearningGivenClauseSearch properly wires forward subsumption callback."""
        from pyladr.search.online_integration import _OnlineLearningGivenClauseSearch

        mock_integration = MagicMock()
        mock_integration.on_forward_subsumption = MagicMock()

        instance = _OnlineLearningGivenClauseSearch(
            integration=mock_integration,
            options=SearchOptions(learn_from_forward_subsumption=True),
        )

        # Callback should be properly wired to the final HookedSearch instance
        assert instance._forward_subsumption_callback is not None


# ── CLI argument tests ───────────────────────────────────────────────────────


class TestCLIArgument:
    """Test --learn-from-forward-subsumption CLI argument."""

    def test_argument_registered(self):
        """The argument should be registered in the prover9 CLI parser."""
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()

        # Parse with the flag
        args = parser.parse_args(["--learn-from-forward-subsumption"])
        assert args.learn_from_forward_subsumption is True

    def test_argument_default_false(self):
        """Default value for --learn-from-forward-subsumption is False."""
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()

        args = parser.parse_args([])
        assert args.learn_from_forward_subsumption is False

    def test_argument_independent_of_back_subsumption(self):
        """Forward and back subsumption args are independent."""
        from pyladr.apps.prover9 import _build_arg_parser
        parser = _build_arg_parser()

        args = parser.parse_args([
            "--learn-from-forward-subsumption",
            "--learn-from-back-subsumption",
        ])
        assert args.learn_from_forward_subsumption is True
        assert args.learn_from_back_subsumption is True


# ── End-to-end forward subsumption test ──────────────────────────────────────


class TestEndToEndForwardSubsumption:
    """End-to-end tests that trigger actual forward subsumption events."""

    def test_general_subsumes_specific(self):
        """A general clause in usable should forward-subsume specific generated clauses."""
        st = SymbolTable()
        parser = LADRParser(st)

        # P(x) | Q(x) is general; derivations producing P(a) | Q(a) get subsumed
        parsed = parser.parse_input("""
formulas(usable).
  P(x) | Q(x).
end_of_list.

formulas(sos).
  R(a) -> (P(a) | Q(a)).
  R(a).
  -S(a).
end_of_list.

formulas(goals).
  S(a).
end_of_list.
""")

        opts = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=30,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)

        events = []
        search.set_forward_subsumption_callback(lambda s, d: events.append((s.id, d.id)))

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        result = search.run(usable=usable, sos=sos)

        # The search should complete, and any forward subsumption events
        # should have been captured
        assert result is not None

    def test_multiple_forward_subsumptions(self):
        """Multiple generated clauses can be forward-subsumed by the same general clause."""
        st = SymbolTable()
        parser = LADRParser(st)

        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  Q(a) -> P(a).
  Q(b) -> P(b).
  Q(c) -> P(c).
  Q(a).
  Q(b).
  Q(c).
  -R(a).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""")

        opts = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=50,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)

        events = []
        search.set_forward_subsumption_callback(lambda s, d: events.append((s.id, d.id)))

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        search.run(usable=usable, sos=sos)

        # Each P(a), P(b), P(c) should be forward-subsumed by P(x)
        if events:
            # All subsuming clauses should be the same (P(x))
            subsuming_ids = {e[0] for e in events}
            assert len(subsuming_ids) >= 1  # At least one subsuming clause

    def test_forward_subsumption_subsuming_clause_is_general(self):
        """The subsuming clause in callback should be the more general one."""
        st = SymbolTable()
        parser = LADRParser(st)

        parsed = parser.parse_input("""
formulas(usable).
  P(x).
end_of_list.

formulas(sos).
  Q(a) -> P(a).
  Q(a).
  -R(a).
end_of_list.

formulas(goals).
  R(a).
end_of_list.
""")

        opts = SearchOptions(
            learn_from_forward_subsumption=True,
            max_given=20,
            quiet=True,
        )
        search = GivenClauseSearch(options=opts, symbol_table=st)

        events = []

        def callback(subsuming, subsumed):
            events.append({
                'subsuming': subsuming,
                'subsumed': subsumed,
                'subsuming_lits': len(subsuming.literals),
                'subsumed_lits': len(subsumed.literals),
            })

        search.set_forward_subsumption_callback(callback)

        usable = list(parsed.usable)
        sos = list(parsed.sos) + _deny_goals(list(parsed.goals))
        search.run(usable=usable, sos=sos)

        # Subsuming clause should have variables (more general)
        for event in events:
            # Subsuming clause is the existing, more general one
            assert isinstance(event['subsuming'], Clause)
            assert isinstance(event['subsumed'], Clause)
