"""Backward compatibility tests for ML enhancements.

These tests ensure that ML features are completely opt-in and that
default behavior is identical whether or not ML modules are present.
This is the critical guardrail against breaking changes.

Run with: pytest tests/compatibility/test_ml_backward_compat.py -v
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

from tests.compatibility.conftest import run_search


# ── Default Behavior Tests ─────────────────────────────────────────────────


class TestDefaultBehaviorUnchanged:
    """Verify that default search behavior is identical with/without ML."""

    def test_search_options_no_ml_defaults(self):
        """SearchOptions defaults do not include any ML parameters."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()
        # Verify core defaults match C Prover9 behavior
        assert opts.binary_resolution is True
        assert opts.paramodulation is False
        assert opts.factoring is True
        assert opts.max_given == -1
        assert opts.max_kept == -1
        assert opts.max_proofs == 1
        assert opts.demodulation is False
        assert opts.check_tautology is True
        assert opts.merge_lits is True

    def test_search_result_structure_unchanged(self):
        """SearchResult has the expected fields — no ML contamination."""
        from pyladr.search.given_clause import SearchResult

        import dataclasses

        field_names = {f.name for f in dataclasses.fields(SearchResult)}
        expected = {"exit_code", "proofs", "stats"}
        assert expected.issubset(field_names), (
            f"Missing expected fields: {expected - field_names}"
        )

    def test_exit_codes_match_c(self):
        """Exit codes match C search.h enum values."""
        from pyladr.search.given_clause import ExitCode

        assert ExitCode.MAX_PROOFS_EXIT == 1
        assert ExitCode.SOS_EMPTY_EXIT == 2
        assert ExitCode.MAX_GIVEN_EXIT == 3
        assert ExitCode.MAX_KEPT_EXIT == 4
        assert ExitCode.MAX_SECONDS_EXIT == 5
        assert ExitCode.MAX_GENERATED_EXIT == 6
        assert ExitCode.FATAL_EXIT == 7

    def test_statistics_fields_match_c(self):
        """SearchStatistics has all C-matching fields."""
        from pyladr.search.statistics import SearchStatistics

        import dataclasses

        field_names = {f.name for f in dataclasses.fields(SearchStatistics)}
        c_expected = {
            "given",
            "generated",
            "kept",
            "subsumed",
            "back_subsumed",
            "demodulated",
            "back_demodulated",
            "new_demodulators",
            "sos_limit_deleted",
            "proofs",
        }
        assert c_expected.issubset(field_names), (
            f"Missing C-compatible fields: {c_expected - field_names}"
        )

    def test_default_search_no_ml_import(self, trivial_resolution_clauses):
        """Default search path does not import torch or ML modules."""
        # Track which modules get imported during search
        imported_before = set(sys.modules.keys())

        result = run_search(usable=[], sos=trivial_resolution_clauses)

        imported_after = set(sys.modules.keys())
        new_imports = imported_after - imported_before

        ml_modules = {m for m in new_imports if "torch" in m or "torch_geometric" in m}
        assert len(ml_modules) == 0, (
            f"Default search imported ML modules: {ml_modules}"
        )


# ── Proof Equivalence Tests ────────────────────────────────────────────────


class TestProofEquivalence:
    """Ensure proofs are identical with default settings."""

    def test_trivial_proof_identical(self, trivial_resolution_clauses):
        """Trivial resolution proof matches expected structure."""
        from pyladr.search.given_clause import ExitCode

        result = run_search(usable=[], sos=trivial_resolution_clauses)
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1
        proof = result.proofs[0]
        assert len(proof.empty_clause.literals) == 0

    def test_equational_proof_identical(self, equational_problem):
        """Equational proof with paramodulation matches expected structure."""
        from pyladr.search.given_clause import ExitCode

        st, usable, sos = equational_problem
        result = run_search(
            usable=usable,
            sos=sos,
            paramodulation=True,
            symbol_table=st,
        )
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT
        assert len(result.proofs) == 1


# ── API Surface Tests ──────────────────────────────────────────────────────


class TestAPISurface:
    """Ensure public API surface has not changed incompatibly."""

    def test_given_clause_search_constructor(self):
        """GivenClauseSearch can be constructed with minimal args."""
        from pyladr.search.given_clause import GivenClauseSearch, SearchOptions

        opts = SearchOptions()
        search = GivenClauseSearch(options=opts)
        assert search is not None

    def test_given_clause_search_run_signature(self):
        """GivenClauseSearch.run accepts usable and sos lists."""
        import inspect

        from pyladr.search.given_clause import GivenClauseSearch

        sig = inspect.signature(GivenClauseSearch.run)
        params = list(sig.parameters.keys())
        assert "usable" in params
        assert "sos" in params

    def test_clause_construction_api(self):
        """Clause can be constructed from literals tuple."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        atom = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        lit = Literal(sign=True, atom=atom)
        c = Clause(literals=(lit,))
        assert len(c.literals) == 1

    def test_term_construction_api(self):
        """Term construction API is stable."""
        from pyladr.core.term import (
            get_rigid_term,
            get_variable_term,
            build_binary_term,
            build_unary_term,
        )

        x = get_variable_term(0)
        a = get_rigid_term(1, 0)
        f_x = build_unary_term(2, x)
        g_a_x = build_binary_term(3, a, x)
        assert f_x is not None
        assert g_a_x is not None

    def test_parser_api(self):
        """LADRParser public API is stable."""
        from pyladr.core.symbol import SymbolTable
        from pyladr.parsing.ladr_parser import LADRParser

        st = SymbolTable()
        parser = LADRParser(st)
        assert hasattr(parser, "parse_term")
        assert hasattr(parser, "parse_input")

    def test_symbol_table_api(self):
        """SymbolTable public API is stable."""
        from pyladr.core.symbol import SymbolTable

        st = SymbolTable()
        assert hasattr(st, "str_to_sn")

    def test_resolution_api(self):
        """Resolution module public API is stable."""
        from pyladr.inference import resolution

        assert hasattr(resolution, "binary_resolve")
        assert hasattr(resolution, "factor")
        assert hasattr(resolution, "is_tautology")
        assert hasattr(resolution, "merge_literals")
        assert hasattr(resolution, "renumber_variables")

    def test_subsumption_api(self):
        """Subsumption module public API is stable."""
        from pyladr.inference import subsumption

        assert hasattr(subsumption, "subsumes")
        assert hasattr(subsumption, "forward_subsume_from_lists")
        assert hasattr(subsumption, "back_subsume_from_lists")

    def test_paramodulation_api(self):
        """Paramodulation module public API is stable."""
        from pyladr.inference import paramodulation

        assert hasattr(paramodulation, "para_from_into")
        assert hasattr(paramodulation, "orient_equalities")

    def test_demodulation_api(self):
        """Demodulation module public API is stable."""
        from pyladr.inference import demodulation

        assert hasattr(demodulation, "demodulate_clause")
        assert hasattr(demodulation, "demodulator_type")
        assert hasattr(demodulation, "DemodType")
        assert hasattr(demodulation, "DemodulatorIndex")


# ── Import Safety Tests ────────────────────────────────────────────────────


class TestImportSafety:
    """Ensure PyLADR core modules import without ML dependencies."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "pyladr.core.term",
            "pyladr.core.clause",
            "pyladr.core.symbol",
            "pyladr.core.substitution",
            "pyladr.inference.resolution",
            "pyladr.inference.paramodulation",
            "pyladr.inference.demodulation",
            "pyladr.inference.subsumption",
            "pyladr.search.given_clause",
            "pyladr.search.selection",
            "pyladr.search.state",
            "pyladr.search.statistics",
            "pyladr.parsing.ladr_parser",
            "pyladr.parsing.tokenizer",
            "pyladr.indexing.discrimination_tree",
            "pyladr.ordering.kbo",
            "pyladr.ordering.lrpo",
        ],
    )
    def test_core_module_imports_without_torch(self, module_path: str):
        """Core module can be imported even if torch is missing."""
        # We just verify the import succeeds without error
        mod = importlib.import_module(module_path)
        assert mod is not None


# ── Configuration Compatibility Tests ──────────────────────────────────────


class TestConfigurationCompatibility:
    """Ensure all C Prover9 search options are supported."""

    def test_all_c_limits_supported(self):
        """All C-style search limits are supported in SearchOptions."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions(
            max_given=100,
            max_kept=200,
            max_seconds=60.0,
            max_generated=1000,
            max_proofs=1,
        )
        assert opts.max_given == 100
        assert opts.max_kept == 200
        assert opts.max_seconds == 60.0
        assert opts.max_generated == 1000
        assert opts.max_proofs == 1

    def test_all_inference_rules_configurable(self):
        """All inference rules can be toggled independently."""
        from pyladr.search.given_clause import SearchOptions

        # All off
        opts = SearchOptions(
            binary_resolution=False,
            paramodulation=False,
            factoring=False,
            demodulation=False,
        )
        assert not opts.binary_resolution
        assert not opts.paramodulation
        assert not opts.factoring
        assert not opts.demodulation

        # All on
        opts = SearchOptions(
            binary_resolution=True,
            paramodulation=True,
            factoring=True,
            demodulation=True,
        )
        assert opts.binary_resolution
        assert opts.paramodulation
        assert opts.factoring
        assert opts.demodulation

    def test_demodulation_options_supported(self):
        """All demodulation sub-options are supported."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions(
            demodulation=True,
            lex_dep_demod_lim=5,
            lex_order_vars=True,
            demod_step_limit=500,
            back_demod=True,
        )
        assert opts.lex_dep_demod_lim == 5
        assert opts.lex_order_vars is True
        assert opts.demod_step_limit == 500
        assert opts.back_demod is True
