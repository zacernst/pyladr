"""Tests ensuring ML features are completely opt-in.

Validates that:
1. SearchOptions defaults have NO ML features enabled
2. Constructing search objects without ML args works identically
3. ML modules can be absent without affecting core functionality
4. Default clause selection is unchanged when ML is not configured
5. No ML imports occur in hot paths unless explicitly enabled

Run with: pytest tests/compatibility/test_ml_opt_in.py -v
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestDefaultBehaviorUnchanged:
    """Verify that default SearchOptions and behavior have no ML influence."""

    def test_search_options_defaults_no_ml(self):
        """SearchOptions() must not have any ML-related fields enabled by default."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()
        assert opts.binary_resolution is True
        assert opts.factoring is True
        assert opts.paramodulation is False
        assert opts.quiet is False
        assert opts.parallel is None

        # Verify no ML-related attributes snuck in as enabled defaults
        for attr_name in dir(opts):
            if attr_name.startswith("_"):
                continue
            if any(kw in attr_name.lower() for kw in ("ml_", "embed", "neural", "gnn", "graph_")):
                val = getattr(opts, attr_name)
                # If ML fields exist, they must default to disabled/None/False
                # Dimension, rate, and dump-path parameters are configuration values,
                # not activation flags — non-zero defaults are expected.
                if attr_name.endswith("_dim") or attr_name.endswith("_rate") or attr_name.endswith("_embeddings"):
                    continue
                if "dump_embeddings" in attr_name:
                    continue
                assert val is None or val is False or val == 0, (
                    f"ML-related option '{attr_name}' has non-disabled default: {val}"
                )

    def test_search_without_ml_args(self):
        """GivenClauseSearch works without any ML arguments."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c1 = Clause(literals=(Literal(sign=True, atom=P),))
        c2 = Clause(literals=(Literal(sign=False, atom=P),))

        # Must work with plain SearchOptions() -- no ML
        opts = SearchOptions(max_given=50, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT

    def test_selection_strategy_default_no_ml(self):
        """Default clause selection uses weight-based strategy, not ML."""
        from pyladr.search.selection import GivenSelection

        sel = GivenSelection()
        # Selection should not reference any ML/embedding concepts
        assert not hasattr(sel, "embedding_model")
        assert not hasattr(sel, "graph_encoder")


class TestMLModuleAbsence:
    """Verify core functionality when ML modules are completely absent."""

    def test_core_imports_without_torch(self):
        """All core pyladr modules import without torch installed."""
        core_modules = [
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
            "pyladr.indexing.discrimination_tree",
            "pyladr.indexing.feature_index",
            "pyladr.ordering.lrpo",
            "pyladr.ordering.kbo",
            "pyladr.parsing.ladr_parser",
            "pyladr.parsing.tokenizer",
        ]

        # Temporarily hide torch to ensure it's not a hard dependency
        hidden = {}
        for mod_name in list(sys.modules.keys()):
            if mod_name == "torch" or mod_name.startswith("torch."):
                hidden[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(sys.modules, {"torch": None, "torch_geometric": None}):
                for mod_path in core_modules:
                    # Force reimport
                    if mod_path in sys.modules:
                        del sys.modules[mod_path]
                    try:
                        importlib.import_module(mod_path)
                    except ImportError as e:
                        if "torch" in str(e).lower():
                            pytest.fail(
                                f"Core module {mod_path} has hard dependency on torch: {e}"
                            )
                        raise
        finally:
            # Restore torch modules
            sys.modules.update(hidden)

    def test_search_runs_without_ml_packages(self):
        """Full search completes without any ML packages available."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term, get_variable_term
        from pyladr.search.given_clause import ExitCode, GivenClauseSearch, SearchOptions

        P_sn, Q_sn, a_sn = 1, 2, 3
        a = get_rigid_term(a_sn, 0)
        x = get_variable_term(0)

        c1 = Clause(literals=(Literal(sign=True, atom=get_rigid_term(P_sn, 1, (a,))),))
        c2 = Clause(literals=(
            Literal(sign=False, atom=get_rigid_term(P_sn, 1, (x,))),
            Literal(sign=True, atom=get_rigid_term(Q_sn, 1, (x,))),
        ))
        c3 = Clause(literals=(Literal(sign=False, atom=get_rigid_term(Q_sn, 1, (a,))),))

        opts = SearchOptions(binary_resolution=True, factoring=True, max_given=50, quiet=True)
        search = GivenClauseSearch(options=opts)
        result = search.run(usable=[], sos=[c1, c2, c3])
        assert result.exit_code == ExitCode.MAX_PROOFS_EXIT


class TestMLGracefulDegradation:
    """Verify ML components degrade gracefully when unavailable."""

    def test_noop_provider_satisfies_protocol(self):
        """NoOpEmbeddingProvider must satisfy the runtime_checkable EmbeddingProvider protocol."""
        from pyladr.ml.embedding_provider import NoOpEmbeddingProvider
        from pyladr.protocols import EmbeddingProvider

        noop = NoOpEmbeddingProvider()
        assert isinstance(noop, EmbeddingProvider)

    def test_create_factory_falls_back_to_noop_without_torch(self):
        """create_embedding_provider returns NoOp when torch is unavailable."""
        import pyladr.ml.embedding_provider as ep_mod

        # Temporarily patch _ML_AVAILABLE to False to simulate torch absence
        original = ep_mod._ML_AVAILABLE
        try:
            ep_mod._ML_AVAILABLE = False
            provider = ep_mod.create_embedding_provider()
            assert isinstance(provider, ep_mod.NoOpEmbeddingProvider)
        finally:
            ep_mod._ML_AVAILABLE = original

    def test_noop_provider_embedding_dim_propagates(self):
        """NoOp provider respects custom embedding_dim from config."""
        from pyladr.ml.embedding_provider import NoOpEmbeddingProvider

        noop = NoOpEmbeddingProvider(embedding_dim=1024)
        assert noop.embedding_dim == 1024
        # Batch still returns correct-length None list
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import get_rigid_term

        P = get_rigid_term(1, 1, (get_rigid_term(2, 0),))
        c = Clause(literals=(Literal(sign=True, atom=P),))
        assert noop.get_embedding(c) is None
        assert noop.get_embeddings_batch([c, c, c]) == [None, None, None]


class TestMLFeatureGating:
    """Verify ML features are properly gated behind explicit opt-in."""

    def test_no_ml_imports_in_core_search_path(self):
        """The core search hot path must not import ML modules."""
        import pyladr.search.given_clause as gc_mod
        import pyladr.search.selection as sel_mod
        import pyladr.search.state as state_mod

        source_modules = [gc_mod, sel_mod, state_mod]

        for mod in source_modules:
            source = importlib.util.find_spec(mod.__name__)
            if source and source.origin:
                with open(source.origin) as f:
                    content = f.read()

                # Check for unconditional torch imports at module level
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("\"\"\""):
                        continue
                    if "import torch" in stripped and "try:" not in content[max(0, content.find(stripped) - 50):content.find(stripped)]:
                        # Allow conditional imports inside try/except or if blocks
                        if not any(
                            kw in content[max(0, content.find(stripped) - 200):content.find(stripped)]
                            for kw in ("try:", "if ", "TYPE_CHECKING")
                        ):
                            pytest.fail(
                                f"{mod.__name__}:{i} has unconditional 'import torch': {stripped}"
                            )
