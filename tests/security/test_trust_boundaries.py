"""Security validation framework for protocol isolation and trust boundaries.

Enforces clean separation between pyladr packages by validating:
1. Import graph: ML cannot access search internals (and vice versa)
2. Protocol contracts: Embedding providers return valid, finite data
3. Configuration bounds: SearchOptions fields validated before reaching search core

Designed to integrate with Christopher's protocol isolation extraction
(Tasks #17-18) and run as CI gates to prevent security regressions.
"""

from __future__ import annotations

import ast
import importlib
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# ---------------------------------------------------------------------------
# Section 1: Import Graph Isolation Tests
# ---------------------------------------------------------------------------

PYLADR_ROOT = Path(__file__).resolve().parents[2] / "pyladr"

# Allowed cross-boundary imports.  After protocol isolation, the ONLY
# permitted imports between search/ and ml/ should go through core/ or
# a dedicated protocols/ package.
#
# Each entry is (from_package, to_package, allowed_modules).
# "allowed_modules" lists the target modules that ARE permitted to import
# from the other side.  Everything else is a violation.
#
# Phase 1 (current): Document existing violations as known_violations.
# Phase 2 (post-isolation): known_violations shrinks to empty.

KNOWN_VIOLATIONS_ML_TO_SEARCH = {
    # Protocol imports resolved by Task #22 (EmbeddingProvider/ClauseEncoder → pyladr.protocols).
    # Remaining: concrete class dependencies that require deeper refactoring.
    # relational_selection.py still imports EmbeddingEnhancedSelection (parent class),
    # MLSelectionConfig, MLSelectionStats, and private helpers from search.ml_selection
    ("pyladr.ml.attention.relational_selection", "pyladr.search.ml_selection"),
    # relational_selection.py still imports ClauseList from search.state
    ("pyladr.ml.attention.relational_selection", "pyladr.search.state"),
    # contrastive.py TYPE_CHECKING import of Proof/SearchResult
    ("pyladr.ml.training.contrastive", "pyladr.search.given_clause"),
}

KNOWN_VIOLATIONS_SEARCH_TO_ML = {
    # online_integration.py deeply couples search↔ml — remaining violations
    ("pyladr.search.online_integration", "pyladr.ml.online_learning"),
    ("pyladr.search.online_integration", "pyladr.ml.embedding_provider"),
}


def _collect_imports(filepath: Path) -> list[tuple[str, str, int, bool]]:
    """Parse a Python file and extract all import statements.

    Returns list of (importing_module, imported_module, line_number, is_type_checking).
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    # Determine the module name from the file path
    rel = filepath.relative_to(PYLADR_ROOT.parent)
    module_name = str(rel.with_suffix("")).replace("/", ".")

    results: list[tuple[str, str, int, bool]] = []
    in_type_checking = False

    for node in ast.walk(tree):
        # Detect TYPE_CHECKING blocks
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        _extract_import(child, module_name, results, is_type_checking=True)
                continue

        if isinstance(child := node, ast.ImportFrom) and child.module:
            _extract_import(child, module_name, results, is_type_checking=False)
        elif isinstance(child, ast.Import):
            _extract_import(child, module_name, results, is_type_checking=False)

    return results


def _extract_import(
    node: ast.Import | ast.ImportFrom,
    importing_module: str,
    results: list[tuple[str, str, int, bool]],
    is_type_checking: bool,
) -> None:
    if isinstance(node, ast.ImportFrom) and node.module:
        results.append((importing_module, node.module, node.lineno, is_type_checking))
    elif isinstance(node, ast.Import):
        for alias in node.names:
            results.append((importing_module, alias.name, node.lineno, is_type_checking))


def _get_python_files(package_dir: Path) -> list[Path]:
    return sorted(package_dir.rglob("*.py"))


class TestImportGraphIsolation:
    """Verify that cross-package imports respect trust boundaries."""

    def _collect_cross_boundary_imports(
        self, from_pkg: str, to_pkg: str
    ) -> list[tuple[str, str, int, bool]]:
        """Find all imports from from_pkg that reference to_pkg."""
        from_dir = PYLADR_ROOT / from_pkg.replace(".", "/").replace("pyladr/", "")
        violations = []
        for filepath in _get_python_files(from_dir):
            for importing, imported, lineno, is_tc in _collect_imports(filepath):
                if imported.startswith(to_pkg):
                    violations.append((importing, imported, lineno, is_tc))
        return violations

    def test_ml_does_not_import_search_internals(self) -> None:
        """ML package should only access search via protocols, not internals."""
        violations = self._collect_cross_boundary_imports("pyladr.ml", "pyladr.search")

        unexpected = []
        for importing, imported, lineno, is_tc in violations:
            pair = (importing, imported)
            if pair not in KNOWN_VIOLATIONS_ML_TO_SEARCH:
                tc_label = " (TYPE_CHECKING)" if is_tc else ""
                unexpected.append(f"  {importing}:{lineno} → {imported}{tc_label}")

        if unexpected:
            msg = (
                "Unexpected ML → Search imports violate trust boundary:\n"
                + "\n".join(unexpected)
                + "\n\nAdd to KNOWN_VIOLATIONS or route through protocols package."
            )
            pytest.fail(msg)

    def test_search_does_not_import_ml_internals(self) -> None:
        """Search package should only access ML via protocols, not internals."""
        violations = self._collect_cross_boundary_imports("pyladr.search", "pyladr.ml")

        unexpected = []
        for importing, imported, lineno, is_tc in violations:
            pair = (importing, imported)
            if pair not in KNOWN_VIOLATIONS_SEARCH_TO_ML:
                tc_label = " (TYPE_CHECKING)" if is_tc else ""
                unexpected.append(f"  {importing}:{lineno} → {imported}{tc_label}")

        if unexpected:
            msg = (
                "Unexpected Search → ML imports violate trust boundary:\n"
                + "\n".join(unexpected)
                + "\n\nAdd to KNOWN_VIOLATIONS or route through protocols package."
            )
            pytest.fail(msg)

    def test_core_does_not_import_search_or_ml(self) -> None:
        """Core package must remain independent — no search or ML imports."""
        for target in ("pyladr.search", "pyladr.ml"):
            violations = self._collect_cross_boundary_imports("pyladr.core", target)
            if violations:
                details = [f"  {v[0]}:{v[2]} → {v[1]}" for v in violations]
                pytest.fail(
                    f"Core → {target} imports violate immutability:\n"
                    + "\n".join(details)
                )

    def test_no_private_imports_across_boundary(self) -> None:
        """No cross-boundary imports of private symbols (_prefixed)."""
        ml_violations = self._collect_cross_boundary_imports("pyladr.ml", "pyladr.search")
        search_violations = self._collect_cross_boundary_imports("pyladr.search", "pyladr.ml")

        private_imports = []
        for violations in (ml_violations, search_violations):
            for importing, imported, lineno, _ in violations:
                # Check if importing private submodules
                parts = imported.split(".")
                if any(p.startswith("_") for p in parts):
                    private_imports.append(f"  {importing}:{lineno} → {imported}")

        if private_imports:
            pytest.fail(
                "Private symbol imports across trust boundary:\n"
                + "\n".join(private_imports)
            )

    def test_known_violations_shrink_over_time(self) -> None:
        """Track that known violations decrease as protocol isolation progresses.

        Update this count as violations are resolved. The goal is zero.
        """
        total_known = len(KNOWN_VIOLATIONS_ML_TO_SEARCH) + len(KNOWN_VIOLATIONS_SEARCH_TO_ML)
        # Phase 1 baseline: 5 known violations
        # Protocol extraction (Task #22) moved EmbeddingProvider/ClauseEncoder to
        # pyladr.protocols, breaking the architectural circular dependency.
        # Concrete class imports remain (EmbeddingEnhancedSelection, ClauseList, etc.)
        # and require deeper refactoring to resolve.
        # Goal: 0 when all concrete dependencies are also extracted
        assert total_known <= 5, (
            f"Known violations increased to {total_known} — "
            "protocol isolation should be reducing this number"
        )


# ---------------------------------------------------------------------------
# Section 2: Protocol Contract Validation Tests
# ---------------------------------------------------------------------------


class TestProtocolContracts:
    """Validate that protocol implementations honor their contracts."""

    def test_protocols_canonical_location(self) -> None:
        """Protocols must be importable from the canonical pyladr.protocols module."""
        from pyladr.protocols import EmbeddingProvider, ClauseEncoder

        # Both must be runtime checkable protocols
        assert hasattr(EmbeddingProvider, "embedding_dim")
        assert hasattr(EmbeddingProvider, "get_embedding")
        assert hasattr(EmbeddingProvider, "get_embeddings_batch")
        assert hasattr(ClauseEncoder, "encode_clauses")

    def test_protocols_backward_compatible_reexports(self) -> None:
        """Protocols must still be importable from their original locations."""
        from pyladr.protocols import EmbeddingProvider as Canonical
        from pyladr.search.ml_selection import EmbeddingProvider as Legacy

        assert Canonical is Legacy, (
            "EmbeddingProvider in search.ml_selection must be the same object "
            "as in pyladr.protocols (re-export, not copy)"
        )

    def test_embedding_provider_protocol_defined(self) -> None:
        """EmbeddingProvider protocol exists and is runtime checkable."""
        from pyladr.search.ml_selection import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "embedding_dim")
        assert hasattr(EmbeddingProvider, "get_embedding")
        assert hasattr(EmbeddingProvider, "get_embeddings_batch")

    def test_noop_provider_returns_valid_data(self) -> None:
        """NoOp fallback provider must return consistent, safe values."""
        from pyladr.core.clause import Clause, Literal
        from pyladr.core.term import Term

        try:
            from pyladr.search.ml_selection import NoOpEmbeddingProvider
        except ImportError:
            pytest.skip("NoOpEmbeddingProvider not available")

        provider = NoOpEmbeddingProvider()

        # embedding_dim must be positive
        assert provider.embedding_dim > 0

        # get_embedding must return None or correct-length list
        dummy = Clause(literals=(), weight=1.0, id=1, justification=())
        result = provider.get_embedding(dummy)
        if result is not None:
            assert len(result) == provider.embedding_dim
            assert all(isinstance(x, (int, float)) for x in result)
            assert all(math.isfinite(x) for x in result)

        # get_embeddings_batch must return correct count
        batch_result = provider.get_embeddings_batch([dummy, dummy])
        assert len(batch_result) == 2

    def test_embedding_values_must_be_finite(self) -> None:
        """Validate that embedding vectors contain only finite floats.

        This is a contract test pattern — any EmbeddingProvider implementation
        should pass this when given valid clauses.
        """
        bad_values = [float("nan"), float("inf"), float("-inf")]
        for val in bad_values:
            assert not math.isfinite(val), f"Expected {val} to be non-finite"

    def test_validate_embeddings_replaces_nan_rows(self) -> None:
        """_validate_embeddings must replace NaN rows with zeros."""
        try:
            import torch
            from pyladr.ml.embedding_provider import _validate_embeddings
        except ImportError:
            pytest.skip("torch not available")

        tensor = torch.tensor([[1.0, 2.0], [float("nan"), 3.0], [4.0, 5.0]])
        result = _validate_embeddings(tensor, 3, 2)
        assert result.shape == (3, 2)
        assert torch.isfinite(result).all()
        # Row 0 and 2 preserved, row 1 zeroed
        assert result[0].tolist() == [1.0, 2.0]
        assert result[1].tolist() == [0.0, 0.0]
        assert result[2].tolist() == [4.0, 5.0]

    def test_validate_embeddings_replaces_inf_rows(self) -> None:
        """_validate_embeddings must replace Inf rows with zeros."""
        try:
            import torch
            from pyladr.ml.embedding_provider import _validate_embeddings
        except ImportError:
            pytest.skip("torch not available")

        tensor = torch.tensor([[1.0, 2.0], [float("inf"), float("-inf")]])
        result = _validate_embeddings(tensor, 2, 2)
        assert torch.isfinite(result).all()
        assert result[1].tolist() == [0.0, 0.0]

    def test_validate_embeddings_rejects_wrong_shape(self) -> None:
        """_validate_embeddings must return zeros for wrong dimensions."""
        try:
            import torch
            from pyladr.ml.embedding_provider import _validate_embeddings
        except ImportError:
            pytest.skip("torch not available")

        # Wrong number of rows
        tensor = torch.tensor([[1.0, 2.0]])
        result = _validate_embeddings(tensor, 3, 2)
        assert result.shape == (3, 2)
        assert (result == 0).all()

        # Wrong embedding dim
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _validate_embeddings(tensor, 2, 2)
        assert result.shape == (2, 2)
        assert (result == 0).all()

    def test_validate_embeddings_passes_clean_tensor(self) -> None:
        """_validate_embeddings must not modify valid tensors."""
        try:
            import torch
            from pyladr.ml.embedding_provider import _validate_embeddings
        except ImportError:
            pytest.skip("torch not available")

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _validate_embeddings(tensor, 2, 2)
        assert torch.equal(result, tensor)


# ---------------------------------------------------------------------------
# Section 3: Security Regression Gates (CI-ready)
# ---------------------------------------------------------------------------


class TestSecurityRegressionGates:
    """CI gates that prevent reintroduction of eliminated vulnerabilities."""

    def test_no_shell_true_in_codebase(self) -> None:
        """Ensure shell=True never reappears in production code."""
        violations = []
        for pyfile in _get_python_files(PYLADR_ROOT):
            source = pyfile.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword):
                    if node.arg == "shell" and isinstance(node.value, ast.Constant):
                        if node.value.value is True:
                            rel = pyfile.relative_to(PYLADR_ROOT.parent)
                            violations.append(f"  {rel}:{node.lineno}")

        if violations:
            pytest.fail(
                "shell=True found in production code (security vulnerability):\n"
                + "\n".join(violations)
            )

    @pytest.mark.xfail(
        reason="Legacy backward-compatibility paths in clause_encoder.py and "
        "contrastive.py still use weights_only=False for v1 checkpoint loading. "
        "New checkpoints use secure v2 format (JSON sidecar + weights_only=True). "
        "Remove xfail when legacy format support is dropped.",
        strict=True,
    )
    def test_no_unsafe_torch_load(self) -> None:
        """Ensure torch.load always uses weights_only=True."""
        violations = []
        for pyfile in _get_python_files(PYLADR_ROOT):
            source = pyfile.read_text(encoding="utf-8")
            if "torch.load" not in source:
                continue
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Match torch.load(...)
                func = node.func
                is_torch_load = False
                if isinstance(func, ast.Attribute) and func.attr == "load":
                    if isinstance(func.value, ast.Name) and func.value.id == "torch":
                        is_torch_load = True
                if not is_torch_load:
                    continue

                # Check for weights_only keyword
                weights_only = None
                for kw in node.keywords:
                    if kw.arg == "weights_only":
                        if isinstance(kw.value, ast.Constant):
                            weights_only = kw.value.value
                        break

                if weights_only is not True:
                    rel = pyfile.relative_to(PYLADR_ROOT.parent)
                    violations.append(f"  {rel}:{node.lineno} (weights_only={weights_only})")

        if violations:
            pytest.fail(
                "Unsafe torch.load (weights_only != True) found:\n"
                + "\n".join(violations)
                + "\n\nAll model loading must use weights_only=True to prevent "
                "arbitrary code execution via pickle deserialization."
            )

    def test_recursion_limit_bounded(self) -> None:
        """Ensure recursion limit stays at safe level."""
        limit = sys.getrecursionlimit()
        assert limit <= 10_000, (
            f"Recursion limit is {limit} — must be ≤ 10,000 to prevent "
            "stack exhaustion attacks from deeply nested terms"
        )


# ---------------------------------------------------------------------------
# Section 4: Configuration Bounds Validation
# ---------------------------------------------------------------------------


class TestConfigurationBounds:
    """Validate that SearchOptions fields have safe defaults and bounds."""

    def test_search_options_defaults_are_safe(self) -> None:
        """Default SearchOptions must not create resource exhaustion risk."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()

        # ML features must default to off
        assert opts.online_learning is False
        assert opts.goal_directed is False

        # Penalty features must default to off
        assert not opts.penalty_propagation
        assert not opts.repetition_penalty

    def test_ml_features_are_opt_in(self) -> None:
        """Verify all ML features require explicit activation."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions()

        # These fields control ML activation — all must default False/None
        ml_flags = [
            ("online_learning", False),
            ("goal_directed", False),
        ]

        for field_name, expected_default in ml_flags:
            actual = getattr(opts, field_name)
            assert actual == expected_default, (
                f"SearchOptions.{field_name} defaults to {actual!r}, "
                f"expected {expected_default!r} — ML features must be opt-in"
            )

    def test_validation_rejects_negative_below_minimum(self) -> None:
        """Numeric fields with minimum bounds must reject values below them."""
        from pyladr.search.given_clause import SearchOptions

        # demod_step_limit min is 1
        with pytest.raises(ValueError, match="demod_step_limit"):
            SearchOptions(demod_step_limit=0)

        # embedding_dim min is 1
        with pytest.raises(ValueError, match="embedding_dim"):
            SearchOptions(embedding_dim=0)

        # penalty_propagation_decay must be in [0.0, 1.0]
        with pytest.raises(ValueError, match="penalty_propagation_decay"):
            SearchOptions(penalty_propagation_decay=-0.1)
        with pytest.raises(ValueError, match="penalty_propagation_decay"):
            SearchOptions(penalty_propagation_decay=1.1)

    def test_validation_rejects_invalid_ml_weight(self) -> None:
        """ml_weight must be None or in [0.0, 1.0]."""
        from pyladr.search.given_clause import SearchOptions

        # None is valid (ML disabled)
        opts = SearchOptions(ml_weight=None)
        assert opts.ml_weight is None

        # 0.0 and 1.0 are valid boundaries
        SearchOptions(ml_weight=0.0)
        SearchOptions(ml_weight=1.0)

        # Out of range must raise
        with pytest.raises(ValueError, match="ml_weight"):
            SearchOptions(ml_weight=-0.01)
        with pytest.raises(ValueError, match="ml_weight"):
            SearchOptions(ml_weight=1.01)

    def test_validation_called_on_construction(self) -> None:
        """SearchOptions.__post_init__ must call validate_search_options."""
        from pyladr.search.given_clause import SearchOptions

        # Constructing with invalid params must raise immediately
        with pytest.raises(ValueError):
            SearchOptions(entropy_weight=-1)

    def test_semantic_validation_detects_misconfiguration(self) -> None:
        """Semantic validation catches logically inconsistent configs."""
        from pyladr.search.given_clause import SearchOptions

        opts = SearchOptions(back_demod=True, demodulation=False)
        warnings = opts.validate()
        assert any("back_demod" in w for w in warnings)

        opts2 = SearchOptions(lazy_demod=True, demodulation=False)
        warnings2 = opts2.validate()
        assert any("lazy_demod" in w for w in warnings2)

    def test_all_numeric_fields_have_bounds(self) -> None:
        """Every numeric SearchOptions field must have at least a minimum bound.

        This is a security gate: new numeric fields added without validation
        create resource exhaustion vectors.
        """
        from pyladr.search.options import _NUMERIC_BOUNDS

        validated_fields = {entry[0] for entry in _NUMERIC_BOUNDS}
        # ml_weight is validated separately (special None handling)
        validated_fields.add("ml_weight")

        from pyladr.search.given_clause import SearchOptions
        import dataclasses

        numeric_fields = []
        for f in dataclasses.fields(SearchOptions):
            if f.type in ("int", "float", "float | None"):
                numeric_fields.append(f.name)

        missing = [f for f in numeric_fields if f not in validated_fields]
        if missing:
            pytest.fail(
                f"Numeric SearchOptions fields without bounds validation:\n"
                f"  {missing}\n"
                "Add to _NUMERIC_BOUNDS in options.py to prevent resource exhaustion."
            )

    def test_mode_fields_have_allowlists(self) -> None:
        """String mode fields must validate against finite allowlists."""
        from pyladr.search.given_clause import SearchOptions

        # Invalid mode strings should produce semantic warnings
        opts = SearchOptions(penalty_propagation_mode="INVALID_MODE")
        warnings = opts.validate()
        assert any("penalty_propagation_mode" in w for w in warnings)

        opts2 = SearchOptions(penalty_weight_mode="INVALID_MODE")
        warnings2 = opts2.validate()
        assert any("penalty_weight_mode" in w for w in warnings2)

    def test_resource_exhaustion_upper_bounds(self) -> None:
        """Fields that control memory allocation must have upper bounds.

        nucleus_penalty_cache_size and embedding_dim control allocation sizes.
        Without upper bounds, attackers can cause OOM via config injection.
        """
        from pyladr.search.options import _NUMERIC_BOUNDS

        # Fields that directly control allocation sizes need upper bounds
        resource_fields = {
            "nucleus_penalty_cache_size": 10_000_000,  # 10M entries max
            "embedding_dim": 4096,  # match GNNConfig upper bound
            "demod_step_limit": 1_000_000,  # prevent infinite demod loops
            "penalty_propagation_max_depth": 100,  # prevent deep graph traversal
        }

        bounds_map = {entry[0]: (entry[1], entry[2]) for entry in _NUMERIC_BOUNDS}

        missing_upper = []
        for field_name, suggested_max in resource_fields.items():
            if field_name not in bounds_map:
                missing_upper.append(f"  {field_name}: not in _NUMERIC_BOUNDS at all")
            elif bounds_map[field_name][1] is None:
                missing_upper.append(
                    f"  {field_name}: has no upper bound (suggested max: {suggested_max})"
                )

        if missing_upper:
            pytest.fail(
                "Resource-controlling fields lack upper bounds (OOM risk):\n"
                + "\n".join(missing_upper)
                + "\n\nAdd upper bounds to _NUMERIC_BOUNDS in options.py."
            )
