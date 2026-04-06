"""Smoke tests to verify the project is properly set up."""

from __future__ import annotations


def test_import_pyladr() -> None:
    import pyladr

    assert pyladr.__version__ == "0.1.0"


def test_import_subpackages() -> None:
    import pyladr.core
    import pyladr.parsing
    import pyladr.inference
    import pyladr.indexing
    import pyladr.ordering
    import pyladr.search
