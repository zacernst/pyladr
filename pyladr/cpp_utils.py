"""Utilities for converting between Python and C++ term representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.core.term import Term as PyTerm


def py_term_to_cpp(t: "PyTerm"):  # returns _pyladr_core.Term
    """Recursively convert a Python Term to its C++ counterpart."""
    from pyladr._pyladr_core import Term as CppTerm
    if t.is_variable:
        return CppTerm.make_variable(t.varnum)
    cpp_args = [py_term_to_cpp(a) for a in t.args]
    return CppTerm.make_rigid(-t.private_symbol, t.arity, cpp_args)


def cpp_term_to_py(cpp_t):
    """Recursively convert a C++ Term back to a Python Term."""
    from pyladr.core.term import get_variable_term, get_rigid_term
    if cpp_t.is_variable:
        return get_variable_term(cpp_t.varnum())
    py_args = tuple(cpp_term_to_py(a) for a in cpp_t.args)
    return get_rigid_term(-cpp_t.private_symbol, cpp_t.arity, py_args)
