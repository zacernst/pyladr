"""Inference rules: resolution, paramodulation, demodulation, subsumption."""

from pyladr.inference.resolution import (
    all_binary_resolvents,
    binary_resolve,
    factor,
    is_tautology,
    merge_literals,
    renumber_variables,
)
from pyladr.inference.paramodulation import (
    orient_equalities,
    para_from_into,
    paramodulate,
    is_eq_atom,
    pos_eq,
    neg_eq,
)
from pyladr.inference.demodulation import (
    DemodType,
    DemodulatorIndex,
    back_demodulatable,
    demodulate_clause,
    demodulate_term,
    demodulator_type,
)
from pyladr.inference.subsumption import (
    back_subsume,
    back_subsume_from_lists,
    forward_subsume,
    forward_subsume_from_lists,
    subsumes,
)

__all__ = [
    "DemodType",
    "DemodulatorIndex",
    "all_binary_resolvents",
    "back_demodulatable",
    "back_subsume",
    "back_subsume_from_lists",
    "binary_resolve",
    "demodulate_clause",
    "demodulate_term",
    "demodulator_type",
    "factor",
    "forward_subsume",
    "forward_subsume_from_lists",
    "is_eq_atom",
    "is_tautology",
    "merge_literals",
    "neg_eq",
    "orient_equalities",
    "para_from_into",
    "paramodulate",
    "pos_eq",
    "renumber_variables",
    "subsumes",
]
