"""SearchOptions validation and sub-configuration types.

Provides:
1. Frozen sub-config dataclasses that group related SearchOptions fields
2. Bounds validation for all numeric parameters
3. Semantic validation for cross-field constraints

Sub-configs are used for structured access and documentation. SearchOptions
itself remains mutable for LADR assign()/set()/clear() directive compatibility.

Note: some search-related config classes use the *Config suffix (e.g.
MLSelectionConfig, GoalDirectedConfig, OnlineIntegrationConfig) while the
main search class uses *Options (SearchOptions). These should be normalized
to *Config in a future refactor.

Usage:
    from pyladr.search.given_clause import SearchOptions
    from pyladr.search.options import validate_search_options

    opts = SearchOptions(max_given=100, penalty_propagation_decay=0.5)
    errors = validate_search_options(opts)  # [] if valid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ── Sub-configuration types ──────────────────────────────────────────────────
#
# These frozen dataclasses group related SearchOptions fields for type
# documentation and structured access. They are NOT used as SearchOptions
# fields — SearchOptions keeps its flat layout for backward compatibility.


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Inference rule configuration."""

    binary_resolution: bool = True
    paramodulation: bool = False
    hyper_resolution: bool = False
    factoring: bool = True
    para_into_vars: bool = False


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Search limit configuration. -1 means no limit."""

    max_given: int = -1
    max_kept: int = -1
    max_seconds: float = -1.0
    max_generated: int = -1
    max_proofs: int = 1
    max_weight: float = -1.0


@dataclass(frozen=True, slots=True)
class DemodulationConfig:
    """Demodulation and rewriting configuration."""

    demodulation: bool = False
    lex_dep_demod_lim: int = 0
    lex_order_vars: bool = False
    demod_step_limit: int = 1000
    back_demod: bool = False


@dataclass(frozen=True, slots=True)
class SimplificationConfig:
    """Clause simplification configuration."""

    check_tautology: bool = True
    merge_lits: bool = True


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Clause selection strategy configuration."""

    priority_sos: bool = True
    lazy_demod: bool = False
    sos_limit: int = -1
    entropy_weight: int = 0
    unification_weight: int = 0


@dataclass(frozen=True, slots=True)
class PenaltyPropagationConfig:
    """Penalty propagation configuration for overly general clauses."""

    enabled: bool = False
    mode: str = "additive"
    decay: float = 0.5
    threshold: float = 5.0
    max_depth: int = 3
    max_penalty: float = 20.0


@dataclass(frozen=True, slots=True)
class RepetitionPenaltyConfig:
    """Subformula repetition penalty configuration."""

    enabled: bool = False
    weight: float = 2.0
    min_size: int = 2
    max_penalty: float = 15.0
    normalize: bool = False


@dataclass(frozen=True, slots=True)
class NucleusPenaltyConfig:
    """Nucleus unification penalty configuration."""

    enabled: bool = False
    threshold: float = 3.0
    weight: float = 1.5
    max_penalty: float = 15.0
    cache_size: int = 10000


@dataclass(frozen=True, slots=True)
class PenaltyWeightConfig:
    """Penalty-based weight adjustment configuration."""

    enabled: bool = False
    threshold: float = 5.0
    multiplier: float = 2.0
    max_weight: float = 1000.0
    mode: str = "exponential"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output and display configuration."""

    print_given: bool = True
    print_kept: bool = False
    print_gen: bool = False
    print_given_stats: bool = False
    quiet: bool = False


@dataclass(frozen=True, slots=True)
class MLConfig:
    """Machine learning integration configuration."""

    online_learning: bool = False
    ml_weight: float | None = None
    embedding_dim: int = 32
    goal_directed: bool = False
    goal_proximity_weight: float = 0.3
    embedding_evolution_rate: float = 0.01


# ── Validation ───────────────────────────────────────────────────────────────

# Bounds specifications: (field_name, min_value, max_value, description)
# None means no bound on that side.
_NUMERIC_BOUNDS: list[tuple[str, float | None, float | None, str]] = [
    # Limits (allow -1 for "no limit")
    ("max_given", -1, None, "max given clauses"),
    ("max_kept", -1, None, "max kept clauses"),
    ("max_seconds", -1.0, None, "max search time"),
    ("max_generated", -1, None, "max generated clauses"),
    ("max_proofs", -1, None, "max proofs (-1/0=unlimited)"),
    ("max_weight", -1.0, None, "max clause weight"),
    ("sos_limit", -1, None, "SOS limit"),
    ("backsub_check", -1, None, "backsub check threshold (-1=never)"),
    # Demodulation
    ("lex_dep_demod_lim", 0, None, "lex-dep demod limit"),
    ("demod_step_limit", 1, 1_000_000, "demod step limit"),
    # Selection weights
    ("entropy_weight", 0, None, "entropy selection weight"),
    ("unification_weight", 0, None, "unification selection weight"),
    # Penalty propagation
    ("penalty_propagation_decay", 0.0, 1.0, "penalty propagation decay"),
    ("penalty_propagation_threshold", 0.0, None, "penalty propagation threshold"),
    ("penalty_propagation_max_depth", 0, 100, "penalty propagation max depth"),
    ("penalty_propagation_max", 0.0, None, "penalty propagation max"),
    # Repetition penalty
    ("repetition_penalty_weight", 0.0, None, "repetition penalty weight"),
    ("repetition_penalty_min_size", 1, None, "repetition penalty min subterm size"),
    ("repetition_penalty_max", 0.0, None, "repetition penalty max"),
    # Nucleus penalty
    ("nucleus_penalty_threshold", 0.0, None, "nucleus penalty threshold"),
    ("nucleus_penalty_weight", 0.0, None, "nucleus penalty weight"),
    ("nucleus_penalty_max", 0.0, None, "nucleus penalty max"),
    ("nucleus_penalty_cache_size", 1, 10_000_000, "nucleus penalty cache size"),
    # Penalty weight
    ("penalty_weight_threshold", 0.0, None, "penalty weight threshold"),
    ("penalty_weight_multiplier", 1.0, None, "penalty weight multiplier"),
    ("penalty_weight_max", 0.0, None, "penalty weight max"),
    # ML
    ("embedding_dim", 1, 4096, "embedding dimension"),
    ("goal_proximity_weight", 0.0, 1.0, "goal proximity weight"),
    ("embedding_evolution_rate", 0.0, 1.0, "embedding evolution rate"),
    # FORTE
    ("forte_weight", 0.0, None, "FORTE embedding selection ratio weight"),
    ("forte_embedding_dim", 1, 4096, "FORTE embedding dimension"),
    ("forte_cache_max_entries", 1, 10_000_000, "FORTE cache size"),
    # Tree2Vec
    ("tree2vec_weight", 0.0, None, "Tree2Vec embedding selection ratio weight"),
    ("tree2vec_maximin_weight", 0.0, None, "Tree2Vec maximin selection ratio weight"),
    ("tree2vec_embedding_dim", 1, 4096, "Tree2Vec embedding dimension"),
    ("tree2vec_cache_max_entries", 1, 10_000_000, "Tree2Vec cache size"),
    ("tree2vec_online_lr", 0.0001, 1.0, "Tree2Vec online learning rate"),
    ("tree2vec_online_update_interval", 1, 10_000, "Tree2Vec online update interval"),
    ("tree2vec_online_batch_size", 1, 10_000, "Tree2Vec online batch size"),
    ("tree2vec_online_max_updates", 0, None, "Tree2Vec max online updates (0=unlimited)"),
    ("tree2vec_goal_proximity_weight", 0.0, 1.0, "Tree2Vec goal proximity weight"),
    ("tree2vec_proximity_report_interval", 1, 100_000, "Tree2Vec proximity report interval"),
    # RNN2Vec
    ("rnn2vec_weight", 0.0, None, "RNN2Vec embedding selection weight"),
    ("rnn2vec_embedding_dim", 1, 4096, "RNN2Vec output embedding dimension"),
    ("rnn2vec_hidden_dim", 1, 4096, "RNN2Vec RNN hidden dimension"),
    ("rnn2vec_input_dim", 1, 4096, "RNN2Vec token embedding dimension"),
    ("rnn2vec_num_layers", 1, 10, "RNN2Vec number of RNN layers"),
    ("rnn2vec_cache_max_entries", 1, 10_000_000, "RNN2Vec embedding cache size"),
    ("rnn2vec_online_lr", 0.00001, 1.0, "RNN2Vec online learning rate"),
    ("rnn2vec_online_update_interval", 1, 10_000, "RNN2Vec online update interval"),
    ("rnn2vec_online_batch_size", 1, 10_000, "RNN2Vec online batch size"),
    ("rnn2vec_online_max_updates", 0, None, "RNN2Vec max online updates (0=unlimited)"),
    ("rnn2vec_training_epochs", 1, 1000, "RNN2Vec training epochs"),
    ("rnn2vec_training_lr", 0.00001, 1.0, "RNN2Vec initial training learning rate"),
    ("rnn2vec_goal_proximity_weight", 0.0, 1.0, "RNN2Vec goal proximity influence weight"),
    ("rnn2vec_random_goal_weight", 0.0, None, "RNN2Vec random-goal selection ratio weight"),
    # Proof-guided selection
    ("proof_guided_weight", 0.0, None, "proof-guided selection ratio weight"),
    ("proof_guided_exploitation_ratio", 0.0, 1.0, "proof-guided exploitation ratio"),
    ("proof_guided_max_patterns", 1, 100_000, "proof-guided max patterns"),
    ("proof_guided_decay_rate", 0.0, 1.0, "proof-guided decay rate"),
    ("proof_guided_min_similarity", 0.0, 1.0, "proof-guided min similarity threshold"),
    ("proof_guided_warmup_proofs", 0, 1000, "proof-guided warmup proofs"),
]

_VALID_PENALTY_PROPAGATION_MODES = frozenset({"additive", "multiplicative", "max"})
_VALID_PENALTY_WEIGHT_MODES = frozenset({"linear", "exponential", "step"})


def validate_search_options(opts: object) -> list[str]:
    """Validate SearchOptions fields, returning a list of error messages.

    Returns an empty list if all fields are valid.

    Args:
        opts: A SearchOptions instance (uses duck typing to avoid circular import).

    Returns:
        List of human-readable validation error strings.
    """
    errors: list[str] = []

    # Numeric bounds
    for field_name, lo, hi, desc in _NUMERIC_BOUNDS:
        val = getattr(opts, field_name, None)
        if val is None:
            continue  # e.g. ml_weight=None is valid
        if lo is not None and val < lo:
            errors.append(f"{field_name}: {val} < {lo} (minimum for {desc})")
        if hi is not None and val > hi:
            errors.append(f"{field_name}: {val} > {hi} (maximum for {desc})")

    ml_weight = getattr(opts, "ml_weight", None)
    if ml_weight is not None and not (0.0 <= ml_weight <= 1.0):
        errors.append(f"ml_weight: {ml_weight} not in [0.0, 1.0]")

    return errors


def validate_search_options_semantic(opts: object) -> list[str]:
    """Semantic cross-field validation returning warnings.

    These are logical inconsistencies that won't crash but indicate
    likely configuration mistakes. Not raised by __post_init__.

    Args:
        opts: A SearchOptions instance.

    Returns:
        List of human-readable warning strings.
    """
    warnings: list[str] = []

    if getattr(opts, "back_demod", False) and not getattr(opts, "demodulation", False):
        warnings.append("back_demod=True has no effect without demodulation=True")

    if getattr(opts, "lazy_demod", False) and not getattr(opts, "demodulation", False):
        warnings.append("lazy_demod=True has no effect without demodulation=True")

    mode = getattr(opts, "penalty_propagation_mode", "additive")
    if mode not in _VALID_PENALTY_PROPAGATION_MODES:
        warnings.append(
            f"penalty_propagation_mode={mode!r} unrecognized, will fall back to additive"
        )

    pw_mode = getattr(opts, "penalty_weight_mode", "exponential")
    if pw_mode not in _VALID_PENALTY_WEIGHT_MODES:
        warnings.append(
            f"penalty_weight_mode={pw_mode!r} unrecognized, will fall back to exponential"
        )

    if getattr(opts, "proof_guided", False) and not getattr(opts, "forte_embeddings", False):
        warnings.append(
            "proof_guided=True has no effect without forte_embeddings=True"
        )

    return warnings
