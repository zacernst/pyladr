"""SearchOptions and validation for search configuration.

Provides:
1. SearchOptions — the main mutable search configuration dataclass
2. Frozen sub-config dataclasses that group related SearchOptions fields
3. Bounds validation for all numeric parameters
4. Semantic validation for cross-field constraints

Sub-configs are used for structured access and documentation. SearchOptions
itself remains mutable for LADR assign()/set()/clear() directive compatibility.

Note: some search-related config classes use the *Config suffix (e.g.
MLSelectionConfig, GoalDirectedConfig, OnlineIntegrationConfig) while the
main search class uses *Options (SearchOptions). These should be normalized
to *Config in a future refactor.

Usage:
    from pyladr.search.options import SearchOptions, validate_search_options

    opts = SearchOptions(max_given=100, penalty_propagation_decay=0.5)
    errors = validate_search_options(opts)  # [] if valid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyladr.parallel.inference_engine import ParallelSearchConfig


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
class NucleusPenaltyConfig:
    """Nucleus unification penalty configuration."""

    enabled: bool = False
    threshold: float = 3.0
    weight: float = 1.5
    max_penalty: float = 15.0
    cache_size: int = 10000


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


# ── Main search options ──────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchOptions:
    """Search options matching C search options.

    Controls inference rules, limits, and output.
    """

    # Inference rules
    binary_resolution: bool = True
    paramodulation: bool = False
    hyper_resolution: bool = False
    factoring: bool = True
    para_into_vars: bool = False

    # Limits (C: max_given, max_kept, max_seconds, max_generated, max_proofs, max_weight)
    max_given: int = -1       # -1 = no limit
    max_kept: int = -1
    max_seconds: float = -1.0
    max_generated: int = -1
    max_proofs: int = 1
    max_weight: float = -1.0  # -1 = no limit
    max_weight_tighten_after: int = 0    # tighten max_weight after N given clauses (0 = disabled)
    max_weight_tighten_to: float = -1.0  # new max_weight cap after the threshold is reached

    # Demodulation
    demodulation: bool = False
    lex_dep_demod_lim: int = 0
    lex_order_vars: bool = False
    demod_step_limit: int = 1000
    back_demod: bool = False

    # Simplification
    check_tautology: bool = True
    merge_lits: bool = True

    # Parallelization
    parallel: ParallelSearchConfig | None = None

    # Priority SOS: heap-based O(log n) weight selection + O(1) removal
    priority_sos: bool = True

    # Lazy demodulation: defer lex-dependent rewrites until selection
    lazy_demod: bool = False

    # Back-subsumption check: at given clause #N, check kept/back_subsumed ratio.
    # If ratio > 20, disable backward subsumption (C: backsub_check, default 500).
    # -1 = never check (always keep back subsumption enabled).
    backsub_check: int = 500

    # SOS limit: max clauses in SOS before culling (C: sos_limit)
    sos_limit: int = -1  # -1 = no limit

    # Weight-based selection ratio: how many times per cycle the lightest
    # clause is chosen.  Age always gets 1 slot; this controls the W slots.
    # Default 4 matches C Prover9's age_factor=5 (1 age + 4 weight = 5 total).
    weight_ratio: int = 4

    # Entropy-based selection: ratio weight for entropy selector in the cycle.
    # 0 = disabled (default). e.g., entropy_weight=2 with default ratio=5
    # gives cycle: 1 age + 4 weight + 2 entropy = 7 total.
    entropy_weight: int = 0

    # Unification penalty selection: ratio weight for penalty selector.
    # 0 = disabled (default). Prefers most specific clauses (lowest penalty).
    # e.g., unification_weight=2 gives cycle: 1 age + 4 weight + 2 penalty.
    unification_weight: int = 0

    # Penalty propagation: derived clauses inherit penalties from overly general parents.
    # Targets "unifies with everything" patterns where variables in antecedents
    # don't appear in consequents (e.g., ¬P(x) ∨ Q resolved with P(t) → Q).
    # Disabled by default for C Prover9 compatibility.
    penalty_propagation: bool = False
    penalty_propagation_mode: str = "additive"  # "additive", "multiplicative", "max"
    penalty_propagation_decay: float = 0.5      # Decay factor per generation (0.0-1.0)
    penalty_propagation_threshold: float = 5.0  # Only propagate if parent penalty >= this
    penalty_propagation_max_depth: int = 3      # Max inheritance depth (0 = unlimited)
    penalty_propagation_max: float = 20.0       # Cap on accumulated penalty

    # Subformula repetition penalty: penalizes clauses with repeated structural patterns.
    # Detects repeated subterms (e.g., multiple i(x,x)) and adds a penalty per extra occurrence.
    # Disabled by default for C Prover9 compatibility.
    repetition_penalty: bool = False
    repetition_penalty_weight: float = 2.0     # Penalty per extra occurrence of a repeated subterm
    repetition_penalty_min_size: int = 2        # Minimum subterm symbol_count to consider
    repetition_penalty_max: float = 15.0        # Cap on total repetition penalty
    repetition_penalty_normalize: bool = False  # Variable-agnostic matching (i(x,x) ≡ i(y,y))

    # Nucleus unification penalty: penalizes clauses whose negative literals act as
    # overly permissive nuclei in hyperresolution — matching too many satellites due
    # to variable-dominated argument positions. Disabled by default for C Prover9 compatibility.
    nucleus_unification_penalty: bool = False
    nucleus_penalty_threshold: float = 3.0    # Min penalty to apply (filters noise)
    nucleus_penalty_weight: float = 1.5       # Base penalty per overly general nucleus literal
    nucleus_penalty_max: float = 15.0         # Cap on total nucleus penalty
    nucleus_penalty_cache_size: int = 10000   # Max cached nucleus patterns (LRU eviction)

    # Penalty weight adjustment: inflates clause weight based on combined penalty score.
    # High-penalty clauses (overly general or structurally redundant) get pushed down
    # the selection queue by increasing their weight. Composable with penalty_propagation
    # and repetition_penalty. Disabled by default for C Prover9 compatibility.
    penalty_weight_enabled: bool = False
    penalty_weight_threshold: float = 5.0      # Only adjust if penalty >= this value
    penalty_weight_multiplier: float = 2.0     # Weight increase factor (>= 1.0)
    penalty_weight_max: float = 1000.0         # Cap on adjusted weight
    penalty_weight_mode: str = "exponential"   # "linear", "exponential", "step"

    # Hints: weight bonus for clauses matching hint patterns.
    # hint_wt is the weight assigned to matched clauses (lower = preferred).
    # A kept clause matches a hint if the hint subsumes the clause.
    hint_wt: float = 1.0

    # Output
    print_given: bool = True
    print_kept: bool = False
    print_gen: bool = False
    print_given_stats: bool = False
    quiet: bool = False

    # Machine Learning Integration
    online_learning: bool = False
    ml_weight: float | None = None  # None = auto-determine
    embedding_dim: int = 32
    goal_directed: bool = False
    goal_proximity_weight: float = 0.3
    embedding_evolution_rate: float = 0.01
    learn_from_back_subsumption: bool = False
    learn_from_forward_subsumption: bool = False

    # FORTE Embedding Integration
    forte_embeddings: bool = False
    forte_weight: float = 0.0   # selection ratio weight (0 = disabled)
    forte_embedding_dim: int = 128
    forte_cache_max_entries: int = 10_000

    # Tree2Vec Embedding Integration
    tree2vec_embeddings: bool = False
    tree2vec_weight: float = 0.0   # selection ratio weight (0 = disabled)
    tree2vec_embedding_dim: int = 64
    tree2vec_cache_max_entries: int = 10_000
    tree2vec_include_position: bool = False  # encode argument position in walk tokens
    tree2vec_include_depth: bool = False     # encode tree depth in walk tokens
    tree2vec_include_var_identity: bool = False  # De Bruijn-style variable identity in walk tokens
    tree2vec_skip_predicate: bool = True     # skip predicate wrapper, walk from args directly
    tree2vec_include_path_length: bool = True   # prepend path length token to PATH walks
    tree2vec_composition: str = "weighted_depth"  # embedding composition: mean, weighted_depth, root_concat
    tree2vec_online_learning: bool = False   # enable online tree2vec updates during search
    tree2vec_online_update_interval: int = 20  # clauses kept between online updates
    tree2vec_online_batch_size: int = 10     # max clauses per online update batch
    tree2vec_online_lr: float = 0.005        # learning rate for online skipgram updates
    tree2vec_online_max_updates: int = 0     # stop updating after N retrainings (0 = unlimited)
    tree2vec_bg_update: bool = True   # run update_online on a daemon thread (False = sync, useful for tests)
    tree2vec_dump_embeddings: str = ""  # path to write SOS embedding JSON after each training; "" = disabled
    tree2vec_goal_proximity: bool = False    # enable goal-directed tree2vec selection
    tree2vec_goal_proximity_weight: float = 0.3  # goal proximity influence weight
    tree2vec_cross_arg_proximity: bool = True  # cross-argument proximity for CD compatibility
    tree2vec_proximity_report_interval: int = 100  # given clauses between proximity trend reports
    tree2vec_maximin_weight: float = 0.0  # selection ratio weight for maximin T2V (0 = disabled)
    tree2vec_model_path: str = ""   # path to pre-trained model; "" = train on initial clauses

    # RNN2Vec Embedding Integration
    rnn2vec_embeddings: bool = False
    rnn2vec_weight: float = 0.0        # selection ratio weight (0 = disabled)
    rnn2vec_rnn_type: str = "gru"      # lstm, gru, elman
    rnn2vec_hidden_dim: int = 64
    rnn2vec_embedding_dim: int = 64
    rnn2vec_input_dim: int = 32
    rnn2vec_num_layers: int = 1
    rnn2vec_bidirectional: bool = False
    rnn2vec_composition: str = "mean"  # last_hidden, mean_pool, attention_pool
    rnn2vec_cache_max_entries: int = 10_000
    rnn2vec_online_learning: bool = False
    rnn2vec_online_update_interval: int = 20
    rnn2vec_online_batch_size: int = 10
    rnn2vec_online_lr: float = 0.001
    rnn2vec_online_max_updates: int = 0
    rnn2vec_model_path: str = ""       # pre-trained model directory; "" = train on initial clauses
    rnn2vec_save_model: str = ""       # save trained model to this directory after training; "" = don't save
    rnn2vec_training_epochs: int = 5
    rnn2vec_training_lr: float = 0.001
    rnn2vec_goal_proximity: bool = False          # enable goal-directed rnn2vec selection
    rnn2vec_goal_proximity_weight: float = 0.3   # goal proximity influence weight
    rnn2vec_random_goal_weight: float = 0.0      # selection ratio for random-goal proximity mode
    rnn2vec_dump_embeddings: str = ""            # path to write SOS embedding JSON; "" = disabled

    # Proof-guided selection: learn from successful proof patterns.
    # Requires FORTE embeddings (forte_embeddings=True).
    # Blends exploitation (similarity to proof patterns) with exploration (diversity).
    proof_guided: bool = False
    proof_guided_weight: float = 0.0               # Selection ratio weight (0 = disabled in cycle)
    proof_guided_exploitation_ratio: float = 0.7   # 0.0=pure exploration, 1.0=pure exploitation
    proof_guided_max_patterns: int = 500            # Max stored proof pattern embeddings
    proof_guided_decay_rate: float = 0.95           # Exponential decay per proof event (0,1]
    proof_guided_min_similarity: float = 0.1        # Minimum cosine similarity threshold
    proof_guided_warmup_proofs: int = 1             # Proofs required before activation

    def __post_init__(self) -> None:
        """Validate option bounds and cross-field constraints."""
        errors = validate_search_options(self)
        if errors:
            raise ValueError(
                "Invalid SearchOptions:\n  " + "\n  ".join(errors)
            )

    def validate(self) -> list[str]:
        """Re-validate after mutation (e.g. LADR assign directives).

        Returns list of error and warning strings, empty if valid.
        Includes both bounds errors and semantic warnings.
        """
        return validate_search_options(self) + validate_search_options_semantic(self)


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
    ("max_weight_tighten_after", 0, None, "given-clause threshold to tighten max_weight"),
    ("max_weight_tighten_to", -1.0, None, "new max_weight cap after threshold"),
    ("sos_limit", -1, None, "SOS limit"),
    ("backsub_check", -1, None, "backsub check threshold (-1=never)"),
    # Demodulation
    ("lex_dep_demod_lim", 0, None, "lex-dep demod limit"),
    ("demod_step_limit", 1, 1_000_000, "demod step limit"),
    # Selection weights
    ("weight_ratio", 0, None, "weight-to-age selection ratio"),
    ("hint_wt", 0.0, None, "weight assigned to hint-matched clauses"),
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
