"""Given-clause search algorithm matching C search.c.

Implements the Otter/Prover9 given-clause algorithm:
1. Move initial clauses into usable/sos lists
2. Process initial clauses (assign IDs, index, check for proofs)
3. Main loop: select given clause → infer → process → repeat
4. Terminate on proof, SOS empty, or resource limits

The algorithm integrates:
- Clause selection (selection.py) matching C giv_select.c
- Binary resolution (inference/resolution.py) matching C resolve.c
- Forward subsumption checking
- Clause simplification (tautology removal, merge)
- Proof detection (empty clause)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from itertools import chain
from typing import Callable

from pyladr.core.clause import Clause, Justification, JustType, Literal
from pyladr.core.symbol import SymbolTable
from pyladr.inference.resolution import (
    all_binary_resolvents,
    binary_resolve,
    factor,
    is_tautology,
    merge_literals,
    renumber_variables,
)
from pyladr.inference.hyper_resolution import all_hyper_resolvents, indexed_hyper_resolution
from pyladr.inference.paramodulation import (
    is_eq_atom,
    orient_equalities,
    para_from_into,
    reset_orientation_state,
)
from pyladr.inference.demodulation import (
    DemodType,
    DemodulatorIndex,
    back_demodulatable,
    demodulate_clause,
    demodulator_type,
)
from pyladr.inference.subsumption import (
    BackSubsumptionIndex,
    back_subsume_from_lists,
    back_subsume_indexed,
    forward_subsume,
    forward_subsume_from_lists,
    subsumes,
)
from pyladr.indexing.literal_index import LiteralIndex
from pyladr.parallel.inference_engine import ParallelInferenceEngine
from pyladr.search.lazy_demod import LazyDemodState
from pyladr.search.penalty_propagation import (
    PenaltyCache,
    PenaltyCombineMode,
    PenaltyPropagationConfig,
    compute_and_cache_penalty,
)
from pyladr.search.penalty_weight import (
    PenaltyWeightConfig,
    PenaltyWeightMode,
    penalty_adjusted_weight,
)
from pyladr.search.nucleus_penalty import (
    NucleusUnificationPenaltyConfig,
    compute_nucleus_unification_penalty,
)
from pyladr.search.nucleus_patterns import NucleusPatternCache, cache_nucleus_patterns
from pyladr.search.repetition_penalty import (
    RepetitionPenaltyConfig,
    compute_repetition_penalty,
)
from pyladr.search.priority_sos import PrioritySOS, _forte_novelty_score
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    _clause_generality_penalty,
    default_clause_weight,
)
from pyladr.search.state import ClauseList, SearchState
from pyladr.search.statistics import SearchStatistics

logger = logging.getLogger(__name__)


# ── Unit conflict index (canonical definition in pyladr.search.unit_conflict)
from pyladr.search.unit_conflict import UnitConflictIndex  # noqa: E402


# ── Exit codes matching C search.h ──────────────────────────────────────────


class ExitCode(IntEnum):
    """Search termination codes matching C exit codes."""

    MAX_PROOFS_EXIT = 1
    SOS_EMPTY_EXIT = 2
    MAX_GIVEN_EXIT = 3
    MAX_KEPT_EXIT = 4
    MAX_SECONDS_EXIT = 5
    MAX_GENERATED_EXIT = 6
    FATAL_EXIT = 7


# ── Search options (canonical definition in pyladr.search.options) ──────────
# Re-exported here for backward compatibility with existing import paths.
from pyladr.search.options import SearchOptions  # noqa: E402


# ── Proof result ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Proof:
    """A proof found during search.

    Contains the empty clause and a trace of clauses used.
    """

    empty_clause: Clause
    clauses: tuple[Clause, ...]

    def __repr__(self) -> str:
        return (
            f"Proof(empty_clause=id:{self.empty_clause.id}, "
            f"length={len(self.clauses)})"
        )


# ── Search result ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of a search invocation."""

    exit_code: ExitCode
    proofs: tuple[Proof, ...]
    stats: SearchStatistics


# ── T2V goal-distance histogram formatting ──────────────────────────────────


def format_t2v_histogram(histogram: dict, proof_num: int | None = 1) -> str:
    """Format a T2V goal-distance histogram as a conditional probability table.

    Distances are in [0, 1]: 0.0 = identical to goal, 1.0 = maximally distant.
    When *proof_num* is ``None`` the header reads "cumulative" instead of
    referencing a single proof number.
    """
    proof_probs = histogram["proof_probs"]
    nonproof_probs = histogram["nonproof_probs"]
    proof_n = histogram["proof_n"]
    nonproof_n = histogram["nonproof_n"]
    lo = histogram["lo"]
    bw = histogram["bucket_width"]
    if proof_num is None:
        n_proofs = histogram.get("n_proofs", "?")
        header = (
            f"T2V goal distance (cumulative, {n_proofs} proofs,"
            f" {proof_n} proof clauses, {nonproof_n} non-proof):"
        )
    else:
        header = f"T2V goal distance at proof {proof_num} ({proof_n} proof clauses, {nonproof_n} non-proof):"
    lines = [
        header,
        "  range            P(range|proof)  P(range|non-proof)",
    ]
    for i in range(5):
        lo_edge = lo + i * bw
        hi_edge = lo + (i + 1) * bw
        label = f"{lo_edge:.2f}-{hi_edge:.2f}"
        lines.append(
            f"  {label}:  {proof_probs[i]:>16.4f}  {nonproof_probs[i]:>18.4f}"
        )
    return "\n".join(lines)


# ── Cross-argument distance helpers ─────────────────────────────────────────

def _get_antecedent_term(clause: "Clause") -> "Term | None":
    """Extract the antecedent term from a clause of the form P(i(x,y)).

    Returns the first arg of the inner term (x), or None if the clause
    doesn't have the expected structure.
    """
    if not clause.literals:
        return None
    atom = clause.literals[0].atom  # P(...)
    if atom.arity < 1:
        return None
    inner = atom.args[0]  # i(x,y)
    if inner.arity < 1:
        return None
    return inner.args[0]  # x


def _t2v_cosine(a: "list[float]", b: "list[float]") -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 1e-12 else 0.0


# ── Main search engine ──────────────────────────────────────────────────────


class GivenClauseSearch:
    """Given-clause search engine matching C search() in search.c.

    The given-clause algorithm:
    1. Initialize: move clauses to usable/sos, process initial clauses
    2. Main loop:
       a. Select a given clause from SOS
       b. Move given clause to usable
       c. Index it for resolution
       d. Generate inferences (given_infer)
       e. Process inferred clauses (cl_process)
       f. Process limbo list (limbo_process)
    3. Terminate when: proof found, SOS empty, or limits exceeded

    Usage:
        search = GivenClauseSearch(options)
        result = search.run(usable_clauses, sos_clauses)
    """

    # ── EmbeddingManager extraction candidates ──────────────────────────
    #
    # The following 22 __slots__ entries and 24 methods belong to embedding
    # management (FORTE, Tree2Vec, RNN2Vec, proof patterns) and should be
    # extracted into pyladr.search.embedding_manager.EmbeddingManager.
    # See embedding_manager.py for the target interface skeleton.
    #
    # Slots to move:
    #   _forte_provider, _forte_embeddings,
    #   _tree2vec_provider, _tree2vec_embeddings,
    #   _t2v_kept_since_update, _t2v_online_batch, _t2v_goal_clauses,
    #   _t2v_goal_provider, _t2v_goal_clause_ids,
    #   _t2v_distance_window, _t2v_distance_prev_avg,
    #   _t2v_initial_goal_count, _t2v_all_given_distances,
    #   _t2v_update_count, _t2v_bg_updater, _t2v_completion_queue,
    #   _t2v_antecedent_embeddings, _t2v_goal_arg_embs, _t2v_goal_ant_embs,
    #   _rnn2vec_provider, _rnn2vec_embeddings,
    #   _r2v_kept_since_update, _r2v_online_batch,
    #   _r2v_update_count, _r2v_bg_updater, _r2v_completion_queue,
    #   _r2v_goal_provider, _r2v_goal_clauses,
    #   _proof_pattern_memory
    #
    # Methods to move (24 methods, ~1,400 lines):
    #   _init_embeddings (L619)
    #   _maybe_init_rnn2vec (L1207)
    #   _r2v_select_most_diverse (L1351)
    #   _r2v_select_random_goal (L1383)
    #   _do_r2v_online_update (L1421)
    #   _on_r2v_update_done (L1450)
    #   _process_r2v_completions (L1454)
    #   _save_r2v_model (L1484)
    #   _do_t2v_online_update (L2281)
    #   _do_t2v_online_update_sync (L2311)
    #   _on_t2v_update_done (L2429)
    #   _process_t2v_completions (L2437)
    #   _dump_t2v_embeddings (L2478)
    #   _dump_r2v_embeddings (L2570)
    #   _t2v_cross_arg_distance (L2646)
    #   _t2v_select_nearest_goal (L2665)
    #   _t2v_select_maximin (L2707)
    #   _record_proof_patterns (L2998)
    #   _compute_t2v_histogram (L3148)
    #   _compute_t2v_cumulative_histogram (L3196)
    #
    # Properties to move:
    #   forte_embeddings, forte_provider, proof_pattern_memory
    #
    # Module-level functions to move:
    #   format_t2v_histogram (L416), _get_antecedent_term (L453),
    #   _t2v_cosine (L470)
    # ──────────────────────────────────────────────────────────────────────

    __slots__ = (
        "_opts", "_state", "_selection", "_proofs", "_all_clauses",
        "_symbol_table", "_demod_index", "_parallel_engine",
        "_subsump_idx", "_back_subsump_idx", "_unit_conflict_idx",
        "_lazy_demod", "_proof_callback", "_penalty_cache",
        "_repetition_config", "_penalty_weight_config",
        "_nucleus_penalty_config", "_nucleus_pattern_cache",
        "_back_subsume_enabled",
        "_back_subsumption_callback", "_forward_subsumption_callback",
        # ── EmbeddingManager candidates (22 slots) ──
        "_forte_provider", "_forte_embeddings",
        "_tree2vec_provider", "_tree2vec_embeddings",
        "_t2v_kept_since_update", "_t2v_online_batch", "_t2v_goal_clauses",
        "_t2v_goal_provider", "_t2v_goal_clause_ids",
        "_t2v_distance_window", "_t2v_distance_prev_avg",
        "_t2v_initial_goal_count", "_t2v_all_given_distances",
        "_t2v_update_count",
        "_t2v_bg_updater",
        "_t2v_completion_queue",
        "_t2v_antecedent_embeddings", "_t2v_goal_arg_embs", "_t2v_goal_ant_embs",
        "_rnn2vec_provider", "_rnn2vec_embeddings",
        "_r2v_kept_since_update", "_r2v_online_batch",
        "_r2v_update_count", "_r2v_bg_updater", "_r2v_completion_queue",
        "_r2v_goal_provider", "_r2v_goal_clauses",
        "_proof_pattern_memory",
        # ── End EmbeddingManager candidates ──
        "_hints",
        "_rep_penalty_cache",
    )

    def __init__(
        self,
        options: SearchOptions | None = None,
        selection: GivenSelection | None = None,
        symbol_table: SymbolTable | None = None,
        proof_callback: Callable[[Proof, int], None] | None = None,
        hints: list[Clause] | None = None,
    ) -> None:
        self._opts = options or SearchOptions()
        self._state = SearchState()
        if self._opts.priority_sos:
            self._state.sos = PrioritySOS("sos")

        # ── Search core ──
        self._selection = self._init_selection(selection)
        self._proofs: list[Proof] = []
        self._all_clauses: dict[int, Clause] = {}
        self._symbol_table = symbol_table or SymbolTable()
        self._demod_index = DemodulatorIndex()
        self._lazy_demod: LazyDemodState | None = (
            LazyDemodState() if self._opts.lazy_demod else None
        )
        self._parallel_engine = ParallelInferenceEngine(self._opts.parallel)
        self._subsump_idx = LiteralIndex(first_only=True)
        self._back_subsump_idx = BackSubsumptionIndex()
        self._unit_conflict_idx = UnitConflictIndex()
        self._back_subsume_enabled = True
        self._proof_callback = proof_callback
        self._back_subsumption_callback: Callable[[Clause, Clause], None] | None = None
        self._forward_subsumption_callback: Callable[[Clause, Clause], None] | None = None
        self._hints: list[Clause] = hints if hints is not None else []

        # ── Embeddings (FORTE, Tree2Vec, proof-guided) ──
        self._init_embeddings()

        # ── Penalties (propagation, repetition, weight adjustment, nucleus) ──
        self._init_penalties()

    # ── Initialization helpers ─────────────────────────────────────────
    #
    # Planned decomposition (see embedding_manager.py for target interface):
    #   - EmbeddingManager: owns 22 _forte_*/_tree2vec_*/_t2v_*/_rnn2vec_*/_r2v_*
    #     slots + 24 methods (~1,400 lines). Skeleton in embedding_manager.py.
    #   - PenaltyManager: owns _penalty_cache, _repetition_config, _penalty_weight_config,
    #     _nucleus_*, _rep_penalty_cache with methods get_clause_penalty(), compute_and_cache()
    #   TODO: Unify the 4 penalty systems (penalty_propagation, repetition_penalty,
    #   nucleus_penalty, penalty_weight) behind a common protocol.
    # Blocked today by deep cross-method access (18 embedding slots read/written in 10+ methods).

    def _init_selection(self, selection: GivenSelection | None) -> GivenSelection:
        """Build the selection strategy from options."""
        if selection is not None:
            return selection
        opts = self._opts
        from pyladr.search.selection import SelectionRule
        has_extra = (
            opts.entropy_weight > 0 or opts.unification_weight > 0
            or opts.forte_weight > 0 or opts.proof_guided_weight > 0
            or opts.tree2vec_weight > 0 or opts.tree2vec_maximin_weight > 0
            or opts.rnn2vec_weight > 0 or opts.rnn2vec_random_goal_weight > 0
        )
        if has_extra or opts.weight_ratio != 4:
            rules = [
                SelectionRule("A", SelectionOrder.AGE, part=1),
                SelectionRule("W", SelectionOrder.WEIGHT, part=opts.weight_ratio),
            ]
            if opts.entropy_weight > 0:
                rules.append(SelectionRule("E", SelectionOrder.ENTROPY, part=opts.entropy_weight))
            if opts.unification_weight > 0:
                rules.append(SelectionRule("U", SelectionOrder.UNIFICATION_PENALTY, part=opts.unification_weight))
            if opts.forte_weight > 0:
                rules.append(SelectionRule("F", SelectionOrder.FORTE, part=opts.forte_weight))
            if opts.proof_guided_weight > 0:
                pg_weight = opts.proof_guided_weight
                if pg_weight != int(pg_weight):
                    logger.warning(
                        "proof_guided_weight=%.2f is not integral, rounding to %d",
                        pg_weight, round(pg_weight),
                    )
                rules.append(SelectionRule("PG", SelectionOrder.PROOF_GUIDED, part=round(pg_weight)))
            if opts.tree2vec_weight > 0:
                rules.append(SelectionRule("T2V", SelectionOrder.TREE2VEC, part=opts.tree2vec_weight))
            if opts.tree2vec_maximin_weight > 0:
                rules.append(SelectionRule("T2VM", SelectionOrder.TREE2VEC_MAXIMIN, part=opts.tree2vec_maximin_weight))
            if opts.rnn2vec_weight > 0:
                rules.append(SelectionRule("R2V", SelectionOrder.RNN2VEC, part=opts.rnn2vec_weight))
            if opts.rnn2vec_random_goal_weight > 0:
                rules.append(SelectionRule("RGP", SelectionOrder.RNN2VEC_RANDOM_GOAL, part=opts.rnn2vec_random_goal_weight))
            return GivenSelection(rules=rules)
        return GivenSelection()

    def _init_embeddings(self) -> None:
        """Initialize FORTE, Tree2Vec, and proof-guided embedding state."""
        opts = self._opts

        # FORTE embedding side structure (clause_id → embedding vector)
        self._forte_embeddings: dict[int, list[float]] = {}
        self._forte_provider: object | None = None
        if opts.forte_embeddings:
            try:
                from pyladr.ml.forte.provider import (
                    ForteEmbeddingProvider,
                    ForteProviderConfig,
                )
                from pyladr.ml.forte.algorithm import ForteConfig

                forte_cfg = ForteProviderConfig(
                    forte_config=ForteConfig(
                        embedding_dim=opts.forte_embedding_dim,
                    ),
                    cache_max_entries=opts.forte_cache_max_entries,
                )
                self._forte_provider = ForteEmbeddingProvider(config=forte_cfg)
                logger.info(
                    "FORTE embeddings enabled (dim=%d, cache=%d)",
                    opts.forte_embedding_dim,
                    opts.forte_cache_max_entries,
                )
            except Exception:
                logger.warning(
                    "Failed to initialize FORTE provider, continuing without",
                    exc_info=True,
                )

        # Tree2Vec embedding side structure (clause_id → embedding vector)
        self._tree2vec_embeddings: dict[int, list[float]] = {}
        self._tree2vec_provider: object | None = None
        self._t2v_kept_since_update: int = 0
        self._t2v_online_batch: list[Clause] = []
        self._t2v_goal_clauses: list[Clause] = []
        self._t2v_goal_clause_ids: list[int] = []
        self._t2v_goal_provider: object | None = None
        self._t2v_distance_window: list[float] = []
        self._t2v_distance_prev_avg: float = 0.0
        self._t2v_initial_goal_count: int = 0
        self._t2v_all_given_distances: dict[int, float] = {}
        self._t2v_update_count: int = 0
        self._t2v_bg_updater: object | None = None
        import queue as _queue
        self._t2v_completion_queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._t2v_antecedent_embeddings: dict[int, list[float]] = {}
        self._t2v_goal_arg_embs: list[list[float]] = []
        self._t2v_goal_ant_embs: list[list[float]] = []
        if opts.tree2vec_embeddings:
            try:
                from pyladr.ml.tree2vec.provider import (
                    Tree2VecEmbeddingProvider,
                    Tree2VecProviderConfig,
                )
                from pyladr.ml.tree2vec.algorithm import Tree2VecConfig
                from pyladr.ml.tree2vec.skipgram import SkipGramConfig
                from pyladr.ml.tree2vec.walks import WalkConfig

                t2v_cfg = Tree2VecProviderConfig(
                    tree2vec_config=Tree2VecConfig(
                        walk_config=WalkConfig(
                            include_position=opts.tree2vec_include_position,
                            include_depth=opts.tree2vec_include_depth,
                            include_path_length=opts.tree2vec_include_path_length,
                            include_var_identity=opts.tree2vec_include_var_identity,
                            skip_predicate_wrapper=opts.tree2vec_skip_predicate,
                        ),
                        skipgram_config=SkipGramConfig(
                            embedding_dim=opts.tree2vec_embedding_dim,
                        ),
                        composition=opts.tree2vec_composition,
                    ),
                    cache_max_entries=opts.tree2vec_cache_max_entries,
                )
                # Training requires input file — will be initialized in run()
                # when clauses are available. Store config for deferred init.
                self._tree2vec_provider = t2v_cfg
                logger.info(
                    "Tree2Vec embeddings enabled (dim=%d, cache=%d)",
                    opts.tree2vec_embedding_dim,
                    opts.tree2vec_cache_max_entries,
                )
            except Exception:
                logger.warning(
                    "Failed to initialize Tree2Vec provider, continuing without",
                    exc_info=True,
                )

        # RNN2Vec embedding side structure (clause_id → embedding vector)
        self._rnn2vec_embeddings: dict[int, list[float]] = {}
        self._rnn2vec_provider: object | None = None
        self._r2v_kept_since_update: int = 0
        self._r2v_online_batch: list[Clause] = []
        self._r2v_update_count: int = 0
        self._r2v_bg_updater: object | None = None
        self._r2v_goal_provider: object | None = None
        self._r2v_goal_clauses: list = []
        import queue as _queue2
        self._r2v_completion_queue: _queue2.SimpleQueue = _queue2.SimpleQueue()
        if opts.rnn2vec_embeddings:
            try:
                from pyladr.ml.rnn2vec.provider import (
                    RNN2VecEmbeddingProvider,
                    RNN2VecProviderConfig,
                )
                from pyladr.ml.rnn2vec.algorithm import RNN2VecConfig
                from pyladr.ml.rnn2vec.encoder import RNNEmbeddingConfig

                r2v_cfg = RNN2VecProviderConfig(
                    rnn2vec_config=RNN2VecConfig(
                        rnn_config=RNNEmbeddingConfig(
                            rnn_type=opts.rnn2vec_rnn_type,
                            hidden_dim=opts.rnn2vec_hidden_dim,
                            embedding_dim=opts.rnn2vec_embedding_dim,
                            input_dim=opts.rnn2vec_input_dim,
                            num_layers=opts.rnn2vec_num_layers,
                            bidirectional=opts.rnn2vec_bidirectional,
                            composition=opts.rnn2vec_composition,
                        ),
                        training_epochs=opts.rnn2vec_training_epochs,
                        learning_rate=opts.rnn2vec_training_lr,
                    ),
                    cache_max_entries=opts.rnn2vec_cache_max_entries,
                )
                # Training requires clauses — will be initialized in run()
                # when clauses are available. Store config for deferred init.
                self._rnn2vec_provider = r2v_cfg
                logger.info(
                    "RNN2Vec embeddings enabled (dim=%d, cache=%d)",
                    opts.rnn2vec_embedding_dim,
                    opts.rnn2vec_cache_max_entries,
                )
            except Exception:
                logger.warning(
                    "Failed to initialize RNN2Vec provider, continuing without",
                    exc_info=True,
                )

        # Wire PrioritySOS to FORTE embeddings for lazy heap init
        if isinstance(self._state.sos, PrioritySOS) and self._forte_provider is not None:
            self._state.sos._forte_embeddings_ref = self._forte_embeddings

        # Proof-guided selection: learn from successful proof patterns
        self._proof_pattern_memory: object | None = None
        if opts.proof_guided and self._forte_provider is not None:
            try:
                from pyladr.search.proof_pattern_memory import (
                    ProofGuidedConfig,
                    ProofPatternMemory,
                )

                pg_config = ProofGuidedConfig(
                    enabled=True,
                    exploitation_ratio=opts.proof_guided_exploitation_ratio,
                    max_patterns=opts.proof_guided_max_patterns,
                    decay_rate=opts.proof_guided_decay_rate,
                    min_similarity_threshold=opts.proof_guided_min_similarity,
                    warmup_proofs=opts.proof_guided_warmup_proofs,
                )
                self._proof_pattern_memory = ProofPatternMemory(config=pg_config)
                logger.info(
                    "Proof-guided selection enabled (exploitation=%.2f, max_patterns=%d, decay=%.2f)",
                    pg_config.exploitation_ratio,
                    pg_config.max_patterns,
                    pg_config.decay_rate,
                )
            except Exception:
                logger.warning(
                    "Failed to initialize proof pattern memory, continuing without",
                    exc_info=True,
                )

        # Wire PrioritySOS to proof-guided scorer
        if (
            isinstance(self._state.sos, PrioritySOS)
            and self._proof_pattern_memory is not None
            and self._forte_provider is not None
        ):
            memory = self._proof_pattern_memory
            embeddings_ref = self._forte_embeddings
            from pyladr.search.proof_pattern_memory import proof_guided_score

            def _proof_guided_scorer(clause_id: int) -> float:
                emb = embeddings_ref.get(clause_id)
                if emb is None:
                    return 0.5  # neutral
                # Diversity score: L1-norm normalized to [0,1]
                l1 = sum(map(abs, emb))
                dim = len(emb)
                max_l1 = dim ** 0.5  # sqrt(dim) for L2-normalized vectors
                diversity = l1 / max_l1 if max_l1 > 0 else 0.5
                return proof_guided_score(emb, memory, diversity, memory.config)

            self._state.sos._proof_guided_scorer = _proof_guided_scorer

    def _init_penalties(self) -> None:
        """Initialize penalty propagation, repetition, weight adjustment, and nucleus state."""
        opts = self._opts

        # Penalty propagation cache (side structure, preserves Clause immutability)
        self._penalty_cache: PenaltyCache | None = None
        if opts.penalty_propagation:
            mode_map = {
                "additive": PenaltyCombineMode.ADDITIVE,
                "multiplicative": PenaltyCombineMode.MULTIPLICATIVE,
                "max": PenaltyCombineMode.MAX,
            }
            pp_config = PenaltyPropagationConfig(
                enabled=True,
                mode=mode_map.get(
                    opts.penalty_propagation_mode,
                    PenaltyCombineMode.ADDITIVE,
                ),
                decay=opts.penalty_propagation_decay,
                threshold=opts.penalty_propagation_threshold,
                max_depth=opts.penalty_propagation_max_depth,
                max_penalty=opts.penalty_propagation_max,
            )
            self._penalty_cache = PenaltyCache(pp_config)

        # Cache for repetition penalty: avoids recomputing between _should_delete and _limbo_process
        self._rep_penalty_cache: dict[int, float] = {}

        # Subformula repetition penalty config
        self._repetition_config: RepetitionPenaltyConfig | None = None
        if opts.repetition_penalty:
            self._repetition_config = RepetitionPenaltyConfig(
                enabled=True,
                base_penalty=opts.repetition_penalty_weight,
                min_subterm_size=opts.repetition_penalty_min_size,
                max_penalty=opts.repetition_penalty_max,
                normalize_variables=opts.repetition_penalty_normalize,
            )

        # Penalty weight adjustment config
        self._penalty_weight_config: PenaltyWeightConfig | None = None
        if opts.penalty_weight_enabled:
            mode_map = {
                "linear": PenaltyWeightMode.LINEAR,
                "exponential": PenaltyWeightMode.EXPONENTIAL,
                "step": PenaltyWeightMode.STEP,
            }
            self._penalty_weight_config = PenaltyWeightConfig(
                enabled=True,
                threshold=opts.penalty_weight_threshold,
                multiplier=opts.penalty_weight_multiplier,
                max_adjusted_weight=opts.penalty_weight_max,
                mode=mode_map.get(
                    opts.penalty_weight_mode,
                    PenaltyWeightMode.EXPONENTIAL,
                ),
            )

        # Nucleus unification penalty config and pattern cache
        self._nucleus_penalty_config: NucleusUnificationPenaltyConfig | None = None
        self._nucleus_pattern_cache: NucleusPatternCache | None = None
        if opts.nucleus_unification_penalty:
            self._nucleus_penalty_config = NucleusUnificationPenaltyConfig(
                enabled=True,
                threshold=opts.nucleus_penalty_threshold,
                base_penalty=opts.nucleus_penalty_weight,
                max_penalty=opts.nucleus_penalty_max,
            )
            self._nucleus_pattern_cache = NucleusPatternCache(
                max_size=opts.nucleus_penalty_cache_size,
            )

    def set_back_subsumption_callback(
        self, callback: Callable[[Clause, Clause], None],
    ) -> None:
        """Register a callback for back-subsumption events."""
        self._back_subsumption_callback = callback

    def set_forward_subsumption_callback(
        self, callback: Callable[[Clause, Clause], None],
    ) -> None:
        """Register a callback for forward-subsumption events."""
        self._forward_subsumption_callback = callback

    def _on_goal_subsumed(self, subsumed: Clause) -> None:
        """Remove a subsumed goal clause from the T2V goal-distance tracking list."""
        if not self._t2v_goal_clause_ids:
            return
        try:
            idx = self._t2v_goal_clause_ids.index(subsumed.id)
        except ValueError:
            return
        goal_scorer = getattr(
            getattr(self, '_t2v_goal_provider', None), '_goal_scorer', None,
        )
        if goal_scorer is not None:
            goal_scorer.remove_goal(idx)
        self._t2v_goal_clause_ids.pop(idx)
        self._t2v_goal_clauses.pop(idx)
        logger.debug(
            "Goal clause %d subsumed, removed from distance tracking (%d goals remaining)",
            subsumed.id, len(self._t2v_goal_clause_ids),
        )

    @property
    def state(self) -> SearchState:
        """Access search state (for testing/inspection)."""
        return self._state

    @property
    def stats(self) -> SearchStatistics:
        return self._state.stats

    @property
    def forte_embeddings(self) -> dict[int, list[float]]:
        """FORTE embedding storage: clause_id → embedding vector."""
        return self._forte_embeddings

    @property
    def forte_provider(self) -> object | None:
        """The active ForteEmbeddingProvider, or None if disabled."""
        return self._forte_provider

    @property
    def proof_pattern_memory(self) -> object | None:
        """The active ProofPatternMemory, or None if disabled."""
        return self._proof_pattern_memory

    def set_proof_callback(self, callback: Callable[[Proof, int], None] | None) -> None:
        """Set or replace the proof callback. Used by online integration."""
        self._proof_callback = callback

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        usable: list[Clause] | None = None,
        sos: list[Clause] | None = None,
        original_goals: list[Clause] | None = None,
    ) -> SearchResult:
        """Run the given-clause search. Matches C search().

        Args:
            usable: Initial usable clauses (already processed, available for inference).
            sos: Initial SOS clauses (candidates for given clause selection).

        Returns:
            SearchResult with exit code, proofs, and statistics.
        """
        self._state.stats.start()

        # Reset module-level orientation state so this search starts clean.
        # Without this, orientation marks from previous searches leak across
        # problem boundaries (the sets grow forever and are never cleared).
        reset_orientation_state()

        # Move initial clauses into state (C: move_clauses_to_clist)
        self._init_clauses(usable or [], sos or [], original_goals or [])

        # Process initial clauses (C: index_and_process_initial_clauses)
        exit_code = self._process_initial_clauses()
        if exit_code is not None:
            return self._make_result(exit_code)

        # Main search loop (C: while(inferences_to_make()) { ... })
        self._state.searching = True
        self._state.stats.search_start_time = self._state.stats.start_time

        try:
            while self._inferences_to_make():
                # Drain any background completions before selecting
                self._process_t2v_completions()
                self._process_r2v_completions()

                exit_code = self._make_inferences()
                if exit_code is not None:
                    return self._make_result(exit_code)

                exit_code = self._limbo_process()
                if exit_code is not None:
                    return self._make_result(exit_code)

            # SOS exhausted
            return self._make_result(ExitCode.SOS_EMPTY_EXIT)
        finally:
            if self._t2v_bg_updater is not None:
                self._t2v_bg_updater.shutdown(drain=True, timeout=5.0)
                self._process_t2v_completions()
            if self._r2v_bg_updater is not None:
                self._r2v_bg_updater.shutdown(drain=True, timeout=5.0)
                self._process_r2v_completions()

    # ── Initialization ──────────────────────────────────────────────────

    def _init_clauses(
        self,
        usable: list[Clause],
        sos: list[Clause],
        original_goals: list[Clause] | None = None,
    ) -> None:
        """Move initial clauses into state lists. Matches C setup in search()."""
        for c in usable:
            if c.id == 0:
                self._state.assign_clause_id(c)
            c.initial = True
            self._all_clauses[c.id] = c
            self._state.usable.append(c)

        for c in sos:
            if c.id == 0:
                self._state.assign_clause_id(c)
            c.initial = True
            self._all_clauses[c.id] = c
            self._state.sos.append(c)

        # Seed initial clauses into penalty cache (depth=0, no inheritance)
        if self._penalty_cache is not None:
            for c in (*usable, *sos):
                compute_and_cache_penalty(
                    c, self._penalty_cache, self._all_clauses,
                )

        # Batch pre-compute FORTE embeddings for initial clauses
        if self._forte_provider is not None:
            initial = [*usable, *sos]
            if initial:
                embeddings = self._forte_provider.get_embeddings_batch(initial)  # type: ignore[union-attr]
                for c, emb in zip(initial, embeddings):
                    if emb is not None:
                        self._forte_embeddings[c.id] = emb
                logger.debug(
                    "FORTE: pre-computed %d/%d initial embeddings",
                    len(self._forte_embeddings), len(initial),
                )

        # Deferred Tree2Vec training: train on initial clauses, then compute embeddings
        if self._tree2vec_provider is not None and not callable(self._tree2vec_provider):
            try:
                from pyladr.ml.tree2vec.provider import Tree2VecEmbeddingProvider
                from pyladr.ml.tree2vec.vampire_parser import VampireCorpus

                t2v_cfg = self._tree2vec_provider  # stored config from __init__
                initial = [*usable, *sos]

                if self._opts.tree2vec_model_path:
                    # Load pre-trained model from disk (offline mode)
                    provider = Tree2VecEmbeddingProvider.from_saved_model(
                        model_path=self._opts.tree2vec_model_path,
                        config=t2v_cfg,
                    )
                    self._tree2vec_provider = provider
                elif initial:
                    # Train Tree2Vec on all initial clauses
                    from pyladr.ml.tree2vec.formula_processor import process_vampire_corpus
                    corpus = VampireCorpus(
                        sos_clauses=tuple(sos),
                        goal_clauses=(),
                        all_terms=(),
                        all_subterms=(),
                        symbol_table=self._symbol_table,
                    )
                    result = process_vampire_corpus(
                        corpus,
                        tree2vec_config=t2v_cfg.tree2vec_config,
                        augmentation_config=t2v_cfg.augmentation_config,
                    )
                    provider = Tree2VecEmbeddingProvider(
                        tree2vec=result.tree2vec,
                        config=t2v_cfg,
                    )
                    self._tree2vec_provider = provider

                if self._tree2vec_provider is not t2v_cfg and initial:
                    # Batch compute initial embeddings
                    provider = self._tree2vec_provider
                    embeddings = provider.get_embeddings_batch(initial)
                    for c, emb in zip(initial, embeddings):
                        if emb is not None:
                            self._tree2vec_embeddings[c.id] = emb
                    logger.debug(
                        "Tree2Vec: pre-computed %d/%d initial embeddings",
                        len(self._tree2vec_embeddings), len(initial),
                    )
                if self._opts.tree2vec_dump_embeddings:
                    self._dump_t2v_embeddings(0)
            except Exception:
                logger.warning(
                    "Failed to initialize Tree2Vec provider, continuing without",
                    exc_info=True,
                )
                self._tree2vec_provider = None

        # Background updater for Tree2Vec online learning
        if (
            self._opts.tree2vec_online_learning
            and self._opts.tree2vec_bg_update
            and self._tree2vec_provider is not None
            and callable(getattr(self._tree2vec_provider, 'bump_model_version', None))
        ):
            try:
                from pyladr.ml.tree2vec.background_updater import BackgroundT2VUpdater
                self._t2v_bg_updater = BackgroundT2VUpdater(
                    provider=self._tree2vec_provider,
                    learning_rate=self._opts.tree2vec_online_lr,
                    max_updates=self._opts.tree2vec_online_max_updates,
                    completion_callback=self._on_t2v_update_done,
                )
                logger.info(
                    "Tree2Vec background updater started (lr=%.5f, max_updates=%d)",
                    self._opts.tree2vec_online_lr,
                    self._opts.tree2vec_online_max_updates,
                )
            except Exception:
                logger.warning(
                    "Failed to start Tree2Vec background updater, using sync mode",
                    exc_info=True,
                )

        # Goal-proximity wrapping for Tree2Vec
        if (
            self._opts.tree2vec_goal_proximity
            and self._tree2vec_provider is not None
            and callable(getattr(self._tree2vec_provider, 'get_embedding', None))
        ):
            from pyladr.search.goal_directed import (
                GoalDirectedConfig,
                GoalDirectedEmbeddingProvider,
                _deskolemize_clause,
            )

            # Use DENY-justified SOS clauses as the proximity reference.
            # These provide natural sign contrast (negative literals) against
            # the positive derived clauses, giving a wide, useful proximity range.
            all_initial = [*(usable or []), *(sos or [])]
            self._t2v_goal_clauses = [
                c for c in all_initial
                if (c.justification
                    and len(c.justification) > 0
                    and c.justification[0].just_type == JustType.DENY)
            ]

            self._t2v_goal_clause_ids = [c.id for c in self._t2v_goal_clauses]
            self._t2v_initial_goal_count = len(self._t2v_goal_clauses)

            if self._t2v_goal_clauses:
                gd_config = GoalDirectedConfig(
                    enabled=True,
                    goal_proximity_weight=self._opts.tree2vec_goal_proximity_weight,
                )
                gd_provider = GoalDirectedEmbeddingProvider(
                    base_provider=self._tree2vec_provider,
                    config=gd_config,
                )
                gd_provider.register_goals(self._t2v_goal_clauses)
                self._t2v_goal_provider = gd_provider
                logger.info(
                    "Tree2Vec goal-distance enabled: %d goals, weight=%.2f",
                    len(self._t2v_goal_clauses),
                    self._opts.tree2vec_goal_proximity_weight,
                )

            # Cross-arg proximity: precompute goal arg and antecedent embeddings
            if self._opts.tree2vec_cross_arg_proximity and self._t2v_goal_clauses:
                t2v_algo = getattr(self._tree2vec_provider, '_tree2vec', None)
                if t2v_algo is not None:
                    for gc in self._t2v_goal_clauses:
                        # Deskolemise: replace Skolem constants with variables
                        # so cross-arg embeddings use the same VAR tokens as
                        # derived clauses, making the comparison meaningful.
                        gc_norm = _deskolemize_clause(gc)
                        # Full arg embedding: the argument of P (i.e., i(x₀,x₁))
                        if gc_norm.literals and gc_norm.literals[0].atom.arity >= 1:
                            arg_emb = t2v_algo.embed_term(gc_norm.literals[0].atom.args[0])
                            if arg_emb is not None:
                                self._t2v_goal_arg_embs.append(arg_emb)
                                # Antecedent: first arg of i(x₀,x₁) → x₀
                                ant_term = _get_antecedent_term(gc_norm)
                                if ant_term is not None:
                                    ant_emb = t2v_algo.embed_term(ant_term)
                                    if ant_emb is not None:
                                        self._t2v_goal_ant_embs.append(ant_emb)
                                    else:
                                        self._t2v_goal_ant_embs.append(arg_emb)
                                else:
                                    self._t2v_goal_ant_embs.append(arg_emb)
                    if self._t2v_goal_arg_embs:
                        logger.info(
                            "Tree2Vec cross-arg distance: %d goal arg/ant embeddings",
                            len(self._t2v_goal_arg_embs),
                        )

        # Deferred RNN2Vec initialization
        self._maybe_init_rnn2vec(usable, sos)

    def _maybe_init_rnn2vec(self, usable, sos) -> None:
        """Deferred RNN2Vec training: train on initial clauses, then compute embeddings."""
        if self._rnn2vec_provider is not None and not callable(self._rnn2vec_provider):
            try:
                from pyladr.ml.rnn2vec.provider import RNN2VecEmbeddingProvider
                from pyladr.ml.tree2vec.vampire_parser import VampireCorpus

                r2v_cfg = self._rnn2vec_provider  # stored config from __init__
                initial = [*usable, *sos]

                if self._opts.rnn2vec_model_path:
                    # Load pre-trained model from disk (offline mode)
                    print(f"% RNN2Vec: loading model from {self._opts.rnn2vec_model_path} ...")
                    provider = RNN2VecEmbeddingProvider.from_saved_model(
                        model_path=self._opts.rnn2vec_model_path,
                        config=r2v_cfg,
                    )
                    self._rnn2vec_provider = provider
                    r2v = getattr(provider, "_rnn2vec", None)
                    if r2v is not None:
                        print(f"% RNN2Vec: model loaded "
                              f"(vocab={r2v.vocab_size}, dim={r2v.embedding_dim})")
                elif initial:
                    # Train RNN2Vec on all initial clauses
                    from pyladr.ml.rnn2vec.formula_processor import process_vampire_corpus as r2v_process
                    n_clauses = len(initial)
                    epochs = r2v_cfg.rnn2vec_config.training_epochs
                    rnn_cfg = r2v_cfg.rnn2vec_config.rnn_config
                    print(f"% RNN2Vec: training on {n_clauses} initial clauses "
                          f"({epochs} epochs, {rnn_cfg.rnn_type.upper()}, "
                          f"h={rnn_cfg.hidden_dim}, dim={rnn_cfg.embedding_dim}) ...")
                    corpus = VampireCorpus(
                        sos_clauses=tuple(sos),
                        goal_clauses=(),
                        all_terms=(),
                        all_subterms=(),
                        symbol_table=self._symbol_table,
                    )
                    result = r2v_process(
                        corpus,
                        rnn2vec_config=r2v_cfg.rnn2vec_config,
                        augmentation_config=r2v_cfg.augmentation_config,
                    )
                    provider = RNN2VecEmbeddingProvider(
                        rnn2vec=result.rnn2vec,
                        config=r2v_cfg,
                    )
                    self._rnn2vec_provider = provider
                    stats = result.training_stats
                    if stats:
                        print(f"% RNN2Vec: training complete "
                              f"(loss={stats.get('loss', 0.0):.4f}, "
                              f"vocab={int(stats.get('vocab_size', 0))})")

                if self._rnn2vec_provider is not r2v_cfg and initial:
                    # Batch compute initial embeddings
                    provider = self._rnn2vec_provider
                    embeddings = provider.get_embeddings_batch(initial)
                    for c, emb in zip(initial, embeddings):
                        if emb is not None:
                            self._rnn2vec_embeddings[c.id] = emb
                    logger.debug(
                        "RNN2Vec: pre-computed %d/%d initial embeddings",
                        len(self._rnn2vec_embeddings), len(initial),
                    )
                    if self._opts.rnn2vec_dump_embeddings:
                        self._dump_r2v_embeddings(0)
            except Exception:
                logger.warning(
                    "Failed to initialize RNN2Vec provider, continuing without",
                    exc_info=True,
                )
                self._rnn2vec_provider = None

        # Background updater for RNN2Vec online learning
        if (
            self._opts.rnn2vec_online_learning
            and self._rnn2vec_provider is not None
            and callable(getattr(self._rnn2vec_provider, 'bump_model_version', None))
        ):
            try:
                from pyladr.ml.rnn2vec.background_updater import BackgroundRNN2VecUpdater
                self._r2v_bg_updater = BackgroundRNN2VecUpdater(
                    provider=self._rnn2vec_provider,
                    learning_rate=self._opts.rnn2vec_online_lr,
                    max_updates=self._opts.rnn2vec_online_max_updates,
                    completion_callback=self._on_r2v_update_done,
                )
                logger.info(
                    "RNN2Vec background updater started (lr=%.5f, max_updates=%d)",
                    self._opts.rnn2vec_online_lr,
                    self._opts.rnn2vec_online_max_updates,
                )
            except Exception:
                logger.warning(
                    "Failed to start RNN2Vec background updater, using sync mode",
                    exc_info=True,
                )

        # Goal-proximity wrapping for RNN2Vec
        need_goals = (
            (self._opts.rnn2vec_goal_proximity or self._opts.rnn2vec_random_goal_weight > 0)
            and self._rnn2vec_provider is not None
            and callable(getattr(self._rnn2vec_provider, 'get_embedding', None))
        )
        if need_goals:
            from pyladr.search.goal_directed import (
                GoalDirectedConfig,
                GoalDirectedEmbeddingProvider,
            )

            all_initial = [*(usable or []), *(sos or [])]
            r2v_goal_clauses = [
                c for c in all_initial
                if (c.justification
                    and len(c.justification) > 0
                    and c.justification[0].just_type == JustType.DENY)
            ]
            self._r2v_goal_clauses = r2v_goal_clauses

            if r2v_goal_clauses:
                gd_config = GoalDirectedConfig(
                    enabled=True,
                    goal_proximity_weight=self._opts.rnn2vec_goal_proximity_weight,
                )
                gd_provider = GoalDirectedEmbeddingProvider(
                    base_provider=self._rnn2vec_provider,
                    config=gd_config,
                )
                gd_provider.register_goals(r2v_goal_clauses)
                self._r2v_goal_provider = gd_provider
                if self._opts.rnn2vec_goal_proximity:
                    print(f"% RNN2Vec: goal-distance enabled "
                          f"({len(r2v_goal_clauses)} goals, "
                          f"weight={self._opts.rnn2vec_goal_proximity_weight:.2f})")
                if self._opts.rnn2vec_random_goal_weight > 0:
                    print(f"% RNN2Vec: random-goal selection enabled "
                          f"({len(r2v_goal_clauses)} goals)")
                logger.info(
                    "RNN2Vec goal-distance enabled: %d goals, weight=%.2f",
                    len(r2v_goal_clauses),
                    self._opts.rnn2vec_goal_proximity_weight,
                )

    def _r2v_select_most_diverse(self, sos) -> "Clause | None":
        """Select the clause in SOS whose RNN2Vec embedding is most diverse.

        Picks the clause with the highest minimum distance to all already-given
        clause embeddings (maximin diversity). Falls back to None if no
        embeddings are available for SOS clauses.
        """
        given_embs = [
            emb for cid, emb in self._rnn2vec_embeddings.items()
            if cid not in {c.id for c in sos}
        ]
        if not given_embs:
            # No given embeddings yet — fall back
            return None

        best_clause = None
        best_min_dist = -1.0

        for c in sos:
            emb = self._rnn2vec_embeddings.get(c.id)
            if emb is None:
                continue
            # Minimum cosine similarity to all given embeddings
            min_dist = min(_t2v_cosine(emb, ge) for ge in given_embs)
            # We want the LEAST similar clause (highest diversity)
            diversity = 1.0 - min_dist
            if diversity > best_min_dist:
                best_min_dist = diversity
                best_clause = c

        return best_clause

    def _r2v_select_random_goal(self, sos) -> "Clause | None":
        """Select the SOS clause nearest to a randomly-chosen unproven goal.

        Picks a goal at random from the registered R2V goal embeddings, then
        returns the SOS clause with minimum cosine distance (maximum similarity)
        to that goal's embedding. Falls back to None if no goal embeddings or
        SOS embeddings are available.
        """
        import random as _random

        if not self._rnn2vec_embeddings:
            return None

        goal_scorer = getattr(self._r2v_goal_provider, '_goal_scorer', None) if self._r2v_goal_provider is not None else None
        if goal_scorer is None:
            return None

        with goal_scorer._lock:
            goal_embs = list(goal_scorer._goal_embeddings)

        if not goal_embs:
            return None

        target_emb = _random.choice(goal_embs)

        best_clause = None
        best_dist = float("inf")
        for c in sos:
            emb = self._rnn2vec_embeddings.get(c.id)
            if emb is None:
                continue
            dist = (1.0 - _t2v_cosine(emb, target_emb)) / 2.0
            if dist < best_dist:
                best_dist = dist
                best_clause = c

        return best_clause

    def _do_r2v_online_update(self) -> None:
        """Trigger an online RNN2Vec update from accumulated batch."""
        batch = list(self._r2v_online_batch)
        self._r2v_online_batch.clear()
        self._r2v_kept_since_update = 0

        if not batch:
            return

        if self._r2v_bg_updater is not None:
            self._r2v_bg_updater.submit(batch)
            self._r2v_update_count += 1
        elif self._rnn2vec_provider is not None and callable(
            getattr(self._rnn2vec_provider, 'bump_model_version', None)
        ):
            # Synchronous fallback
            try:
                self._rnn2vec_provider._rnn2vec.update_online(
                    batch, learning_rate=self._opts.rnn2vec_online_lr
                )
                self._rnn2vec_provider.bump_model_version()
                self._r2v_update_count += 1
                if self._opts.rnn2vec_dump_embeddings:
                    self._dump_r2v_embeddings(self._r2v_update_count)
                if self._opts.rnn2vec_save_model:
                    self._save_r2v_model(self._r2v_update_count)
            except Exception:
                logger.warning("RNN2Vec online update failed", exc_info=True)

    def _on_r2v_update_done(self, update_count: int, stats: dict) -> None:
        """Callback from BackgroundRNN2VecUpdater when an update completes."""
        self._r2v_completion_queue.put((update_count, stats))

    def _process_r2v_completions(self) -> None:
        """Drain completion notifications from the R2V background updater."""
        if self._r2v_bg_updater is None:
            return
        import queue as _queue
        while True:
            try:
                update_num, stats = self._r2v_completion_queue.get_nowait()
            except _queue.Empty:
                break
            self._r2v_update_count = update_num
            pairs = stats.get("pairs_trained", 0)
            loss = stats.get("loss", 0.0)
            oov = stats.get("oov_skipped", 0)
            vocab_ext = stats.get("vocab_extended", 0)
            ext_str = f", vocab_extended={vocab_ext}" if vocab_ext > 0 else ""
            logger.info(
                "RNN2Vec bg update #%d complete: pairs=%d, oov_skipped=%d, loss=%.4f%s",
                update_num, pairs, oov, loss, ext_str,
            )
            if not self._opts.quiet:
                print(
                    f"\nNOTE: R2V bg update #{update_num} done:"
                    f" pairs={pairs}, oov_skipped={oov}{ext_str}, loss={loss:.4f}"
                )
            if self._opts.rnn2vec_dump_embeddings:
                self._dump_r2v_embeddings(update_num)
            if self._opts.rnn2vec_save_model:
                self._save_r2v_model(update_num)

    def _save_r2v_model(self, update_number: int) -> None:
        """Save the current RNN2Vec model to disk."""
        r2v = getattr(self._rnn2vec_provider, "_rnn2vec", None)
        if r2v is None:
            return
        try:
            r2v.save(self._opts.rnn2vec_save_model)
            if not self._opts.quiet:
                print(f"% RNN2Vec: model saved (update #{update_number}) "
                      f"→ {self._opts.rnn2vec_save_model}")
        except Exception:
            logger.warning("Failed to save RNN2Vec model to %r",
                           self._opts.rnn2vec_save_model, exc_info=True)

    def _process_initial_clauses(self) -> ExitCode | None:
        """Process initial clauses. Matches C index_and_process_initial_clauses().

        C Prover9 processes each initial SOS clause as a "given" clause in
        insertion order (FIFO), generating all inferences between initial
        clauses BEFORE the ratio-based selection loop begins. This is shown
        in C output as "given #N (I,wt=...)" lines.

        Steps:
        1. Orient equalities, weigh, and index usable clauses
        2. Process each SOS clause as "initial given": move to usable,
           index, generate inferences, process results
        3. Only after ALL initials are processed does the main loop start
        """
        # Orient equalities if paramodulation is enabled
        if self._opts.paramodulation:
            for c in self._state.usable:
                oriented = orient_equalities(c, self._symbol_table)
                if oriented is not c:
                    c.literals = oriented.literals
            for c in self._state.sos:
                oriented = orient_equalities(c, self._symbol_table)
                if oriented is not c:
                    c.literals = oriented.literals

        # Index usable clauses (they're already available for inference)
        for c in self._state.usable:
            c.weight = default_clause_weight(c)
            self._state.index_clashable(c, insert=True)
            self._subsump_idx.update(c, insert=True)
            self._back_subsump_idx.insert(c)
            self._unit_conflict_idx.insert(c)

        # Weigh SOS clauses and check for immediate empty clause
        for c in self._state.sos:
            c.weight = default_clause_weight(c)
            if c.is_empty:
                return self._handle_proof(c)

        # Process each initial SOS clause as a "given" clause in FIFO order.
        # This matches C Prover9's index_and_process_initial_clauses() which
        # selects each initial clause, moves it to usable, indexes it, and
        # generates inferences. The "(I)" selection type in C output.
        initial_sos = list(self._state.sos)
        for c in initial_sos:
            self._state.sos.remove(c)

            # Capture previous given's inference count and ID before overwriting
            prev_id = self._state.stats._current_given_id
            prev_count = self._state.stats.get_given_inference_count(
                prev_id
            ) if prev_id != 0 else -1

            self._state.stats.given += 1
            self._state.stats.begin_given(c.id, len(self._state.usable))

            c.given_selection = "I"

            # Print given clause with (I) for Initial, matching C format
            if self._opts.print_given and not self._opts.quiet:
                wt = int(c.weight) if c.weight == int(c.weight) else c.weight
                clause_str = self._format_clause_std(c)
                extras = self._format_selection_extras(c)
                prev_info = ""
                if self._opts.print_given_stats and prev_count >= 0:
                    compatible, available, percentage = self._state.stats.get_given_compatibility_stats(prev_id)
                    if available > 0:
                        prev_info = f"  [prev: {prev_count} inferences from {compatible}/{available} clauses ({percentage:.0f}%)]"
                    else:
                        prev_info = f"  [prev generated: {prev_count} inference{'s' if prev_count != 1 else ''}]"
                print(
                    f"given #{self._state.stats.given} "
                    f"(I,wt={wt}{extras}): {clause_str}{prev_info}"
                )

            # Move to usable and index (C: clist_append + index_clashable)
            self._state.usable.append(c)
            self._state.index_clashable(c, insert=True)
            self._subsump_idx.update(c, insert=True)
            self._back_subsump_idx.insert(c)
            self._unit_conflict_idx.insert(c)

            # Check demodulator status
            if self._opts.demodulation:
                if self._opts.paramodulation:
                    oriented = orient_equalities(c, self._symbol_table)
                    if oriented is not c:
                        c.literals = oriented.literals
                dtype = demodulator_type(
                    c, self._symbol_table, self._opts.lex_dep_demod_lim,
                )
                if dtype != DemodType.NOT_DEMODULATOR:
                    self._demod_index.insert(c, dtype)
                    self._state.stats.new_demodulators += 1
                    self._state.demods.append(c)
                    if self._lazy_demod is not None:
                        self._lazy_demod.bump_version()

            # Generate inferences from this initial clause
            exit_code = self._given_infer(c)
            if exit_code is not None:
                return exit_code

            # Process limbo (back-subsumption, move to SOS)
            exit_code = self._limbo_process()
            if exit_code is not None:
                return exit_code

        return None

    # ── Main loop components ────────────────────────────────────────────

    def _inferences_to_make(self) -> bool:
        """Check if there are clauses available for inference.

        Matches C inferences_to_make() → givens_available().
        """
        return not self._state.sos.is_empty

    def _make_inferences(self) -> ExitCode | None:
        """Select given clause and make inferences. Matches C make_inferences().

        1. Select given clause from SOS
        2. Check limits
        3. Move to usable, index
        4. Generate inferences (given_infer)
        """
        # Select given clause (C: get_given_clause2)
        # T2V variants: if the ratio cycle wants T2V/T2VM and we have
        # embeddings, use goal-proximity scoring instead of the age/FORTE fallback.
        current_order = self._selection._get_current_rule().order
        if self._tree2vec_embeddings and not self._state.sos.is_empty and current_order in (
            SelectionOrder.TREE2VEC,
            SelectionOrder.TREE2VEC_MAXIMIN,
        ):
            if current_order == SelectionOrder.TREE2VEC_MAXIMIN:
                given = self._t2v_select_maximin(self._state.sos)
                t2v_label = "T2VM"
            else:
                given = self._t2v_select_nearest_goal(self._state.sos)
                t2v_label = "T2V"
            if given is not None:
                self._state.sos.remove(given)
                rule = self._selection._get_current_rule()
                rule.selected += 1
                self._selection._advance_cycle()
                selection_type = t2v_label
            else:
                given, selection_type = self._selection.select_given(
                    self._state.sos, self._state.stats.given
                )
        elif self._rnn2vec_embeddings and not self._state.sos.is_empty and current_order in (
            SelectionOrder.RNN2VEC,
            SelectionOrder.RNN2VEC_RANDOM_GOAL,
        ):
            if current_order == SelectionOrder.RNN2VEC_RANDOM_GOAL:
                given = self._r2v_select_random_goal(self._state.sos)
                r2v_label = "RGP"
            else:
                # RNN2Vec diversity: pick clause most dissimilar from already-given
                given = self._r2v_select_most_diverse(self._state.sos)
                r2v_label = "R2V"
            if given is not None:
                self._state.sos.remove(given)
                rule = self._selection._get_current_rule()
                rule.selected += 1
                self._selection._advance_cycle()
                selection_type = r2v_label
            else:
                given, selection_type = self._selection.select_given(
                    self._state.sos, self._state.stats.given
                )
        else:
            given, selection_type = self._selection.select_given(
                self._state.sos,
                self._state.stats.given,
            )

        if given is None:
            return None  # SOS became empty during selection

        given.given_selection = selection_type

        # Lazy demodulation: ensure clause is fully reduced before use.
        # Must happen BEFORE distance assignment so that given_distance and
        # _t2v_all_given_distances are stored on the actual clause object
        # that will appear in the proof trace (demodulate_clause returns a new
        # Clause with the same id but different literals).
        if self._lazy_demod is not None and self._lazy_demod.needs_reduction(given):
            given = self._lazy_demod.ensure_fully_reduced(
                given, self._demod_index, self._symbol_table,
                self._opts.lex_order_vars, self._opts.demod_step_limit,
            )

        # Goal distance: compute and store on clause, track for trend reporting.
        # Done after lazy demodulation so the value lands on the right object.
        if self._opts.tree2vec_goal_proximity and self._t2v_goal_provider is not None:
            emb = self._tree2vec_embeddings.get(given.id)
            goal_scorer = getattr(self._t2v_goal_provider, '_goal_scorer', None)
            if emb is not None and goal_scorer is not None:
                gd = None
                # Cross-arg distance scoring
                if self._opts.tree2vec_cross_arg_proximity and self._t2v_goal_arg_embs:
                    gd = self._t2v_cross_arg_distance(emb, given.id)
                if gd is None:
                    gd = goal_scorer.nearest_goal_distance(emb)
                given.given_distance = gd
                self._t2v_distance_window.append(gd)
                self._t2v_all_given_distances[given.id] = gd

        # Capture previous given's inference count and ID before overwriting
        prev_id = self._state.stats._current_given_id
        prev_count = self._state.stats.get_given_inference_count(
            prev_id
        ) if prev_id != 0 else -1

        self._state.stats.given += 1
        # Don't count the given clause itself as "available" for meaningful compatibility metrics
        available_others = len(self._state.usable)  # Given not yet added to usable at this point
        self._state.stats.begin_given(given.id, available_others)

        # Check max_given limit (C: over_parm_limit check)
        if (
            self._opts.max_given > 0
            and self._state.stats.given > self._opts.max_given
        ):
            return ExitCode.MAX_GIVEN_EXIT

        # Tighten max_weight after a given-clause threshold (one-shot).
        if (
            self._opts.max_weight_tighten_after > 0
            and self._state.stats.given == self._opts.max_weight_tighten_after
            and self._opts.max_weight_tighten_to > 0
        ):
            old = self._opts.max_weight
            self._opts.max_weight = self._opts.max_weight_tighten_to
            self._opts.max_weight_tighten_after = 0  # disable so it only fires once
            if not self._opts.quiet:
                print(
                    f"\nNOTE: max_weight tightened from "
                    f"{old if old > 0 else 'unlimited'} → "
                    f"{self._opts.max_weight_tighten_to} "
                    f"at given #{self._state.stats.given}."
                )

        # Maybe disable back subsumption (C: backsub_check heuristic).
        # At given clause #N, check kept/back_subsumed ratio.
        # If ratio > 20, backward subsumption is unproductive — disable it.
        if (
            self._back_subsume_enabled
            and self._opts.backsub_check > 0
            and self._state.stats.given == self._opts.backsub_check
        ):
            bs = self._state.stats.back_subsumed
            ratio = (self._state.stats.kept // bs) if bs > 0 else 2**31
            if ratio > 20:
                self._back_subsume_enabled = False
                if not self._opts.quiet:
                    elapsed = self._state.stats.elapsed_seconds()
                    print(
                        f"\nNOTE: Back_subsumption disabled, ratio of kept"
                        f" to back_subsumed is {ratio} ({elapsed:.2f} sec)."
                    )

        # Periodic T2V goal-distance trend report
        if (
            self._opts.tree2vec_goal_proximity
            and self._t2v_goal_provider is not None
            and self._opts.tree2vec_proximity_report_interval > 0
            and len(self._t2v_distance_window) >= self._opts.tree2vec_proximity_report_interval
            and not self._opts.quiet
        ):
            avg = sum(self._t2v_distance_window) / len(self._t2v_distance_window)
            if self._t2v_distance_prev_avg > 0:
                trend = avg - self._t2v_distance_prev_avg
                # Negative trend = avg distance decreasing = getting closer to goals
                trend_str = f", trend={trend:+.2f}"
            else:
                trend_str = ""
            remaining = len(self._t2v_goal_clause_ids)
            print(
                f"\nNOTE: T2V goal distance at given #{self._state.stats.given}:"
                f" avg={avg:.2f}{trend_str},"
                f" goals_remaining={remaining}/{self._t2v_initial_goal_count}."
            )
            self._t2v_distance_prev_avg = avg
            self._t2v_distance_window.clear()

        # Check max_seconds limit
        if (
            self._opts.max_seconds > 0
            and self._state.stats.elapsed_seconds() > self._opts.max_seconds
        ):
            return ExitCode.MAX_SECONDS_EXIT

        # Print given clause (C: print_given) — output directly to stdout
        # matching C fwrite_clause() behavior in the search loop.
        if self._opts.print_given and not self._opts.quiet:
            wt = int(given.weight) if given.weight == int(given.weight) else given.weight
            clause_str = self._format_clause_std(given)
            extras = self._format_selection_extras(given)
            prev_info = ""
            if self._opts.print_given_stats and prev_count >= 0:
                compatible, available, percentage = self._state.stats.get_given_compatibility_stats(prev_id)
                if available > 0:
                    prev_info = f"  [prev: {prev_count} inferences from {compatible}/{available} clauses ({percentage:.0f}%)]"
                else:
                    prev_info = f"  [prev generated: {prev_count} inference{'s' if prev_count != 1 else ''}]"
            print(
                f"given #{self._state.stats.given} "
                f"({selection_type},wt={wt}{extras}): {clause_str}{prev_info}"
            )

        # Move to usable and index (C: clist_append + index_clashable)
        self._state.usable.append(given)
        self._state.index_clashable(given, insert=True)

        # Generate inferences (C: given_infer)
        return self._given_infer(given)

    def _given_infer(self, given: Clause) -> ExitCode | None:
        """Generate inferences from the given clause. Matches C given_infer().

        Uses parallel inference engine when conditions are met (enough usable
        clauses and free-threading available). Falls back to sequential otherwise.
        """
        usable_snapshot = list(self._state.usable)

        if self._parallel_engine.should_parallelize(len(usable_snapshot)):
            result = self._given_infer_parallel(given, usable_snapshot)
        else:
            result = self._given_infer_sequential(given, usable_snapshot)

        count = self._state.stats.get_given_inference_count(given.id)
        logger.debug(
            "given clause %d generated %d inferences", given.id, count,
        )
        return result

    def _given_infer_parallel(
        self, given: Clause, usable_snapshot: list[Clause],
    ) -> ExitCode | None:
        """Parallel inference: generate all inferences, then process sequentially."""
        inferences = self._parallel_engine.generate_inferences(
            given, usable_snapshot,
            binary_resolution=self._opts.binary_resolution,
            paramodulation=self._opts.paramodulation,
            factoring=self._opts.factoring,
            para_into_vars=self._opts.para_into_vars,
            symbol_table=self._symbol_table,
        )
        for clause in inferences:
            exit_code = self._cl_process(clause)
            if exit_code is not None:
                return exit_code
        return None

    def _given_infer_sequential(
        self, given: Clause, usable_snapshot: list[Clause],
    ) -> ExitCode | None:
        """Sequential inference generation (original algorithm)."""
        if self._opts.binary_resolution:
            for usable_clause in usable_snapshot:
                # Track attempted partnership with this specific clause
                self._state.stats.record_attempted_partnership(usable_clause.id)

                # Track if this clause pair produces any inferences
                pair_successful = False

                resolvents = all_binary_resolvents(given, usable_clause)
                if resolvents:
                    pair_successful = True
                    # Debug: print(f"DEBUG: {given.id} successfully resolved with {usable_clause.id}, generated {len(resolvents)} resolvents")
                for resolvent in resolvents:
                    exit_code = self._cl_process(resolvent)
                    if exit_code is not None:
                        return exit_code

                if usable_clause is not given:
                    resolvents2 = all_binary_resolvents(usable_clause, given)
                    if resolvents2:
                        pair_successful = True
                    for resolvent in resolvents2:
                        exit_code = self._cl_process(resolvent)
                        if exit_code is not None:
                            return exit_code

                # Record successful partnership with this specific clause
                if pair_successful:
                    self._state.stats.record_successful_partnership(usable_clause.id)

        if self._opts.paramodulation:
            st = self._symbol_table
            for usable_clause in usable_snapshot:
                # Track attempted paramodulation partnership with this specific clause
                self._state.stats.record_attempted_partnership(usable_clause.id)

                # Track if this clause pair produces any paramodulation inferences
                para_successful = False

                paras = para_from_into(
                    given, usable_clause, False, st,
                    self._opts.para_into_vars,
                )
                if paras:
                    para_successful = True
                for p in paras:
                    exit_code = self._cl_process(p)
                    if exit_code is not None:
                        return exit_code

                if usable_clause is not given:
                    paras2 = para_from_into(
                        usable_clause, given, True, st,
                        self._opts.para_into_vars,
                    )
                    if paras2:
                        para_successful = True
                    for p in paras2:
                        exit_code = self._cl_process(p)
                        if exit_code is not None:
                            return exit_code

                # Record successful partnership with this specific clause
                if para_successful:
                    self._state.stats.record_successful_partnership(usable_clause.id)

        if self._opts.hyper_resolution:
            # Extract nucleus patterns before hyperresolution (when enabled)
            if self._nucleus_pattern_cache is not None:
                cache_nucleus_patterns(given, self._nucleus_pattern_cache)

            # For hyper-resolution, we attempt inference with all usable clauses
            for usable_clause in usable_snapshot:
                self._state.stats.record_attempted_partnership(usable_clause.id)

            resolvents = indexed_hyper_resolution(given, self._state.clashable_idx, usable_snapshot)
            # CONSERVATIVE FIX: Hyper-resolution partnerships are complex to track precisely.
            # For now, if any resolvents were generated, mark successful partnerships with
            # all clauses that could potentially be involved. This may overcount slightly,
            # but is mathematically consistent (successes ≤ attempts).
            if resolvents:
                # Mark success with all usable clauses - conservative but consistent
                for usable_clause in usable_snapshot:
                    self._state.stats.record_successful_partnership(usable_clause.id)
            for resolvent in resolvents:
                exit_code = self._cl_process(resolvent)
                if exit_code is not None:
                    return exit_code

        if self._opts.factoring:
            # Factoring attempts inference with itself (self-partnership)
            self._state.stats.record_attempted_partnership(given.id)

            factors = factor(given)
            if factors:  # Track successful factoring as self-partnership
                self._state.stats.record_successful_partnership(given.id)
            for f in factors:
                exit_code = self._cl_process(f)
                if exit_code is not None:
                    return exit_code

        return None

    def _cl_process(self, c: Clause) -> ExitCode | None:
        """Process a newly inferred clause. Matches C cl_process().

        1. Increment generated count
        2. Simplify (demod, orient, merge, unit deletion)
        3. Check for empty clause → proof
        4. Delete checks (tautology, limits, forward subsumption)
        5. Keep: assign ID, index, add to limbo
        """
        # Check max_generated limit
        self._state.stats.record_generated()
        if (
            self._opts.max_generated > 0
            and self._state.stats.generated > self._opts.max_generated
        ):
            return ExitCode.MAX_GENERATED_EXIT

        if self._opts.print_gen:
            logger.debug("generated: %s", c.to_str())

        # Simplify (C: cl_process_simplify)
        c = self._simplify(c)

        # Check for empty clause (C: handle_proof_and_maybe_exit)
        if c.is_empty:
            return self._handle_proof(c)

        # Delete checks (C: cl_process_delete)
        if self._should_delete(c):
            return None

        # Keep the clause (C: cl_process_keep)
        return self._keep_clause(c)

    def _simplify(self, c: Clause) -> Clause:
        """Simplify a clause. Matches C cl_process_simplify().

        Applies demodulation (if enabled), then merge literals.
        """
        # Renumber variables first (C: renumber_variables before demod if lex_order_vars)
        # This ensures variable numbers stay in 0..MAX_VARS-1 range for matching
        c = renumber_variables(c)

        # Forward demodulation (C: demodulate_clause in cl_process_simplify)
        if self._opts.demodulation and not self._demod_index.is_empty:
            if self._lazy_demod is not None:
                # Lazy demod: skip eager demod, mark for deferred reduction
                self._lazy_demod.mark_partially_reduced(c)
                self._lazy_demod.stats.deferred_demods += 1
            else:
                c, steps = demodulate_clause(
                    c, self._demod_index, self._symbol_table,
                    self._opts.lex_order_vars, self._opts.demod_step_limit,
                )
                if steps:
                    self._state.stats.demodulated += 1

        # Simplify trivial equality literals (C: simplify_literals2):
        # Negative -(t=t) is always false → remove the literal
        c = self._simplify_eq_literals(c)

        if self._opts.merge_lits:
            c = merge_literals(c)

        return c

    def _simplify_eq_literals(self, c: Clause) -> Clause:
        """Remove negative reflexive equalities -(t=t). Matches C simplify_literals2().

        -(t=t) is always false, so the literal can be removed from the clause.
        """
        new_lits: list[Literal] = []
        changed = False
        for lit in c.literals:
            if (
                not lit.sign
                and is_eq_atom(lit.atom, self._symbol_table)
                and lit.atom.args[0].term_ident(lit.atom.args[1])
            ):
                changed = True
                continue
            new_lits.append(lit)

        if not changed:
            return c

        return Clause(
            literals=tuple(new_lits),
            id=c.id,
            weight=c.weight,
            justification=c.justification,
            is_formula=c.is_formula,
        )

    def _apply_hint_weight(self, c: Clause) -> None:
        """Apply hint weight bonus if any hint subsumes this clause.

        If a hint subsumes c, set c.weight = min(c.weight, hint_wt).
        This makes hint-matched clauses lighter, thus preferred by
        weight-based selection.
        """
        if not self._hints:
            return
        for hint in self._hints:
            if subsumes(hint, c):
                c.weight = min(c.weight, self._opts.hint_wt)
                return

    def _should_delete(self, c: Clause) -> bool:
        """Check if clause should be deleted. Matches C cl_process_delete().

        Checks: tautology, weight limits, forward subsumption.
        """
        # Tautology check (C: true_clause)
        if self._opts.check_tautology and is_tautology(c):
            self._state.stats.subsumed += 1
            return True

        # Weight the clause
        c.weight = default_clause_weight(c)

        # Penalty weight adjustment: inflate weight for high-penalty clauses
        if self._penalty_weight_config is not None:
            penalty = self._get_clause_penalty(c)
            adjusted = penalty_adjusted_weight(
                c.weight, penalty, self._penalty_weight_config,
            )
            if adjusted != c.weight:
                c.weight = adjusted
                self._state.stats.penalty_weight_adjusted += 1

        # Hint weight bonus: if a hint subsumes this clause, reduce weight
        self._apply_hint_weight(c)

        # Max weight check (C: cl_process_delete weight limit)
        if self._opts.max_weight > 0 and c.weight > self._opts.max_weight:
            self._state.stats.subsumed += 1
            return True

        # Max kept limit check
        if (
            self._opts.max_kept > 0
            and self._state.stats.kept >= self._opts.max_kept
        ):
            self._state.stats.sos_limit_deleted += 1
            return True

        # Forward subsumption check: is c subsumed by an existing clause?
        if self._forward_subsumed(c):
            self._state.stats.subsumed += 1
            return True

        return False

    def _get_clause_penalty(self, c: Clause) -> float:
        """Get combined penalty for a clause.

        Checks the penalty propagation cache first (O(1)), then falls back
        to computing the intrinsic generality penalty. Adds repetition
        penalty when configured.
        """
        penalty = 0.0

        # Try penalty cache first (populated by penalty propagation)
        if self._penalty_cache is not None:
            penalty = self._penalty_cache.get_combined(c.id)
        else:
            # No cache: compute intrinsic generality penalty
            penalty = _clause_generality_penalty(c)

        # Add repetition penalty if configured (cache for reuse in _limbo_process)
        if self._repetition_config is not None:
            rep = compute_repetition_penalty(c, self._repetition_config)
            if rep > 0.0 and c.id != 0:
                self._rep_penalty_cache[c.id] = rep
            penalty += rep

        # Add nucleus unification penalty if configured
        if self._nucleus_penalty_config is not None and self._nucleus_pattern_cache is not None:
            penalty += compute_nucleus_unification_penalty(
                c, self._nucleus_penalty_config,
            )

        return penalty

    def _forward_subsumed(self, c: Clause) -> bool:
        """Check if c is forward subsumed by any existing clause.

        Uses indexed subsumption via LiteralIndex (matching C forward_subsumption
        with lindex). Retrieves generalizations of c's literals from the index.
        """
        subsumer = forward_subsume(
            c, self._subsump_idx.pos, self._subsump_idx.neg,
        )
        if subsumer is not None:
            if (
                self._opts.learn_from_forward_subsumption
                and self._forward_subsumption_callback is not None
            ):
                self._forward_subsumption_callback(subsumer, c)
            return True
        return False

    def _keep_clause(self, c: Clause) -> ExitCode | None:
        """Keep a clause. Matches C cl_process_keep().

        Assigns ID, renumbers variables, indexes, and puts in limbo.
        """
        self._state.stats.kept += 1
        c.normal_vars = True

        if c.id == 0:
            self._state.assign_clause_id(c)

        self._all_clauses[c.id] = c

        # Compute and cache penalty propagation (after ID assignment)
        if self._penalty_cache is not None:
            compute_and_cache_penalty(c, self._penalty_cache, self._all_clauses)

        # Compute and cache FORTE embedding for kept clause
        if self._forte_provider is not None:
            emb = self._forte_provider.get_embedding(c)  # type: ignore[union-attr]
            if emb is not None:
                self._forte_embeddings[c.id] = emb

        # Compute and cache Tree2Vec embedding for kept clause
        # Use goal-directed provider when available for proximity-enhanced embeddings
        t2v_emb_provider = self._t2v_goal_provider or self._tree2vec_provider
        if t2v_emb_provider is not None and callable(getattr(t2v_emb_provider, 'get_embedding', None)):
            emb = t2v_emb_provider.get_embedding(c)  # type: ignore[union-attr]
            if emb is not None:
                self._tree2vec_embeddings[c.id] = emb

            # Cross-arg proximity: also embed the antecedent term
            if self._opts.tree2vec_cross_arg_proximity and self._tree2vec_provider is not None:
                ant_term = _get_antecedent_term(c)
                if ant_term is not None:
                    t2v_algo = getattr(self._tree2vec_provider, '_tree2vec', None)
                    if t2v_algo is not None:
                        ant_emb = t2v_algo.embed_term(ant_term)
                        if ant_emb is not None:
                            self._t2v_antecedent_embeddings[c.id] = ant_emb

        # Online Tree2Vec learning: accumulate batch, trigger update at interval
        if (
            self._opts.tree2vec_online_learning
            and self._tree2vec_provider is not None
            and callable(getattr(self._tree2vec_provider, 'bump_model_version', None))
        ):
            max_updates = self._opts.tree2vec_online_max_updates
            if max_updates == 0 or self._t2v_update_count < max_updates:
                if len(self._t2v_online_batch) < self._opts.tree2vec_online_batch_size:
                    self._t2v_online_batch.append(c)
                self._t2v_kept_since_update += 1
                if self._t2v_kept_since_update >= self._opts.tree2vec_online_update_interval:
                    self._do_t2v_online_update()

        # Compute and cache RNN2Vec embedding for kept clause
        if self._rnn2vec_provider is not None and callable(getattr(self._rnn2vec_provider, 'get_embedding', None)):
            emb = self._rnn2vec_provider.get_embedding(c)
            if emb is not None:
                self._rnn2vec_embeddings[c.id] = emb

        # Online RNN2Vec learning: accumulate batch, trigger update at interval
        if (
            self._opts.rnn2vec_online_learning
            and self._rnn2vec_provider is not None
            and callable(getattr(self._rnn2vec_provider, 'bump_model_version', None))
        ):
            max_updates = self._opts.rnn2vec_online_max_updates
            if max_updates == 0 or self._r2v_update_count < max_updates:
                if len(self._r2v_online_batch) < self._opts.rnn2vec_online_batch_size:
                    self._r2v_online_batch.append(c)
                self._r2v_kept_since_update += 1
                if self._r2v_kept_since_update >= self._opts.rnn2vec_online_update_interval:
                    self._do_r2v_online_update()

        if self._opts.print_kept:
            logger.info("kept:      %s", self._format_clause_std(c))

        # Check max_kept limit
        if (
            self._opts.max_kept > 0
            and self._state.stats.kept > self._opts.max_kept
        ):
            return ExitCode.MAX_KEPT_EXIT

        # Unit conflict check (C: cl_process_conflict)
        exit_code = self._unit_conflict(c)
        if exit_code is not None:
            return exit_code

        # Check if new clause is a demodulator (C: cl_process_new_demod)
        if self._opts.demodulation:
            # Orient equalities first (must use return value for flipped atoms)
            if self._opts.paramodulation:
                oriented = orient_equalities(c, self._symbol_table)
                if oriented is not c:
                    c.literals = oriented.literals
            dtype = demodulator_type(
                c, self._symbol_table, self._opts.lex_dep_demod_lim,
            )
            if dtype != DemodType.NOT_DEMODULATOR:
                self._demod_index.insert(c, dtype)
                self._state.stats.new_demodulators += 1
                self._state.demods.append(c)
                if self._lazy_demod is not None:
                    self._lazy_demod.bump_version()

        # Index for forward subsumption (C: index_literals / lindex_update_first)
        self._subsump_idx.update(c, insert=True)
        self._back_subsump_idx.insert(c)
        self._unit_conflict_idx.insert(c)
        # Add to limbo (C: clist_append(c, Glob.limbo))
        self._state.limbo.append(c)

        return None

    def _do_t2v_online_update(self) -> None:
        """Trigger a Tree2Vec online update.

        If the background updater is running, submits the batch to its queue
        and returns immediately.  Otherwise falls back to the synchronous
        implementation.
        """
        batch = self._t2v_online_batch
        if not batch:
            self._t2v_kept_since_update = 0
            return

        if self._t2v_bg_updater is not None:
            accepted = self._t2v_bg_updater.submit(list(batch))
            if accepted:
                logger.debug(
                    "Tree2Vec online update #%d submitted to background thread "
                    "(given=%d, batch=%d clauses)",
                    self._t2v_update_count + 1,
                    self._state.stats.given,
                    len(batch),
                )
            else:
                logger.debug("Tree2Vec background updater: batch dropped (queue full or limit reached)")
        else:
            self._do_t2v_online_update_sync()

        self._t2v_online_batch = []
        self._t2v_kept_since_update = 0

    def _do_t2v_online_update_sync(self) -> None:
        """Perform an online Tree2Vec update from the accumulated batch (synchronous)."""
        batch = self._t2v_online_batch
        if not batch:
            self._t2v_kept_since_update = 0
            return

        self._t2v_update_count += 1
        update_num = self._t2v_update_count
        provider = self._tree2vec_provider
        lr = self._opts.tree2vec_online_lr
        logger.info(
            "Tree2Vec online update #%d starting: given=%d, batch=%d clauses, lr=%.5f",
            update_num, self._state.stats.given, len(batch), lr,
        )
        try:
            stats = provider._tree2vec.update_online(  # type: ignore[union-attr]
                batch,
                learning_rate=lr,
            )
            new_version = provider.bump_model_version()  # type: ignore[union-attr]
            pairs = stats.get("pairs_trained", 0)
            oov = stats.get("oov_skipped", 0)
            vocab_ext = stats.get("vocab_extended", 0)
            loss = stats.get("loss", 0.0)

            logger.info(
                "Tree2Vec online update #%d done: pairs=%d, oov_skipped=%d, loss=%.4f,"
                " model_v=%d%s",
                update_num, pairs, oov, loss, new_version,
                f", vocab_extended={vocab_ext}" if vocab_ext > 0 else "",
            )

            # Re-embed all SOS clauses with the updated model
            sos_reembedded = 0
            t2v_emb_provider = self._t2v_goal_provider or self._tree2vec_provider
            if t2v_emb_provider is not None and callable(getattr(t2v_emb_provider, 'get_embedding', None)):
                for c in self._state.sos:
                    emb = t2v_emb_provider.get_embedding(c)  # type: ignore[union-attr]
                    if emb is not None:
                        self._tree2vec_embeddings[c.id] = emb
                        sos_reembedded += 1
            logger.info(
                "Tree2Vec online update #%d: re-embedded %d/%d SOS clauses",
                update_num, sos_reembedded, len(self._state.sos),
            )

            # Re-embed SOS antecedent terms for cross-arg proximity
            if self._opts.tree2vec_cross_arg_proximity:
                t2v_algo = getattr(provider, '_tree2vec', None)
                if t2v_algo is not None:
                    ant_reembedded = 0
                    for c in self._state.sos:
                        ant_term = _get_antecedent_term(c)
                        if ant_term is not None:
                            ant_emb = t2v_algo.embed_term(ant_term)
                            if ant_emb is not None:
                                self._t2v_antecedent_embeddings[c.id] = ant_emb
                                ant_reembedded += 1
                    logger.info(
                        "Tree2Vec online update #%d: re-embedded %d antecedent terms (cross-arg)",
                        update_num, ant_reembedded,
                    )
                    # Re-embed goal arg/ant embeddings
                    if self._t2v_goal_clauses:
                        self._t2v_goal_arg_embs.clear()
                        self._t2v_goal_ant_embs.clear()
                        for gc in self._t2v_goal_clauses:
                            gc_norm = _deskolemize_clause(gc)
                            if gc_norm.literals and gc_norm.literals[0].atom.arity >= 1:
                                arg_emb = t2v_algo.embed_term(gc_norm.literals[0].atom.args[0])
                                if arg_emb is not None:
                                    self._t2v_goal_arg_embs.append(arg_emb)
                                    ant_term = _get_antecedent_term(gc_norm)
                                    if ant_term is not None:
                                        a_emb = t2v_algo.embed_term(ant_term)
                                        self._t2v_goal_ant_embs.append(a_emb if a_emb is not None else arg_emb)
                                    else:
                                        self._t2v_goal_ant_embs.append(arg_emb)
                        logger.info(
                            "Tree2Vec online update #%d: re-embedded %d goal cross-arg embeddings",
                            update_num, len(self._t2v_goal_arg_embs),
                        )

            # Re-embed goal clauses if goal-proximity is active
            if self._opts.tree2vec_goal_proximity and self._t2v_goal_clauses:
                goal_provider = getattr(self, '_t2v_goal_provider', None)
                if goal_provider is not None and hasattr(goal_provider, 'register_goals'):
                    goal_provider.register_goals(self._t2v_goal_clauses)
                    n_goals = len(self._t2v_goal_clauses)
                    logger.info(
                        "Tree2Vec online update #%d: re-embedded %d goal clause embeddings",
                        update_num, n_goals,
                    )

            if not self._opts.quiet:
                ext_str = f", vocab_extended={vocab_ext}" if vocab_ext > 0 else ""
                print(
                    f"\nNOTE: T2V online update #{update_num}"
                    f" (given={self._state.stats.given}, batch={len(batch)} clauses):"
                    f" pairs={pairs}, oov_skipped={oov}{ext_str},"
                    f" loss={loss:.4f}, model_v={new_version},"
                    f" sos_reembedded={sos_reembedded}."
                )
        except Exception:
            logger.warning(
                "Tree2Vec online update #%d failed, continuing",
                update_num,
                exc_info=True,
            )
            if not self._opts.quiet:
                print(f"\nNOTE: T2V online update #{update_num} failed, continuing.")

        self._t2v_online_batch = []
        self._t2v_kept_since_update = 0
        if self._opts.tree2vec_dump_embeddings:
            self._dump_t2v_embeddings(update_num)

    def _on_t2v_update_done(self, update_count: int, stats: dict) -> None:
        """Completion callback called from the background updater thread.

        Must only post to thread-safe structures — no direct mutation of
        main-thread state here.
        """
        self._t2v_completion_queue.put((update_count, stats))

    def _process_t2v_completions(self) -> None:
        """Drain completion notifications from the background updater.

        Called from the main thread at the start of each given-clause step
        to process any updates that finished while the main loop was running.
        """
        if self._t2v_bg_updater is None:
            return
        import queue as _queue
        while True:
            try:
                update_num, stats = self._t2v_completion_queue.get_nowait()
            except _queue.Empty:
                break
            self._t2v_update_count = update_num
            # Clear stale antecedent/goal embeddings — they recompute lazily
            self._t2v_antecedent_embeddings.clear()
            self._t2v_goal_arg_embs.clear()
            self._t2v_goal_ant_embs.clear()
            # Re-register goals now that model weights have changed
            if self._opts.tree2vec_goal_proximity and self._t2v_goal_clauses:
                goal_provider = getattr(self, "_t2v_goal_provider", None)
                if goal_provider is not None and hasattr(goal_provider, "register_goals"):
                    goal_provider.register_goals(self._t2v_goal_clauses)
            pairs = stats.get("pairs_trained", 0)
            loss = stats.get("loss", 0.0)
            oov = stats.get("oov_skipped", 0)
            vocab_ext = stats.get("vocab_extended", 0)
            ext_str = f", vocab_extended={vocab_ext}" if vocab_ext > 0 else ""
            logger.info(
                "Tree2Vec bg update #%d complete: pairs=%d, oov_skipped=%d, loss=%.4f%s",
                update_num, pairs, oov, loss, ext_str,
            )
            if not self._opts.quiet:
                print(
                    f"\nNOTE: T2V bg update #{update_num} done:"
                    f" pairs={pairs}, oov_skipped={oov}{ext_str}, loss={loss:.4f}"
                )
            if self._opts.tree2vec_dump_embeddings:
                self._dump_t2v_embeddings(update_num)

    def _dump_t2v_embeddings(self, update_number: int) -> None:
        """Write SOS clause embeddings (plus any goal clauses) to a JSON file.

        Goal clauses are identified by JustType.DENY justification and are
        always included regardless of whether tree2vec_goal_proximity is
        enabled or whether they have already been processed as given clauses.
        Overwrites the file on every call so the latest state is always readable.
        """
        import json
        from pathlib import Path
        from datetime import datetime

        path = self._opts.tree2vec_dump_embeddings
        provider = self._tree2vec_provider
        if provider is None or not callable(getattr(provider, "get_embedding", None)):
            return

        # Clause IDs that appear in any proof found so far
        proof_ids: set[int] = {
            c.id for proof in self._proofs for c in proof.clauses
        }

        def _is_goal(clause: Clause) -> bool:
            """True for negated-goal (DENY) clauses regardless of any options."""
            return bool(
                clause.justification
                and clause.justification[0].just_type == JustType.DENY
            )

        # Collect SOS clauses first, tracking which IDs are already present
        seen_ids: set[int] = set()
        to_dump: list[Clause] = []
        for clause in self._state.sos:
            seen_ids.add(clause.id)
            to_dump.append(clause)

        # Add goal clauses that have already been processed (moved to usable)
        # so they always appear in the output even after being given.
        for clause in self._state.usable:
            if clause.id not in seen_ids and _is_goal(clause):
                seen_ids.add(clause.id)
                to_dump.append(clause)

        # Model metadata
        t2v = getattr(provider, "_tree2vec", None)
        model_meta: dict = {
            "update_number": update_number,
            "model_version": getattr(provider, "model_version", None),
            "vocab_size": t2v.vocab_size if t2v is not None else None,
            "embedding_dim": t2v.embedding_dim if t2v is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

        from pyladr.search.goal_directed import _deskolemize_clause

        entries: list[dict] = []
        for clause in to_dump:
            goal = _is_goal(clause)
            if goal:
                # Deskolemize before embedding: replace Skolem constants with
                # variables and force positive sign, matching what the goal-
                # proximity scorer uses.  This ensures the embedded vector
                # represents the pure structural shape (e.g. P(i(x,y))) rather
                # than specific constant identities the model has never seen.
                embed_clause = _deskolemize_clause(clause)
            else:
                embed_clause = clause
            emb = provider.get_embedding(embed_clause)  # type: ignore[union-attr]
            entries.append({
                "id": clause.id,
                "clause": self._format_clause_std(clause),
                "weight": clause.weight,
                "embedding": emb,
                "in_proof": clause.id in proof_ids,
                "is_goal": goal,
            })

        blob = {
            "format_version": 1,
            "model": model_meta,
            "clauses": entries,
        }
        try:
            Path(path).write_text(json.dumps(blob, indent=2), encoding="utf-8")
            n_goal = sum(1 for e in entries if e["is_goal"])
            logger.debug(
                "T2V embeddings dumped: %d clauses (%d goal) → %s (update #%d)",
                len(entries), n_goal, path, update_number,
            )
        except OSError as exc:
            logger.warning("Failed to write T2V embedding dump to %r: %s", path, exc)

    def _dump_r2v_embeddings(self, update_number: int) -> None:
        """Write RNN2Vec SOS clause embeddings to a JSON file.

        Same format as _dump_t2v_embeddings. Goal clauses are deskolemized
        before embedding to match what the goal-proximity scorer uses.
        Overwrites the file on every call so the latest state is readable.
        """
        import json
        from pathlib import Path
        from datetime import datetime

        path = self._opts.rnn2vec_dump_embeddings
        provider = self._rnn2vec_provider
        if provider is None or not callable(getattr(provider, "get_embedding", None)):
            return

        proof_ids: set[int] = {
            c.id for proof in self._proofs for c in proof.clauses
        }

        def _is_goal(clause: Clause) -> bool:
            return bool(
                clause.justification
                and clause.justification[0].just_type == JustType.DENY
            )

        seen_ids: set[int] = set()
        to_dump: list[Clause] = []
        for clause in self._state.sos:
            seen_ids.add(clause.id)
            to_dump.append(clause)
        for clause in self._state.usable:
            if clause.id not in seen_ids and _is_goal(clause):
                seen_ids.add(clause.id)
                to_dump.append(clause)

        r2v = getattr(provider, "_rnn2vec", None)
        model_meta: dict = {
            "update_number": update_number,
            "model_version": getattr(provider, "model_version", None),
            "vocab_size": r2v.vocab_size if r2v is not None else None,
            "embedding_dim": r2v.embedding_dim if r2v is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

        from pyladr.search.goal_directed import _deskolemize_clause

        entries: list[dict] = []
        for clause in to_dump:
            goal = _is_goal(clause)
            embed_clause = _deskolemize_clause(clause) if goal else clause
            emb = provider.get_embedding(embed_clause)  # type: ignore[union-attr]
            entries.append({
                "id": clause.id,
                "clause": self._format_clause_std(clause),
                "weight": clause.weight,
                "embedding": emb,
                "in_proof": clause.id in proof_ids,
                "is_goal": goal,
            })

        blob = {
            "format_version": 1,
            "model": model_meta,
            "clauses": entries,
        }
        try:
            Path(path).write_text(json.dumps(blob, indent=2), encoding="utf-8")
            n_goal = sum(1 for e in entries if e["is_goal"])
            logger.debug(
                "R2V embeddings dumped: %d clauses (%d goal) → %s (update #%d)",
                len(entries), n_goal, path, update_number,
            )
        except OSError as exc:
            logger.warning("Failed to write R2V embedding dump to %r: %s", path, exc)

    def _t2v_cross_arg_distance(self, emb_full: "list[float]", clause_id: int) -> float | None:
        """Compute cross-argument cosine distance for a clause.

        Returns (1 - max_cross_sim) / 2 mapped to [0, 1] where 0 = very
        similar to some goal arg/antecedent, or None if cross-arg scoring
        is not applicable (no antecedent or no goal embeddings).
        """
        if not self._t2v_goal_arg_embs:
            return None
        emb_ant = self._t2v_antecedent_embeddings.get(clause_id)
        if emb_ant is None:
            return None
        best_sim = -1.0
        for goal_arg, goal_ant in zip(self._t2v_goal_arg_embs, self._t2v_goal_ant_embs):
            sim1 = _t2v_cosine(emb_ant, goal_arg)   # x1 vs i(x2,y2)
            sim2 = _t2v_cosine(goal_ant, emb_full)   # x2 vs i(x1,y1)
            best_sim = max(best_sim, sim1, sim2)
        return (1.0 - best_sim) / 2.0

    def _t2v_select_nearest_goal(self, sos: object) -> "Clause | None":
        """Select the SOS clause with the smallest cosine distance to any goal.

        When cross-arg distance scoring is enabled, uses cross-argument scoring
        for clauses with extractable antecedents. Falls back to the standard
        goal-distance scorer for clauses without antecedents.
        """
        if not self._tree2vec_embeddings:
            return None

        use_cross_arg = (
            self._opts.tree2vec_cross_arg_proximity
            and self._t2v_goal_arg_embs
        )

        # Get goal scorer from the goal-directed provider if present
        goal_scorer = None
        goal_provider = getattr(self, "_t2v_goal_provider", None)
        if goal_provider is not None:
            goal_scorer = getattr(goal_provider, "_goal_scorer", None)

        best_clause: "Clause | None" = None
        best_dist = float("inf")
        for c in sos:  # type: ignore[union-attr]
            emb = self._tree2vec_embeddings.get(c.id)
            if emb is None:
                continue
            dist: float | None = None
            if use_cross_arg:
                dist = self._t2v_cross_arg_distance(emb, c.id)
            if dist is None:
                # Fallback to standard goal scorer or weight-based
                if goal_scorer is not None:
                    dist = goal_scorer.nearest_goal_distance(emb)
                else:
                    dist = c.weight / (1.0 + c.weight)
            if dist < best_dist:
                best_dist = dist
                best_clause = c

        return best_clause

    def _t2v_select_maximin(self, sos: object) -> "Clause | None":
        """Select the SOS clause with the smallest farthest-goal distance (minimax distance).

        For each candidate clause computes the cosine distance to its *farthest*
        goal (i.e. max distance across all goals), then returns the clause with
        the smallest such maximum distance.  This is the minimax-distance
        criterion: it favours clauses broadly close to every goal over those
        specialised to just one goal.

        Equivalently, in proximity terms this is a maximin selection: the
        clause with the highest floor similarity across all goals.

        Falls back to weight-based scoring when no goal scorer is available.
        """
        if not self._tree2vec_embeddings:
            return None

        goal_scorer = None
        goal_provider = getattr(self, "_t2v_goal_provider", None)
        if goal_provider is not None:
            goal_scorer = getattr(goal_provider, "_goal_scorer", None)

        best_clause: "Clause | None" = None
        best_dist = float("inf")
        for c in sos:  # type: ignore[union-attr]
            emb = self._tree2vec_embeddings.get(c.id)
            if emb is None:
                continue
            if goal_scorer is not None:
                dist = goal_scorer.farthest_goal_distance(emb)
            else:
                dist = c.weight / (1.0 + c.weight)
            if dist < best_dist:
                best_dist = dist
                best_clause = c

        return best_clause

    def _unit_conflict(self, c: Clause) -> ExitCode | None:
        """Check for unit conflict. Matches C cl_process_conflict().

        If c is a unit clause, check if its complement exists in usable
        or limbo. If so, we can derive the empty clause.
        Uses O(1) hash-based lookup via UnitConflictIndex.
        """
        other = self._unit_conflict_idx.find_complement(c)
        if other is None:
            return None

        # Found unit conflict — derive empty clause
        self._state.stats.unit_conflicts += 1
        just = Justification(
            just_type=JustType.BINARY_RES,
            clause_ids=(c.id, other.id),
        )
        empty = Clause(
            literals=(),
            justification=(just,),
        )
        self._state.assign_clause_id(empty)
        self._all_clauses[empty.id] = empty
        return self._handle_proof(empty)

    def _limbo_process(self) -> ExitCode | None:
        """Process limbo list. Matches C limbo_process().

        Move limbo clauses to SOS. Apply backward subsumption:
        each new clause may subsume previously kept clauses.
        """
        while not self._state.limbo.is_empty:
            c = self._state.limbo.pop_first()
            if c is None:
                continue

            # Backward subsumption: remove clauses subsumed by c
            # Uses hash-based BackSubsumptionIndex for O(1) candidate retrieval
            # Gated by backsub_check heuristic (C: back_subsume flag)
            back_subsumed = (
                back_subsume_indexed(c, self._back_subsump_idx)
                if self._back_subsume_enabled
                else []
            )
            for victim in back_subsumed:
                self._state.stats.back_subsumed += 1
                # Remove from whichever list it's in and deindex
                self._state.index_clashable(victim, insert=False)
                self._subsump_idx.update(victim, insert=False)
                self._back_subsump_idx.remove(victim)
                self._unit_conflict_idx.remove(victim)
                self._state.usable.remove(victim)
                self._state.sos.remove(victim)
                self._state.disabled.append(victim)
                # Evict from penalty cache to bound memory growth
                if self._penalty_cache is not None:
                    self._penalty_cache.remove(victim.id)
                # Evict FORTE/Tree2Vec embeddings for disabled clause
                self._forte_embeddings.pop(victim.id, None)
                self._tree2vec_embeddings.pop(victim.id, None)
                logger.debug(
                    "back subsumed: %s by %s",
                    self._format_clause_std(victim),
                    self._format_clause_std(c),
                )
                if (
                    self._opts.learn_from_back_subsumption
                    and self._back_subsumption_callback is not None
                ):
                    self._back_subsumption_callback(c, victim)
                if (
                    self._opts.tree2vec_goal_proximity
                    and self._t2v_goal_provider is not None
                ):
                    self._on_goal_subsumed(victim)

            # Back-demodulation: new demodulators rewrite kept clauses
            if self._opts.back_demod and self._opts.demodulation:
                dtype = demodulator_type(
                    c, self._symbol_table, self._opts.lex_dep_demod_lim,
                )
                if dtype != DemodType.NOT_DEMODULATOR:
                    # Find clauses in usable/sos that can be rewritten
                    # Use chain() to avoid materializing full list
                    rewritable = back_demodulatable(
                        c, dtype,
                        chain(self._state.usable, self._state.sos),
                        self._symbol_table,
                        self._opts.lex_order_vars,
                    )
                    for victim in rewritable:
                        self._state.stats.back_demodulated += 1
                        # Disable original, reprocess the rewritten version
                        self._state.index_clashable(victim, insert=False)
                        self._subsump_idx.update(victim, insert=False)
                        self._back_subsump_idx.remove(victim)
                        self._unit_conflict_idx.remove(victim)
                        self._state.usable.remove(victim)
                        self._state.sos.remove(victim)
                        self._state.disabled.append(victim)
                        # Evict from penalty cache to bound memory growth
                        if self._penalty_cache is not None:
                            self._penalty_cache.remove(victim.id)
                        # Evict FORTE/Tree2Vec embeddings for disabled clause
                        self._forte_embeddings.pop(victim.id, None)
                        self._tree2vec_embeddings.pop(victim.id, None)

                        # Create copy with back-demod justification
                        back_just = Justification(
                            just_type=JustType.BACK_DEMOD,
                            clause_id=victim.id,
                        )
                        rewritten = Clause(
                            literals=victim.literals,
                            justification=victim.justification + (back_just,),
                        )
                        # Re-process (will be demodulated by _simplify)
                        exit_code = self._cl_process(rewritten)
                        if exit_code is not None:
                            return exit_code

            # SOS limit check (C: sos_limit — discard if SOS is full)
            if (
                self._opts.sos_limit > 0
                and len(self._state.sos) >= self._opts.sos_limit
            ):
                self._state.stats.sos_limit_deleted += 1
            else:
                # Pass combined penalty and FORTE score to PrioritySOS for heap ordering
                penalty_val: float | None = None
                if self._penalty_cache is not None:
                    rec = self._penalty_cache.get(c.id)
                    if rec is not None:
                        penalty_val = rec.combined_penalty
                if self._repetition_config is not None:
                    rep = self._rep_penalty_cache.pop(c.id, None)
                    if rep is None:
                        rep = compute_repetition_penalty(c, self._repetition_config)
                    if rep > 0.0:
                        penalty_val = (penalty_val or 0.0) + rep
                # Compute FORTE score from stored embedding
                forte_val: float | None = None
                if self._forte_provider is not None:
                    emb = self._forte_embeddings.get(c.id)
                    if emb is not None:
                        forte_val = _forte_novelty_score(emb)
                if isinstance(self._state.sos, PrioritySOS) and (penalty_val is not None or forte_val is not None):
                    self._state.sos.append(c, penalty_override=penalty_val, forte_score=forte_val)
                else:
                    self._state.sos.append(c)
        return None

    # ── Proof handling ──────────────────────────────────────────────────

    def _handle_proof(self, empty: Clause) -> ExitCode | None:
        """Handle discovery of an empty clause (proof). Matches C handle_proof_and_maybe_exit().

        Collects the proof trace and checks if we've found enough proofs.
        """
        self._state.stats.proofs += 1
        self._state.stats.empty_clauses_found += 1

        if empty.id == 0:
            self._state.assign_clause_id(empty)
            self._all_clauses[empty.id] = empty

        # Build proof trace (all clauses in derivation)
        proof_clauses = self._trace_proof(empty)

        # Remove proven goal clauses from T2V proximity list
        if self._opts.tree2vec_goal_proximity and self._t2v_goal_clause_ids:
            for c in proof_clauses:
                if c.id in self._t2v_goal_clause_ids:
                    self._on_goal_subsumed(c)

        proof = Proof(
            empty_clause=empty,
            clauses=tuple(proof_clauses),
        )
        self._proofs.append(proof)

        logger.info("PROOF FOUND (proof %d)", self._state.stats.proofs)
        print(f"PROOF FOUND (proof {self._state.stats.proofs})")

        # Print per-proof T2V goal-distance histogram
        if self._opts.tree2vec_goal_proximity and self._t2v_goal_provider is not None:
            histogram = self._compute_t2v_histogram(proof)
            if histogram is not None and not self._opts.quiet:
                print(format_t2v_histogram(histogram, self._state.stats.proofs))

            # Print cumulative histogram when 2+ proofs have been found
            if len(self._proofs) > 1 and not self._opts.quiet:
                cumulative = self._compute_t2v_cumulative_histogram()
                if cumulative is not None:
                    print(format_t2v_histogram(cumulative, proof_num=None))

        # Record proof clause embeddings in proof pattern memory
        if self._proof_pattern_memory is not None and self._forte_provider is not None:
            try:
                self._record_proof_patterns(proof)
            except Exception:
                logger.debug("Failed to record proof patterns", exc_info=True)

        if self._proof_callback is not None:
            try:
                self._proof_callback(proof, self._state.stats.proofs)
            except Exception:
                logger.exception("proof_callback raised an exception")

        if (
            self._opts.max_proofs > 0
            and self._state.stats.proofs >= self._opts.max_proofs
        ):
            return ExitCode.MAX_PROOFS_EXIT

        return None

    def _trace_proof(self, empty: Clause) -> list[Clause]:
        """Trace back through justifications to find all clauses in the proof.

        Matches C proof_id_set() / get_clause_by_id() pattern.
        """
        visited: set[int] = set()
        proof_clauses: list[Clause] = []
        stack = [empty]

        while stack:
            c = stack.pop()
            if c.id in visited:
                continue
            visited.add(c.id)
            proof_clauses.append(c)

            # Follow justification chain
            for just in c.justification:
                # Follow clause_ids references
                for cid in just.clause_ids:
                    if cid in self._all_clauses and cid not in visited:
                        stack.append(self._all_clauses[cid])
                # Follow single clause_id
                if just.clause_id > 0 and just.clause_id in self._all_clauses:
                    if just.clause_id not in visited:
                        stack.append(self._all_clauses[just.clause_id])
                # Follow para justification
                if just.para is not None:
                    for pid in (just.para.from_id, just.para.into_id):
                        if pid in self._all_clauses and pid not in visited:
                            stack.append(self._all_clauses[pid])

        # Sort by ID for deterministic output
        proof_clauses.sort(key=lambda c: c.id)
        return proof_clauses

    def _record_proof_patterns(self, proof: Proof) -> None:
        """Record embeddings of proof clauses into proof pattern memory.

        For each clause in the proof trace, obtains its FORTE embedding
        (from the cache or computed fresh) and records it as a successful
        proof pattern. Clauses that already have cached embeddings are
        preferred (they were actively used during search).
        """
        provider = self._forte_provider
        memory = self._proof_pattern_memory
        embeddings_cache = self._forte_embeddings

        proof_embeddings: list[list[float]] = []
        for clause in proof.clauses:
            # Prefer cached embedding (clause was actively used in search)
            emb = embeddings_cache.get(clause.id)
            if emb is None:
                # Compute fresh embedding for proof clauses not in cache
                emb = provider.get_embedding(clause)  # type: ignore[union-attr]
            if emb is not None:
                proof_embeddings.append(emb)

        if proof_embeddings:
            memory.record_proof(proof_embeddings)  # type: ignore[union-attr]
            logger.info(
                "Recorded %d proof pattern embeddings (total patterns: %d, proofs: %d)",
                len(proof_embeddings),
                memory.pattern_count,  # type: ignore[union-attr]
                memory.proof_count,  # type: ignore[union-attr]
            )

    # ── Clause formatting ──────────────────────────────────────────────

    def _format_clause_std(self, clause: Clause) -> str:
        """Format clause matching C CL_FORM_STD (ID, literals, justification)."""
        st = self._symbol_table
        parts: list[str] = []
        if clause.id > 0:
            parts.append(f"{clause.id} ")
        if clause.is_empty:
            parts.append("$F")
        else:
            lit_strs = []
            for lit in clause.literals:
                atom_str = lit.atom.to_str(st)
                lit_strs.append(atom_str if lit.sign else f"-{atom_str}")
            parts.append(" | ".join(lit_strs))
        if clause.justification:
            just = clause.justification[0]
            parts.append(f".  [{just.just_type.name.lower()}].")
        else:
            parts.append(".")
        return "".join(parts)

    def _format_selection_extras(self, clause: Clause) -> str:
        """Format conditional selection metric extras for display."""
        from pyladr.search.selection import _clause_generality_penalty
        parts: list[str] = []
        if self._opts.entropy_weight > 0:
            entropy = self._calculate_structural_entropy(clause)
            parts.append(f",ent={entropy:.2f}")
        if self._opts.unification_weight > 0:
            penalty = _clause_generality_penalty(clause)
            parts.append(f",pen={penalty:.2f}")
        if self._penalty_cache is not None:
            rec = self._penalty_cache.get(clause.id)
            if rec is not None and rec.inherited_penalty > 0:
                parts.append(f",ipen={rec.inherited_penalty:.2f}")
        if self._opts.tree2vec_goal_proximity and self._t2v_goal_provider is not None:
            emb = self._tree2vec_embeddings.get(clause.id)
            if emb is not None:
                gd: float | None = None
                # Use cross-arg distance when enabled — matches the selection metric
                if self._opts.tree2vec_cross_arg_proximity and self._t2v_goal_arg_embs:
                    gd = self._t2v_cross_arg_distance(emb, clause.id)
                if gd is None:
                    goal_scorer = getattr(self._t2v_goal_provider, '_goal_scorer', None)
                    if goal_scorer is not None:
                        gd = goal_scorer.nearest_goal_distance(emb)
                if gd is not None:
                    parts.append(f",gd={gd:.4f}")
        if self._opts.rnn2vec_goal_proximity and self._r2v_goal_provider is not None:
            emb = self._rnn2vec_embeddings.get(clause.id)
            if emb is not None:
                goal_scorer = getattr(self._r2v_goal_provider, '_goal_scorer', None)
                if goal_scorer is not None:
                    r2v_gd = goal_scorer.nearest_goal_distance(emb)
                    if r2v_gd is not None:
                        parts.append(f",r2v_gd={r2v_gd:.4f}")
        return "".join(parts)

    def _calculate_structural_entropy(self, clause: Clause) -> float:
        """Calculate Shannon entropy of clause interpreted as tree structure.

        Node types: Clause, Literal, Predicate, Function, Variable, Constant
        Formula: H = -∑ p(v) log₂ p(v) where p(v) is probability of node type v
        """
        node_counts = {
            'clause': 0,
            'literal': 0,
            'predicate': 0,
            'function': 0,
            'variable': 0,
            'constant': 0
        }

        # Count clause node
        node_counts['clause'] = 1

        # Count literals
        node_counts['literal'] = len(clause.literals)

        # Count predicate, function, variable, constant nodes in all terms
        for literal in clause.literals:
            self._count_term_nodes(literal.atom, node_counts, is_predicate=True)

        # Calculate entropy
        total_nodes = sum(node_counts.values())
        if total_nodes <= 1:
            return 0.0

        entropy = 0.0
        for count in node_counts.values():
            if count > 0:
                p = count / total_nodes
                entropy -= p * math.log2(p)

        return entropy

    def _count_term_nodes(self, term, node_counts: dict, is_predicate: bool = False) -> None:
        """Recursively count nodes in a term tree."""
        from pyladr.core.term import Term

        if term.is_variable:
            node_counts['variable'] += 1
        elif term.is_constant:
            node_counts['constant'] += 1
        else:
            # Complex term - distinguish between predicate and function
            if is_predicate:
                node_counts['predicate'] += 1
            else:
                node_counts['function'] += 1

            # Recursively count argument nodes
            for arg in term.args:
                self._count_term_nodes(arg, node_counts, is_predicate=False)

    # ── Result construction ─────────────────────────────────────────────

    def _compute_t2v_histogram(self, proof: object) -> dict | None:
        """Compute conditional probability histogram: P(range|proof) vs P(range|non-proof).

        Splits all given clause goal-distances into proof vs non-proof populations,
        buckets both, and normalizes to probabilities.
        """
        if not self._t2v_all_given_distances:
            return None

        proof_ids = {c.id for c in proof.clauses}
        proof_scores: list[float] = []
        nonproof_scores: list[float] = []
        all_scores: list[float] = []

        for cid, score in self._t2v_all_given_distances.items():
            all_scores.append(score)
            if cid in proof_ids:
                proof_scores.append(score)
            else:
                nonproof_scores.append(score)

        if not all_scores:
            return None

        lo = min(all_scores)
        hi = max(all_scores)
        if hi == lo:
            lo = max(0.0, lo - 0.05)
            hi = min(1.0, hi + 0.05)
        bucket_width = (hi - lo) / 5

        def _bucket_and_normalize(scores: list[float]) -> list[float]:
            counts = [0, 0, 0, 0, 0]
            for s in scores:
                counts[min(4, int((s - lo) / bucket_width))] += 1
            n = len(scores)
            return [c / n if n > 0 else 0.0 for c in counts]

        return {
            "proof_probs": _bucket_and_normalize(proof_scores),
            "nonproof_probs": _bucket_and_normalize(nonproof_scores),
            "proof_n": len(proof_scores),
            "nonproof_n": len(nonproof_scores),
            "lo": lo,
            "hi": hi,
            "bucket_width": bucket_width,
        }

    def _compute_t2v_cumulative_histogram(self) -> dict | None:
        """Compute cumulative goal-distance histogram across all proofs found so far.

        Same logic as ``_compute_t2v_histogram`` but unions clause IDs from
        every proof in ``self._proofs``.
        """
        if not self._t2v_all_given_distances or not self._proofs:
            return None

        proof_ids: set[int] = set()
        for p in self._proofs:
            for c in p.clauses:
                proof_ids.add(c.id)

        proof_scores: list[float] = []
        nonproof_scores: list[float] = []
        all_scores: list[float] = []

        for cid, score in self._t2v_all_given_distances.items():
            all_scores.append(score)
            if cid in proof_ids:
                proof_scores.append(score)
            else:
                nonproof_scores.append(score)

        if not all_scores:
            return None

        lo = min(all_scores)
        hi = max(all_scores)
        if hi == lo:
            lo = max(0.0, lo - 0.05)
            hi = min(1.0, hi + 0.05)
        bucket_width = (hi - lo) / 5

        def _bucket_and_normalize(scores: list[float]) -> list[float]:
            counts = [0, 0, 0, 0, 0]
            for s in scores:
                counts[min(4, int((s - lo) / bucket_width))] += 1
            n = len(scores)
            return [c / n if n > 0 else 0.0 for c in counts]

        return {
            "proof_probs": _bucket_and_normalize(proof_scores),
            "nonproof_probs": _bucket_and_normalize(nonproof_scores),
            "proof_n": len(proof_scores),
            "nonproof_n": len(nonproof_scores),
            "n_proofs": len(self._proofs),
            "lo": lo,
            "hi": hi,
            "bucket_width": bucket_width,
        }

    def _make_result(self, exit_code: ExitCode) -> SearchResult:
        """Construct search result. Matches C collect_prover_results()."""
        self._state.return_code = exit_code

        # Compute T2V goal-distance histogram for the last proof (for end-of-search stats)
        if self._opts.tree2vec_goal_proximity and self._t2v_goal_provider is not None and self._proofs:
            if len(self._proofs) > 1:
                self._state.stats.t2v_distance_histogram = self._compute_t2v_cumulative_histogram()
            else:
                self._state.stats.t2v_distance_histogram = self._compute_t2v_histogram(self._proofs[-1])

        return SearchResult(
            exit_code=exit_code,
            proofs=tuple(self._proofs),
            stats=self._state.stats,
        )
