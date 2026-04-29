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
from dataclasses import dataclass, field
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
from pyladr.search.priority_sos import PrioritySOS
from pyladr.search.selection import (
    GivenSelection,
    SelectionOrder,
    _clause_generality_penalty,
    default_clause_weight,
)
from pyladr.search.clause_formatting import format_clause_std
from pyladr.search.proof_tracing import trace_proof
from pyladr.search.state import ClauseList, SearchState
from pyladr.search.statistics import SearchStatistics

logger = logging.getLogger(__name__)


# ── Unit conflict index (canonical definition in pyladr.search.unit_conflict)
from pyladr.search.unit_conflict import UnitConflictIndex  # noqa: E402


# ── Result types (canonical definition in pyladr.search.result_types) ───────
# Re-exported here for backward compatibility with existing import paths.
from pyladr.search.result_types import ExitCode, Proof, SearchResult  # noqa: E402

# ── Search options (canonical definition in pyladr.search.options) ──────────
# Re-exported here for backward compatibility with existing import paths.
from pyladr.search.options import SearchOptions  # noqa: E402


from pyladr.search.embedding_helpers import (  # noqa: E402
    _cosine,
    compute_cumulative_distance_histogram,
    compute_distance_histogram,
    format_distance_histogram,
)
from pyladr.search.r2v_subsystem import R2VSearchSubsystem  # noqa: E402


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

    __slots__ = (
        "_opts", "_state", "_selection", "_proofs", "_all_clauses",
        "_symbol_table", "_demod_index", "_parallel_engine",
        "_subsump_idx", "_back_subsump_idx", "_unit_conflict_idx",
        "_lazy_demod", "_proof_callback", "_penalty_cache",
        "_repetition_config", "_penalty_weight_config",
        "_nucleus_penalty_config", "_nucleus_pattern_cache",
        "_back_subsume_enabled",
        "_back_subsumption_callback", "_forward_subsumption_callback",
        # ── RNN2Vec search subsystem ──
        "_r2v",
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

    def _init_selection(self, selection: GivenSelection | None) -> GivenSelection:
        """Build the selection strategy from options."""
        if selection is not None:
            return selection
        opts = self._opts
        from pyladr.search.selection import SelectionRule
        has_extra = (
            opts.unification_weight > 0
            or opts.rnn2vec_weight > 0 or opts.rnn2vec_random_goal_weight > 0
        )
        if has_extra or opts.weight_ratio != 4:
            rules = [
                SelectionRule("A", SelectionOrder.AGE, part=1),
                SelectionRule("W", SelectionOrder.WEIGHT, part=opts.weight_ratio),
            ]
            if opts.unification_weight > 0:
                rules.append(SelectionRule("U", SelectionOrder.UNIFICATION_PENALTY, part=opts.unification_weight))
            if opts.rnn2vec_weight > 0:
                rules.append(SelectionRule("R2V", SelectionOrder.RNN2VEC, part=round(opts.rnn2vec_weight)))
            if opts.rnn2vec_random_goal_weight > 0:
                rules.append(SelectionRule("RGP", SelectionOrder.RNN2VEC_RANDOM_GOAL, part=round(opts.rnn2vec_random_goal_weight)))
            return GivenSelection(rules=rules)
        return GivenSelection()

    def _init_embeddings(self) -> None:
        """Initialize RNN2Vec search subsystem."""
        self._r2v = R2VSearchSubsystem(
            self._opts,
            state_fn=lambda: (
                self._state.sos,
                self._state.usable,
                self._proofs,
                self._format_clause_std,
            ),
        )

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

    @property
    def state(self) -> SearchState:
        """Access search state (for testing/inspection)."""
        return self._state

    @property
    def stats(self) -> SearchStatistics:
        return self._state.stats

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
                self._r2v.process_completions()

                exit_code = self._make_inferences()
                if exit_code is not None:
                    return self._make_result(exit_code)

                exit_code = self._limbo_process()
                if exit_code is not None:
                    return self._make_result(exit_code)

            # SOS exhausted
            return self._make_result(ExitCode.SOS_EMPTY_EXIT)
        finally:
            self._r2v.shutdown()

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

        # Deferred RNN2Vec initialization
        self._r2v.maybe_init(usable, sos, self._symbol_table)

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
        # RNN2Vec variants: if the ratio cycle wants R2V/RGP and we have
        # embeddings, use embedding-based scoring instead of the age fallback.
        current_order = self._selection._get_current_rule().order
        r2v_given, r2v_label = self._r2v.maybe_select_given(self._state.sos, current_order)
        if r2v_given is not None:
            self._state.sos.remove(r2v_given)
            rule = self._selection._get_current_rule()
            rule.selected += 1
            self._selection._advance_cycle()
            given, selection_type = r2v_given, r2v_label
        else:
            given, selection_type = self._selection.select_given(
                self._state.sos,
                self._state.stats.given,
            )

        if given is None:
            return None  # SOS became empty during selection

        given.given_selection = selection_type

        # Lazy demodulation: ensure clause is fully reduced before use.
        # Must happen BEFORE distance assignment so that given_distance is
        # stored on the actual clause object that will appear in the proof
        # trace (demodulate_clause returns a new Clause with the same id but
        # different literals).
        if self._lazy_demod is not None and self._lazy_demod.needs_reduction(given):
            given = self._lazy_demod.ensure_fully_reduced(
                given, self._demod_index, self._symbol_table,
                self._opts.lex_order_vars, self._opts.demod_step_limit,
            )

        # R2V goal distance tracking for histogram reporting.
        self._r2v.record_given_distance(given)

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

        # Update RNN2Vec state for kept clause
        self._r2v.on_clause_kept(c, self._all_clauses)

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
                # Evict RNN2Vec embedding for disabled clause
                self._r2v.on_clause_evicted(victim.id)
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
                        # Evict RNN2Vec embedding for disabled clause
                        self._r2v.on_clause_evicted(victim.id)

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
                # Pass combined penalty to PrioritySOS for heap ordering
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
                if isinstance(self._state.sos, PrioritySOS) and penalty_val is not None:
                    self._state.sos.append(c, penalty_override=penalty_val)
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
        proof_clauses = trace_proof(empty, self._all_clauses)

        proof = Proof(
            empty_clause=empty,
            clauses=tuple(proof_clauses),
        )
        self._proofs.append(proof)

        logger.info("PROOF FOUND (proof %d)", self._state.stats.proofs)
        print(f"PROOF FOUND (proof {self._state.stats.proofs})")

        # Print per-proof and cumulative R2V goal-distance histograms
        self._r2v.on_proof_found(proof, self._proofs, self._opts.quiet)

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

    # ── Clause formatting ──────────────────────────────────────────────

    def _format_clause_std(self, clause: Clause) -> str:
        """Format clause matching C CL_FORM_STD (ID, literals, justification)."""
        return format_clause_std(self._symbol_table, clause)

    def _format_selection_extras(self, clause: Clause) -> str:
        """Format conditional selection metric extras for display."""
        from pyladr.search.selection import _clause_generality_penalty
        parts: list[str] = []
        if self._opts.unification_weight > 0:
            penalty = _clause_generality_penalty(clause)
            parts.append(f",pen={penalty:.2f}")
        if self._penalty_cache is not None:
            rec = self._penalty_cache.get(clause.id)
            if rec is not None and rec.inherited_penalty > 0:
                parts.append(f",ipen={rec.inherited_penalty:.2f}")
        parts.append(self._r2v.format_extras(clause))
        return "".join(parts)

    # ── Result construction ─────────────────────────────────────────────

    def _make_result(self, exit_code: ExitCode) -> SearchResult:
        """Construct search result. Matches C collect_prover_results()."""
        self._state.return_code = exit_code

        return SearchResult(
            exit_code=exit_code,
            proofs=tuple(self._proofs),
            stats=self._state.stats,
        )
