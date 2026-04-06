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
from enum import IntEnum, auto
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
from pyladr.inference.paramodulation import (
    orient_equalities,
    para_from_into,
)
from pyladr.inference.demodulation import (
    DemodType,
    DemodulatorIndex,
    back_demodulatable,
    demodulate_clause,
    demodulator_type,
)
from pyladr.inference.subsumption import (
    back_subsume_from_lists,
    forward_subsume_from_lists,
    subsumes,
)
from pyladr.parallel.inference_engine import ParallelInferenceEngine, ParallelSearchConfig
from pyladr.search.selection import GivenSelection, default_clause_weight
from pyladr.search.state import ClauseList, SearchState
from pyladr.search.statistics import SearchStatistics

logger = logging.getLogger(__name__)


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


# ── Search options ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchOptions:
    """Search options matching C search options.

    Controls inference rules, limits, and output.
    """

    # Inference rules
    binary_resolution: bool = True
    paramodulation: bool = False
    factoring: bool = True
    para_into_vars: bool = False

    # Limits (C: max_given, max_kept, max_seconds, max_generated, max_proofs)
    max_given: int = -1       # -1 = no limit
    max_kept: int = -1
    max_seconds: float = -1.0
    max_generated: int = -1
    max_proofs: int = 1

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

    # Output
    print_given: bool = True
    print_kept: bool = False
    print_gen: bool = False
    quiet: bool = False


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
    )

    def __init__(
        self,
        options: SearchOptions | None = None,
        selection: GivenSelection | None = None,
        symbol_table: SymbolTable | None = None,
    ) -> None:
        self._opts = options or SearchOptions()
        self._state = SearchState()
        self._selection = selection or GivenSelection()
        self._proofs: list[Proof] = []
        self._all_clauses: dict[int, Clause] = {}
        self._symbol_table = symbol_table or SymbolTable()
        self._demod_index = DemodulatorIndex()
        self._parallel_engine = ParallelInferenceEngine(self._opts.parallel)

    @property
    def state(self) -> SearchState:
        """Access search state (for testing/inspection)."""
        return self._state

    @property
    def stats(self) -> SearchStatistics:
        return self._state.stats

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        usable: list[Clause] | None = None,
        sos: list[Clause] | None = None,
    ) -> SearchResult:
        """Run the given-clause search. Matches C search().

        Args:
            usable: Initial usable clauses (already processed, available for inference).
            sos: Initial SOS clauses (candidates for given clause selection).

        Returns:
            SearchResult with exit code, proofs, and statistics.
        """
        self._state.stats.start()

        # Move initial clauses into state (C: move_clauses_to_clist)
        self._init_clauses(usable or [], sos or [])

        # Process initial clauses (C: index_and_process_initial_clauses)
        exit_code = self._process_initial_clauses()
        if exit_code is not None:
            return self._make_result(exit_code)

        # Main search loop (C: while(inferences_to_make()) { ... })
        self._state.searching = True
        self._state.stats.search_start_time = self._state.stats.start_time

        while self._inferences_to_make():
            exit_code = self._make_inferences()
            if exit_code is not None:
                return self._make_result(exit_code)

            exit_code = self._limbo_process()
            if exit_code is not None:
                return self._make_result(exit_code)

        # SOS exhausted
        return self._make_result(ExitCode.SOS_EMPTY_EXIT)

    # ── Initialization ──────────────────────────────────────────────────

    def _init_clauses(
        self,
        usable: list[Clause],
        sos: list[Clause],
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

    def _process_initial_clauses(self) -> ExitCode | None:
        """Process initial clauses. Matches C index_and_process_initial_clauses().

        1. Weigh and assign IDs to all initial clauses
        2. Index usable clauses for resolution
        3. Check for empty clauses (immediate proof)
        """
        # Orient equalities if paramodulation is enabled
        if self._opts.paramodulation:
            for c in self._state.usable:
                orient_equalities(c, self._symbol_table)
            for c in self._state.sos:
                orient_equalities(c, self._symbol_table)

        # Index usable clauses (they're already available for inference)
        for c in self._state.usable:
            c.weight = default_clause_weight(c)
            self._state.index_clashable(c, insert=True)

        # Process SOS clauses: weigh, check for empty clause
        for c in self._state.sos:
            c.weight = default_clause_weight(c)

            # Check for empty clause (immediate proof)
            if c.is_empty:
                return self._handle_proof(c)

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
        given, selection_type = self._selection.select_given(
            self._state.sos,
            self._state.stats.given,
        )

        if given is None:
            return None  # SOS became empty during selection

        self._state.stats.given += 1

        # Check max_given limit (C: over_parm_limit check)
        if (
            self._opts.max_given > 0
            and self._state.stats.given > self._opts.max_given
        ):
            return ExitCode.MAX_GIVEN_EXIT

        # Check max_seconds limit
        if (
            self._opts.max_seconds > 0
            and self._state.stats.elapsed_seconds() > self._opts.max_seconds
        ):
            return ExitCode.MAX_SECONDS_EXIT

        # Print given clause (C: print_given)
        if self._opts.print_given and not self._opts.quiet:
            wt = int(given.weight) if given.weight == int(given.weight) else given.weight
            logger.info(
                "given #%d (%s,wt=%s): %s",
                self._state.stats.given,
                selection_type,
                wt,
                given.to_str(),
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
            return self._given_infer_parallel(given, usable_snapshot)
        return self._given_infer_sequential(given, usable_snapshot)

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
                resolvents = all_binary_resolvents(given, usable_clause)
                for resolvent in resolvents:
                    exit_code = self._cl_process(resolvent)
                    if exit_code is not None:
                        return exit_code

                if usable_clause is not given:
                    resolvents2 = all_binary_resolvents(usable_clause, given)
                    for resolvent in resolvents2:
                        exit_code = self._cl_process(resolvent)
                        if exit_code is not None:
                            return exit_code

        if self._opts.paramodulation:
            st = self._symbol_table
            for usable_clause in usable_snapshot:
                paras = para_from_into(
                    given, usable_clause, False, st,
                    self._opts.para_into_vars,
                )
                for p in paras:
                    exit_code = self._cl_process(p)
                    if exit_code is not None:
                        return exit_code

                if usable_clause is not given:
                    paras2 = para_from_into(
                        usable_clause, given, True, st,
                        self._opts.para_into_vars,
                    )
                    for p in paras2:
                        exit_code = self._cl_process(p)
                        if exit_code is not None:
                            return exit_code

        if self._opts.factoring:
            factors = factor(given)
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
        self._state.stats.generated += 1
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
        from pyladr.inference.paramodulation import is_eq_atom

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

    def _forward_subsumed(self, c: Clause) -> bool:
        """Check if c is forward subsumed by any existing clause.

        Uses full matching-based subsumption (matching C forward_subsumption).
        Checks usable, sos, and limbo lists for a clause that subsumes c.
        """
        subsumer = forward_subsume_from_lists(
            c,
            [self._state.usable, self._state.sos, self._state.limbo],
        )
        return subsumer is not None

    def _keep_clause(self, c: Clause) -> ExitCode | None:
        """Keep a clause. Matches C cl_process_keep().

        Assigns ID, renumbers variables, indexes, and puts in limbo.
        """
        self._state.stats.kept += 1
        c.normal_vars = True

        if c.id == 0:
            self._state.assign_clause_id(c)

        self._all_clauses[c.id] = c

        if self._opts.print_kept:
            logger.info("kept:      %s", c.to_str())

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
            # Orient equalities first
            if self._opts.paramodulation:
                orient_equalities(c, self._symbol_table)
            dtype = demodulator_type(
                c, self._symbol_table, self._opts.lex_dep_demod_lim,
            )
            if dtype != DemodType.NOT_DEMODULATOR:
                self._demod_index.insert(c, dtype)
                self._state.stats.new_demodulators += 1
                self._state.demods.append(c)

        # Index for forward subsumption (C: index_literals)
        # Add to limbo (C: clist_append(c, Glob.limbo))
        self._state.limbo.append(c)

        return None

    def _unit_conflict(self, c: Clause) -> ExitCode | None:
        """Check for unit conflict. Matches C cl_process_conflict().

        If c is a unit clause, check if its complement exists in usable
        or limbo. If so, we can derive the empty clause.
        """
        if not c.is_unit:
            return None

        c_lit = c.literals[0]

        # Search for complement in usable clauses
        for other in self._state.usable:
            if not other.is_unit:
                continue
            o_lit = other.literals[0]
            if c_lit.sign != o_lit.sign and c_lit.atom.term_ident(o_lit.atom):
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

        return None

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
            back_subsumed = back_subsume_from_lists(
                c,
                [self._state.usable, self._state.sos],
            )
            for victim in back_subsumed:
                self._state.stats.back_subsumed += 1
                # Remove from whichever list it's in and deindex
                self._state.index_clashable(victim, insert=False)
                self._state.usable.remove(victim)
                self._state.sos.remove(victim)
                self._state.disabled.append(victim)
                logger.debug(
                    "back subsumed: %s by %s",
                    victim.to_str(),
                    c.to_str(),
                )

            # Back-demodulation: new demodulators rewrite kept clauses
            if self._opts.back_demod and self._opts.demodulation:
                dtype = demodulator_type(
                    c, self._symbol_table, self._opts.lex_dep_demod_lim,
                )
                if dtype != DemodType.NOT_DEMODULATOR:
                    # Find clauses in usable/sos that can be rewritten
                    all_kept = list(self._state.usable) + list(self._state.sos)
                    rewritable = back_demodulatable(
                        c, dtype, all_kept, self._symbol_table,
                        self._opts.lex_order_vars,
                    )
                    for victim in rewritable:
                        self._state.stats.back_demodulated += 1
                        # Disable original, reprocess the rewritten version
                        self._state.index_clashable(victim, insert=False)
                        self._state.usable.remove(victim)
                        self._state.sos.remove(victim)
                        self._state.disabled.append(victim)

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
        proof = Proof(
            empty_clause=empty,
            clauses=tuple(proof_clauses),
        )
        self._proofs.append(proof)

        logger.info("PROOF FOUND (proof %d)", self._state.stats.proofs)

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

    # ── Result construction ─────────────────────────────────────────────

    def _make_result(self, exit_code: ExitCode) -> SearchResult:
        """Construct search result. Matches C collect_prover_results()."""
        self._state.return_code = exit_code
        return SearchResult(
            exit_code=exit_code,
            proofs=tuple(self._proofs),
            stats=self._state.stats,
        )
