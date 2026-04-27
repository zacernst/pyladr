"""EmbeddingManager: target interface for extraction from GivenClauseSearch.

This is a SKELETON ONLY — method bodies are stubs (`...`). No logic has been
moved from given_clause.py yet. The purpose is to document the intended
interface so that the full extraction can proceed incrementally.

The EmbeddingManager will own all embedding-related state and methods that are
currently spread across 22 __slots__ entries and 24 methods (~1,400 lines) in
GivenClauseSearch. After extraction, GivenClauseSearch will hold a single
`_embedding_mgr: EmbeddingManager` slot and delegate to it.

Extraction plan:
  1. This skeleton (current step)
  2. Move _init_embeddings logic → EmbeddingManager.__init__
  3. Move FORTE methods/state
  4. Move Tree2Vec methods/state
  5. Move RNN2Vec methods/state
  6. Move proof pattern memory
  7. Move module-level helpers (format_t2v_histogram, _t2v_cosine, etc.)
  8. Replace 22 slots with single _embedding_mgr slot
  9. Update GivenClauseSearch callers to delegate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pyladr.core.clause import Clause
    from pyladr.search.given_clause import Proof, SearchOptions
    from pyladr.search.selection import GivenSelection
    from pyladr.search.state import ClauseList


class EmbeddingManager:
    """Manages all embedding subsystems (FORTE, Tree2Vec, RNN2Vec, proof patterns).

    Extracted from GivenClauseSearch to reduce the god-object from ~3,300 lines.
    Owns 22 slots and 24 methods that were previously on GivenClauseSearch.

    Usage (after full extraction)::

        mgr = EmbeddingManager(options, selection)
        # During search init:
        mgr.init_rnn2vec(usable, sos)
        # During clause keeping:
        mgr.on_clause_kept(clause)
        # During selection:
        clause = mgr.t2v_select_nearest_goal(sos)
        # After proof:
        mgr.record_proof_patterns(proof)
    """

    __slots__ = (
        # FORTE
        "_forte_provider",
        "_forte_embeddings",
        # Tree2Vec
        "_tree2vec_provider",
        "_tree2vec_embeddings",
        "_t2v_kept_since_update",
        "_t2v_online_batch",
        "_t2v_goal_clauses",
        "_t2v_goal_provider",
        "_t2v_goal_clause_ids",
        "_t2v_distance_window",
        "_t2v_distance_prev_avg",
        "_t2v_initial_goal_count",
        "_t2v_all_given_distances",
        "_t2v_update_count",
        "_t2v_bg_updater",
        "_t2v_completion_queue",
        "_t2v_antecedent_embeddings",
        "_t2v_goal_arg_embs",
        "_t2v_goal_ant_embs",
        # RNN2Vec
        "_rnn2vec_provider",
        "_rnn2vec_embeddings",
        "_r2v_kept_since_update",
        "_r2v_online_batch",
        "_r2v_update_count",
        "_r2v_bg_updater",
        "_r2v_completion_queue",
        "_r2v_goal_provider",
        "_r2v_goal_clauses",
        # Proof patterns
        "_proof_pattern_memory",
        # Config
        "_opts",
    )

    def __init__(self, options: SearchOptions, selection: GivenSelection) -> None:
        """Initialize all embedding subsystems based on options.

        Corresponds to GivenClauseSearch._init_embeddings (L619).
        """
        ...

    # ── FORTE ─────────────────────────────────────────────────────────

    @property
    def forte_embeddings(self) -> dict[int, list[float]]:
        """FORTE embedding storage: clause_id -> embedding vector."""
        ...

    @property
    def forte_provider(self) -> object | None:
        """The active ForteEmbeddingProvider, or None if disabled."""
        ...

    # ── Tree2Vec ──────────────────────────────────────────────────────

    def t2v_select_nearest_goal(self, sos: ClauseList) -> Clause | None:
        """Select SOS clause nearest to any goal by Tree2Vec embedding distance.

        Corresponds to GivenClauseSearch._t2v_select_nearest_goal (L2665).
        """
        ...

    def t2v_select_maximin(self, sos: ClauseList) -> Clause | None:
        """Select SOS clause with highest floor similarity across all goals.

        Corresponds to GivenClauseSearch._t2v_select_maximin (L2707).
        """
        ...

    def t2v_cross_arg_distance(
        self, emb_full: list[float], clause_id: int,
    ) -> float | None:
        """Cross-argument distance for CD compatibility.

        Corresponds to GivenClauseSearch._t2v_cross_arg_distance (L2646).
        """
        ...

    def do_t2v_online_update(self) -> None:
        """Trigger async Tree2Vec online update.

        Corresponds to GivenClauseSearch._do_t2v_online_update (L2281).
        """
        ...

    def do_t2v_online_update_sync(self) -> None:
        """Synchronous Tree2Vec online update (for tests).

        Corresponds to GivenClauseSearch._do_t2v_online_update_sync (L2311).
        """
        ...

    def on_t2v_update_done(self, update_count: int, stats: dict) -> None:
        """Callback when async Tree2Vec update completes.

        Corresponds to GivenClauseSearch._on_t2v_update_done (L2429).
        """
        ...

    def process_t2v_completions(self) -> None:
        """Process completed async Tree2Vec updates from the queue.

        Corresponds to GivenClauseSearch._process_t2v_completions (L2437).
        """
        ...

    def dump_t2v_embeddings(self, update_number: int) -> None:
        """Write Tree2Vec SOS embeddings to JSON file.

        Corresponds to GivenClauseSearch._dump_t2v_embeddings (L2478).
        """
        ...

    def compute_t2v_histogram(self, proof: Proof) -> dict | None:
        """Compute T2V goal-distance histogram for a single proof.

        Corresponds to GivenClauseSearch._compute_t2v_histogram (L3148).
        """
        ...

    def compute_t2v_cumulative_histogram(self) -> dict | None:
        """Compute cumulative T2V histogram across all proofs.

        Corresponds to GivenClauseSearch._compute_t2v_cumulative_histogram (L3196).
        """
        ...

    # ── RNN2Vec ───────────────────────────────────────────────────────

    def init_rnn2vec(self, usable: list[Clause], sos: list[Clause]) -> None:
        """Initialize RNN2Vec from initial clauses if enabled.

        Corresponds to GivenClauseSearch._maybe_init_rnn2vec (L1207).
        """
        ...

    def r2v_select_most_diverse(self, sos: ClauseList) -> Clause | None:
        """Select SOS clause most diverse from current embeddings.

        Corresponds to GivenClauseSearch._r2v_select_most_diverse (L1351).
        """
        ...

    def r2v_select_random_goal(self, sos: ClauseList) -> Clause | None:
        """Select SOS clause nearest to a randomly-chosen unproven goal.

        Corresponds to GivenClauseSearch._r2v_select_random_goal (L1383).
        """
        ...

    def do_r2v_online_update(self) -> None:
        """Trigger async RNN2Vec online update.

        Corresponds to GivenClauseSearch._do_r2v_online_update (L1421).
        """
        ...

    def on_r2v_update_done(self, update_count: int, stats: dict) -> None:
        """Callback when async RNN2Vec update completes.

        Corresponds to GivenClauseSearch._on_r2v_update_done (L1450).
        """
        ...

    def process_r2v_completions(self) -> None:
        """Process completed async RNN2Vec updates from the queue.

        Corresponds to GivenClauseSearch._process_r2v_completions (L1454).
        """
        ...

    def save_r2v_model(self, update_number: int) -> None:
        """Save RNN2Vec model checkpoint.

        Corresponds to GivenClauseSearch._save_r2v_model (L1484).
        """
        ...

    def dump_r2v_embeddings(self, update_number: int) -> None:
        """Write RNN2Vec SOS embeddings to JSON file.

        Corresponds to GivenClauseSearch._dump_r2v_embeddings (L2570).
        """
        ...

    # ── Proof patterns ────────────────────────────────────────────────

    @property
    def proof_pattern_memory(self) -> object | None:
        """The active ProofPatternMemory, or None if disabled."""
        ...

    def record_proof_patterns(self, proof: Proof) -> None:
        """Record embedding patterns from a successful proof for exploitation.

        Corresponds to GivenClauseSearch._record_proof_patterns (L2998).
        """
        ...


# ── Module-level helpers (to move from given_clause.py) ──────────────


def format_t2v_histogram(histogram: dict, proof_num: int | None = 1) -> str:
    """Format a T2V goal-distance histogram as a conditional probability table.

    Corresponds to given_clause.format_t2v_histogram (L416).
    """
    ...


def t2v_cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Corresponds to given_clause._t2v_cosine (L470).
    """
    ...


def get_antecedent_term(clause: Clause) -> object | None:
    """Extract the antecedent term from a clause of the form P(i(x,y)).

    Corresponds to given_clause._get_antecedent_term (L453).
    """
    ...
