"""RNN2Vec search subsystem for GivenClauseSearch.

Owns all RNN2Vec embedding state and exposes an event-style interface
that GivenClauseSearch calls at key moments in the search loop.
"""
from __future__ import annotations

import logging
import queue
from typing import TYPE_CHECKING, Callable

from pyladr.core.clause import Clause, JustType
from pyladr.search.embedding_helpers import (
    _cosine,
    compute_cumulative_distance_histogram,
    compute_distance_histogram,
    format_distance_histogram,
)

if TYPE_CHECKING:
    from pyladr.search.options import SearchOptions

logger = logging.getLogger(__name__)


class R2VSearchSubsystem:
    """Manages RNN2Vec embedding-based clause selection within GivenClauseSearch.

    Owns all R2V state: provider, embedding cache, online batch, background
    updater, and goal-distance provider. GivenClauseSearch holds one instance
    and calls the event-style interface at key moments in the search loop.
    """

    __slots__ = (
        "_provider",
        "_embeddings",
        "_kept_since_update",
        "_online_batch",
        "_update_count",
        "_bg_updater",
        "_goal_provider",
        "_goal_clauses",
        "_all_given_distances",
        "_completion_queue",
        "_opts",
        "_state_fn",
    )

    def __init__(
        self,
        opts: SearchOptions,
        state_fn: Callable[[], tuple] | None = None,
    ) -> None:
        """Initialize from search options.

        Args:
            opts: Search options.
            state_fn: Callable returning (sos, usable, proofs, format_fn) for
                      _dump_embeddings when called from background contexts.
        """
        self._opts = opts
        self._state_fn = state_fn

        self._embeddings: dict[int, list[float]] = {}
        self._provider: object | None = None
        self._kept_since_update: int = 0
        self._online_batch: list[Clause] = []
        self._update_count: int = 0
        self._bg_updater: object | None = None
        self._goal_provider: object | None = None
        self._goal_clauses: list = []
        self._all_given_distances: dict[int, float] = {}
        self._completion_queue: queue.SimpleQueue = queue.SimpleQueue()

        if opts.rnn2vec_embeddings:
            try:
                from pyladr.ml.rnn2vec.provider import (
                    build_provider_config_from_search_options,
                )
                self._provider = build_provider_config_from_search_options(opts)
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

    def maybe_init(self, usable, sos, symbol_table) -> None:
        """Deferred initialization: train on initial clauses and compute embeddings."""
        if self._provider is not None and not callable(self._provider):
            try:
                from pyladr.ml.rnn2vec.provider import RNN2VecEmbeddingProvider

                r2v_cfg = self._provider
                initial = [*usable, *sos]

                if self._opts.rnn2vec_model_path:
                    print(f"% RNN2Vec: loading model from {self._opts.rnn2vec_model_path} ...")
                    provider = RNN2VecEmbeddingProvider.from_saved_model(
                        model_path=self._opts.rnn2vec_model_path,
                        config=r2v_cfg,
                    )
                    self._provider = provider
                    r2v = getattr(provider, "_rnn2vec", None)
                    if r2v is not None:
                        print(f"% RNN2Vec: model loaded "
                              f"(vocab={r2v.vocab_size}, dim={r2v.embedding_dim})")
                elif initial:
                    n_clauses = len(initial)
                    epochs = r2v_cfg.rnn2vec_config.training_epochs
                    rnn_cfg = r2v_cfg.rnn2vec_config.rnn_config
                    print(f"% RNN2Vec: training on {n_clauses} initial clauses "
                          f"({epochs} epochs, {rnn_cfg.rnn_type.upper()}, "
                          f"h={rnn_cfg.hidden_dim}, dim={rnn_cfg.embedding_dim}) ...")
                    provider, stats = RNN2VecEmbeddingProvider.from_sos_clauses(
                        sos_clauses=sos,
                        symbol_table=symbol_table,
                        config=r2v_cfg,
                    )
                    self._provider = provider
                    if stats:
                        print(f"% RNN2Vec: training complete "
                              f"(loss={stats.get('loss', 0.0):.4f}, "
                              f"vocab={int(stats.get('vocab_size', 0))})")

                if self._provider is not r2v_cfg and initial:
                    provider = self._provider
                    embeddings = provider.get_embeddings_batch(initial)
                    for c, emb in zip(initial, embeddings):
                        if emb is not None:
                            self._embeddings[c.id] = emb
                    logger.debug(
                        "RNN2Vec: pre-computed %d/%d initial embeddings",
                        len(self._embeddings), len(initial),
                    )
                    if self._opts.rnn2vec_dump_embeddings:
                        self._dump_embeddings(0)
            except Exception as _r2v_exc:
                import traceback as _tb
                print(
                    f"% RNN2Vec init error: {type(_r2v_exc).__name__}: {_r2v_exc}\n"
                    f"% {''.join(_tb.format_exc()).strip()}",
                )
                logger.warning(
                    "Failed to initialize RNN2Vec provider, continuing without",
                    exc_info=True,
                )
                self._provider = None

        # Background updater for online learning
        if (
            self._opts.rnn2vec_online_learning
            and self._provider is not None
            and callable(getattr(self._provider, 'bump_model_version', None))
        ):
            try:
                from pyladr.ml.rnn2vec.background_updater import BackgroundRNN2VecUpdater
                self._bg_updater = BackgroundRNN2VecUpdater(
                    provider=self._provider,
                    learning_rate=self._opts.rnn2vec_online_lr,
                    max_updates=self._opts.rnn2vec_online_max_updates,
                    completion_callback=self._on_update_done,
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

        # Goal-proximity wrapping
        wants_goals = (
            self._opts.rnn2vec_goal_proximity or self._opts.rnn2vec_random_goal_weight > 0
        )
        need_goals = (
            wants_goals
            and self._provider is not None
            and callable(getattr(self._provider, 'get_embedding', None))
        )
        if wants_goals and not need_goals:
            print(
                "% RNN2Vec: goal proximity requested but provider not ready "
                f"(provider={str(self._provider)[:80]})"
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
            self._goal_clauses = r2v_goal_clauses

            if not r2v_goal_clauses:
                print(
                    f"% RNN2Vec: no DENY-justified goal clauses found in "
                    f"{len(all_initial)} initial clauses; "
                    f"goal proximity disabled."
                )

            if r2v_goal_clauses:
                gd_config = GoalDirectedConfig(
                    enabled=True,
                    goal_proximity_weight=self._opts.rnn2vec_goal_proximity_weight,
                    ancestor_tracking=self._opts.rnn2vec_ancestor_tracking,
                    ancestor_proximity_threshold=self._opts.rnn2vec_ancestor_proximity_threshold,
                    ancestor_max_count=self._opts.rnn2vec_ancestor_max_count,
                    ancestor_decay=self._opts.rnn2vec_ancestor_decay,
                    ancestor_min_weight=self._opts.rnn2vec_ancestor_min_weight,
                    ancestor_max_depth=self._opts.rnn2vec_ancestor_max_depth,
                )
                gd_provider = GoalDirectedEmbeddingProvider(
                    base_provider=self._provider,
                    config=gd_config,
                )
                gd_provider.register_goals(r2v_goal_clauses)
                self._goal_provider = gd_provider
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

    def maybe_select_given(self, sos, order) -> tuple[Clause | None, str | None]:
        """Attempt R2V-based selection from SOS.

        Returns (clause, label) on success, or (None, None) if R2V is
        unavailable or produced no candidate — caller falls back to standard
        selection.
        """
        from pyladr.search.selection import SelectionOrder
        if not self._embeddings or sos.is_empty:
            return None, None
        if order == SelectionOrder.RNN2VEC_RANDOM_GOAL:
            given = self._select_random_goal(sos)
            label = "RGP"
        elif order == SelectionOrder.RNN2VEC:
            given = self._select_most_diverse(sos)
            label = "R2V"
        else:
            return None, None
        if given is None:
            return None, None
        return given, label

    def record_given_distance(self, given: Clause) -> None:
        """Record goal distance for the selected given clause (histogram support)."""
        if not (self._opts.rnn2vec_random_goal_weight > 0 or self._opts.rnn2vec_goal_proximity):
            return
        if self._goal_provider is None:
            return
        emb = self._embeddings.get(given.id)
        goal_scorer = getattr(self._goal_provider, '_goal_scorer', None)
        if emb is not None and goal_scorer is not None:
            gd = goal_scorer.nearest_goal_distance(emb)
            if gd is not None:
                self._all_given_distances[given.id] = gd

    def on_clause_kept(self, c: Clause, all_clauses: dict) -> None:
        """Update R2V state when a clause is kept.

        Computes embedding, tracks productive ancestors, and accumulates
        online learning batch.
        """
        # Compute and cache embedding
        if self._provider is not None and callable(getattr(self._provider, 'get_embedding', None)):
            emb = self._provider.get_embedding(c)
            if emb is not None:
                self._embeddings[c.id] = emb

        # Record productive ancestors for goal-proximity tracking
        if (
            self._goal_provider is not None
            and self._goal_provider._config.ancestor_tracking
            and c.justification
        ):
            emb = self._embeddings.get(c.id)
            if emb is not None and self._goal_provider.num_goals > 0:
                parent_ids: list[int] = []
                for just in c.justification:
                    if just.clause_id > 0:
                        parent_ids.append(just.clause_id)
                    parent_ids.extend(just.clause_ids)
                    if just.para is not None:
                        parent_ids.append(just.para.from_id)
                        parent_ids.append(just.para.into_id)
                seen: set[int] = set()
                parent_clauses: list[Clause] = []
                for pid in parent_ids:
                    if pid <= 0 or pid in seen:
                        continue
                    seen.add(pid)
                    parent = all_clauses.get(pid)
                    if parent is not None:
                        parent_clauses.append(parent)
                if parent_clauses:
                    self._goal_provider.try_expand_from_clause(emb, parent_clauses)

        # Online learning: accumulate batch, trigger update at interval
        if (
            self._opts.rnn2vec_online_learning
            and self._provider is not None
            and callable(getattr(self._provider, 'bump_model_version', None))
        ):
            max_updates = self._opts.rnn2vec_online_max_updates
            if max_updates == 0 or self._update_count < max_updates:
                if len(self._online_batch) < self._opts.rnn2vec_online_batch_size:
                    self._online_batch.append(c)
                self._kept_since_update += 1
                if self._kept_since_update >= self._opts.rnn2vec_online_update_interval:
                    self._do_online_update()

    def on_clause_evicted(self, clause_id: int) -> None:
        """Remove embedding for a disabled clause."""
        self._embeddings.pop(clause_id, None)

    def on_proof_found(self, proof, all_proofs: list, quiet: bool) -> None:
        """Print per-proof and cumulative R2V goal-distance histograms."""
        if not (self._opts.rnn2vec_random_goal_weight > 0 or self._opts.rnn2vec_goal_proximity):
            return
        if self._goal_provider is None:
            return
        r2v_histogram = compute_distance_histogram(self._all_given_distances, proof)
        if r2v_histogram is not None and not quiet:
            print(format_distance_histogram(
                r2v_histogram, len(all_proofs), label="R2V"
            ))
        if len(all_proofs) > 1 and not quiet:
            cumulative = compute_cumulative_distance_histogram(
                self._all_given_distances, all_proofs
            )
            if cumulative is not None:
                print(format_distance_histogram(cumulative, proof_num=None, label="R2V"))

    def process_completions(self) -> None:
        """Drain completion notifications from the R2V background updater."""
        if self._bg_updater is None:
            return
        while True:
            try:
                update_num, stats = self._completion_queue.get_nowait()
            except queue.Empty:
                break
            self._update_count = update_num
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
                self._dump_embeddings(update_num)
            if self._opts.rnn2vec_save_model:
                self._save_model(update_num)

    def shutdown(self, drain_timeout: float = 5.0) -> None:
        """Shut down background updater and process remaining completions."""
        if self._bg_updater is not None:
            self._bg_updater.shutdown(drain=True, timeout=drain_timeout)
            self.process_completions()

    def format_extras(self, clause: Clause) -> str:
        """Format R2V goal-distance metric for the selection display line."""
        if not (self._opts.rnn2vec_goal_proximity and self._goal_provider is not None):
            return ""
        emb = self._embeddings.get(clause.id)
        if emb is None:
            return ""
        goal_scorer = getattr(self._goal_provider, '_goal_scorer', None)
        if goal_scorer is None:
            return ""
        r2v_gd = goal_scorer.nearest_goal_distance(emb)
        if r2v_gd is None:
            return ""
        return f",r2v_gd={r2v_gd:.4f}"

    # ── Internal selection helpers ─────────────────────────────────────────

    def _select_most_diverse(self, sos) -> Clause | None:
        """Select SOS clause most dissimilar from all already-given embeddings (maximin diversity)."""
        given_embs = [
            emb for cid, emb in self._embeddings.items()
            if cid not in {c.id for c in sos}
        ]
        if not given_embs:
            return None

        best_clause = None
        best_min_dist = -1.0

        for c in sos:
            emb = self._embeddings.get(c.id)
            if emb is None:
                continue
            min_dist = min(_cosine(emb, ge) for ge in given_embs)
            diversity = 1.0 - min_dist
            if diversity > best_min_dist:
                best_min_dist = diversity
                best_clause = c

        return best_clause

    def _select_random_goal(self, sos) -> Clause | None:
        """Select SOS clause nearest to a randomly-chosen unproven goal embedding."""
        import random as _random

        if not self._embeddings:
            return None

        goal_scorer = (
            getattr(self._goal_provider, '_goal_scorer', None)
            if self._goal_provider is not None else None
        )
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
            emb = self._embeddings.get(c.id)
            if emb is None:
                continue
            dist = (1.0 - _cosine(emb, target_emb)) / 2.0
            if dist < best_dist:
                best_dist = dist
                best_clause = c

        return best_clause

    def _do_online_update(self) -> None:
        """Trigger an online RNN2Vec update from the accumulated batch."""
        batch = list(self._online_batch)
        self._online_batch.clear()
        self._kept_since_update = 0

        if not batch:
            return

        if self._bg_updater is not None:
            self._bg_updater.submit(batch)
            self._update_count += 1
        elif self._provider is not None and callable(
            getattr(self._provider, 'bump_model_version', None)
        ):
            try:
                self._provider._rnn2vec.update_online(
                    batch, learning_rate=self._opts.rnn2vec_online_lr
                )
                self._provider.bump_model_version()
                self._update_count += 1
                if self._opts.rnn2vec_dump_embeddings:
                    self._dump_embeddings(self._update_count)
                if self._opts.rnn2vec_save_model:
                    self._save_model(self._update_count)
            except Exception:
                logger.warning("RNN2Vec online update failed", exc_info=True)

    def _on_update_done(self, update_count: int, stats: dict) -> None:
        """Callback from BackgroundRNN2VecUpdater when an update completes."""
        self._completion_queue.put((update_count, stats))

    def _save_model(self, update_number: int) -> None:
        """Save the current RNN2Vec model to disk."""
        r2v = getattr(self._provider, "_rnn2vec", None)
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

    def _dump_embeddings(self, update_number: int) -> None:
        """Write RNN2Vec SOS clause embeddings to a JSON file."""
        import json
        from datetime import datetime
        from pathlib import Path

        path = self._opts.rnn2vec_dump_embeddings
        provider = self._provider
        if provider is None or not callable(getattr(provider, "get_embedding", None)):
            return
        if self._state_fn is None:
            return

        sos, usable, proofs, format_fn = self._state_fn()
        proof_ids: set[int] = {c.id for proof in proofs for c in proof.clauses}

        def _is_goal(clause: Clause) -> bool:
            return bool(
                clause.justification
                and clause.justification[0].just_type == JustType.DENY
            )

        seen_ids: set[int] = set()
        to_dump: list[Clause] = []
        for clause in sos:
            seen_ids.add(clause.id)
            to_dump.append(clause)
        for clause in usable:
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
            emb = provider.get_embedding(embed_clause)
            entries.append({
                "id": clause.id,
                "clause": format_fn(clause),
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
