"""Tests for offline Tree2Vec training mode (save/load/from_saved_model/CLI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pyladr.core.clause import Clause, Literal
from pyladr.core.term import Term, get_rigid_term, get_variable_term
from pyladr.ml.tree2vec.algorithm import Tree2Vec, Tree2VecConfig
from pyladr.ml.tree2vec.skipgram import SkipGramConfig
from pyladr.ml.tree2vec.walks import WalkConfig


# ── Helpers ─────────────────────────────────────────────────────────────────

SYM_P = 1
SYM_I = 2
SYM_N = 3


def var(n: int) -> Term:
    return get_variable_term(n)


def n_term(arg: Term) -> Term:
    return get_rigid_term(SYM_N, 1, (arg,))


def i_term(left: Term, right: Term) -> Term:
    return get_rigid_term(SYM_I, 2, (left, right))


def P(arg: Term) -> Term:
    return get_rigid_term(SYM_P, 1, (arg,))


def make_clause(*literals: Literal) -> Clause:
    return Clause(literals=literals)


def make_literal(sign: bool, atom: Term) -> Literal:
    return Literal(sign=sign, atom=atom)


def make_test_clauses() -> list[Clause]:
    """Small set of clauses for fast training."""
    x, y = var(0), var(1)
    return [
        make_clause(make_literal(True, P(i_term(x, y)))),
        make_clause(make_literal(False, P(n_term(x)))),
        make_clause(make_literal(True, P(i_term(n_term(x), x)))),
    ]


def make_trained_model() -> Tree2Vec:
    config = Tree2VecConfig(
        skipgram_config=SkipGramConfig(embedding_dim=8, num_epochs=2, seed=42),
    )
    t2v = Tree2Vec(config)
    t2v.train(make_test_clauses())
    return t2v


# ── Tests ────────────────────────────────────────────────────────────────────


def test_save_load_roundtrip_embeddings_match(tmp_path):
    """Save then load — embeddings must match within floating-point tolerance."""
    t2v = make_trained_model()
    clauses = make_test_clauses()

    # Compute reference embeddings before save
    original_embs = [t2v.embed_clause(c) for c in clauses]

    model_file = tmp_path / "model.t2v.json"
    t2v.save(model_file)

    loaded = Tree2Vec.load(model_file)

    # Check metadata
    assert loaded.vocab_size == t2v.vocab_size
    assert loaded.embedding_dim == t2v.embedding_dim
    assert loaded.trained is True

    # Check embeddings match
    for i, clause in enumerate(clauses):
        orig = original_embs[i]
        loaded_emb = loaded.embed_clause(clause)
        if orig is None:
            assert loaded_emb is None
        else:
            assert loaded_emb is not None
            assert len(loaded_emb) == len(orig)
            for a, b in zip(orig, loaded_emb):
                assert abs(a - b) < 1e-9, f"Embedding mismatch at position {i}"


def test_save_untrained_raises(tmp_path):
    """save() on an untrained model must raise RuntimeError."""
    t2v = Tree2Vec()
    with pytest.raises(RuntimeError, match="untrained"):
        t2v.save(tmp_path / "model.t2v.json")


def test_load_wrong_version_raises(tmp_path):
    """load() with wrong format_version must raise ValueError."""
    bad_file = tmp_path / "bad.t2v.json"
    bad_file.write_text(json.dumps({"format_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="format version"):
        Tree2Vec.load(bad_file)


def test_from_saved_model_factory(tmp_path):
    """from_saved_model() returns a working provider."""
    from pyladr.ml.tree2vec.provider import Tree2VecEmbeddingProvider

    t2v = make_trained_model()
    model_file = tmp_path / "model.t2v.json"
    t2v.save(model_file)

    provider = Tree2VecEmbeddingProvider.from_saved_model(str(model_file))
    assert provider.embedding_dim == t2v.embedding_dim

    clause = make_test_clauses()[0]
    emb = provider.get_embedding(clause)
    # Should produce an embedding (not None)
    assert emb is not None
    assert len(emb) == t2v.embedding_dim


def test_search_options_model_path_field():
    """tree2vec_model_path field exists on SearchOptions and defaults to empty string."""
    from pyladr.search.given_clause import SearchOptions

    opts = SearchOptions()
    assert hasattr(opts, "tree2vec_model_path")
    assert opts.tree2vec_model_path == ""

    opts2 = SearchOptions(tree2vec_model_path="/some/path.t2v.json")
    assert opts2.tree2vec_model_path == "/some/path.t2v.json"


def test_cli_train_corpus_flag(tmp_path):
    """--tree2vec-train-corpus exits 0 and creates the .t2v.json file."""
    from pyladr.apps.prover9 import run_prover

    fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_group.in"
    if not fixture.exists():
        pytest.skip(f"Fixture not found: {fixture}")

    output_file = str(fixture) + ".t2v.json"
    try:
        exit_code = run_prover(["pyprover9", "--tree2vec-train-corpus", str(fixture)])
        assert exit_code == 0
        assert Path(output_file).exists(), "Model file was not created"

        # Check the file is valid JSON with expected structure
        data = json.loads(Path(output_file).read_text(encoding="utf-8"))
        assert data["format_version"] == Tree2Vec.SAVE_FORMAT_VERSION
        assert "trainer" in data
    finally:
        Path(output_file).unlink(missing_ok=True)


def test_cli_load_model_flag(tmp_path):
    """Train a model then run search with --tree2vec-load-model; must not crash."""
    from pyladr.apps.prover9 import run_prover

    fixture = Path(__file__).parent.parent / "fixtures" / "inputs" / "simple_group.in"
    if not fixture.exists():
        pytest.skip(f"Fixture not found: {fixture}")

    model_file = tmp_path / "model.t2v.json"

    # Step 1: train
    exit_code = run_prover(["pyprover9", "--tree2vec-train-corpus", str(fixture)])
    assert exit_code == 0
    # Move the auto-generated file to tmp_path
    auto_path = Path(str(fixture) + ".t2v.json")
    if auto_path.exists():
        auto_path.rename(model_file)
    assert model_file.exists(), "Trained model file not found"

    # Step 2: search with frozen model
    exit_code = run_prover([
        "pyprover9",
        "-f", str(fixture),
        "--tree2vec-load-model", str(model_file),
        "--tree2vec-weight", "1",
        "-max_seconds", "5",
    ])
    # Any exit code except 1 (fatal error) is acceptable
    assert exit_code != 1, f"Search crashed with fatal error (exit code {exit_code})"
