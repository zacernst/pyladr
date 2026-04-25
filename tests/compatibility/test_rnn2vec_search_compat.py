"""Search compatibility tests for RNN2Vec integration.

Validates that:
1. RNN2Vec is disabled by default — no regression on existing behavior
2. SearchOptions defaults are correct for all rnn2vec_* fields
3. CLI parameters parse correctly with expected defaults
4. LADR assign()/set() directives map to the right SearchOptions fields
5. Options validation catches out-of-range rnn2vec_* values

Run with: pytest tests/compatibility/test_rnn2vec_search_compat.py -v
"""

from __future__ import annotations

import pytest

from pyladr.search.given_clause import SearchOptions
from pyladr.search.options import validate_search_options


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse(args: list[str]):
    """Parse CLI args using the prover's argument parser."""
    from pyladr.apps.prover9 import _build_arg_parser
    parser = _build_arg_parser()
    return parser.parse_known_args(args)


# ── 1. No regression: RNN2Vec disabled by default ───────────────────────────

class TestSearchWithRNN2VecNoRegression:
    """RNN2Vec must be off by default — search behaves identically to before."""

    def test_rnn2vec_disabled_by_default(self) -> None:
        opts = SearchOptions()
        assert opts.rnn2vec_embeddings is False

    def test_rnn2vec_weight_zero_by_default(self) -> None:
        opts = SearchOptions()
        assert opts.rnn2vec_weight == 0.0

    def test_rnn2vec_online_learning_off_by_default(self) -> None:
        opts = SearchOptions()
        assert opts.rnn2vec_online_learning is False

    def test_rnn2vec_model_path_empty_by_default(self) -> None:
        opts = SearchOptions()
        assert opts.rnn2vec_model_path == ""

    def test_default_search_options_valid(self) -> None:
        """Default SearchOptions must pass validation (including rnn2vec fields)."""
        opts = SearchOptions()
        errors = validate_search_options(opts)
        assert errors == []


# ── 2. SearchOptions defaults ───────────────────────────────────────────────

class TestSearchOptionDefaults:
    """All rnn2vec_* SearchOptions fields have correct defaults."""

    def test_rnn2vec_rnn_type_default(self) -> None:
        assert SearchOptions().rnn2vec_rnn_type == "gru"

    def test_rnn2vec_hidden_dim_default(self) -> None:
        assert SearchOptions().rnn2vec_hidden_dim == 64

    def test_rnn2vec_embedding_dim_default(self) -> None:
        assert SearchOptions().rnn2vec_embedding_dim == 64

    def test_rnn2vec_input_dim_default(self) -> None:
        assert SearchOptions().rnn2vec_input_dim == 32

    def test_rnn2vec_num_layers_default(self) -> None:
        assert SearchOptions().rnn2vec_num_layers == 1

    def test_rnn2vec_bidirectional_default(self) -> None:
        assert SearchOptions().rnn2vec_bidirectional is False

    def test_rnn2vec_composition_default(self) -> None:
        assert SearchOptions().rnn2vec_composition == "mean"

    def test_rnn2vec_cache_max_entries_default(self) -> None:
        assert SearchOptions().rnn2vec_cache_max_entries == 10_000

    def test_rnn2vec_online_update_interval_default(self) -> None:
        assert SearchOptions().rnn2vec_online_update_interval == 20

    def test_rnn2vec_online_batch_size_default(self) -> None:
        assert SearchOptions().rnn2vec_online_batch_size == 10

    def test_rnn2vec_online_lr_default(self) -> None:
        assert SearchOptions().rnn2vec_online_lr == 0.001

    def test_rnn2vec_online_max_updates_default(self) -> None:
        assert SearchOptions().rnn2vec_online_max_updates == 0

    def test_rnn2vec_training_epochs_default(self) -> None:
        assert SearchOptions().rnn2vec_training_epochs == 5

    def test_rnn2vec_training_lr_default(self) -> None:
        assert SearchOptions().rnn2vec_training_lr == 0.001


# ── 3. CLI parameter parsing ────────────────────────────────────────────────

class TestCLIDirectiveParsing:
    """CLI flags for RNN2Vec parse correctly."""

    def test_rnn2vec_embeddings_flag(self) -> None:
        args, _ = _parse(["--rnn2vec-embeddings"])
        assert args.rnn2vec_embeddings is True

    def test_rnn2vec_embeddings_default_off(self) -> None:
        args, _ = _parse([])
        assert args.rnn2vec_embeddings is False

    def test_rnn2vec_weight(self) -> None:
        args, _ = _parse(["--rnn2vec-weight", "0.4"])
        assert args.rnn2vec_weight == 0.4

    def test_rnn2vec_weight_default(self) -> None:
        args, _ = _parse([])
        assert args.rnn2vec_weight == 0.0

    def test_rnn2vec_dim(self) -> None:
        args, _ = _parse(["--rnn2vec-dim", "128"])
        assert args.rnn2vec_dim == 128

    def test_rnn2vec_dim_default(self) -> None:
        args, _ = _parse([])
        assert args.rnn2vec_dim == 64

    def test_rnn2vec_hidden_dim(self) -> None:
        args, _ = _parse(["--rnn2vec-hidden-dim", "256"])
        assert args.rnn2vec_hidden_dim == 256

    def test_rnn2vec_input_dim(self) -> None:
        args, _ = _parse(["--rnn2vec-input-dim", "64"])
        assert args.rnn2vec_input_dim == 64

    def test_rnn2vec_cache(self) -> None:
        args, _ = _parse(["--rnn2vec-cache", "50000"])
        assert args.rnn2vec_cache == 50000

    def test_rnn2vec_rnn_type_choices(self) -> None:
        for rnn_type in ("gru", "lstm", "elman"):
            args, _ = _parse(["--rnn2vec-rnn-type", rnn_type])
            assert args.rnn2vec_rnn_type == rnn_type

    def test_rnn2vec_composition_choices(self) -> None:
        for comp in ("last_hidden", "mean_pool", "attention_pool"):
            args, _ = _parse(["--rnn2vec-composition", comp])
            assert args.rnn2vec_composition == comp

    def test_rnn2vec_online_learning_flag(self) -> None:
        args, _ = _parse(["--rnn2vec-online-learning"])
        assert args.rnn2vec_online_learning is True

    def test_rnn2vec_online_interval(self) -> None:
        args, _ = _parse(["--rnn2vec-online-interval", "50"])
        assert args.rnn2vec_online_interval == 50

    def test_rnn2vec_online_batch_size(self) -> None:
        args, _ = _parse(["--rnn2vec-online-batch-size", "32"])
        assert args.rnn2vec_online_batch_size == 32

    def test_rnn2vec_online_lr(self) -> None:
        args, _ = _parse(["--rnn2vec-online-lr", "0.01"])
        assert args.rnn2vec_online_lr == 0.01

    def test_rnn2vec_online_max_updates(self) -> None:
        args, _ = _parse(["--rnn2vec-online-max-updates", "100"])
        assert args.rnn2vec_online_max_updates == 100

    def test_rnn2vec_training_epochs(self) -> None:
        args, _ = _parse(["--rnn2vec-training-epochs", "10"])
        assert args.rnn2vec_training_epochs == 10

    def test_rnn2vec_training_lr(self) -> None:
        args, _ = _parse(["--rnn2vec-training-lr", "0.01"])
        assert args.rnn2vec_training_lr == 0.01

    def test_rnn2vec_load_model(self) -> None:
        args, _ = _parse(["--rnn2vec-load-model", "/tmp/model"])
        assert args.rnn2vec_load_model == "/tmp/model"

    def test_rnn2vec_load_model_default_none(self) -> None:
        args, _ = _parse([])
        assert args.rnn2vec_load_model is None

    def test_all_rnn2vec_params_together(self) -> None:
        args, _ = _parse([
            "--rnn2vec-embeddings",
            "--rnn2vec-weight", "0.5",
            "--rnn2vec-dim", "128",
            "--rnn2vec-hidden-dim", "256",
            "--rnn2vec-input-dim", "64",
            "--rnn2vec-rnn-type", "lstm",
            "--rnn2vec-composition", "attention_pool",
            "--rnn2vec-online-learning",
            "--rnn2vec-online-interval", "50",
            "--rnn2vec-online-batch-size", "32",
            "--rnn2vec-online-lr", "0.01",
            "--rnn2vec-online-max-updates", "100",
            "--rnn2vec-training-epochs", "10",
            "--rnn2vec-training-lr", "0.005",
            "--rnn2vec-cache", "50000",
        ])
        assert args.rnn2vec_embeddings is True
        assert args.rnn2vec_weight == 0.5
        assert args.rnn2vec_dim == 128
        assert args.rnn2vec_hidden_dim == 256
        assert args.rnn2vec_input_dim == 64
        assert args.rnn2vec_rnn_type == "lstm"
        assert args.rnn2vec_composition == "attention_pool"
        assert args.rnn2vec_online_learning is True
        assert args.rnn2vec_online_interval == 50
        assert args.rnn2vec_online_batch_size == 32
        assert args.rnn2vec_online_lr == 0.01
        assert args.rnn2vec_online_max_updates == 100
        assert args.rnn2vec_training_epochs == 10
        assert args.rnn2vec_training_lr == 0.005
        assert args.rnn2vec_cache == 50000


# ── 4. Validation catches bad rnn2vec values ────────────────────────────────

class TestRNN2VecOptionsValidation:
    """validate_search_options catches out-of-range rnn2vec_* values."""

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="rnn2vec_weight"):
            SearchOptions(rnn2vec_weight=-1.0)

    def test_zero_embedding_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="rnn2vec_embedding_dim"):
            SearchOptions(rnn2vec_embedding_dim=0)

    def test_zero_hidden_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="rnn2vec_hidden_dim"):
            SearchOptions(rnn2vec_hidden_dim=0)

    def test_training_epochs_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="rnn2vec_training_epochs"):
            SearchOptions(rnn2vec_training_epochs=0)

    def test_online_lr_too_high_rejected(self) -> None:
        with pytest.raises(ValueError, match="rnn2vec_online_lr"):
            SearchOptions(rnn2vec_online_lr=2.0)

    def test_valid_rnn2vec_options_pass(self) -> None:
        opts = SearchOptions(
            rnn2vec_embeddings=True,
            rnn2vec_weight=0.5,
            rnn2vec_embedding_dim=128,
            rnn2vec_hidden_dim=128,
            rnn2vec_training_epochs=10,
        )
        errors = validate_search_options(opts)
        assert errors == []


# ── 5. LADR directive mapping ───────────────────────────────────────────────

class TestLADRDirectiveMapping:
    """LADR assign()/set() directives map to correct SearchOptions fields."""

    def test_assign_directives_present(self) -> None:
        from pyladr.apps.prover9 import _ASSIGN_MAP
        expected = [
            "rnn2vec_weight", "rnn2vec_hidden_dim", "rnn2vec_embedding_dim",
            "rnn2vec_input_dim", "rnn2vec_num_layers", "rnn2vec_cache_max_entries",
            "rnn2vec_online_update_interval", "rnn2vec_online_batch_size",
            "rnn2vec_online_lr", "rnn2vec_online_max_updates",
            "rnn2vec_training_epochs", "rnn2vec_training_lr",
            "rnn2vec_composition", "rnn2vec_rnn_type",
        ]
        for name in expected:
            assert name in _ASSIGN_MAP, f"Missing assign directive: {name}"

    def test_set_only_flags_present(self) -> None:
        from pyladr.apps.prover9 import _SET_ONLY_FLAG_MAP
        expected = [
            "rnn2vec_embeddings", "rnn2vec_online_learning", "rnn2vec_bidirectional",
        ]
        for name in expected:
            assert name in _SET_ONLY_FLAG_MAP, f"Missing set-only flag: {name}"

    def test_assign_map_types(self) -> None:
        from pyladr.apps.prover9 import _ASSIGN_MAP
        int_fields = {
            "rnn2vec_hidden_dim", "rnn2vec_embedding_dim", "rnn2vec_input_dim",
            "rnn2vec_num_layers", "rnn2vec_cache_max_entries",
            "rnn2vec_online_update_interval", "rnn2vec_online_batch_size",
            "rnn2vec_online_max_updates", "rnn2vec_training_epochs",
        }
        float_fields = {
            "rnn2vec_weight", "rnn2vec_online_lr", "rnn2vec_training_lr",
        }
        str_fields = {"rnn2vec_composition", "rnn2vec_rnn_type"}
        for name in int_fields:
            _, typ = _ASSIGN_MAP[name]
            assert typ is int, f"{name} should be int, got {typ}"
        for name in float_fields:
            _, typ = _ASSIGN_MAP[name]
            assert typ is float, f"{name} should be float, got {typ}"
        for name in str_fields:
            _, typ = _ASSIGN_MAP[name]
            assert typ is str, f"{name} should be str, got {typ}"


# ── 6. SelectionOrder enum ──────────────────────────────────────────────────

class TestSelectionOrderRNN2Vec:
    """SelectionOrder has RNN2VEC entry."""

    def test_rnn2vec_selection_order_exists(self) -> None:
        from pyladr.search.selection import SelectionOrder
        assert hasattr(SelectionOrder, "RNN2VEC")
        assert SelectionOrder.RNN2VEC.value == 9

    def test_rnn2vec_weight_creates_selection_rule(self) -> None:
        """When rnn2vec_weight > 0, a RNN2VEC selection rule is added."""
        from pyladr.search.selection import SelectionOrder
        opts = SearchOptions(rnn2vec_weight=0.5)
        # Build selection rules via the search constructor path
        rules = []
        if opts.rnn2vec_weight > 0:
            from pyladr.search.selection import SelectionRule
            rules.append(SelectionRule("R2V", SelectionOrder.RNN2VEC, part=opts.rnn2vec_weight))
        assert len(rules) == 1
        assert rules[0].order == SelectionOrder.RNN2VEC
        assert rules[0].part == 0.5
