# RNN2Vec Test Coverage Plan

**Author:** Edsger (Testing Specialist)
**Date:** 2026-04-21
**Status:** Planning Document — No Code

This document defines the complete test coverage plan for the RNN2Vec embedding system. It mirrors and extends the patterns established by the existing Tree2Vec test suite (`tests/unit/test_tree2vec.py`, `tests/unit/test_online_tree2vec.py`) and the GNN embedding provider tests (`tests/unit/test_embedding_provider.py`).

---

## 1. Unit Test Plan: Core RNN2Vec, RNN Encoder, and Provider

### 1.1 Test File: `tests/unit/test_rnn2vec.py`

Core algorithm tests, paralleling `test_tree2vec.py`.

#### `class TestRNN2VecConfig`
| Method | Purpose |
|--------|---------|
| `test_default_config_values` | Verify all defaults: `rnn_type="lstm"`, `hidden_dim=128`, `embedding_dim=64`, `num_layers=1`, `bidirectional=False`, `dropout=0.0`, `normalize=True`, `composition="last_hidden"` |
| `test_frozen_config` | Config is immutable after creation |
| `test_embedding_dim_with_bidirectional` | When `bidirectional=True`, effective hidden is `2 * hidden_dim`; verify `embedding_dim` projection accounts for this |
| `test_config_with_gru` | `rnn_type="gru"` accepted without error |
| `test_config_with_vanilla_rnn` | `rnn_type="rnn"` accepted without error |
| `test_invalid_rnn_type_rejected` | Unknown `rnn_type` raises `ValueError` |

#### `class TestRNN2Vec`
| Method | Purpose |
|--------|---------|
| `test_train_and_embed` | Train on vampire.in-style clauses, verify `trained` flag, stats dict has `vocab_size > 0` |
| `test_embed_term_returns_correct_dim` | `embed_term()` returns `list[float]` of length `embedding_dim` |
| `test_embed_clause_returns_correct_dim` | `embed_clause()` returns `list[float]` of length `embedding_dim` |
| `test_embed_clauses_batch_parallel_to_input` | `embed_clauses_batch()` returns list same length as input |
| `test_untrained_returns_none` | Before `train()`, `embed_term()` and `embed_clause()` return `None` |
| `test_similarity_structurally_similar` | α-equivalent terms (same structure, different var IDs) get similarity > 0.99 |
| `test_similarity_structurally_different` | Structurally distinct terms get similarity < structurally similar |
| `test_normalized_embeddings_unit_norm` | When `normalize=True`, embedding L2 norm ≈ 1.0 (within 1e-6) |
| `test_unnormalized_embeddings` | When `normalize=False`, norm is NOT constrained to 1.0 |
| `test_train_from_terms` | `train_from_terms()` works without clause wrapper |
| `test_deterministic_training` | Same config + same seed → identical embeddings |
| `test_most_similar_tokens` | After training, `most_similar_tokens()` returns `list[tuple[str, float]]` |
| `test_get_token_embedding` | Known token returns `list[float]`; unknown returns `None` |

#### `class TestRNN2VecComposition`
| Method | Purpose |
|--------|---------|
| `test_last_hidden_composition` | `composition="last_hidden"` — uses final RNN hidden state |
| `test_mean_pooling_composition` | `composition="mean_pooling"` — averages all RNN timestep outputs |
| `test_attention_pooling_composition` | `composition="attention_pooling"` — learned attention over timesteps |
| `test_composition_dimension_consistent` | All composition strategies produce same `embedding_dim` |

#### `class TestRNN2VecSerialization`
| Method | Purpose |
|--------|---------|
| `test_save_raises_if_untrained` | `save()` on untrained model raises `RuntimeError` |
| `test_save_load_round_trip` | Save to tmpdir, load back, verify identical embeddings on same inputs |
| `test_load_invalid_version_raises` | Loading file with wrong `format_version` raises `ValueError` |
| `test_save_load_preserves_config` | Loaded model has identical config to original |
| `test_save_load_preserves_vocab` | Loaded model has identical vocabulary |

### 1.2 Test File: `tests/unit/test_rnn_encoder.py`

Low-level RNN cell and encoder tests.

#### `class TestLSTMCell`
| Method | Purpose |
|--------|---------|
| `test_output_shape` | Given `(input_dim, hidden_dim)`, cell output is `(hidden_dim,)` for both h and c |
| `test_forget_gate_range` | Forget gate values are in [0, 1] (sigmoid output) |
| `test_input_gate_range` | Input gate values are in [0, 1] |
| `test_output_gate_range` | Output gate values are in [0, 1] |
| `test_cell_state_update` | `c_new = f * c_old + i * g` — verify formula numerically for known inputs |
| `test_hidden_state_from_cell` | `h_new = o * tanh(c_new)` — verify formula |
| `test_zero_input_zero_hidden` | Zero inputs produce zero-ish output (only bias terms contribute) |
| `test_deterministic_with_seed` | Same inputs produce identical outputs |

#### `class TestGRUCell`
| Method | Purpose |
|--------|---------|
| `test_output_shape` | Cell output is `(hidden_dim,)` |
| `test_reset_gate_range` | Reset gate in [0, 1] |
| `test_update_gate_range` | Update gate in [0, 1] |
| `test_hidden_state_formula` | `h = (1 - z) * n + z * h_old` — verify numerically |
| `test_deterministic_with_seed` | Reproducible outputs |

#### `class TestVanillaRNNCell`
| Method | Purpose |
|--------|---------|
| `test_output_shape` | Cell output is `(hidden_dim,)` |
| `test_tanh_bounded` | Output values in [-1, 1] |
| `test_deterministic_with_seed` | Reproducible outputs |

#### `class TestRNNEncoder`
| Method | Purpose |
|--------|---------|
| `test_forward_single_sequence` | Encode one token sequence → hidden state of correct shape |
| `test_forward_empty_sequence` | Empty sequence returns zero vector or None (graceful) |
| `test_forward_single_token` | Single-token sequence returns valid hidden state |
| `test_hidden_dim_matches_config` | Output dimension matches `hidden_dim` (or `2 * hidden_dim` if bidirectional) |
| `test_multi_layer_stacking` | `num_layers > 1` produces valid output of same hidden dim |
| `test_bidirectional_doubles_dim` | Bidirectional output is `2 * hidden_dim` before projection |
| `test_dropout_no_effect_eval_mode` | With dropout > 0 in eval mode, output is deterministic |

### 1.3 Test File: `tests/unit/test_rnn2vec_provider.py`

Provider wrapping the RNN2Vec algorithm.

#### `class TestRNN2VecEmbeddingProviderBasic`
| Method | Purpose |
|--------|---------|
| `test_embedding_dim_property` | `embedding_dim` matches RNN2Vec config |
| `test_get_embedding_returns_list_or_none` | Type check on return value |
| `test_get_embedding_correct_length` | Embedding length == `embedding_dim` |
| `test_get_embeddings_batch_returns_list` | Batch returns `list[list[float] | None]` |
| `test_get_embeddings_batch_empty_input` | Empty list → empty list |
| `test_untrained_returns_none` | Before training, all embeddings are `None` |
| `test_untrained_batch_returns_all_none` | Batch on untrained model → `[None, None, ...]` |

#### `class TestRNN2VecProviderCache`
| Method | Purpose |
|--------|---------|
| `test_cache_populated_after_get` | Cache size increases after embedding request |
| `test_cache_hit_on_second_call` | Second call for same clause hits cache (stats verify) |
| `test_alpha_equivalent_clauses_share_cache` | P(x0) and P(x1) produce same structural hash → 1 cache entry |
| `test_cache_eviction_at_capacity` | When cache exceeds `max_entries`, LRU entries evicted |
| `test_cache_disabled` | `enable_cache=False` → embeddings recomputed each call |
| `test_cache_stats_snapshot` | `stats.snapshot()` returns dict with hits, misses, evictions, hit_rate |

#### `class TestRNN2VecProviderVersioning`
| Method | Purpose |
|--------|---------|
| `test_version_starts_at_zero` | Initial `model_version == 0` |
| `test_bump_increments_version` | `bump_model_version()` returns 1, then 2, etc. |
| `test_stale_cache_after_bump` | After version bump, cached embeddings treated as misses |
| `test_invalidate_all_clears_cache` | `invalidate_all()` drops all entries |

#### `class TestRNN2VecProviderFactory`
| Method | Purpose |
|--------|---------|
| `test_from_vampire_file` | Construct from `.in` file, verify `trained == True` |
| `test_from_saved_model` | Load from saved JSON, verify `trained == True` |
| `test_from_corpus` | Construct from pre-parsed corpus |

#### `class TestRNN2VecProviderThreadSafety`
| Method | Purpose |
|--------|---------|
| `test_concurrent_reads` | 4 threads reading embeddings concurrently — no crash |
| `test_concurrent_reads_during_version_bump` | Reads during version bump — no crash, results valid |

---

## 2. EmbeddingProvider Protocol Compliance Tests

### Test File: `tests/unit/test_rnn2vec_protocol_compliance.py`

#### `class TestRNN2VecProtocolCompliance`
| Method | Purpose |
|--------|---------|
| `test_isinstance_check` | `isinstance(provider, EmbeddingProvider)` is `True` (runtime_checkable) |
| `test_has_embedding_dim_property` | `hasattr(provider, 'embedding_dim')` and returns `int` |
| `test_has_get_embedding_method` | Method exists and accepts `Clause` argument |
| `test_has_get_embeddings_batch_method` | Method exists and accepts `list[Clause]` |
| `test_get_embedding_returns_list_float_or_none` | Return type is exactly `list[float]` or `None` — NOT `list[int]`, NOT `numpy.ndarray` |
| `test_embedding_length_matches_dim` | `len(get_embedding(c)) == embedding_dim` for all non-None results |
| `test_batch_length_matches_input` | `len(get_embeddings_batch(cs)) == len(cs)` |
| `test_batch_element_types` | Each batch element is `list[float]` or `None` |
| `test_graceful_degradation_untrained` | Returns `None` when model not trained — does NOT raise |
| `test_graceful_degradation_error` | If internal error occurs, returns `None` — does NOT propagate exception |
| `test_empty_batch_returns_empty` | `get_embeddings_batch([]) == []` |

---

## 3. Behavioral Correctness Tests

### Test File: `tests/unit/test_rnn2vec_behavioral.py`

#### `class TestAlphaEquivalence`
| Method | Purpose |
|--------|---------|
| `test_alpha_equivalent_terms_same_embedding` | `i(x0, n(x0))` and `i(x1, n(x1))` produce identical embeddings |
| `test_alpha_equivalent_clauses_same_embedding` | `P(i(x, y))` with var(0),var(1) vs var(2),var(3) → same embedding |
| `test_variable_renaming_preserves_similarity` | Similarity between α-equivalent pairs is 1.0 (or > 0.999) |

#### `class TestNormalization`
| Method | Purpose |
|--------|---------|
| `test_normalized_unit_norm` | All embeddings with `normalize=True` have L2 norm within 1e-6 of 1.0 |
| `test_normalization_idempotent` | Normalizing an already-normalized vector doesn't change it |
| `test_zero_vector_graceful` | If composition produces zero vector, normalization returns zero (not NaN/crash) |

#### `class TestSaveLoadRoundTrip`
| Method | Purpose |
|--------|---------|
| `test_embeddings_identical_after_reload` | For 10 test clauses, `emb_before == emb_after_load` within 1e-10 |
| `test_config_preserved_after_reload` | `rnn_type`, `hidden_dim`, `embedding_dim`, `composition`, `normalize` all match |
| `test_vocab_preserved_after_reload` | Same tokens map to same IDs after reload |
| `test_online_update_then_save_load` | Train → online update → save → load → embeddings match |

#### `class TestOnlineUpdateChangesEmbeddings`
| Method | Purpose |
|--------|---------|
| `test_embeddings_drift_after_update` | After `update_online()` with novel clauses, some embeddings change (L2 diff > 1e-10) |
| `test_untrained_update_is_noop` | `update_online()` on untrained model returns `pairs_trained == 0` |
| `test_multiple_updates_accumulate` | Each successive update continues to shift embeddings |

---

## 4. RNN Cell Unit Tests

### (Covered in `tests/unit/test_rnn_encoder.py` §1.2 above)

Additional numerical precision tests:

#### `class TestLSTMNumericalPrecision`
| Method | Purpose |
|--------|---------|
| `test_gate_computation_manual` | Hand-compute gates for a known 2-dim input/hidden, verify match |
| `test_large_input_no_overflow` | Input values of ±100 produce valid (not NaN/Inf) output |
| `test_gradient_flow` | For BPTT: verify gradient of loss w.r.t. input embedding is non-zero after 3 timesteps |

#### `class TestGRUNumericalPrecision`
| Method | Purpose |
|--------|---------|
| `test_gate_computation_manual` | Hand-compute gates, verify match |
| `test_large_input_no_overflow` | Stability under large inputs |

---

## 5. Comparison/Integration Tests: RNN2Vec and Tree2Vec Coexistence

### Test File: `tests/unit/test_rnn2vec_tree2vec_coexistence.py`

#### `class TestProviderCoexistence`
| Method | Purpose |
|--------|---------|
| `test_both_satisfy_protocol` | Both `RNN2VecEmbeddingProvider` and `Tree2VecEmbeddingProvider` pass `isinstance(p, EmbeddingProvider)` |
| `test_same_clause_both_providers` | Same clause fed to both providers — both return valid embeddings (dimensions may differ) |
| `test_provider_swap_during_search` | Swap provider type between search runs — no crash, proofs still found |
| `test_embedding_dim_independent` | Each provider reports its own `embedding_dim`, not coupled to the other |

#### `class TestSelectionIntegration`
| Method | Purpose |
|--------|---------|
| `test_ml_selection_accepts_rnn2vec` | `EmbeddingEnhancedSelection` works with `RNN2VecEmbeddingProvider` |
| `test_ml_selection_accepts_tree2vec` | Baseline: existing Tree2Vec still works in selection |
| `test_goal_directed_with_rnn2vec` | `GoalDirectedEmbeddingProvider` wrapping `RNN2VecEmbeddingProvider` works |

---

## 6. Performance Regression Test Plan

### Test File: `tests/benchmarks/bench_rnn2vec_performance.py`

#### `class TestRNN2VecTrainingPerformance`
| Method | Purpose | Threshold |
|--------|---------|-----------|
| `test_training_time_small_corpus` | Train on 50 clauses (vampire.in-scale) — measure wall time | < 5s |
| `test_training_time_medium_corpus` | Train on 500 clauses — measure wall time | < 30s |
| `test_embedding_throughput` | Embed 1000 clauses, report clauses/sec | > 500 cls/s |
| `test_single_embedding_latency` | Measure p50, p99 latency for single clause embedding | p99 < 5ms |
| `test_batch_embedding_throughput` | `embed_clauses_batch(100)` — report batch throughput | > 2000 cls/s |

#### `class TestRNN2VecMemoryFootprint`
| Method | Purpose | Threshold |
|--------|---------|-----------|
| `test_model_memory_usage` | Measure memory of trained model (vocab=100, hidden=128, layers=1) | < 10 MB |
| `test_cache_memory_at_capacity` | Cache with 100k entries — measure memory | < 500 MB |

#### `class TestRNN2VecVsTree2VecComparison`
| Method | Purpose |
|--------|---------|
| `test_training_time_comparison` | Train both on same corpus, report ratio |
| `test_embedding_throughput_comparison` | Embed same 1000 clauses with both, report ratio |
| `test_embedding_quality_comparison` | α-equivalent similarity, structurally different dissimilarity — compare |

**Note:** These benchmarks are informational (logged/reported), not hard pass/fail in CI. Use `@pytest.mark.benchmark` or custom timing fixtures. Thresholds should be calibrated after initial implementation and tuned per CI hardware.

---

## 7. Compatibility Test Plan: No C Prover9 Regression

### Test File: `tests/compatibility/test_rnn2vec_search_compat.py`

These tests ensure that enabling RNN2Vec embeddings does not cause any C Prover9 compatibility regression.

#### `class TestSearchWithRNN2VecNoRegression`
| Method | Purpose |
|--------|---------|
| `test_simple_proof_found` | `_SIMPLE_PROOF_INPUT` finds proof with RNN2Vec enabled |
| `test_exit_code_matches_baseline` | Exit code with RNN2Vec == exit code without |
| `test_proof_still_found_vampire_in` | `vampire.in` problem finds proof with RNN2Vec enabled |
| `test_no_proof_timeout_matches` | Unsolvable problem: timeout behavior unchanged |
| `test_rnn2vec_disabled_by_default` | Default `SearchOptions()` has `rnn2vec_embeddings=False` — zero overhead |
| `test_set_rnn2vec_embeddings_parsed` | `set(rnn2vec_embeddings).` in input text enables RNN2Vec |
| `test_assign_rnn2vec_parameters_parsed` | `assign(rnn2vec_hidden_dim, 64).` etc. parsed correctly |

#### `class TestSearchOptionDefaults`
| Method | Purpose |
|--------|---------|
| `test_rnn2vec_defaults` | `rnn2vec_embeddings=False`, `rnn2vec_online_learning=False`, etc. |
| `test_rnn2vec_flags_no_effect_when_disabled` | Online learning flags ignored when `rnn2vec_embeddings=False` |
| `test_tree2vec_and_rnn2vec_mutual_exclusion` | Cannot enable both `tree2vec_embeddings` and `rnn2vec_embeddings` simultaneously (or: define precedence) |

#### `class TestCProver9Equivalence`
(Only runs when C binary is available — `@pytest.mark.c_prover9`)

| Method | Purpose |
|--------|---------|
| `test_simple_group_proof_found` | C and Python both find proof for `simple_group.in` |
| `test_exit_codes_match` | Exit codes identical for 5 standard test inputs |
| `test_rnn2vec_does_not_change_non_ml_search` | With `rnn2vec_embeddings=False`, search is byte-for-byte identical to baseline |

---

## 8. Edge Case Coverage

### Test File: `tests/unit/test_rnn2vec_edge_cases.py`

#### `class TestEdgeCaseEmptyClauses`
| Method | Purpose |
|--------|---------|
| `test_empty_clause_no_literals` | `Clause(literals=())` → `embed_clause()` returns `None` (no atoms to encode) |
| `test_empty_clause_in_batch` | Batch with empty clause → corresponding entry is `None`, others valid |
| `test_empty_clause_list_for_training` | `train([])` → stats show `vocab_size == 0`, no crash |

#### `class TestEdgeCasePropositional`
| Method | Purpose |
|--------|---------|
| `test_propositional_atom_zero_arity` | `P` (arity 0, no args) → valid embedding (single token "CONST:N") |
| `test_propositional_clause` | Clause with only propositional literals → valid embedding |
| `test_mixed_propositional_and_first_order` | Clause with both → valid embedding |

#### `class TestEdgeCaseDeepTrees`
| Method | Purpose |
|--------|---------|
| `test_deeply_nested_term_depth_10` | `n(n(n(n(n(n(n(n(n(n(x))))))))))` — 10 levels deep → valid embedding, no stack overflow |
| `test_deeply_nested_term_depth_50` | 50 levels → still valid (RNN processes as sequence, no recursion limit) |
| `test_deep_tree_embedding_varies` | Depth-10 and depth-2 terms produce different embeddings |

#### `class TestEdgeCaseLargeClauses`
| Method | Purpose |
|--------|---------|
| `test_clause_with_100_literals` | 100-literal clause → valid embedding, completes in < 1s |
| `test_clause_with_wide_term` | Term with arity 20 → valid embedding |
| `test_large_batch_1000_clauses` | Batch of 1000 → all valid, completes in reasonable time |

#### `class TestEdgeCaseOOVTokens`
| Method | Purpose |
|--------|---------|
| `test_novel_symbol_after_training` | New symbol ID not in training vocab → handled gracefully (zero embedding or skip) |
| `test_online_update_extends_vocab` | After online update with novel tokens, they get embeddings (if `vocab_extension=True`) |
| `test_online_update_skips_oov_when_disabled` | With `vocab_extension=False`, OOV tokens skipped, stats reported |
| `test_all_oov_clause` | Clause with entirely novel symbols → returns `None` or degraded embedding, no crash |

#### `class TestEdgeCaseSingletonInputs`
| Method | Purpose |
|--------|---------|
| `test_single_variable_term` | `var(0)` alone → valid embedding (single-token sequence) |
| `test_single_constant_term` | Constant (arity 0) → valid embedding |
| `test_single_literal_clause` | One-literal clause → valid embedding |
| `test_training_on_one_clause` | Train on single clause → `trained == True`, embeddings available |

---

## Test Infrastructure Notes

### Fixtures and Helpers

All RNN2Vec test files should share a common helper module (or conftest fixtures):

```python
# tests/conftest.py additions (or tests/rnn2vec_helpers.py)

@pytest.fixture
def vampire_clauses() -> list[Clause]:
    """Standard vampire.in-domain clauses for testing."""
    ...

@pytest.fixture
def trained_rnn2vec(vampire_clauses) -> RNN2Vec:
    """Pre-trained RNN2Vec model on vampire.in domain."""
    ...

@pytest.fixture
def rnn2vec_provider(trained_rnn2vec) -> RNN2VecEmbeddingProvider:
    """Provider wrapping trained RNN2Vec."""
    ...
```

### Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.slow` | Training and benchmark tests (> 5s) |
| `@pytest.mark.benchmark` | Performance measurement tests |
| `@pytest.mark.c_prover9` | Requires C Prover9 binary |

### Coverage Targets

| Module | Target Coverage |
|--------|----------------|
| `pyladr/ml/rnn2vec/algorithm.py` | ≥ 95% |
| `pyladr/ml/rnn2vec/encoder.py` | ≥ 95% |
| `pyladr/ml/rnn2vec/provider.py` | ≥ 90% |
| `pyladr/ml/rnn2vec/config.py` | 100% |

### Test Count Summary

| File | Tests | Category |
|------|-------|----------|
| `test_rnn2vec.py` | ~26 | Core algorithm |
| `test_rnn_encoder.py` | ~22 | RNN cells + encoder |
| `test_rnn2vec_provider.py` | ~20 | Provider + cache |
| `test_rnn2vec_protocol_compliance.py` | 11 | Protocol conformance |
| `test_rnn2vec_behavioral.py` | ~12 | Correctness invariants |
| `test_rnn2vec_tree2vec_coexistence.py` | ~7 | Coexistence + selection |
| `bench_rnn2vec_performance.py` | ~7 | Benchmarks |
| `test_rnn2vec_search_compat.py` | ~10 | Compatibility |
| `test_rnn2vec_edge_cases.py` | ~18 | Edge cases |
| **Total** | **~133** | |
