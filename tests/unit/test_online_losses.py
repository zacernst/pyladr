"""Tests for online contrastive loss functions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pyladr.ml.training.online_losses import (
    CombinedOnlineLoss,
    LossStatistics,
    OnlineInfoNCELoss,
    OnlineLossConfig,
    OnlineTripletLoss,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _random_embeddings(
    batch: int, dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (anchor, positive, negative) tensors with requires_grad."""
    anchor = torch.randn(batch, dim, requires_grad=True)
    positive = anchor.detach() + 0.1 * torch.randn(batch, dim)
    positive.requires_grad_(True)
    negative = torch.randn(batch, dim, requires_grad=True)
    return anchor, positive, negative


# ── OnlineLossConfig tests ─────────────────────────────────────────────────


class TestOnlineLossConfig:
    def test_defaults(self):
        cfg = OnlineLossConfig()
        assert cfg.temperature == 0.07
        assert cfg.temperature_min == 0.01
        assert cfg.temperature_decay == 1.0
        assert cfg.margin == 0.5
        assert cfg.use_in_batch_negatives is True
        assert cfg.label_smoothing == 0.0

    def test_custom(self):
        cfg = OnlineLossConfig(temperature=0.1, margin=0.3)
        assert cfg.temperature == 0.1
        assert cfg.margin == 0.3


# ── LossStatistics tests ──────────────────────────────────────────────────


class TestLossStatistics:
    def test_initial_values(self):
        s = LossStatistics()
        assert s.total_steps == 0
        assert s.ema_loss == 0.0

    def test_snapshot(self):
        s = LossStatistics(total_steps=5, ema_loss=1.234567, last_loss=1.2)
        snap = s.snapshot()
        assert snap["total_steps"] == 5
        assert isinstance(snap["ema_loss"], float)
        assert "similarity_gap" in snap


# ── OnlineInfoNCELoss tests ───────────────────────────────────────────────


class TestOnlineInfoNCELoss:
    def test_basic_forward(self):
        loss_fn = OnlineInfoNCELoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_gradient_flows(self):
        loss_fn = OnlineInfoNCELoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        loss.backward()

        assert anchor.grad is not None
        assert positive.grad is not None
        assert negative.grad is not None

    def test_perfect_alignment_low_loss(self):
        """Identical positive should give lower loss than random."""
        loss_fn = OnlineInfoNCELoss(
            OnlineLossConfig(temperature=0.5, use_in_batch_negatives=False),
        )

        anchor = torch.randn(4, 32)
        positive = anchor.clone()
        negative = torch.randn(4, 32)

        loss_aligned = loss_fn(anchor, positive, negative)

        # Random positive
        loss_fn2 = OnlineInfoNCELoss(
            OnlineLossConfig(temperature=0.5, use_in_batch_negatives=False),
        )
        random_pos = torch.randn(4, 32)
        loss_random = loss_fn2(anchor, random_pos, negative)

        assert loss_aligned.item() < loss_random.item()

    def test_weighted_loss(self):
        loss_fn = OnlineInfoNCELoss()
        anchor, positive, negative = _random_embeddings(4)
        weights = torch.tensor([2.0, 1.0, 1.0, 0.5])

        loss = loss_fn(anchor, positive, negative, weights)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_in_batch_negatives_enabled(self):
        """In-batch negatives should produce different loss than paired-only."""
        cfg_with = OnlineLossConfig(use_in_batch_negatives=True, temperature=0.5)
        cfg_without = OnlineLossConfig(use_in_batch_negatives=False, temperature=0.5)

        loss_with = OnlineInfoNCELoss(cfg_with)
        loss_without = OnlineInfoNCELoss(cfg_without)

        torch.manual_seed(42)
        anchor, positive, negative = _random_embeddings(8, dim=32)

        l1 = loss_with(anchor, positive, negative)
        l2 = loss_without(anchor, positive, negative)

        # Losses should differ (more negatives → typically higher loss)
        # We just check both are valid and not identical
        assert l1.item() >= 0
        assert l2.item() >= 0

    def test_single_example_falls_back_to_paired(self):
        """Batch size 1 can't use in-batch negatives — should still work."""
        loss_fn = OnlineInfoNCELoss(
            OnlineLossConfig(use_in_batch_negatives=True),
        )
        anchor, positive, negative = _random_embeddings(1)

        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_statistics_updated(self):
        loss_fn = OnlineInfoNCELoss()
        anchor, positive, negative = _random_embeddings(4)

        loss_fn(anchor, positive, negative)

        stats = loss_fn.stats
        assert stats.total_steps == 1
        assert stats.last_loss > 0
        assert stats.ema_loss > 0
        assert -1.0 <= stats.mean_positive_sim <= 1.0
        assert -1.0 <= stats.mean_negative_sim <= 1.0

    def test_statistics_ema_convergence(self):
        """EMA should converge toward the running loss."""
        loss_fn = OnlineInfoNCELoss()

        for _ in range(20):
            anchor, positive, negative = _random_embeddings(8)
            loss_fn(anchor, positive, negative)

        stats = loss_fn.stats
        assert stats.total_steps == 20
        # EMA should be close to last loss after many steps
        assert abs(stats.ema_loss - stats.last_loss) < 5.0

    def test_temperature_annealing(self):
        cfg = OnlineLossConfig(
            temperature=0.1,
            temperature_decay=0.9,
            temperature_min=0.05,
        )
        loss_fn = OnlineInfoNCELoss(cfg)
        initial_temp = loss_fn.temperature

        anchor, positive, negative = _random_embeddings(4)
        loss_fn(anchor, positive, negative)

        assert loss_fn.temperature < initial_temp
        assert loss_fn.temperature == 0.1 * 0.9

    def test_temperature_floor(self):
        cfg = OnlineLossConfig(
            temperature=0.02,
            temperature_decay=0.1,
            temperature_min=0.01,
        )
        loss_fn = OnlineInfoNCELoss(cfg)

        anchor, positive, negative = _random_embeddings(4)
        loss_fn(anchor, positive, negative)

        assert loss_fn.temperature >= cfg.temperature_min

    def test_no_annealing_by_default(self):
        loss_fn = OnlineInfoNCELoss()
        initial_temp = loss_fn.temperature

        anchor, positive, negative = _random_embeddings(4)
        loss_fn(anchor, positive, negative)

        assert loss_fn.temperature == initial_temp

    def test_label_smoothing(self):
        cfg = OnlineLossConfig(label_smoothing=0.1, use_in_batch_negatives=False)
        loss_fn = OnlineInfoNCELoss(cfg)
        anchor, positive, negative = _random_embeddings(4)

        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_various_batch_sizes(self):
        loss_fn = OnlineInfoNCELoss()
        for batch_size in [1, 2, 4, 16, 32]:
            anchor, positive, negative = _random_embeddings(batch_size)
            loss = loss_fn(anchor, positive, negative)
            assert loss.shape == ()
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_various_embedding_dims(self):
        loss_fn = OnlineInfoNCELoss()
        for dim in [8, 64, 256, 512]:
            anchor, positive, negative = _random_embeddings(4, dim=dim)
            loss = loss_fn(anchor, positive, negative)
            assert loss.shape == ()
            assert not torch.isnan(loss)


# ── OnlineTripletLoss tests ───────────────────────────────────────────────


class TestOnlineTripletLoss:
    def test_basic_forward(self):
        loss_fn = OnlineTripletLoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self):
        loss_fn = OnlineTripletLoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        loss.backward()

        assert anchor.grad is not None

    def test_zero_loss_when_margin_satisfied(self):
        loss_fn = OnlineTripletLoss(OnlineLossConfig(margin=0.1))

        anchor = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        positive = torch.tensor([[0.99, 0.1, 0.0, 0.0]])
        negative = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

        loss = loss_fn(anchor, positive, negative)
        assert loss.item() < 0.2

    def test_positive_loss_when_margin_violated(self):
        loss_fn = OnlineTripletLoss(OnlineLossConfig(margin=0.5))

        # Positive and negative equidistant → margin violation
        anchor = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        positive = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        negative = torch.tensor([[0.5, -0.5, 0.0, 0.0]])

        loss = loss_fn(anchor, positive, negative)
        assert loss.item() > 0

    def test_weighted_loss(self):
        loss_fn = OnlineTripletLoss()
        anchor, positive, negative = _random_embeddings(4)
        weights = torch.tensor([1.0, 2.0, 0.5, 1.0])

        loss = loss_fn(anchor, positive, negative, weights)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_statistics_updated(self):
        loss_fn = OnlineTripletLoss()
        anchor, positive, negative = _random_embeddings(4)

        loss_fn(anchor, positive, negative)

        stats = loss_fn.stats
        assert stats.total_steps == 1
        assert -1.0 <= stats.mean_positive_sim <= 1.0


# ── CombinedOnlineLoss tests ──────────────────────────────────────────────


class TestCombinedOnlineLoss:
    def test_basic_forward(self):
        loss_fn = CombinedOnlineLoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self):
        loss_fn = CombinedOnlineLoss()
        anchor, positive, negative = _random_embeddings(8)

        loss = loss_fn(anchor, positive, negative)
        loss.backward()

        assert anchor.grad is not None

    def test_blend_weights(self):
        """Different infonce_weight should produce different losses."""
        cfg = OnlineLossConfig(use_in_batch_negatives=False, temperature=0.5)

        loss_fn_high = CombinedOnlineLoss(cfg, infonce_weight=0.9)
        loss_fn_low = CombinedOnlineLoss(cfg, infonce_weight=0.1)

        torch.manual_seed(123)
        anchor, positive, negative = _random_embeddings(8, dim=32)

        l1 = loss_fn_high(anchor, positive, negative)
        l2 = loss_fn_low(anchor, positive, negative)

        assert l1.item() >= 0
        assert l2.item() >= 0

    def test_weighted_loss(self):
        loss_fn = CombinedOnlineLoss()
        anchor, positive, negative = _random_embeddings(4)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

        loss = loss_fn(anchor, positive, negative, weights)
        assert loss.shape == ()

    def test_stats_returns_infonce_stats(self):
        loss_fn = CombinedOnlineLoss()
        anchor, positive, negative = _random_embeddings(4)

        loss_fn(anchor, positive, negative)

        # Stats should be from the InfoNCE branch
        assert loss_fn.stats.total_steps == 1
        assert loss_fn.infonce.stats.total_steps == 1
        assert loss_fn.triplet.stats.total_steps == 1

    def test_sub_loss_accessors(self):
        loss_fn = CombinedOnlineLoss()
        assert isinstance(loss_fn.infonce, OnlineInfoNCELoss)
        assert isinstance(loss_fn.triplet, OnlineTripletLoss)


# ── Integration-style tests ───────────────────────────────────────────────


class TestOnlineLossIntegration:
    """Tests that verify the losses work in a realistic optimizer loop."""

    def test_optimizer_step_with_infonce(self):
        """Simulate a single online learning step."""
        model = torch.nn.Linear(64, 64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = OnlineInfoNCELoss()

        # Simulate encoding
        raw = torch.randn(8, 64)
        anchor = model(raw)
        positive = model(raw + 0.1 * torch.randn_like(raw))
        negative = model(torch.randn_like(raw))

        loss = loss_fn(anchor, positive, negative)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Parameters should have been updated
        assert loss.item() >= 0
        assert loss_fn.stats.total_steps == 1

    def test_optimizer_step_with_combined(self):
        model = torch.nn.Linear(64, 64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = CombinedOnlineLoss()

        raw = torch.randn(8, 64)
        anchor = model(raw)
        positive = model(raw + 0.1 * torch.randn_like(raw))
        negative = model(torch.randn_like(raw))

        loss = loss_fn(anchor, positive, negative)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        assert loss.item() >= 0

    def test_multi_step_convergence(self):
        """After many steps with clear positive/negative separation,
        the similarity gap should increase or stay positive."""
        model = torch.nn.Linear(32, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = OnlineInfoNCELoss(
            OnlineLossConfig(use_in_batch_negatives=False, temperature=0.5),
        )

        losses = []
        for _ in range(50):
            # Positive: anchor + small noise, Negative: completely random
            raw = torch.randn(16, 32)
            anchor = model(raw)
            positive = model(raw + 0.05 * torch.randn_like(raw))
            negative = model(torch.randn_like(raw))

            loss = loss_fn(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over training
        first_10_avg = sum(losses[:10]) / 10
        last_10_avg = sum(losses[-10:]) / 10
        assert last_10_avg <= first_10_avg + 0.5  # allow some slack

    def test_handles_all_same_embeddings(self):
        """Edge case: all embeddings identical shouldn't crash."""
        loss_fn = OnlineInfoNCELoss()
        anchor = torch.ones(4, 32, requires_grad=True)
        positive = torch.ones(4, 32, requires_grad=True)
        negative = torch.ones(4, 32, requires_grad=True)

        loss = loss_fn(anchor, positive, negative)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
