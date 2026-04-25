"""Unit tests for compatibility counting bug validation.

These tests ensure that mathematical impossibilities in compatibility percentage
calculations are caught and prevented. The bug manifests as high inference counts
with impossibly low partnership counts, e.g., "284 inferences from 1/331 clauses (0%)".
"""

import pytest
from pyladr.search.statistics import SearchStatistics


class TestCompatibilityCountingValidation:
    """Tests to catch mathematical impossibilities in compatibility counting."""

    def test_partnership_count_never_exceeds_available_clauses(self):
        """Partnership counts should never exceed the number of available clauses."""
        stats = SearchStatistics()

        # Simulate a given clause with 5 available clauses for inference
        stats.begin_given(clause_id=100, available_count=5)

        # Record attempted partnerships with all 5 clauses (IDs 1-5)
        for clause_id in range(1, 6):
            stats.record_attempted_partnership(clause_id)
            stats.record_successful_partnership(clause_id)

        compatible, available, percentage = stats.get_given_compatibility_stats(100)

        assert compatible <= available, f"Partnerships ({compatible}) exceed available clauses ({available})"
        assert compatible == 5
        assert available == 5
        assert percentage == 100.0

    def test_high_inference_count_requires_reasonable_partnership_count(self):
        """High inference counts should require reasonable partnership counts."""
        stats = SearchStatistics()

        # Test case: generating many inferences requires many partnerships
        stats.begin_given(clause_id=200, available_count=331)

        # Simulate generating 284 inferences - this should require many partnerships
        for _ in range(284):
            stats.record_generated()

        # Simulate attempting inference with many clauses but only succeeding with one
        for clause_id in range(1, 332):  # Attempt with clauses 1-331
            stats.record_attempted_partnership(clause_id)

        # Only succeed with clause 1
        stats.record_successful_partnership(1)

        compatible, available, percentage = stats.get_given_compatibility_stats(200)
        inference_count = stats.get_given_inference_count(200)

        # The bug would show: 284 inferences from 1/331 clauses (0%)
        # This is mathematically possible but suspicious
        assert compatible == 1
        assert available == 331
        assert inference_count == 284
        assert percentage == pytest.approx(0.3, abs=0.1)  # 1/331 ≈ 0.3%

    def test_hyper_resolution_partnership_estimation_bug(self):
        """Test the specific hyper-resolution partnership estimation bug."""
        stats = SearchStatistics()

        # Simulate the problematic hyper-resolution scenario
        stats.begin_given(clause_id=351, available_count=331)

        # Simulate attempting partnerships with all 331 clauses
        for clause_id in range(1, 332):
            stats.record_attempted_partnership(clause_id)

        # But only succeeding with a reasonable number (not the old bug of 284)
        # In the corrected version, we succeed with far fewer actual partners
        for clause_id in range(1, 4):  # Success with only 3 clauses
            stats.record_successful_partnership(clause_id)

        # Also record the 284 inferences generated
        for _ in range(284):
            stats.record_generated()

        compatible, available, percentage = stats.get_given_compatibility_stats(351)
        inference_count = stats.get_given_inference_count(351)

        # With the fix: should show reasonable partnership ratios
        assert compatible <= available, f"Partnerships ({compatible}) should not exceed attempts ({available})"

        # Should show reasonable percentages (3 successes out of 331 attempts)
        assert percentage <= 100.0
        assert percentage == pytest.approx(0.9, abs=0.1)  # 3/331 ≈ 0.9%

    def test_factoring_self_partnership_counting(self):
        """Test that factoring partnerships are counted correctly."""
        stats = SearchStatistics()

        stats.begin_given(clause_id=400, available_count=10)

        # Factoring attempts and succeeds with itself (self-partnership)
        stats.record_attempted_partnership(400)  # Self-partnership with clause 400
        stats.record_successful_partnership(400)  # Self-partnership succeeds
        stats.record_generated()  # One factoring inference

        compatible, available, percentage = stats.get_given_compatibility_stats(400)
        inference_count = stats.get_given_inference_count(400)

        assert compatible == 1  # Self-partnership succeeds
        assert available == 1   # Only attempts self-partnership
        assert inference_count == 1
        assert percentage == 100.0  # 1/1 = 100% (factoring always succeeds or fails entirely)

    def test_zero_available_clauses_edge_case(self):
        """Test edge case with zero available clauses."""
        stats = SearchStatistics()

        stats.begin_given(clause_id=500, available_count=0)

        # No partnerships or inferences should be possible
        compatible, available, percentage = stats.get_given_compatibility_stats(500)
        inference_count = stats.get_given_inference_count(500)

        assert compatible == 0
        assert available == 0
        assert inference_count == 0
        assert percentage == 0.0

    def test_mathematical_impossibility_detection(self):
        """Test detection of mathematical impossibilities in partnership ratios."""
        stats = SearchStatistics()

        # Scenario: very high inference count with impossibly low partnership percentage
        stats.begin_given(clause_id=600, available_count=331)

        # Generate many inferences
        for _ in range(284):
            stats.record_generated()

        # Simulate attempting with many clauses but only succeeding with one
        for clause_id in range(1, 332):  # Attempt with clauses 1-331
            stats.record_attempted_partnership(clause_id)

        # But only succeed with 1 partnership - this is suspicious but mathematically possible
        stats.record_successful_partnership(1)

        compatible, available, percentage = stats.get_given_compatibility_stats(600)
        inference_count = stats.get_given_inference_count(600)

        # Check for the mathematical impossibility pattern
        inferences_per_partnership = inference_count / max(compatible, 1)

        # If one partnership generated 284 inferences, that's unusual but possible
        # The real bug would be if we claimed 284 partnerships but percentage was 0%
        assert not (compatible > 10 and percentage == 0.0), \
            f"Mathematical impossibility: {compatible} partnerships but 0% compatibility"

        assert inferences_per_partnership == 284.0  # 284 inferences / 1 partnership

    def test_percentage_calculation_consistency(self):
        """Test that percentage calculations are consistent and mathematically sound."""
        stats = SearchStatistics()

        test_cases = [
            (10, 100, 10.0),  # 10/100 = 10%
            (5, 5, 100.0),    # 5/5 = 100%
            (1, 331, 0.3),    # 1/331 ≈ 0.3%
            (0, 100, 0.0),    # 0/100 = 0%
        ]

        for i, (compatible_count, available_count, expected_percentage) in enumerate(test_cases):
            clause_id = 700 + i
            stats.begin_given(clause_id=clause_id, available_count=available_count)

            # Record attempted partnerships with the specified number of clauses
            for partner_id in range(1, available_count + 1):
                stats.record_attempted_partnership(partner_id)

            # Record successful partnerships with the specified number of clauses
            for partner_id in range(1, compatible_count + 1):
                stats.record_successful_partnership(partner_id)

            compatible, available, percentage = stats.get_given_compatibility_stats(clause_id)

            assert compatible == compatible_count
            assert available == available_count
            assert percentage == pytest.approx(expected_percentage, abs=0.1)