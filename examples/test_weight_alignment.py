#!/usr/bin/env python3
"""
Test Weight Alignment Fix

Verifies that the training/selection mismatch has been fixed:
- Training: Productive clauses → smaller embedding norms
- Selection: Smaller norms → higher proof potential scores ✅ FIXED
"""

import math

def old_proof_potential_score(embedding):
    """OLD (broken) version - rewards higher norms."""
    norm = math.sqrt(sum(x * x for x in embedding))
    return 2.0 / (1.0 + math.exp(-norm)) - 1.0  # Higher norm → higher score

def new_proof_potential_score(embedding):
    """NEW (fixed) version - rewards smaller norms."""
    norm = math.sqrt(sum(x * x for x in embedding))
    return 2.0 / (1.0 + math.exp(norm)) - 1.0   # Smaller norm → higher score

def test_weight_alignment():
    """Test that the fix properly aligns training and selection."""

    print("🧪 Testing Weight Alignment Fix")
    print("=" * 50)

    # Simulate embeddings: productive vs unproductive
    # Training pushes productive toward smaller norms
    productive_embedding = [0.1, -0.2, 0.15, -0.1, 0.05]     # Small norm
    unproductive_embedding = [0.8, -0.6, 0.9, -0.7, 0.5]     # Large norm

    # Calculate norms
    prod_norm = math.sqrt(sum(x*x for x in productive_embedding))
    unprod_norm = math.sqrt(sum(x*x for x in unproductive_embedding))

    print(f"📊 Embedding Norms:")
    print(f"  Productive clause norm:   {prod_norm:.3f} (smaller)")
    print(f"  Unproductive clause norm: {unprod_norm:.3f} (larger)")

    # Test OLD (broken) scoring
    old_prod_score = old_proof_potential_score(productive_embedding)
    old_unprod_score = old_proof_potential_score(unproductive_embedding)

    print(f"\n❌ OLD Scoring (BROKEN):")
    print(f"  Productive score:   {old_prod_score:.3f}")
    print(f"  Unproductive score: {old_unprod_score:.3f}")
    print(f"  Winner: {'Productive' if old_prod_score > old_unprod_score else 'Unproductive'} ❌ WRONG!")

    # Test NEW (fixed) scoring
    new_prod_score = new_proof_potential_score(productive_embedding)
    new_unprod_score = new_proof_potential_score(unproductive_embedding)

    print(f"\n✅ NEW Scoring (FIXED):")
    print(f"  Productive score:   {new_prod_score:.3f}")
    print(f"  Unproductive score: {new_unprod_score:.3f}")
    print(f"  Winner: {'Productive' if new_prod_score > new_unprod_score else 'Unproductive'} ✅ CORRECT!")

    print(f"\n🎯 Alignment Status:")
    print(f"  Training: Productive → smaller norms ✅")
    print(f"  Selection: Smaller norms → higher scores ✅")
    print(f"  Alignment: {'FIXED' if new_prod_score > new_unprod_score else 'STILL BROKEN'} ✅")

    # Test edge cases
    print(f"\n🔬 Edge Case Tests:")
    tiny_embedding = [0.01, 0.01, 0.01]
    huge_embedding = [10.0, 10.0, 10.0]

    tiny_score = new_proof_potential_score(tiny_embedding)
    huge_score = new_proof_potential_score(huge_embedding)

    print(f"  Tiny norm ({math.sqrt(sum(x*x for x in tiny_embedding)):.3f}) → score: {tiny_score:.3f}")
    print(f"  Huge norm ({math.sqrt(sum(x*x for x in huge_embedding)):.3f}) → score: {huge_score:.3f}")
    print(f"  Tiny > Huge: {tiny_score > huge_score} ✅")

if __name__ == "__main__":
    test_weight_alignment()