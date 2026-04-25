#!/usr/bin/env python3
"""
Analyze the corrected model's embeddings to confirm goal-aware training worked
"""

import math
from pathlib import Path

# Core PyLADR imports
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.apps.prover9 import _deny_goals

# ML imports
try:
    from pyladr.ml.embedding_provider import EmbeddingProvider, EmbeddingProviderConfig
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def analyze_corrected_embeddings():
    """Analyze embeddings from the corrected model."""

    if not ML_AVAILABLE:
        print("❌ ML dependencies required")
        return 1

    print("🎯 Corrected Model Embedding Analysis")
    print("=" * 60)
    print("Target: Variable pattern should have smallest embedding norm")
    print("Pattern: -P(i(x,i(i(y,i(x,z)),i(i(n(z),i(i(n(u),v),y)),i(u,z)))))")

    # Parse the problem
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open("examples/vampire.in") as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, symbol_table)
    all_clauses = usable + sos

    print(f"\n📋 Analyzing {len(all_clauses)} clauses:")

    # Load corrected model
    config = EmbeddingProviderConfig(model_path="vampire_corrected_model.pt")
    provider = EmbeddingProvider(symbol_table=symbol_table, config=config)

    embeddings = []
    for i, clause in enumerate(all_clauses):
        clause_str = clause.to_str(symbol_table)
        embedding = provider.get_embedding(clause)

        if embedding:
            norm = math.sqrt(sum(x*x for x in embedding))
            embeddings.append((clause_str, norm, i+1))

            # Check if this is the target variable pattern
            is_target = "i(x,i(i(y,i(x,z)),i(i(n(z),i(i(n(u),v),y)),i(u,z)))" in clause_str

            print(f"\n🔍 Clause {i+1}: {clause_str}")
            print(f"  Embedding norm: {norm:.4f}")
            if is_target:
                print(f"  🎯 TARGET PATTERN FOUND! This should have the smallest norm")
            print(f"  Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")

    # Summary statistics
    embeddings.sort(key=lambda x: x[1])  # Sort by norm
    print(f"\n📊 Embedding Norms (sorted from smallest to largest):")
    print("─" * 50)
    for clause_str, norm, clause_num in embeddings:
        is_target = "i(x,i(i(y,i(x,z)),i(i(n(z),i(i(n(u),v),y)),i(u,z)))" in clause_str
        marker = "🎯 TARGET" if is_target else ""
        print(f"  Clause {clause_num}: {norm:.4f} {marker}")
        if len(clause_str) > 60:
            print(f"    {clause_str[:60]}...")
        else:
            print(f"    {clause_str}")

    smallest_norm_clause = embeddings[0]
    print(f"\n💡 Analysis:")
    print(f"  Smallest norm: {smallest_norm_clause[1]:.4f} (Clause {smallest_norm_clause[2]})")
    print(f"  Target pattern found: {'YES' if any('i(x,i(i(y,i(x,z))' in e[0] for e in embeddings) else 'NO'}")

    # Check if target has smallest norm
    target_embeddings = [e for e in embeddings if "i(x,i(i(y,i(x,z))" in e[0]]
    if target_embeddings:
        target_norm = target_embeddings[0][1]
        is_smallest = target_norm == smallest_norm_clause[1]
        print(f"  Target is smallest: {'✅ YES' if is_smallest else '❌ NO'}")
        if is_smallest:
            print(f"  🎉 PERFECT! Goal-aware training + weight alignment working!")
        else:
            print(f"  🤔 Target norm: {target_norm:.4f} vs smallest: {smallest_norm_clause[1]:.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(analyze_corrected_embeddings())