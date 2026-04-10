#!/usr/bin/env python3
"""
Quick demonstration of vampire.in model training concept.

This script shows how the training process works without running full training.
Perfect for understanding the ML training pipeline quickly.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core PyLADR imports
from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.apps.prover9 import _deny_goals, _apply_settings

# ML training imports
try:
    import torch
    from pyladr.ml.embedding_provider import EmbeddingProvider
    from pyladr.ml.graph.clause_encoder import GNNConfig, HeterogeneousClauseGNN
    from pyladr.ml.training.contrastive import InferencePair, PairLabel
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ ML dependencies not available: {e}")
    ML_AVAILABLE = False


def demonstrate_training_concept():
    """Demonstrate the training concept without full training."""

    if not ML_AVAILABLE:
        print("❌ ML dependencies required for this demo")
        return 1

    if not Path("vampire.in").exists():
        print("❌ vampire.in not found")
        return 1

    logger.info("🧛 Vampire.in Training Concept Demo")
    logger.info("=" * 50)

    # Step 1: Parse the problem
    logger.info("📋 Step 1: Parsing vampire.in")
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open("vampire.in") as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos = _deny_goals(parsed, symbol_table)
    initial_clauses = usable + sos

    logger.info(f"  Parsed: {len(usable)} usable, {len(sos)} SOS clauses")
    logger.info("  Initial clauses:")
    for i, clause in enumerate(initial_clauses):
        logger.info(f"    {i+1}: {clause.to_str(symbol_table)} (weight: {clause.weight:.1f})")

    # Step 2: Create training pairs
    logger.info("\n🔄 Step 2: Creating training pairs")
    training_pairs = []

    # Label clauses based on characteristics (this is what the training script does)
    productive_threshold = 15.0

    for clause in initial_clauses:
        is_productive = (
            clause.weight <= productive_threshold or
            len(clause.literals) <= 2 or
            any(lit.atom.symbol.name == "P" for lit in clause.literals)
        )

        label = PairLabel.PRODUCTIVE if is_productive else PairLabel.UNPRODUCTIVE

        pair = InferencePair(
            parent1=clause,
            parent2=None,
            child=clause,
            label=label,
            proof_depth=0 if is_productive else -1
        )
        training_pairs.append(pair)

    productive_count = sum(1 for p in training_pairs if p.label == PairLabel.PRODUCTIVE)
    unproductive_count = len(training_pairs) - productive_count

    logger.info(f"  Created {len(training_pairs)} training pairs:")
    logger.info(f"    Productive: {productive_count}")
    logger.info(f"    Unproductive: {unproductive_count}")

    # Step 3: Show embeddings
    logger.info("\n🧠 Step 3: Generating clause embeddings")

    try:
        # Create embedding provider (this uses random initialization)
        gnn_config = GNNConfig(hidden_dim=64, embedding_dim=128, num_layers=2)
        provider = EmbeddingProvider(symbol_table=symbol_table, gnn_config=gnn_config)

        logger.info(f"  Created embedding provider (dim={provider.embedding_dim})")

        # Generate embeddings for a few clauses
        for i, pair in enumerate(training_pairs[:3]):
            clause = pair.parent1
            embedding = provider.get_embedding(clause)

            if embedding:
                logger.info(f"  Clause {i+1} ({pair.label.name}): {clause.to_str(symbol_table)}")
                logger.info(f"    Embedding: [{embedding[0]:.3f}, {embedding[1]:.3f}, ..., {embedding[-1]:.3f}]")
            else:
                logger.info(f"  Clause {i+1}: Could not generate embedding")

    except Exception as e:
        logger.error(f"  Error generating embeddings: {e}")

    # Step 4: Show training concept
    logger.info("\n📚 Step 4: Training concept")
    logger.info("  In full training, the system would:")
    logger.info("    1. Convert each clause to a heterogeneous graph")
    logger.info("    2. Run GNN forward pass to get embeddings")
    logger.info("    3. Apply contrastive loss:")
    logger.info("       - Productive clauses → similar embeddings")
    logger.info("       - Unproductive clauses → different embeddings")
    logger.info("    4. Backprop + optimize GNN weights")
    logger.info("    5. Repeat for many epochs")

    # Step 5: Expected improvements
    logger.info("\n🎯 Step 5: Expected improvements after training")
    logger.info("  A trained model would:")
    logger.info("    - Recognize productive clause patterns from vampire.in")
    logger.info("    - Prioritize clauses similar to those that led to progress")
    logger.info("    - Focus on structural features that matter for this problem")
    logger.info("    - Potentially find proofs faster or with fewer given clauses")

    # Step 6: Usage
    logger.info("\n🚀 Step 6: How to run full training")
    logger.info("  To actually train a model:")
    logger.info("    python examples/train_vampire_model.py --epochs 50 --attempts 20")
    logger.info("  ")
    logger.info("  To use a trained model:")
    logger.info("    python examples/simple_ml_usage.py vampire.in --model your_model.pt")

    logger.info("\n✅ Training concept demonstration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(demonstrate_training_concept())