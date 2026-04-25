#!/usr/bin/env python3
"""
Simple ML Usage Example

This script shows the most straightforward way to use PyLADR's ML features
for clause embeddings and enhanced search.

Usage:
    python simple_ml_usage.py vampire.in
"""

import sys
import logging
from pathlib import Path

# Configure logging to show INFO level messages (including given clauses)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import PyLADR components
try:
    from pyladr.core.symbol import SymbolTable
    from pyladr.parsing.ladr_parser import LADRParser
    from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
    from pyladr.apps.prover9 import _deny_goals, _apply_settings
    PYLADR_AVAILABLE = True
except ImportError as e:
    print(f"Error: PyLADR not available: {e}")
    print("Make sure you're running from the PyLADR directory")
    PYLADR_AVAILABLE = False

# ML imports with graceful handling
try:
    from pyladr.ml.embedding_provider import EmbeddingProvider
    from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig
    ML_AVAILABLE = True
except ImportError:
    print("ML dependencies not available. Install with: pip install -e '.[ml]'")
    ML_AVAILABLE = False


def demonstrate_embeddings(clauses, symbol_table, model_path=None):
    """Demonstrate how to generate clause embeddings."""
    if not ML_AVAILABLE:
        print("❌ Cannot demonstrate embeddings - ML dependencies missing")
        return

    print("🧠 Generating clause embeddings...")

    try:
        if model_path and Path(model_path).exists():
            # Load a pre-trained model
            print(f"📁 Loading trained model from {model_path}")
            from pyladr.ml.embedding_provider import EmbeddingProviderConfig
            config = EmbeddingProviderConfig(model_path=model_path)
            provider = EmbeddingProvider(symbol_table=symbol_table, config=config)
        else:
            # Create a simple embedding provider with random initialization
            from pyladr.ml.graph.clause_encoder import GNNConfig

            # Create a small model for demonstration
            gnn_config = GNNConfig(
                hidden_dim=64,
                embedding_dim=128,
                num_layers=2,
                dropout=0.0,  # No dropout for deterministic demo
            )

            provider = EmbeddingProvider(symbol_table=symbol_table, gnn_config=gnn_config)

        print(f"📊 Embedding provider created (dim={provider.embedding_dim})")

        # Generate embeddings for first few clauses
        for i, clause in enumerate(clauses[:3]):
            print(f"\n🔍 Clause {i+1}: {clause.to_str(symbol_table)}")

            embedding = provider.get_embedding(clause)
            if embedding:
                print(f"  ✅ Embedding generated: {len(embedding)}-dim vector")
                print(f"  📈 Sample values: {embedding[:5]} ... {embedding[-5:]}")
            else:
                print(f"  ❌ Could not generate embedding")

        # Show cache statistics
        cache_stats = provider.get_cache_stats() if hasattr(provider, 'get_cache_stats') else {}
        print(f"\n📊 Cache stats: {cache_stats}")

    except Exception as e:
        print(f"❌ Error demonstrating embeddings: {e}")


def run_with_ml_selection(problem_file: str, model_path: str = None):
    """Run theorem proving with ML-enhanced clause selection."""

    if not ML_AVAILABLE:
        print("❌ Cannot run ML selection - dependencies missing")
        return run_traditional(problem_file)

    print("🧠 Running with ML-enhanced clause selection...")

    # Parse the problem
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open(problem_file) as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    print(f"📋 Parsed problem: {len(usable)} usable, {len(sos)} SOS clauses")

    # Demonstrate embeddings on initial clauses
    all_clauses = usable + sos
    demonstrate_embeddings(all_clauses, symbol_table, model_path)

    # Configure search with ML enhancements
    opts = SearchOptions(
        binary_resolution=True,
        factoring=True,
        max_given=1000,  # Limit for demo
        max_seconds=300,
        print_given=True,  # Show progress
        quiet=False
    )

    _apply_settings(parsed, opts, symbol_table)

    try:
        # Create embedding provider
        if model_path and Path(model_path).exists():
            # Use trained model
            from pyladr.ml.embedding_provider import EmbeddingProviderConfig
            config = EmbeddingProviderConfig(model_path=model_path)
            provider = EmbeddingProvider(symbol_table=symbol_table, config=config)
        else:
            # Use random initialization
            from pyladr.ml.graph.clause_encoder import GNNConfig
            gnn_config = GNNConfig(hidden_dim=64, embedding_dim=128, num_layers=2)
            provider = EmbeddingProvider(symbol_table=symbol_table, gnn_config=gnn_config)

        # Configure ML selection
        ml_config = MLSelectionConfig(
            enabled=True,
            ml_weight=0.4,  # 40% ML vs 60% traditional
            diversity_weight=0.7
        )

        # Create search engine
        engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)

        # Replace selector with ML-enhanced version
        if hasattr(engine, '_given_selector'):
            original_selector = engine._given_selector
            engine._given_selector = EmbeddingEnhancedSelection(
                embedding_provider=provider,
                config=ml_config,
                base_selector=original_selector
            )
            print("✅ ML-enhanced selection enabled")

        # Run the search
        print("\n🔍 Starting ML-guided search...")
        result = engine.run(usable=usable, sos=sos)

        # Report results
        print(f"\n📊 ML-Guided Search Results:")
        print(f"  Exit Code: {result.exit_code}")
        print(f"  Proof Found: {'Yes' if len(result.proofs) > 0 else 'No'}")
        print(f"  Given Clauses: {result.stats.given}")
        print(f"  Generated: {result.stats.generated}")
        print(f"  Kept: {result.stats.kept}")

        # ML-specific statistics
        if hasattr(engine, '_given_selector') and hasattr(engine._given_selector, 'get_ml_stats'):
            ml_stats = engine._given_selector.get_ml_stats()
            print(f"  ML Selections: {ml_stats.get('ml_selections', 0)}")
            print(f"  Traditional Selections: {ml_stats.get('traditional_selections', 0)}")
            print(f"  Average ML Score: {ml_stats.get('avg_ml_score', 0):.3f}")

    except Exception as e:
        print(f"❌ ML search failed: {e}")
        print("Falling back to traditional search...")
        return run_traditional(problem_file)


def run_traditional(problem_file: str):
    """Run traditional theorem proving for comparison."""

    print("📚 Running traditional clause selection...")

    # Parse the problem
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open(problem_file) as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    # Configure traditional search
    opts = SearchOptions(
        binary_resolution=True,
        factoring=True,
        max_given=1000,  # Same limit as ML version
        max_seconds=300,
        print_given=True,
        quiet=False
    )

    _apply_settings(parsed, opts, symbol_table)

    # Run search
    engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)
    print("\n🔍 Starting traditional search...")
    result = engine.run(usable=usable, sos=sos)

    # Report results
    print(f"\n📊 Traditional Results:")
    print(f"  Exit Code: {result.exit_code}")
    print(f"  Proof Found: {'Yes' if len(result.proofs) > 0 else 'No'}")
    print(f"  Given Clauses: {result.stats.given}")
    print(f"  Generated: {result.stats.generated}")
    print(f"  Kept: {result.stats.kept}")


def main():
    if not PYLADR_AVAILABLE:
        return 1

    if len(sys.argv) < 2:
        print("Usage: python simple_ml_usage.py <problem_file> [--model <model_path>]")
        print("\nExample:")
        print("  python simple_ml_usage.py vampire.in")
        print("  python simple_ml_usage.py vampire.in --model vampire_model.pt")
        print("\nThis script demonstrates basic ML-enhanced theorem proving.")
        return 1

    problem_file = sys.argv[1]
    model_path = None

    # Simple argument parsing for --model
    if len(sys.argv) >= 4 and sys.argv[2] == '--model':
        model_path = sys.argv[3]
        if model_path and not Path(model_path).exists():
            print(f"Error: Model file '{model_path}' not found")
            return 1
    if not Path(problem_file).exists():
        print(f"Error: Problem file '{problem_file}' not found")
        return 1

    print("🚀 PyLADR Simple ML Usage Demo")
    print("=" * 40)
    print(f"Problem file: {problem_file}")

    if ML_AVAILABLE:
        print("✅ ML features available")
        if model_path:
            print(f"📁 Using trained model: {model_path}")
        run_with_ml_selection(problem_file, model_path)
    else:
        print("❌ ML features unavailable - running traditional search only")
        run_traditional(problem_file)

    print("\n✨ Demo complete!")
    print("\nKey takeaways:")
    print("  • Clause embeddings are generated automatically")
    print("  • ML selection blends with traditional weight/age selection")
    print("  • Everything falls back gracefully when ML is unavailable")
    print("  • The search behavior is identical when ML ratio = 0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
