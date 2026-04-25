#!/usr/bin/env python3
"""
ML-Guided Clause Selection Demonstration

This script demonstrates the use of PyLADR's ML-enhanced clause selection
on the challenging vampire.in problem. It compares traditional search with
ML-guided search and shows detailed performance metrics.

Usage:
    python ml_guided_vampire_demo.py [--ml-ratio 0.3] [--max-given 500]

Requirements:
    pip install -e '.[ml]'  # Install with ML dependencies
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging to show INFO level messages (including given clauses)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Core PyLADR imports
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.apps.prover9 import _deny_goals, _apply_settings

# ML-specific imports with graceful fallback
try:
    import torch
    from pyladr.ml.embedding_provider import EmbeddingProvider, NoOpEmbeddingProvider
    from pyladr.ml.graph.clause_encoder import GNNConfig, HeterogeneousClauseGNN
    from pyladr.search.ml_selection import EmbeddingEnhancedSelection, MLSelectionConfig
    from pyladr.search.inference_guidance import EmbeddingGuidedInference, InferenceGuidanceConfig
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("Running in traditional-only mode. Install with: pip install -e '.[ml]'")
    ML_AVAILABLE = False


def create_mock_embedding_provider(symbol_table: SymbolTable) -> Any:
    """Create a mock embedding provider for demonstration purposes.

    In practice, you would load a pre-trained model or train one on your domain.
    """
    if not ML_AVAILABLE:
        return None

    try:
        # For demo: create a simple embedding provider with random initialization
        # In production, you'd load: EmbeddingProvider.load_model("path/to/trained_model.pt")

        gnn_config = GNNConfig(
            hidden_dim=128,      # Smaller for demo
            embedding_dim=256,   # Reasonable size
            num_layers=2,        # Fewer layers for speed
            dropout=0.1,
        )

        # Create embedding provider using the factory function
        provider = EmbeddingProvider(
            symbol_table=symbol_table,
            gnn_config=gnn_config
        )

        return provider

    except Exception as e:
        print(f"Could not create embedding provider: {e}")
        print("Falling back to NoOp provider")
        return NoOpEmbeddingProvider(embedding_dim=256)


def run_traditional_search(problem_file: str, max_given: int = 500) -> Dict[str, Any]:
    """Run traditional clause selection search."""

    print("🔍 Running Traditional Search...")

    # Parse input
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open(problem_file) as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    # Configure traditional search
    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=False,  # Keep it simpler for demo
        factoring=True,
        demodulation=False,
        max_given=max_given,
        max_seconds=60,  # 1 minute limit
        print_given=True,  # Show given clauses
        quiet=False
    )

    _apply_settings(parsed, opts, symbol_table)

    # Run search
    engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)

    start_time = time.time()
    result = engine.run(usable=usable, sos=sos)
    end_time = time.time()

    # Collect metrics
    return {
        'approach': 'traditional',
        'exit_code': result.exit_code,
        'proof_found': len(result.proofs) > 0,
        'given_clauses': result.stats.given,
        'kept_clauses': result.stats.kept,
        'generated_clauses': result.stats.generated,
        'search_time': end_time - start_time,
        'inferences_per_second': result.stats.generated / max(end_time - start_time, 0.001),
        'final_sos_size': len(engine._state.sos) if hasattr(engine, '_state') and hasattr(engine._state, 'sos') else 0,
        'final_usable_size': len(engine._state.usable) if hasattr(engine, '_state') and hasattr(engine._state, 'usable') else 0
    }


def run_ml_guided_search(problem_file: str, max_given: int = 500, ml_ratio: float = 0.3) -> Dict[str, Any]:
    """Run ML-guided clause selection search."""

    if not ML_AVAILABLE:
        print("❌ ML dependencies not available - skipping ML-guided search")
        return {'approach': 'ml_guided', 'error': 'ML not available'}

    print("🧠 Running ML-Guided Search...")

    # Parse input
    symbol_table = SymbolTable()
    parser = LADRParser(symbol_table)

    with open(problem_file) as f:
        input_text = f.read()

    parsed = parser.parse_input(input_text)
    usable, sos, _denied = _deny_goals(parsed, symbol_table)

    # Create embedding provider
    embedding_provider = create_mock_embedding_provider(symbol_table)
    if embedding_provider is None:
        return {'approach': 'ml_guided', 'error': 'Could not create embedding provider'}

    # Configure ML-enhanced search
    opts = SearchOptions(
        binary_resolution=True,
        paramodulation=False,  # Keep it simpler for demo
        factoring=True,
        demodulation=False,
        max_given=max_given,
        max_seconds=60,  # 1 minute limit
        print_given=True,  # Show given clauses
        quiet=False
    )

    _apply_settings(parsed, opts, symbol_table)

    # Create ML-enhanced selection strategy
    ml_config = MLSelectionConfig(
        enabled=True,
        ml_weight=ml_ratio,
        diversity_weight=0.6,
        fallback_on_error=True
    )

    # Create inference guidance
    guidance_config = InferenceGuidanceConfig(
        enabled=True,
        compatibility_threshold=0.3,
        max_candidates=100,
        early_termination_count=20
    )

    # Run search with ML enhancements
    engine = GivenClauseSearch(options=opts, symbol_table=symbol_table)

    # Replace default selection with ML-enhanced version
    if hasattr(engine, '_selector'):
        engine._selector = EmbeddingEnhancedSelection(
            embedding_provider=embedding_provider,
            config=ml_config,
            base_selector=engine._selector
        )

    # Add inference guidance
    if hasattr(engine, '_inference_guidance'):
        engine._inference_guidance = InferenceGuidance(
            embedding_provider=embedding_provider,
            config=guidance_config
        )

    start_time = time.time()
    result = engine.run(usable=usable, sos=sos)
    end_time = time.time()

    # Collect metrics including ML-specific stats
    ml_stats = {}
    if hasattr(engine, '_given_selector') and hasattr(engine._given_selector, 'get_ml_stats'):
        ml_stats = engine._given_selector.get_ml_stats()

    return {
        'approach': 'ml_guided',
        'exit_code': result.exit_code,
        'proof_found': len(result.proofs) > 0,
        'given_clauses': result.stats.given,
        'kept_clauses': result.stats.kept,
        'generated_clauses': result.stats.generated,
        'search_time': end_time - start_time,
        'inferences_per_second': result.stats.generated / max(end_time - start_time, 0.001),
        'final_sos_size': len(engine._state.sos) if hasattr(engine, '_state') and hasattr(engine._state, 'sos') else 0,
        'final_usable_size': len(engine._state.usable) if hasattr(engine, '_state') and hasattr(engine._state, 'usable') else 0,
        'ml_selection_ratio': ml_ratio,
        'ml_stats': ml_stats,
        'embedding_stats': {
            'cache_hit_rate': getattr(embedding_provider, 'cache_hit_rate', 0.0) if embedding_provider else 0.0,
            'embeddings_computed': getattr(embedding_provider, 'embeddings_computed', 0) if embedding_provider else 0,
            'avg_embedding_time': getattr(embedding_provider, 'avg_embedding_time', 0.0) if embedding_provider else 0.0,
        }
    }


def compare_results(traditional: Dict[str, Any], ml_guided: Dict[str, Any]) -> None:
    """Compare and display results from both approaches."""

    print("\n" + "="*80)
    print("🏆 COMPARISON RESULTS")
    print("="*80)

    if 'error' in ml_guided:
        print(f"❌ ML-Guided search failed: {ml_guided['error']}")
        print("\n📊 Traditional Results:")
        print_single_result(traditional)
        return

    # Basic comparison
    print(f"{'Metric':<25} {'Traditional':<15} {'ML-Guided':<15} {'Improvement':<15}")
    print("-" * 70)

    # Given clauses (fewer is better)
    trad_given = traditional['given_clauses']
    ml_given = ml_guided['given_clauses']
    given_improvement = ((trad_given - ml_given) / trad_given * 100) if trad_given > 0 else 0
    print(f"{'Given Clauses':<25} {trad_given:<15} {ml_given:<15} {given_improvement:>+.1f}%")

    # Search time
    trad_time = traditional['search_time']
    ml_time = ml_guided['search_time']
    time_improvement = ((trad_time - ml_time) / trad_time * 100) if trad_time > 0 else 0
    print(f"{'Search Time (s)':<25} {trad_time:<15.2f} {ml_time:<15.2f} {time_improvement:>+.1f}%")

    # Inferences per second (higher is better)
    trad_ips = traditional['inferences_per_second']
    ml_ips = ml_guided['inferences_per_second']
    ips_improvement = ((ml_ips - trad_ips) / trad_ips * 100) if trad_ips > 0 else 0
    print(f"{'Inferences/sec':<25} {trad_ips:<15.1f} {ml_ips:<15.1f} {ips_improvement:>+.1f}%")

    # Generated clauses
    trad_gen = traditional['generated_clauses']
    ml_gen = ml_guided['generated_clauses']
    gen_improvement = ((trad_gen - ml_gen) / trad_gen * 100) if trad_gen > 0 else 0
    print(f"{'Generated Clauses':<25} {trad_gen:<15} {ml_gen:<15} {gen_improvement:>+.1f}%")

    print("\n🧠 ML-Specific Metrics:")
    if 'ml_stats' in ml_guided and ml_guided['ml_stats']:
        ml_stats = ml_guided['ml_stats']
        print(f"  • ML Selection Ratio: {ml_guided['ml_selection_ratio']:.1%}")
        print(f"  • ML Selections Made: {ml_stats.get('ml_selections', 'N/A')}")
        print(f"  • Fallback Count: {ml_stats.get('fallback_count', 'N/A')}")
        print(f"  • Average ML Score: {ml_stats.get('avg_ml_score', 'N/A'):.3f}")

    if 'embedding_stats' in ml_guided:
        embed_stats = ml_guided['embedding_stats']
        print(f"  • Cache Hit Rate: {embed_stats.get('cache_hit_rate', 0):.1%}")
        print(f"  • Embeddings Computed: {embed_stats.get('embeddings_computed', 0)}")
        print(f"  • Avg Embedding Time: {embed_stats.get('avg_embedding_time', 0):.3f}ms")

    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if given_improvement > 0:
        print(f"  ✅ ML search found solution with {given_improvement:.1f}% fewer given clauses")
    elif given_improvement < -5:
        print(f"  ⚠️  ML search used {-given_improvement:.1f}% more given clauses")
    else:
        print(f"  ➡️  Similar performance ({given_improvement:+.1f}% difference)")

    if time_improvement > 0:
        print(f"  ⚡ ML search was {time_improvement:.1f}% faster")
    elif time_improvement < -10:
        print(f"  🐌 ML search was {-time_improvement:.1f}% slower (including ML overhead)")

    # Proof status
    if traditional['proof_found'] and ml_guided['proof_found']:
        print(f"  🏆 Both approaches found proofs!")
    elif ml_guided['proof_found'] and not traditional['proof_found']:
        print(f"  🌟 Only ML approach found a proof!")
    elif traditional['proof_found'] and not ml_guided['proof_found']:
        print(f"  📚 Only traditional approach found a proof")
    else:
        print(f"  🔍 Neither approach found a proof within the time limit")


def print_single_result(result: Dict[str, Any]) -> None:
    """Print results for a single approach."""
    approach = result['approach'].title()
    print(f"  Exit Code: {result['exit_code']}")
    print(f"  Proof Found: {'Yes' if result['proof_found'] else 'No'}")
    print(f"  Given Clauses: {result['given_clauses']}")
    print(f"  Generated Clauses: {result['generated_clauses']}")
    print(f"  Search Time: {result['search_time']:.2f}s")
    print(f"  Inferences/sec: {result['inferences_per_second']:.1f}")


def save_results(traditional: Dict[str, Any], ml_guided: Dict[str, Any]) -> None:
    """Save detailed results to JSON files for further analysis."""
    timestamp = int(time.time())

    # Save traditional results
    trad_file = f"vampire_traditional_{timestamp}.json"
    with open(trad_file, 'w') as f:
        json.dump(traditional, f, indent=2, default=str)

    # Save ML results
    if 'error' not in ml_guided:
        ml_file = f"vampire_ml_guided_{timestamp}.json"
        with open(ml_file, 'w') as f:
            json.dump(ml_guided, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {trad_file}, {ml_file}")
    else:
        print(f"\n💾 Traditional results saved to: {trad_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate ML-guided clause selection on vampire.in",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_guided_vampire_demo.py
  python ml_guided_vampire_demo.py --ml-ratio 0.5 --max-given 1000
  python ml_guided_vampire_demo.py --traditional-only

This script demonstrates PyLADR's ML-enhanced clause selection capabilities
by comparing traditional search with ML-guided search on the challenging
vampire.in problem from automated theorem proving.
        """)

    parser.add_argument('--ml-ratio', type=float, default=0.3,
                        help='Ratio of selections to use ML guidance (0.0-1.0, default: 0.3)')
    parser.add_argument('--max-given', type=int, default=500,
                        help='Maximum given clauses before termination (default: 500)')
    parser.add_argument('--traditional-only', action='store_true',
                        help='Run only traditional search (skip ML)')
    parser.add_argument('--problem-file', default='vampire.in',
                        help='Problem file to use (default: vampire.in)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detailed results to JSON files')

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.ml_ratio <= 1.0:
        print("Error: --ml-ratio must be between 0.0 and 1.0")
        return 1

    if not Path(args.problem_file).exists():
        print(f"Error: Problem file '{args.problem_file}' not found")
        print("Make sure vampire.in is in the current directory")
        return 1

    print("🧛 PyLADR ML-Guided Clause Selection Demo")
    print("=" * 50)
    print(f"Problem: {args.problem_file}")
    print(f"Max Given: {args.max_given}")
    if not args.traditional_only:
        print(f"ML Ratio: {args.ml_ratio:.1%}")
    print()

    # Run traditional search
    traditional_result = run_traditional_search(args.problem_file, args.max_given)

    # Run ML-guided search (unless disabled)
    if args.traditional_only:
        print("\n📊 Traditional Search Results:")
        print_single_result(traditional_result)
        if args.save_results:
            save_results(traditional_result, {'approach': 'skipped'})
    else:
        ml_result = run_ml_guided_search(args.problem_file, args.max_given, args.ml_ratio)

        # Compare results
        compare_results(traditional_result, ml_result)

        if args.save_results:
            save_results(traditional_result, ml_result)

    print(f"\n✨ Demo complete! The vampire.in problem is notoriously challenging -")
    print(f"   don't expect quick proofs, but ML guidance can help find them more efficiently.")

    return 0


if __name__ == "__main__":
    sys.exit(main())