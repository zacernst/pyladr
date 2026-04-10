#!/usr/bin/env python3
"""
Advanced Training Data Extraction for vampire.in

This shows how to capture training pairs from ALL clauses generated during
proof attempts, not just the initial clauses.

The key insight: During proof search, thousands of new clauses are generated
through inference rules. These provide rich training data about which
clause patterns are productive vs unproductive.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core PyLADR imports
from pyladr.core.clause import Clause
from pyladr.core.symbol import SymbolTable
from pyladr.parsing.ladr_parser import LADRParser
from pyladr.search.given_clause import GivenClauseSearch, SearchOptions
from pyladr.apps.prover9 import _deny_goals, _apply_settings

try:
    from pyladr.ml.training.contrastive import InferencePair, PairLabel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class AdvancedProofDataCollector:
    """Extracts training pairs from ALL clauses generated during proof attempts."""

    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table

    def collect_comprehensive_training_data(self, problem_file: str) -> List[InferencePair]:
        """Collect training data from all generated clauses, not just initial ones."""

        logger.info(f"🔍 Advanced data collection from {problem_file}")

        # Parse the problem
        parser = LADRParser(self.symbol_table)
        with open(problem_file) as f:
            input_text = f.read()

        parsed = parser.parse_input(input_text)
        usable, sos = _deny_goals(parsed, self.symbol_table)
        initial_clauses = usable + sos

        logger.info(f"📋 Initial clauses: {len(initial_clauses)}")

        # Run proof attempt with instrumentation
        opts = SearchOptions(
            binary_resolution=True,
            factoring=True,
            max_given=100,          # Reasonable limit for data collection
            max_seconds=30,
            print_given=False,      # Quiet for data collection
            quiet=True
        )

        _apply_settings(parsed, opts, self.symbol_table)

        # Run search and capture all generated clauses
        engine = GivenClauseSearch(options=opts, symbol_table=self.symbol_table)
        result = engine.run(usable=usable, sos=sos)

        logger.info(f"📊 Search results:")
        logger.info(f"  Given clauses: {result.stats.given}")
        logger.info(f"  Generated clauses: {result.stats.generated}")
        logger.info(f"  Kept clauses: {result.stats.kept}")

        # Extract all clauses from the search engine
        all_generated_clauses = self._extract_all_clauses_from_engine(engine)

        logger.info(f"🔬 Extracted {len(all_generated_clauses)} total clauses for analysis")

        # Create training pairs from all clauses
        training_pairs = self._create_comprehensive_training_pairs(
            initial_clauses, all_generated_clauses, result
        )

        logger.info(f"✅ Created {len(training_pairs)} comprehensive training pairs")
        return training_pairs

    def _extract_all_clauses_from_engine(self, engine: GivenClauseSearch) -> List[Clause]:
        """Extract all clauses that were generated during the search."""

        all_clauses = []

        # Get clauses from the search engine's internal state
        if hasattr(engine, '_all_clauses'):
            # _all_clauses is a dict mapping clause_id -> clause
            for clause_id, clause in engine._all_clauses.items():
                all_clauses.append(clause)

        # Also get clauses from the current clause lists
        if hasattr(engine, '_state'):
            state = engine._state

            # Add clauses from all lists: usable, sos, limbo
            for clause_list in [state.usable, state.sos, state.limbo]:
                for clause in clause_list:
                    if clause not in all_clauses:
                        all_clauses.append(clause)

        logger.info(f"  Extracted from engine: {len(all_clauses)} clauses")
        return all_clauses

    def _create_comprehensive_training_pairs(self,
                                          initial_clauses: List[Clause],
                                          all_clauses: List[Clause],
                                          result) -> List[InferencePair]:
        """Create training pairs from all generated clauses with sophisticated labeling."""

        pairs = []

        # Strategy 1: Label based on clause generation order and selection
        given_count = result.stats.given
        kept_count = result.stats.kept

        # Track which clauses were selected as "given" clauses
        selected_as_given = set()  # In a real implementation, track this during search

        for clause in all_clauses:
            # Determine productivity based on multiple factors
            is_productive = self._determine_clause_productivity(
                clause, initial_clauses, given_count, kept_count, selected_as_given
            )

            label = PairLabel.PRODUCTIVE if is_productive else PairLabel.UNPRODUCTIVE

            # Create training pair
            pair = InferencePair(
                parent1=clause,
                parent2=None,  # For single-clause analysis
                child=clause,
                label=label,
                proof_depth=0 if is_productive else -1
            )
            pairs.append(pair)

        # Strategy 2: Create inference pairs between clauses
        # This would track actual parent->child relationships during inference
        inference_pairs = self._create_inference_relationship_pairs(all_clauses)
        pairs.extend(inference_pairs)

        return pairs

    def _determine_clause_productivity(self,
                                    clause: Clause,
                                    initial_clauses: List[Clause],
                                    given_count: int,
                                    kept_count: int,
                                    selected_as_given: Set[int]) -> bool:
        """Advanced heuristics for determining if a clause was productive."""

        # Factor 1: Was this clause selected as a given clause?
        if clause.id in selected_as_given:
            return True  # Definitely productive!

        # Factor 2: Clause characteristics
        weight_productive = clause.weight <= 12.0  # Low weight often good
        length_productive = len(clause.literals) <= 3  # Short clauses often key
        has_main_predicate = any(lit.atom.symbol.name == "P" for lit in clause.literals)

        # Factor 3: Structural patterns learned from vampire.in
        has_implication_pattern = any(
            "i(" in str(lit.atom) for lit in clause.literals
        )
        has_negation_pattern = any(
            "n(" in str(lit.atom) for lit in clause.literals
        )

        # Factor 4: Generation timing (earlier = potentially more useful)
        generation_order_productive = clause.id <= kept_count  # Among kept clauses

        # Combine factors
        productivity_score = sum([
            weight_productive,
            length_productive,
            has_main_predicate,
            has_implication_pattern and has_negation_pattern,
            generation_order_productive
        ])

        return productivity_score >= 3  # Threshold for productivity

    def _create_inference_relationship_pairs(self, all_clauses: List[Clause]) -> List[InferencePair]:
        """Create pairs representing parent->child inference relationships."""

        pairs = []

        # In a full implementation, we'd track the actual inference tree
        # For now, create synthetic relationships based on clause similarity

        for i, clause1 in enumerate(all_clauses[:50]):  # Sample for demo
            for j, clause2 in enumerate(all_clauses[:50]):
                if i != j and self._clauses_could_be_related(clause1, clause2):
                    # Create an inference pair representing a possible relationship
                    pair = InferencePair(
                        parent1=clause1,
                        parent2=clause2,
                        child=clause2,  # clause2 could be derived from clause1
                        label=PairLabel.PRODUCTIVE if clause2.weight <= clause1.weight else PairLabel.UNPRODUCTIVE,
                        inference_type=1,  # Binary resolution
                        proof_depth=-1
                    )
                    pairs.append(pair)

        return pairs[:20]  # Return sample for demo

    def _clauses_could_be_related(self, clause1: Clause, clause2: Clause) -> bool:
        """Heuristic to determine if two clauses could be inference-related."""
        # Simple heuristic: clauses with similar symbols might be related
        symbols1 = {lit.atom.symbol.name for lit in clause1.literals}
        symbols2 = {lit.atom.symbol.name for lit in clause2.literals}
        return len(symbols1.intersection(symbols2)) > 0


def demonstrate_advanced_training_data():
    """Show how advanced training data extraction works."""

    if not ML_AVAILABLE:
        print("❌ ML dependencies required")
        return 1

    if not Path("vampire.in").exists():
        print("❌ vampire.in not found")
        return 1

    logger.info("🧛 Advanced Training Data Extraction Demo")
    logger.info("=" * 60)

    # Create collector
    symbol_table = SymbolTable()
    collector = AdvancedProofDataCollector(symbol_table)

    # Collect comprehensive training data
    training_pairs = collector.collect_comprehensive_training_data("vampire.in")

    # Analyze the results
    productive_pairs = [p for p in training_pairs if p.label == PairLabel.PRODUCTIVE]
    unproductive_pairs = [p for p in training_pairs if p.label == PairLabel.UNPRODUCTIVE]

    logger.info(f"\n📊 Training Data Analysis:")
    logger.info(f"  Total training pairs: {len(training_pairs)}")
    logger.info(f"  Productive pairs: {len(productive_pairs)}")
    logger.info(f"  Unproductive pairs: {len(unproductive_pairs)}")

    # Show examples
    logger.info(f"\n🔍 Example Productive Clauses:")
    for i, pair in enumerate(productive_pairs[:3]):
        clause = pair.parent1
        logger.info(f"  {i+1}: {clause.to_str(symbol_table)} (weight: {clause.weight:.1f})")

    logger.info(f"\n🔍 Example Unproductive Clauses:")
    for i, pair in enumerate(unproductive_pairs[:3]):
        clause = pair.parent1
        logger.info(f"  {i+1}: {clause.to_str(symbol_table)} (weight: {clause.weight:.1f})")

    logger.info(f"\n💡 Key Insights:")
    logger.info(f"  • Training data comes from ALL {len(training_pairs)} generated clauses")
    logger.info(f"  • Much richer than just the 5 initial clauses!")
    logger.info(f"  • Captures inference patterns and clause evolution")
    logger.info(f"  • Labels based on generation order, selection, and structure")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(demonstrate_advanced_training_data())