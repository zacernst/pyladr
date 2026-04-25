"""Cross-clause attention mechanisms for relational scoring.

This package implements multi-head attention over clause embeddings to enable
relational scoring during given-clause selection. Instead of scoring each clause
independently, cross-clause attention models inter-clause relationships —
complementarity, subsumption potential, and proof path diversity.

All components are strictly opt-in and fall back gracefully when disabled.
"""
