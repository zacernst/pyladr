# Online Learning Example Problems

These problems demonstrate where PyLADR's `--online-learning` flag provides
advantages over traditional static clause selection.

## Quick start

```bash
# Run any problem with online learning enabled:
python3 -m pyladr.apps.prover9 --online-learning -f examples/sample_problems/online_learning/lattice_distributivity.in

# Compare with traditional mode:
python3 -m pyladr.apps.prover9 -f examples/sample_problems/online_learning/lattice_distributivity.in

# Or use the comparison script:
python3 examples/compare_online_learning.py
```

## Problem categories

### Equational reasoning (best for online learning)
- `lattice_distributivity.in` — Prove distributivity in a lattice (medium)
- `group_inverse_product.in` — Inverse of a product in group theory (easy)
- `ring_nilpotent.in` — Nilpotent element property in rings (hard)
- `boolean_sheffer.in` — Boolean algebra from the Sheffer stroke (hard)

### First-order logic
- `set_theory_subset.in` — Subset transitivity (easy)
- `order_theory_lattice.in` — Lattice properties from partial order axioms (medium)

### Why online learning helps

Online learning is most beneficial when:
1. **The search space is large** — more clauses means more opportunity to learn
   which clause patterns lead to productive inferences.
2. **Similar subproblems recur** — equational reasoning often generates structurally
   similar clauses, so the model can generalize from early successes.
3. **Traditional heuristics plateau** — when weight-based selection cycles through
   unproductive clauses, learned embeddings can break the deadlock.

For trivial problems (proved in < 10 given clauses), the overhead of online
learning exceeds its benefit. The sweet spot is problems requiring 50-500
given clauses.
