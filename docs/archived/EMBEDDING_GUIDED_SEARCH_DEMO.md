# Graph-Based Clause Embedding: AI-Guided Search Demonstration

**Version**: 1.0
**Date**: 2026-04-07
**Objective**: Demonstrate how to use PyLADR's graph neural network embeddings to guide theorem proving searches for novel proof discovery.

---

## Overview

This demonstration shows how PyLADR's graph-based clause embeddings can enhance theorem proving by:
- **Learning semantic clause representations** through heterogeneous graph neural networks
- **Guiding given-clause selection** based on proof potential rather than just weight/age
- **Discovering novel proof strategies** through embedding-guided search
- **Maintaining full compatibility** with original C Prover9/LADR

---

## Prerequisites

### 1. Install ML Dependencies

```bash
# Install PyLADR with ML support
pip install -e ".[ml]"

# Verify installation
python -c "import torch, torch_geometric; print('ML dependencies ready')"
```

### 2. Download Pre-trained Model (Optional)

```bash
# For this demo, we'll use online learning, but you can download pre-trained models:
# wget https://pyladr-models.example.com/clause_embeddings_v1.0.pt -O models/clause_embeddings.pt
```

---

## Demonstration 1: Basic Embedding-Guided Search

### Problem: Group Theory - Commutativity from Associativity

**File: `examples/group_commutativity.in`**
```prover9
% Can we prove commutativity from associativity alone? (Spoiler: No, but interesting search!)

formulas(assumptions).
  % Group axioms
  e * x = x.           % Left identity
  x * e = x.           % Right identity
  i(x) * x = e.        % Left inverse
  x * i(x) = e.        % Right inverse
  (x * y) * z = x * (y * z).  % Associativity
end_of_list.

formulas(goals).
  x * y = y * x.       % Commutativity (unprovable from these axioms)
end_of_list.
```

### Standard Search (Baseline)

```bash
# Traditional weight/age-based search
pyprover9 -f examples/group_commutativity.in \
          --max-seconds 60 \
          --max-given 1000 \
          --verbose

# Expected: No proof found, explores 800+ clauses following weight/age heuristics
```

### ML-Enhanced Search

```bash
# Enable embedding-guided search
pyprover9 -f examples/group_commutativity.in \
          --max-seconds 60 \
          --max-given 1000 \
          --enable-embeddings \
          --embedding-selection-weight 0.7 \
          --online-learning \
          --verbose

# Expected: Explores different clause combinations, discovers interesting lemmas
```

**Key Configuration Options:**
- `--enable-embeddings`: Activate graph neural network clause embeddings
- `--embedding-selection-weight 0.7`: Blend 70% ML guidance + 30% traditional weight/age
- `--online-learning`: Continuously improve model during search
- `--embedding-cache-size 10000`: Cache embeddings for frequently seen clause patterns

---

## Demonstration 2: Novel Proof Discovery

### Problem: Boolean Algebra - Huntington's Axiom H4

**File: `examples/huntington_h4.in`**
```prover9
% Prove Huntington's 4th axiom from the other three
% This is a challenging problem where ML guidance can find novel proof paths

formulas(assumptions).
  % Huntington's Boolean algebra axioms H1, H2, H3
  x + y = y + x.                    % H1: Commutativity of +
  (x + y) + z = x + (y + z).        % H2: Associativity of +
  n(n(x) + y) + n(n(x) + n(y)) = x. % H3: Huntington's axiom

  % Standard Boolean definitions
  x * y = n(n(x) + n(y)).           % Define * via De Morgan
  -x = n(x).                        % Negation notation
end_of_list.

formulas(goals).
  x + (y * z) = (x + y) * (x + z).  % H4: Distributivity (to be proved)
end_of_list.
```

### Traditional vs ML-Guided Comparison

**Traditional Search:**
```bash
pyprover9 -f examples/huntington_h4.in --max-given 2000 --report-given

# Typical output (weight-based selection):
# Given #1: x + y = y + x [weight=7]
# Given #2: (x + y) + z = x + (y + z) [weight=11]
# Given #3: n(n(x) + y) + n(n(x) + n(y)) = x [weight=21]
# ... follows predictable weight progression
```

**ML-Guided Search:**
```bash
pyprover9 -f examples/huntington_h4.in --enable-embeddings \
          --embedding-selection-weight 0.8 --max-given 2000 --report-given

# Enhanced output (embedding-guided selection):
# Given #1: x + y = y + x [weight=7, embedding_score=0.92]
# Given #2: n(n(x) + n(y)) * n(n(x) + z) = n(n(x)) [weight=15, embedding_score=0.88]
# Given #3: (x + y) + z = x + (y + z) [weight=11, embedding_score=0.85]
# ... discovers structural patterns leading to novel lemma combinations
```

---

## Demonstration 3: Online Learning in Action

### Problem: Lattice Theory - Modular Law

**File: `examples/modular_lattice.in`**
```prover9
% Modular lattice: demonstrate online learning improving search over time

formulas(assumptions).
  % Lattice axioms
  x v x = x.                        % Idempotent (join)
  x ^ x = x.                        % Idempotent (meet)
  x v y = y v x.                    % Commutative (join)
  x ^ y = y ^ x.                    % Commutative (meet)
  (x v y) v z = x v (y v z).        % Associative (join)
  (x ^ y) ^ z = x ^ (y ^ z).        % Associative (meet)
  x v (x ^ y) = x.                  % Absorption
  x ^ (x v y) = x.                  % Absorption

  % Modular law premise
  x <= z -> x v (y ^ z) = (x v y) ^ z.  % Key modular property
  x <= y <-> x v y = y.             % Order definition
end_of_list.

formulas(goals).
  % Prove a consequence of modularity
  (a ^ b) v (a ^ c) <= a ^ (b v (a ^ c)).
end_of_list.
```

### Observing Online Learning

```bash
# Run with detailed ML logging
pyprover9 -f examples/modular_lattice.in \
          --enable-embeddings \
          --online-learning \
          --embedding-learning-rate 0.01 \
          --ml-verbose \
          --max-given 1500

# Watch the learning progress:
# [ML] Epoch 1: Average embedding quality: 0.34
# [ML] Discovered productive pattern: lattice_absorption + order_def -> high success
# [ML] Epoch 50: Average embedding quality: 0.61 (improving!)
# [ML] Epoch 100: Average embedding quality: 0.73
# [ML] Novel inference strategy learned: prioritize meet/join combinations
```

---

## Demonstration 4: Advanced Configuration

### Multi-Criteria Embedding Selection

**File: `examples/advanced_config.in`**
```prover9
% Complex number theory problem requiring sophisticated guidance

formulas(assumptions).
  % Complex number axioms (simplified)
  (a + b*i) + (c + d*i) = (a + c) + (b + d)*i.    % Addition
  (a + b*i) * (c + d*i) = (a*c - b*d) + (a*d + b*c)*i.  % Multiplication
  i * i = -1.                                       % i squared

  % Field properties
  x + 0 = x.
  x * 1 = x.
  exists y (x + y = 0).                            % Additive inverse
  x != 0 -> exists y (x * y = 1).                 % Multiplicative inverse
end_of_list.

formulas(goals).
  % Prove: |z|^2 = z * conjugate(z) = a^2 + b^2 for z = a + b*i
  (a + b*i) * (a - b*i) = a*a + b*b.
end_of_list.
```

### Advanced ML Configuration

```bash
pyprover9 -f examples/advanced_config.in \
          --enable-embeddings \
          --embedding-model-path models/complex_numbers_v2.pt \
          --embedding-selection-weight 0.9 \
          --embedding-diversity-bonus 0.2 \
          --embedding-cache-size 20000 \
          --online-learning \
          --contrastive-learning-weight 0.3 \
          --inference-guidance-threshold 0.6 \
          --ml-batch-size 32 \
          --gpu-acceleration \
          --max-given 3000
```

**Configuration Explanation:**
- `--embedding-diversity-bonus 0.2`: Encourages exploring diverse clause types
- `--contrastive-learning-weight 0.3`: Balance between proof patterns and structural similarity
- `--inference-guidance-threshold 0.6`: Only use ML guidance when confidence > 60%
- `--gpu-acceleration`: Leverage GPU for faster embedding computation

---

## Demonstration 5: Comparative Analysis

### Benchmark Problem: Robbins Algebra

**File: `examples/robbins_conjecture.in`**
```prover9
% Famous Robbins conjecture (solved by EQP in 1996)
% Demonstrate embedding guidance finding alternative proof paths

formulas(assumptions).
  x + y = y + x.                    % Commutativity
  (x + y) + z = x + (y + z).        % Associativity
  n(n(x + y) + n(x + n(y))) = x.    % Robbins axiom
end_of_list.

formulas(goals).
  exists x (x + x = x).             % Idempotent element exists
end_of_list.
```

### Side-by-Side Comparison

```bash
# Run both searches simultaneously for comparison
pyprover9 -f examples/robbins_conjecture.in --max-given 5000 --output traditional.out &
pyprover9 -f examples/robbins_conjecture.in --enable-embeddings \
          --embedding-selection-weight 0.75 --max-given 5000 --output ml_guided.out &

wait  # Wait for both to complete

# Compare results
echo "=== Traditional Search ==="
grep "Given #" traditional.out | head -20

echo "=== ML-Guided Search ==="
grep "Given #" ml_guided.out | head -20

echo "=== Proof Comparison ==="
diff <(grep "PROOF" traditional.out) <(grep "PROOF" ml_guided.out)
```

### Expected Differences

**Traditional Search Pattern:**
```
Given #1: x + y = y + x [age=1, weight=7]
Given #2: (x + y) + z = x + (y + z) [age=2, weight=11]
Given #3: n(n(x + y) + n(x + n(y))) = x [age=3, weight=21]
Given #4: n(n(x + x) + n(x + n(x))) = x [age=4, weight=19]
...predictable weight progression...
```

**ML-Guided Search Pattern:**
```
Given #1: x + y = y + x [age=1, weight=7, emb=0.94]
Given #2: n(n(x + x) + n(x + n(x))) = x [age=4, weight=19, emb=0.91]
Given #3: n(x + n(x)) + x = x + x [age=12, weight=15, emb=0.89]
Given #4: (x + y) + z = x + (y + z) [age=2, weight=11, emb=0.87]
...discovers key lemmas earlier through embedding similarity...
```

---

## Performance Monitoring

### Real-Time ML Statistics

```bash
# Enable comprehensive ML monitoring
pyprover9 -f examples/your_problem.in \
          --enable-embeddings \
          --ml-stats-interval 100 \
          --embedding-diagnostics

# Monitor output:
# [ML-Stats] Given #100: Cache hit rate: 87%, Avg embedding time: 2.3ms
# [ML-Stats] Given #200: Model confidence: 0.78, Novel patterns found: 12
# [ML-Stats] Given #300: Learning rate adapted to: 0.008, Convergence: 0.23
# [ML-Stats] Given #400: GPU utilization: 45%, Memory usage: 1.2GB
```

### Performance Comparison

```python
# Python script: analyze_search_efficiency.py
import json
from pathlib import Path

def compare_search_efficiency(traditional_log, ml_log):
    """Compare search efficiency between traditional and ML-guided approaches."""

    with open(traditional_log) as f:
        trad_data = json.load(f)
    with open(ml_log) as f:
        ml_data = json.load(f)

    print("=== Search Efficiency Comparison ===")
    print(f"Traditional: {trad_data['given_clauses']} given clauses")
    print(f"ML-Guided:   {ml_data['given_clauses']} given clauses")
    print(f"Efficiency gain: {(1 - ml_data['given_clauses']/trad_data['given_clauses']):.1%}")
    print(f"Novel lemmas found: {ml_data['novel_lemmas']} vs {trad_data['novel_lemmas']}")
    print(f"Proof diversity: {ml_data['proof_branching_factor']:.2f}")

# Usage:
# python analyze_search_efficiency.py traditional.json ml_guided.json
```

---

## Troubleshooting Guide

### Common Issues

**1. GPU Memory Errors**
```bash
# Reduce batch size and cache size
pyprover9 --embedding-cache-size 5000 --ml-batch-size 16 --cpu-only
```

**2. Model Convergence Issues**
```bash
# Use pre-trained model or adjust learning rate
pyprover9 --embedding-learning-rate 0.001 --warm-start-epochs 50
```

**3. Compatibility Verification**
```bash
# Verify ML results match traditional results on known problems
python -m pyladr.tools.verify_compatibility examples/known_proofs/
```

---

## Advanced Use Cases

### Custom Domain Models

```python
# custom_model_training.py
from pyladr.ml.training.contrastive import ContrastiveLearner
from pyladr.ml.graph.clause_encoder import HeterogeneousClauseGNN

# Train domain-specific model
learner = ContrastiveLearner(
    model_dim=512,
    domain="group_theory",  # specialized for group theory problems
    use_axiom_embeddings=True
)

model = learner.train_from_problems(
    problem_dir="examples/group_theory/",
    epochs=100,
    validation_split=0.2
)

model.save("models/group_theory_embeddings.pt")
```

### Integration with External Tools

```bash
# Export embeddings for external analysis
pyprover9 -f problem.in --enable-embeddings --export-embeddings embeddings.h5

# Use with external ML tools
python external_analysis.py --embeddings embeddings.h5 --analyze-clustering
```

---

## Conclusion

This demonstration shows how PyLADR's graph-based clause embeddings transform theorem proving from heuristic-driven search to AI-guided discovery. Key benefits:

- **Novel proof strategies** discovered through semantic clause understanding
- **Improved search efficiency** via embedding-guided clause selection
- **Continuous learning** that adapts to problem domains
- **Full compatibility** with existing Prover9/LADR infrastructure

The ML-enhanced search maintains the reliability and correctness of classical automated theorem proving while opening new frontiers for mathematical discovery through AI guidance.

**Next Steps:**
1. Try the examples with your own theorem proving problems
2. Experiment with different embedding weights and online learning rates
3. Train domain-specific models for your area of mathematics
4. Contribute novel proof discoveries back to the PyLADR community!