#!/usr/bin/env python3
"""
Custom domain-specific model training example.
Demonstrates how to train embeddings specialized for specific mathematical domains.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import torch
    from pyladr.ml.training.contrastive import ContrastiveLearner
    from pyladr.ml.graph.clause_encoder import HeterogeneousClauseGNN
except ImportError as e:
    print(f"Error: Missing ML dependencies. Install with: pip install -e '.[ml]'")
    print(f"Specific error: {e}")
    sys.exit(1)


def train_domain_model(
    problem_dir: str,
    domain: str = "general",
    model_dim: int = 512,
    epochs: int = 100,
    output_path: Optional[str] = None
) -> None:
    """Train a domain-specific clause embedding model."""

    problem_path = Path(problem_dir)
    if not problem_path.exists():
        print(f"Error: Problem directory '{problem_dir}' not found")
        return

    print(f"Training {domain} domain model...")
    print(f"Problem directory: {problem_path}")
    print(f"Model dimension: {model_dim}")
    print(f"Training epochs: {epochs}")

    # Initialize contrastive learner
    learner = ContrastiveLearner(
        model_dim=model_dim,
        domain=domain,
        use_axiom_embeddings=True,
        learning_rate=0.001,
        batch_size=32,
        negative_samples=5
    )

    try:
        # Train model from problem files
        model = learner.train_from_problems(
            problem_dir=str(problem_path),
            epochs=epochs,
            validation_split=0.2,
            checkpoint_interval=10,
            verbose=True
        )

        # Save trained model
        if output_path is None:
            output_path = f"models/{domain}_embeddings.pt"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        print(f"Model saved to: {output_path}")

        # Print training statistics
        stats = learner.get_training_stats()
        print(f"\nTraining Results:")
        print(f"Final loss: {stats.get('final_loss', 'N/A'):.4f}")
        print(f"Best validation accuracy: {stats.get('best_val_accuracy', 'N/A'):.2%}")
        print(f"Training time: {stats.get('training_time_minutes', 'N/A'):.1f} minutes")
        print(f"Patterns learned: {stats.get('unique_patterns_learned', 'N/A')}")

    except Exception as e:
        print(f"Error during training: {e}")
        return


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python custom_model_training.py <problem_directory> [domain] [model_dim] [epochs]")
        print("\nExamples:")
        print("  python custom_model_training.py examples/group_theory/ group_theory")
        print("  python custom_model_training.py examples/boolean_algebra/ boolean 256 50")
        print("\nThis script trains domain-specific clause embeddings from theorem proving problems.")
        sys.exit(1)

    problem_dir = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "general"
    model_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    train_domain_model(problem_dir, domain, model_dim, epochs)


if __name__ == "__main__":
    main()