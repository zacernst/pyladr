#!/usr/bin/env python3
"""
Complete workflow: Train a model on vampire.in and test the improvement.

This script demonstrates the complete ML training and evaluation workflow:
1. Train a specialized model for vampire.in
2. Test the trained model vs random initialization
3. Show performance improvements

Usage:
    python train_and_test_vampire.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = time.time() - start_time

    print(f"Completed in {duration:.1f}s with exit code {result.returncode}")

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    else:
        print(f"✅ Success: {description}")
        return True

def main():
    print("🧛 Vampire.in Training & Testing Workflow")
    print("=" * 50)

    # Check if vampire.in exists
    if not Path("vampire.in").exists():
        print("❌ vampire.in not found. Make sure you're in the PyLADR directory.")
        return 1

    model_path = "vampire_trained_model.pt"

    # Step 1: Train the model
    print("\n📚 Phase 1: Training Model")
    print("This will collect training data and train a specialized model...")

    train_cmd = [
        sys.executable, "examples/train_vampire_model.py",
        "--problem", "vampire.in",
        "--epochs", "20",              # Reduced for demo
        "--attempts", "10",            # Fewer attempts for faster training
        "--embedding-dim", "128",
        "--output", model_path
    ]

    if not run_command(train_cmd, "Training vampire.in-specific model"):
        return 1

    # Step 2: Test random initialization (baseline)
    print("\n📊 Phase 2: Baseline Test (Random Model)")
    print("Testing with randomly initialized model...")

    baseline_cmd = [
        sys.executable, "examples/simple_ml_usage.py",
        "vampire.in"
    ]

    if not run_command(baseline_cmd, "Baseline test with random model"):
        print("⚠️  Baseline test failed, but continuing...")

    # Step 3: Test trained model
    print("\n🧠 Phase 3: Trained Model Test")
    print(f"Testing with trained model: {model_path}")

    if not Path(model_path).exists():
        print(f"❌ Trained model {model_path} not found!")
        return 1

    trained_cmd = [
        sys.executable, "examples/simple_ml_usage.py",
        "vampire.in", "--model", model_path
    ]

    if not run_command(trained_cmd, "Testing with trained model"):
        return 1

    # Step 4: Comparison test
    print("\n🏆 Phase 4: Performance Comparison")
    print("Running side-by-side comparison...")

    comparison_cmd = [
        sys.executable, "examples/ml_guided_vampire_demo.py",
        "--max-given", "50",           # Quick comparison
        "--ml-ratio", "0.5"            # 50% ML guidance
    ]

    if not run_command(comparison_cmd, "Performance comparison"):
        print("⚠️  Comparison test failed, but model training completed successfully!")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 Training & Testing Complete!")
    print("=" * 60)
    print(f"✅ Trained model saved: {model_path}")

    if Path(model_path.replace('.pt', '_summary.json')).exists():
        print(f"📋 Training summary: {model_path.replace('.pt', '_summary.json')}")

    print("\n📖 Usage Instructions:")
    print("To use your trained model in demos:")
    print(f"  python examples/simple_ml_usage.py vampire.in --model {model_path}")
    print(f"  python examples/ml_guided_vampire_demo.py --model {model_path}")

    print("\n🔬 Expected Improvements:")
    print("- Better clause selection for vampire.in-style problems")
    print("- More focused search based on learned patterns")
    print("- Potential reduction in search time/generated clauses")

    print("\n💡 Next Steps:")
    print("- Train on more problems for broader domain coverage")
    print("- Experiment with different hyperparameters")
    print("- Try transfer learning to related logical domains")

    return 0

if __name__ == "__main__":
    sys.exit(main())