#!/usr/bin/env python3
"""
Test script to verify PyLADR ML setup is working correctly.
Run this to check that all dependencies are installed and basic functionality works.
"""

import sys
from pathlib import Path


def test_basic_imports():
    """Test that PyLADR core modules can be imported."""
    print("Testing PyLADR core imports...")
    try:
        from pyladr.core.clause import Clause
        from pyladr.core.term import Term  
        from pyladr.core.symbol import Symbol
        from pyladr.search.selection import GivenSelection
        print("✓ Core PyLADR imports successful")
        return True
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False


def test_ml_imports():
    """Test that ML dependencies can be imported."""
    print("Testing ML imports...")
    try:
        import torch
        import torch_geometric
        print(f"✓ PyTorch {torch.__version__} available")
        print(f"✓ PyTorch Geometric {torch_geometric.__version__} available")
        return True
    except ImportError as e:
        print(f"✗ ML import failed: {e}")
        print("  Install with: pip install -e '.[ml]'")
        return False


def test_ml_components():
    """Test that PyLADR ML components can be imported."""
    print("Testing PyLADR ML components...")
    try:
        from pyladr.ml.graph.clause_graph import ClauseGraph
        from pyladr.ml.graph.clause_encoder import HeterogeneousClauseGNN  
        from pyladr.ml.embeddings.cache import EmbeddingCache
        from pyladr.search.ml_selection import EmbeddingEnhancedSelection
        print("✓ PyLADR ML components available")
        return True
    except ImportError as e:
        print(f"✗ ML component import failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability for accelerated embeddings."""
    print("Testing GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("⚠ CUDA not available - will use CPU for embeddings")
            return False
    except:
        return False


def test_example_files():
    """Verify example files are present."""
    print("Testing example files...")
    examples = [
        "group_commutativity.in",
        "huntington_h4.in", 
        "modular_lattice.in",
        "advanced_config.in",
        "robbins_conjecture.in"
    ]
    
    missing = []
    for example in examples:
        if not Path(example).exists():
            missing.append(example)
    
    if missing:
        print(f"✗ Missing example files: {missing}")
        return False
    else:
        print(f"✓ All {len(examples)} example files present")
        return True


def main():
    """Run all tests and report results."""
    print("=== PyLADR ML Setup Verification ===\n")
    
    tests = [
        ("Core PyLADR", test_basic_imports),
        ("ML Dependencies", test_ml_imports), 
        ("ML Components", test_ml_components),
        ("GPU Support", test_gpu_availability),
        ("Example Files", test_example_files)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
        print()
    
    print("=== Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Setup verification successful! You're ready to use PyLADR ML features.")
        print("\nTry: python -c \"from pyladr.search.ml_selection import EmbeddingEnhancedSelection; print('ML ready!')\"")
    else:
        print(f"\n⚠ Setup incomplete. {total-passed} issues need to be resolved.")
        if not results["ML Dependencies"]:
            print("Next step: pip install -e '.[ml]'")
        
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
