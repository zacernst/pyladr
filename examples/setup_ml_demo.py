#!/usr/bin/env python3
"""
Setup script for ML-guided theorem proving demo.

This script helps set up the environment and dependencies needed
for the ML-guided clause selection demonstration.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import torch
        print("✅ PyTorch found")
    except ImportError:
        missing.append("torch")

    try:
        import torch_geometric
        print("✅ PyTorch Geometric found")
    except ImportError:
        missing.append("torch-geometric")

    try:
        from pyladr.ml.embedding_provider import EmbeddingProvider
        print("✅ PyLADR ML modules found")
    except ImportError:
        missing.append("pyladr[ml]")

    return missing


def install_ml_dependencies():
    """Install ML dependencies."""
    print("📦 Installing ML dependencies...")

    commands = [
        [sys.executable, "-m", "pip", "install", "torch"],
        [sys.executable, "-m", "pip", "install", "torch-geometric"],
        [sys.executable, "-m", "pip", "install", "-e", ".[ml]"]
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=Path.cwd())
            print("✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            return False

    return True


def create_sample_problems():
    """Create some sample problems for testing."""
    problems_dir = Path("examples/sample_problems")
    problems_dir.mkdir(exist_ok=True)

    # Simple group theory problem
    group_problem = problems_dir / "simple_group.in"
    with open(group_problem, 'w') as f:
        f.write("""% Simple group theory - prove identity element is unique
formulas(assumptions).
  % Group axioms
  x * (y * z) = (x * y) * z.     % Associativity
  x * e = x.                     % Right identity
  exists y (x * y = e).          % Right inverse
end_of_list.

formulas(goals).
  % Prove left identity: e * x = x
  e * x = x.
end_of_list.
""")

    # Boolean algebra problem
    bool_problem = problems_dir / "boolean_algebra.in"
    with open(bool_problem, 'w') as f:
        f.write("""% Boolean algebra - prove De Morgan's law
formulas(assumptions).
  % Boolean algebra axioms
  x + (y + z) = (x + y) + z.     % Addition associative
  x * (y * z) = (x * y) * z.     % Multiplication associative
  x + y = y + x.                 % Addition commutative
  x * y = y * x.                 % Multiplication commutative
  x + (x * y) = x.               % Absorption
  x * (x + y) = x.               % Absorption
  x + (-x) = 1.                  % Complement
  x * (-x) = 0.                  % Complement
end_of_list.

formulas(goals).
  % Prove De Morgan's law: -(x + y) = -x * -y
  -(x + y) = (-x) * (-y).
end_of_list.
""")

    print(f"✅ Created sample problems in {problems_dir}")
    return True


def check_vampire_file():
    """Check if vampire.in exists and create it if not."""
    vampire_file = Path("vampire.in")

    if vampire_file.exists():
        print("✅ vampire.in found")
        return True

    print("📝 Creating vampire.in...")
    with open(vampire_file, 'w') as f:
        f.write("""set(auto).

formulas(sos).
-P(x) | -P(i(x,y)) | P(y).
P(i(i(x,y),i(i(y,z),i(x,z)))).
P(i(i(n(x),x),x)).
P(i(x,i(n(x),y))).
-P(i(a,i(i(b,i(a,c)),i(i(n(c),i(i(n(d),e),b)),i(d,c))))). % HARD!
end_of_list.
""")

    print("✅ vampire.in created")
    return True


def main():
    print("🔧 PyLADR ML Demo Setup")
    print("=" * 30)

    # Check current dependencies
    missing = check_dependencies()

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        response = input("Would you like to install them? (y/n): ").lower().strip()

        if response == 'y':
            if not install_ml_dependencies():
                print("\n❌ Installation failed. Please install manually:")
                print("  pip install torch torch-geometric")
                print("  pip install -e '.[ml]'")
                return 1

            print("\n✅ Dependencies installed successfully!")
        else:
            print("\n⚠️  ML features will not be available without dependencies.")
            print("You can still run traditional search.")
    else:
        print("\n✅ All dependencies found!")

    # Setup files
    print("\n📁 Setting up demo files...")
    check_vampire_file()
    create_sample_problems()

    print("\n🎯 Setup complete! You can now run:")
    print("  python examples/ml_guided_vampire_demo.py")
    print("  python examples/ml_guided_vampire_demo.py --help")

    print("\nExample commands:")
    print("  # Basic demo with 30% ML guidance")
    print("  python examples/ml_guided_vampire_demo.py")
    print("")
    print("  # More aggressive ML guidance")
    print("  python examples/ml_guided_vampire_demo.py --ml-ratio 0.7")
    print("")
    print("  # Traditional only for comparison")
    print("  python examples/ml_guided_vampire_demo.py --traditional-only")
    print("")
    print("  # Save detailed results")
    print("  python examples/ml_guided_vampire_demo.py --save-results")

    return 0


if __name__ == "__main__":
    sys.exit(main())