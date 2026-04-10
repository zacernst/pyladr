#!/usr/bin/env python3
"""
Debug exactly what's being proven in the vampire.in variants
"""

import subprocess
import sys

def test_vampire_variant(file_path, description):
    """Test a vampire.in variant and show detailed output."""

    print(f"\n🔍 Testing: {description}")
    print(f"File: {file_path}")
    print("=" * 60)

    # Show the file content
    try:
        with open(file_path) as f:
            content = f.read()
        print("📝 Problem content:")
        print(content)
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        return

    print("\n🧮 Running PyProver9 with detailed output...")
    print("-" * 40)

    # Run with detailed output
    result = subprocess.run([
        "python3", "-m", "pyladr.apps.prover9",
        "-f", file_path,
        "--max-given", "20"
    ], capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)

    # Check if proof found
    output = result.stdout + result.stderr
    if "PROOF" in output or "proof" in output:
        print("✅ PROOF FOUND")
    else:
        print("❌ NO PROOF FOUND")

def main():
    print("🧛 Vampire.in Proof Debug Analysis")
    print("=" * 60)

    # Test original hard version
    test_vampire_variant("vampire.in", "Original hard vampire.in (constants)")

    # Test our variable version
    test_vampire_variant("examples/vampire.in", "Modified vampire.in (variables)")

    print(f"\n💡 Analysis Questions:")
    print("1. Is the variable version actually equivalent to the constant version?")
    print("2. Does using variables make unification trivially easy?")
    print("3. Are we testing the right logical problem?")
    print("4. Is there a difference in proof complexity?")

if __name__ == "__main__":
    main()