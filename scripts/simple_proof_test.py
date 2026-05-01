#!/usr/bin/env python3
"""Simple proof test for entropy display."""

import subprocess
import sys
import tempfile

def test_simple_proof():
    """Test proof entropy display with a simple contradiction."""

    # Create a simple input that will generate a contradiction
    input_content = """
    set(auto).
    assign(max_proofs, 1).
    assign(max_given, 10).

    formulas(sos).
    P(a).
    -P(a).
    end_of_list.
    """

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(input_content)
        temp_file = f.name

    print("=== Testing proof entropy display ===")
    print("Input:")
    print(input_content)

    print("\\n=== Running prover9 to generate proof ===")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pyladr.apps.prover9', '-f', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

    except subprocess.TimeoutExpired:
        print("Process timed out")
    except Exception as e:
        print(f"Error: {e}")

    # Cleanup
    import os
    os.unlink(temp_file)

if __name__ == "__main__":
    test_simple_proof()