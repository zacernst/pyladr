#!/usr/bin/env python3
"""
Monitor the ML model tests on the hard vampire.in problem
"""

import subprocess
import time
import sys

def check_task_progress(task_id, description):
    """Check progress of a background task."""

    try:
        result = subprocess.run([
            "python3", "-c",
            f"from pathlib import Path; "
            f"import subprocess; "
            f"result = subprocess.run(['claude', 'task-output', '{task_id}', '--no-block'], capture_output=True, text=True); "
            f"print(result.stdout)"
        ], capture_output=True, text=True, timeout=5)

        output = result.stdout
        if "given #" in output:
            lines = output.split('\n')
            given_lines = [line for line in lines if line.strip().startswith('given #')]
            if given_lines:
                latest = given_lines[-1]
                print(f"  {description}: {latest[:80]}...")
        elif "PROOF FOUND" in output:
            print(f"  {description}: ✅ PROOF FOUND!")
            return "completed_success"
        elif "Exit Code:" in output and "Proof Found: No" in output:
            print(f"  {description}: ❌ No proof found (completed)")
            return "completed_failure"
        else:
            print(f"  {description}: Running...")

    except Exception as e:
        print(f"  {description}: Error checking - {e}")

    return "running"

def main():
    print("🧛 Monitoring Hard Vampire.in ML Tests")
    print("=" * 60)
    print("Testing our ML models on the ORIGINAL hard problem with constants")
    print("Expected: Very challenging, 200+ clauses needed, may not find proof")
    print()

    # Task IDs from the background processes
    tasks = [
        ("b1odls5mz", "Goal-Aware Model"),
        ("bgmj841dv", "Enhanced Model")
    ]

    completed = set()

    while len(completed) < len(tasks):
        print(f"\n⏰ {time.strftime('%H:%M:%S')} - Progress Check:")

        for task_id, description in tasks:
            if task_id not in completed:
                status = check_task_progress(task_id, description)
                if status.startswith("completed"):
                    completed.add(task_id)

        if len(completed) < len(tasks):
            print(f"\n💡 Still running {len(tasks) - len(completed)} tests...")
            print("   This is expected - the hard problem requires extensive search")
            time.sleep(30)  # Check every 30 seconds

    print(f"\n🎯 All tests completed!")
    print("Use TaskOutput to get full results from each model")

if __name__ == "__main__":
    main()