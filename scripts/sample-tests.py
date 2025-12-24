#!/usr/bin/env python3
"""Randomly sample test functions for quality review.

Usage:
    python scripts/sample-tests.py [--count N] [--seed SEED] [--slow] [--changed]

Options:
    --count N    Number of tests to sample (default: 10)
    --seed SEED  Random seed for reproducibility
    --slow       Include slow tests in sampling
    --changed    Also show tests changed in current git diff
"""

import argparse
import ast
import random
import subprocess
import sys
from pathlib import Path


def get_test_functions(test_dir: Path, include_slow: bool = False) -> list[tuple[Path, str, int]]:
    """Find all test functions in the test directory.

    Returns list of (file_path, function_name, line_number) tuples.
    """
    tests = []

    for py_file in test_dir.glob("test_*.py"):
        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Check if marked as slow
                is_slow = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == "slow":
                            is_slow = True
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == "slow":
                                is_slow = True

                if is_slow and not include_slow:
                    continue

                tests.append((py_file, node.name, node.lineno))

    return tests


def get_changed_tests() -> list[tuple[Path, str]]:
    """Get tests that have been modified in the current git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []

    changed_tests = []
    for file_path in changed_files:
        if not file_path.startswith("tests/test_") or not file_path.endswith(".py"):
            continue

        path = Path(file_path)
        if not path.exists():
            continue

        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                changed_tests.append((path, node.name))

    return changed_tests


def get_test_source(file_path: Path, func_name: str) -> str | None:
    """Extract the source code of a test function."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            lines = source.split("\n")
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            return "\n".join(lines[start:end])

    return None


def main():
    parser = argparse.ArgumentParser(description="Sample tests for quality review")
    parser.add_argument("--count", type=int, default=10, help="Number of tests to sample")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--changed", action="store_true", help="Show changed tests too")
    parser.add_argument("--show-source", action="store_true", help="Show test source code")
    args = parser.parse_args()

    test_dir = Path("tests")
    if not test_dir.exists():
        print("Error: tests/ directory not found", file=sys.stderr)
        sys.exit(1)

    # Get all tests
    all_tests = get_test_functions(test_dir, include_slow=args.slow)

    if not all_tests:
        print("No tests found", file=sys.stderr)
        sys.exit(1)

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    else:
        # Use a seed based on current date for daily consistency
        import datetime
        seed = int(datetime.date.today().strftime("%Y%m%d"))
        random.seed(seed)
        print(f"Using date-based seed: {seed}")

    # Sample tests
    count = min(args.count, len(all_tests))
    sampled = random.sample(all_tests, count)

    print(f"\n{'='*60}")
    print(f"SAMPLED TESTS FOR REVIEW ({count} of {len(all_tests)} total)")
    print(f"{'='*60}\n")

    for i, (file_path, func_name, line_no) in enumerate(sampled, 1):
        print(f"{i}. {file_path}:{line_no}")
        print(f"   {func_name}")
        if args.show_source:
            source = get_test_source(file_path, func_name)
            if source:
                print()
                for line in source.split("\n"):
                    print(f"   | {line}")
        print()

    # Show changed tests if requested
    if args.changed:
        changed = get_changed_tests()
        if changed:
            print(f"\n{'='*60}")
            print(f"CHANGED TESTS (from git diff)")
            print(f"{'='*60}\n")

            for file_path, func_name in changed:
                print(f"  {file_path}: {func_name}")
            print()

    # Print review checklist
    print(f"{'='*60}")
    print("REVIEW CHECKLIST")
    print(f"{'='*60}")
    print("""
For each test, verify:

[ ] Comprehensibility
    - Test name clearly describes what's being tested
    - Test has a docstring or is self-explanatory
    - Assertions are clear about what's expected

[ ] Realistic failure modes
    - Test could actually fail if the code is broken
    - Test isn't just checking implementation details
    - Test exercises meaningful behavior

[ ] Robustness
    - No unnecessary sleeps or timing dependencies
    - No flaky assertions (ordering, floating point equality)
    - Mocks are minimal and focused

[ ] Code quality
    - No redundant if/else branches
    - No unnecessary `or` conditions that can never trigger
    - No commented-out code
    - No TODO comments that should be addressed
""")


if __name__ == "__main__":
    main()
