#!/usr/bin/env python3
"""Check for import cycles in the shrinkray package.

This script analyzes the import graph and reports any cyclic dependencies.
Import cycles can cause issues with module loading and make the codebase
harder to understand and maintain.

Exit codes:
    0: No cycles found
    1: Cycles detected or error occurred
"""

import sys
from pathlib import Path

# Import the graph building function from the existing script
from update_import_graph import build_import_graph, get_short_name


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find all cycles in the import graph using DFS.

    Returns a list of cycles, where each cycle is a list of module names
    forming the cycle (first and last element are the same).
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()  # Nodes in current recursion stack
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in sorted(graph.get(node, set())):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle - extract it from the path
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                # Normalize cycle to start from the lexicographically smallest node
                # This helps deduplicate cycles that start at different points
                min_idx = cycle.index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)

        path.pop()
        rec_stack.remove(node)

    for node in sorted(graph.keys()):
        if node not in visited:
            dfs(node)

    return cycles


def format_cycle(cycle: list[str]) -> str:
    """Format a cycle for display, using short module names."""
    short_names = [get_short_name(m) for m in cycle]
    return " -> ".join(short_names)


def main() -> int:
    src_dir = Path("src")
    if not src_dir.exists():
        print("Error: src/ directory not found", file=sys.stderr)
        return 1

    graph = build_import_graph(src_dir)
    cycles = find_cycles(graph)

    if cycles:
        print(f"Found {len(cycles)} import cycle(s):\n", file=sys.stderr)
        for cycle in sorted(cycles, key=lambda c: (len(c), c)):
            print(f"  {format_cycle(cycle)}", file=sys.stderr)
        print(
            "\nImport cycles can cause issues with module loading and should be resolved.",
            file=sys.stderr,
        )
        return 1

    print("No import cycles detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
