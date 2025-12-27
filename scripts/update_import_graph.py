#!/usr/bin/env python3
"""Generate a Mermaid diagram of the internal import graph for src/shrinkray/.

Only shows imports between modules within the package, not external dependencies.
"""

import ast
import sys
from pathlib import Path


def get_module_name(path: Path, src_dir: Path) -> str:
    """Convert a file path to a module name relative to src/."""
    rel_path = path.relative_to(src_dir)
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def get_short_name(module: str) -> str:
    """Get a short display name for a module."""
    # Remove the shrinkray prefix for brevity
    if module.startswith("shrinkray."):
        return module[len("shrinkray.") :]
    return module


def resolve_relative_imports(
    path: Path, src_dir: Path, package_prefix: str
) -> set[str]:
    """Extract and resolve all imports including relative ones."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()

    imports: set[str] = set()
    current_module = get_module_name(path, src_dir)
    current_parts = current_module.split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package_prefix):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                # Absolute import
                if node.module and node.module.startswith(package_prefix):
                    imports.add(node.module)
            else:
                # Relative import
                # Go up 'level' directories from current module
                if node.level <= len(current_parts):
                    base_parts = current_parts[: -node.level]
                    if node.module:
                        resolved = ".".join(base_parts + [node.module])
                    else:
                        resolved = ".".join(base_parts)
                    if resolved.startswith(package_prefix):
                        imports.add(resolved)

    return imports


def normalize_to_file_module(module: str, all_modules: set[str]) -> str | None:
    """Normalize a module path to match an actual file module.

    If we import shrinkray.passes.definitions, we want to link to that module.
    If we import shrinkray.passes, we want to link to shrinkray.passes (the __init__).
    """
    if module in all_modules:
        return module
    # Check if it's a submodule import - find the closest parent
    parts = module.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in all_modules:
            return candidate
        parts.pop()
    return None


def build_import_graph(src_dir: Path) -> dict[str, set[str]]:
    """Build a graph of imports between modules."""
    package_prefix = "shrinkray"
    graph: dict[str, set[str]] = {}

    # First pass: collect all module names
    all_modules: set[str] = set()
    py_files: list[Path] = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        module = get_module_name(py_file, src_dir)
        all_modules.add(module)
        py_files.append(py_file)

    # Second pass: extract imports
    for py_file in py_files:
        module = get_module_name(py_file, src_dir)
        imports = resolve_relative_imports(py_file, src_dir, package_prefix)

        # Normalize imports to actual modules
        normalized_imports: set[str] = set()
        for imp in imports:
            normalized = normalize_to_file_module(imp, all_modules)
            if normalized and normalized != module:
                normalized_imports.add(normalized)

        graph[module] = normalized_imports

    return graph


def generate_mermaid(graph: dict[str, set[str]]) -> str:
    """Generate a Mermaid flowchart from the import graph."""
    lines = ["```mermaid", "flowchart TD"]

    # Create node definitions with short names
    node_ids: dict[str, str] = {}
    for i, module in enumerate(sorted(graph.keys())):
        node_id = f"n{i}"
        node_ids[module] = node_id
        short_name = get_short_name(module)
        lines.append(f"    {node_id}[{short_name}]")

    lines.append("")

    # Create edges
    for module in sorted(graph.keys()):
        for dep in sorted(graph[module]):
            if dep in node_ids:
                lines.append(f"    {node_ids[module]} --> {node_ids[dep]}")

    lines.append("```")
    return "\n".join(lines)


def generate_dot(graph: dict[str, set[str]]) -> str:
    """Generate a Graphviz DOT file from the import graph."""
    lines = ["digraph imports {", "    rankdir=TB;", "    node [shape=box];", ""]

    # Create node definitions with short names
    for module in sorted(graph.keys()):
        short_name = get_short_name(module)
        # Escape quotes in node names
        escaped_name = short_name.replace('"', '\\"')
        lines.append(f'    "{escaped_name}";')

    lines.append("")

    # Create edges
    for module in sorted(graph.keys()):
        src_name = get_short_name(module).replace('"', '\\"')
        for dep in sorted(graph[module]):
            if dep in graph:  # Only include edges to known modules
                dst_name = get_short_name(dep).replace('"', '\\"')
                lines.append(f'    "{src_name}" -> "{dst_name}";')

    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    src_dir = Path("src")
    if not src_dir.exists():
        print("Error: src/ directory not found", file=sys.stderr)
        return 1

    notes_dir = Path("notes")
    notes_dir.mkdir(exist_ok=True)

    md_file = notes_dir / "import-graph.md"
    dot_file = notes_dir / "import-graph.dot"

    graph = build_import_graph(src_dir)
    mermaid = generate_mermaid(graph)
    dot = generate_dot(graph)

    md_content = f"""# Import Graph

This diagram shows the internal import dependencies between modules in `src/shrinkray/`.

External dependencies (trio, attrs, etc.) are not shown.

{mermaid}

---
*This file is auto-generated by `scripts/update_import_graph.py`*
"""

    updated_files: list[str] = []

    # Update markdown file if changed
    if not md_file.exists() or md_file.read_text() != md_content:
        md_file.write_text(md_content)
        updated_files.append(str(md_file))

    # Update DOT file if changed
    if not dot_file.exists() or dot_file.read_text() != dot:
        dot_file.write_text(dot)
        updated_files.append(str(dot_file))

    if updated_files:
        print(f"Updated: {', '.join(updated_files)}")
    else:
        print("All import graph files are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
