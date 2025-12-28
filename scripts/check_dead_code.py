#!/usr/bin/env python3
"""Dead code detection for shrinkray.

Builds a dependency graph of module-level symbols, identifies entry points (roots),
and reports symbols that are not reachable from any root.

For source code:
- Roots are entry points defined in pyproject.toml

For test code:
- Roots are pytest fixtures (decorated with @pytest.fixture)
- Roots are test functions (starting with test_) that reference source symbols

Uses a file-based cache to avoid re-parsing unchanged files.

Exit codes:
    0: No dead code found
    1: Dead code detected or error occurred
"""

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider


CACHE_FILE = Path(".cache/dead_code_cache.json")
SCRIPT_FILE = Path(__file__).resolve()
SRC_DIR = Path("src")
TEST_DIR = Path("tests")
PACKAGE_PREFIX = "shrinkray"
TEST_PREFIX = "tests"

# Entry points from pyproject.toml [project.scripts]
SOURCE_ROOTS = {
    "shrinkray.__main__.main",
    "shrinkray.__main__.worker_main",
}


@dataclass
class Symbol:
    """A module-level symbol (function, class, or variable)."""

    module: str  # e.g., "shrinkray.reducer"
    name: str  # e.g., "ShrinkRay"
    kind: str  # "function", "class", "variable"
    line: int
    file: Path
    is_fixture: bool = False  # True if decorated with @pytest.fixture
    is_test: bool = False  # True if name starts with "test_"
    is_pytest_hook: bool = False  # True if name starts with "pytest_"
    is_pytestmark: bool = False  # True if name is "pytestmark"
    uses_fixtures: bool = False  # True if test function has parameters (uses fixtures)
    invokes_shrinkray_subprocess: bool = False  # True if calls subprocess with "shrinkray"
    marked_used: bool = False  # True if has "# used" comment

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class ModuleInfo:
    """Parsed information about a single module."""

    module_name: str
    file: Path
    symbols: dict[str, Symbol] = field(default_factory=dict)  # name -> Symbol
    # For each symbol, what names does it reference?
    references: dict[str, set[str]] = field(default_factory=dict)
    # imported_name -> (source_module, original_name or None if same)
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    # Names that are referenced at module level (outside any definition)
    module_level_refs: set[str] = field(default_factory=set)


def get_module_name(path: Path, base_dir: Path) -> str:
    """Convert a file path to a module name relative to base directory.

    For src/shrinkray/foo.py -> shrinkray.foo
    For tests/test_foo.py -> tests.test_foo
    """
    rel_path = path.relative_to(base_dir)
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def is_tracked_module(module: str) -> bool:
    """Check if a module should be tracked (shrinkray or tests)."""
    return module.startswith(PACKAGE_PREFIX) or module.startswith(TEST_PREFIX)


class ReferenceCollector(cst.CSTVisitor):
    """Collect all Name references and imports within a CST node."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.references: set[str] = set()
        # Track imports found inside this scope
        self.local_imports: dict[str, tuple[str, str | None]] = {}

    def visit_Name(self, node: cst.Name) -> None:
        self.references.add(node.value)

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        # For x.y.z, we only care about the root 'x'
        # Walk to find the leftmost Name, but also visit any Call nodes we encounter
        current: cst.BaseExpression = node
        while isinstance(current, cst.Attribute):
            current = current.value
        if isinstance(current, cst.Name):
            self.references.add(current.value)
        elif isinstance(current, cst.Call):
            # Visit the call's func and args
            current.visit(self)
        # Don't visit children normally - we handled the chain
        return False

    def visit_Import(self, node: cst.Import) -> None:
        # import x, y, z
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                name_node = alias.name
                if isinstance(name_node, cst.Name):
                    module = name_node.value
                elif isinstance(name_node, cst.Attribute):
                    # import a.b.c
                    parts = []
                    current: cst.BaseExpression = name_node
                    while isinstance(current, cst.Attribute):
                        parts.append(current.attr.value)
                        current = current.value
                    if isinstance(current, cst.Name):
                        parts.append(current.value)
                    module = ".".join(reversed(parts))
                else:
                    continue

                if is_tracked_module(module):
                    if alias.asname and isinstance(alias.asname.name, cst.Name):
                        local_name = alias.asname.name.value
                    else:
                        # For 'import a.b.c', the local name is 'a' (the root)
                        local_name = module.split(".")[0]
                    self.local_imports[local_name] = (module, None)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # from x import y
        if isinstance(node.names, cst.ImportStar):
            return

        # Resolve the module
        if node.module is None:
            base_module = ""
        elif isinstance(node.module, cst.Name):
            base_module = node.module.value
        elif isinstance(node.module, cst.Attribute):
            parts = []
            current: cst.BaseExpression = node.module
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
            base_module = ".".join(reversed(parts))
        else:
            return

        # Handle relative imports
        if len(node.relative) > 0:
            level = len(node.relative)
            parts = self.module_name.split(".")
            if level <= len(parts):
                base_parts = parts[:-level]
                if base_module:
                    source_module = ".".join(base_parts + [base_module])
                else:
                    source_module = ".".join(base_parts)
            else:
                return  # Invalid relative import
        else:
            source_module = base_module

        if not is_tracked_module(source_module):
            return

        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                if isinstance(alias.name, cst.Name):
                    original_name = alias.name.value
                    local_name = (
                        alias.asname.name.value
                        if alias.asname and isinstance(alias.asname.name, cst.Name)
                        else original_name
                    )
                    self.local_imports[local_name] = (source_module, original_name)


def collect_references(
    node: cst.CSTNode, module_name: str
) -> tuple[set[str], dict[str, tuple[str, str | None]]]:
    """Collect all name references and imports in a CST node."""
    collector = ReferenceCollector(module_name)
    node.visit(collector)
    return collector.references, collector.local_imports


class ShrinkraySubprocessDetector(cst.CSTVisitor):
    """Detect if code contains function calls with 'shrinkray' as a string argument."""

    def __init__(self) -> None:
        self.found = False
        self._in_call = False

    def visit_Call(self, node: cst.Call) -> bool:
        self._in_call = True
        return True  # Continue visiting children

    def leave_Call(self, node: cst.Call) -> None:
        self._in_call = False

    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        # Only match if we're inside a function call
        if not self._in_call:
            return
        # node.value includes quotes, extract the actual string content
        # Handle both single and double quotes
        content = node.value[1:-1]  # Remove surrounding quotes
        # Match "shrinkray" or "shrinkray.something"
        if content == "shrinkray" or content.startswith("shrinkray."):
            self.found = True


def invokes_shrinkray_subprocess(node: cst.CSTNode) -> bool:
    """Check if a node contains function call with 'shrinkray' string argument."""
    detector = ShrinkraySubprocessDetector()
    node.visit(detector)
    return detector.found


def has_used_comment(node: cst.CSTNode) -> bool:
    """Check if a node has a '# used' comment marking it as intentionally kept."""
    # Check leading lines for comment (FunctionDef, ClassDef have this)
    if hasattr(node, "leading_lines"):
        for line in node.leading_lines:  # type: ignore[union-attr]
            if isinstance(line, cst.EmptyLine) and line.comment:
                comment_text = line.comment.value.lower()
                if "# used" in comment_text:
                    return True

    # Check trailing_whitespace.comment (for SimpleStatementLine, FunctionDef body, etc.)
    if hasattr(node, "trailing_whitespace"):
        tw = node.trailing_whitespace  # type: ignore[union-attr]
        if tw and hasattr(tw, "comment") and tw.comment:
            comment_text = tw.comment.value.lower()
            if "# used" in comment_text:
                return True

    # For FunctionDef/ClassDef, check the body's header for trailing comment on def line
    if hasattr(node, "body") and hasattr(node.body, "header"):  # type: ignore[union-attr]
        header = node.body.header  # type: ignore[union-attr]
        if hasattr(header, "comment") and header.comment:
            comment_text = header.comment.value.lower()
            if "# used" in comment_text:
                return True

    return False


class ModuleAnalyzer(cst.CSTVisitor):
    """Analyze a module to extract symbols and their references."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, module_name: str, file: Path) -> None:
        self.info = ModuleInfo(module_name=module_name, file=file)
        self._at_module_level = True
        self._current_stmt_has_used = False  # Track # used comment on parent statement

    def _get_line(self, node: cst.CSTNode) -> int:
        pos = cast(CodeRange, self.metadata[PositionProvider][node])
        return pos.start.line

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool:
        """Track if current statement line has a # used comment."""
        if not self._at_module_level:
            return False
        # Check for trailing # used comment
        tw = node.trailing_whitespace
        if tw and hasattr(tw, "comment") and tw.comment:
            if "# used" in tw.comment.value.lower():
                self._current_stmt_has_used = True
            else:
                self._current_stmt_has_used = False
        else:
            self._current_stmt_has_used = False
        return True  # Continue visiting children

    def leave_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        """Reset the flag after processing the statement."""
        self._current_stmt_has_used = False

    def visit_Import(self, node: cst.Import) -> bool:
        if not self._at_module_level:
            return False
        # import x, y, z
        if isinstance(node.names, cst.ImportStar):
            return False
        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                name_node = alias.name
                if isinstance(name_node, cst.Name):
                    module = name_node.value
                elif isinstance(name_node, cst.Attribute):
                    parts = []
                    current: cst.BaseExpression = name_node
                    while isinstance(current, cst.Attribute):
                        parts.append(current.attr.value)
                        current = current.value
                    if isinstance(current, cst.Name):
                        parts.append(current.value)
                    module = ".".join(reversed(parts))
                else:
                    continue

                if is_tracked_module(module):
                    if alias.asname and isinstance(alias.asname.name, cst.Name):
                        local_name = alias.asname.name.value
                    else:
                        # For 'import a.b.c', the local name is 'a' (the root)
                        local_name = module.split(".")[0]
                    self.info.imports[local_name] = (module, None)
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        if not self._at_module_level:
            return False
        if isinstance(node.names, cst.ImportStar):
            return False

        # Resolve the module
        if node.module is None:
            base_module = ""
        elif isinstance(node.module, cst.Name):
            base_module = node.module.value
        elif isinstance(node.module, cst.Attribute):
            parts = []
            current: cst.BaseExpression = node.module
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
            base_module = ".".join(reversed(parts))
        else:
            return False

        # Handle relative imports
        if len(node.relative) > 0:
            level = len(node.relative)
            parts = self.info.module_name.split(".")
            if level <= len(parts):
                base_parts = parts[:-level]
                if base_module:
                    source_module = ".".join(base_parts + [base_module])
                else:
                    source_module = ".".join(base_parts)
            else:
                return False
        else:
            source_module = base_module

        if not is_tracked_module(source_module):
            return False

        for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
                if isinstance(alias.name, cst.Name):
                    original_name = alias.name.value
                    local_name = (
                        alias.asname.name.value
                        if alias.asname and isinstance(alias.asname.name, cst.Name)
                        else original_name
                    )
                    self.info.imports[local_name] = (source_module, original_name)
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if not self._at_module_level:
            return False
        self._process_function(node)
        return False  # Don't recurse - we handle the body ourselves

    def _is_pytest_fixture(self, node: cst.FunctionDef) -> bool:
        """Check if function has @pytest.fixture decorator."""
        for decorator in node.decorators:
            dec = decorator.decorator
            # Handle @pytest.fixture
            if isinstance(dec, cst.Attribute):
                if isinstance(dec.value, cst.Name) and dec.value.value == "pytest":
                    if dec.attr.value == "fixture":
                        return True
            # Handle @pytest.fixture(...)
            elif isinstance(dec, cst.Call):
                func = dec.func
                if isinstance(func, cst.Attribute):
                    if isinstance(func.value, cst.Name) and func.value.value == "pytest":
                        if func.attr.value == "fixture":
                            return True
            # Handle @fixture (if imported as 'from pytest import fixture')
            elif isinstance(dec, cst.Name) and dec.value == "fixture":
                return True
        return False

    def _process_function(self, node: cst.FunctionDef) -> None:
        is_fixture = self._is_pytest_fixture(node)
        is_test = node.name.value.startswith("test_")
        is_pytest_hook = node.name.value.startswith("pytest_")
        marked_used = has_used_comment(node)
        has_shrinkray_subprocess = invokes_shrinkray_subprocess(node)

        # Check if function has parameters (uses fixtures)
        has_params = bool(
            node.params.params
            or node.params.posonly_params
            or node.params.kwonly_params
        )

        symbol = Symbol(
            module=self.info.module_name,
            name=node.name.value,
            kind="function",
            line=self._get_line(node),
            file=self.info.file,
            is_fixture=is_fixture,
            is_test=is_test,
            is_pytest_hook=is_pytest_hook,
            uses_fixtures=is_test and has_params,
            invokes_shrinkray_subprocess=has_shrinkray_subprocess,
            marked_used=marked_used,
        )
        self.info.symbols[node.name.value] = symbol

        # Collect references from the entire function
        refs, local_imports = collect_references(node, self.info.module_name)

        # Remove parameter names
        param_names: set[str] = set()
        for param in node.params.params:
            param_names.add(param.name.value)
        for param in node.params.posonly_params:
            param_names.add(param.name.value)
        for param in node.params.kwonly_params:
            param_names.add(param.name.value)
        if node.params.star_arg and isinstance(node.params.star_arg, cst.Param):
            param_names.add(node.params.star_arg.name.value)
        if node.params.star_kwarg:
            param_names.add(node.params.star_kwarg.name.value)

        refs -= param_names
        refs.discard(node.name.value)

        # Store both module-level imports and local imports for resolution
        self.info.references[node.name.value] = refs
        # Store local imports as additional imports for this symbol
        if local_imports:
            # We'll handle these during resolution by merging them
            if not hasattr(self.info, "local_imports"):
                self.info.local_imports = {}  # type: ignore[attr-defined]
            self.info.local_imports[node.name.value] = local_imports  # type: ignore[attr-defined]

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if not self._at_module_level:
            return False

        symbol = Symbol(
            module=self.info.module_name,
            name=node.name.value,
            kind="class",
            line=self._get_line(node),
            file=self.info.file,
            marked_used=has_used_comment(node),
        )
        self.info.symbols[node.name.value] = symbol

        # Collect references from the entire class
        refs, local_imports = collect_references(node, self.info.module_name)
        refs.discard(node.name.value)
        self.info.references[node.name.value] = refs

        if local_imports:
            if not hasattr(self.info, "local_imports"):
                self.info.local_imports = {}  # type: ignore[attr-defined]
            self.info.local_imports[node.name.value] = local_imports  # type: ignore[attr-defined]

        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        if not self._at_module_level:
            return False

        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self._add_variable(
                    target.target.value,
                    self._get_line(node),
                    node.value,
                    marked_used=self._current_stmt_has_used,
                )
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if not self._at_module_level:
            return False

        if isinstance(node.target, cst.Name):
            refs, _ = collect_references(node.annotation, self.info.module_name)
            if node.value:
                value_refs, _ = collect_references(node.value, self.info.module_name)
                refs.update(value_refs)
            self._add_variable(
                node.target.value,
                self._get_line(node),
                node.value,
                refs,
                marked_used=self._current_stmt_has_used,
            )
        return False

    def _add_variable(
        self,
        name: str,
        lineno: int,
        value: cst.BaseExpression | None,
        extra_refs: set[str] | None = None,
        marked_used: bool = False,
    ) -> None:
        # Skip dunder names
        if name.startswith("__") and name.endswith("__"):
            return

        symbol = Symbol(
            module=self.info.module_name,
            name=name,
            kind="variable",
            line=lineno,
            file=self.info.file,
            is_pytestmark=name == "pytestmark",
            marked_used=marked_used,
        )
        self.info.symbols[name] = symbol

        refs = extra_refs or set()
        if value:
            value_refs, _ = collect_references(value, self.info.module_name)
            refs.update(value_refs)
        refs.discard(name)
        self.info.references[name] = refs

    def visit_If(self, node: cst.If) -> bool:
        # Handle module-level if statements (like TYPE_CHECKING blocks)
        if self._at_module_level:
            self._visit_module_level_block(node.body)
            if node.orelse:
                if isinstance(node.orelse, cst.If):
                    self.visit_If(node.orelse)
                elif isinstance(node.orelse, cst.Else):
                    self._visit_module_level_block(node.orelse.body)
        return False

    def _visit_module_level_block(self, block: cst.BaseSuite) -> None:
        """Visit statements in a block while maintaining module-level context."""
        if isinstance(block, cst.IndentedBlock):
            for stmt in block.body:
                if isinstance(stmt, cst.SimpleStatementLine):
                    for small_stmt in stmt.body:
                        if isinstance(small_stmt, cst.Import):
                            self.visit_Import(small_stmt)
                        elif isinstance(small_stmt, cst.ImportFrom):
                            self.visit_ImportFrom(small_stmt)


def analyze_module(path: Path, src_dir: Path) -> ModuleInfo:
    """Parse a Python file and extract symbol information."""
    module_name = get_module_name(path, src_dir)
    source = path.read_text()
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    analyzer = ModuleAnalyzer(module_name, path)
    wrapper.visit(analyzer)
    return analyzer.info


def build_dependency_graph(
    modules: dict[str, ModuleInfo],
) -> tuple[dict[str, Symbol], dict[str, set[str]]]:
    """Build a dependency graph from module information.

    Returns:
        all_symbols: qualified_name -> Symbol
        graph: qualified_name -> set of qualified names it depends on
    """
    all_symbols: dict[str, Symbol] = {}
    graph: dict[str, set[str]] = {}

    # First pass: collect all symbols
    for module_info in modules.values():
        for name, symbol in module_info.symbols.items():
            all_symbols[symbol.qualified_name] = symbol

    # Second pass: resolve references to qualified names
    for module_info in modules.values():
        # Get local imports if they exist
        local_imports_map: dict[str, dict[str, tuple[str, str | None]]] = getattr(
            module_info, "local_imports", {}
        )

        for name, refs in module_info.references.items():
            qualified = f"{module_info.module_name}.{name}"
            resolved_refs: set[str] = set()

            # Merge module-level imports with local imports for this symbol
            combined_imports = dict(module_info.imports)
            if name in local_imports_map:
                combined_imports.update(local_imports_map[name])

            for ref in refs:
                resolved = resolve_reference(ref, combined_imports, module_info.module_name, all_symbols)
                if resolved:
                    resolved_refs.add(resolved)

            graph[qualified] = resolved_refs

        # Module-level references
        if module_info.module_level_refs:
            resolved_refs = set()
            for ref in module_info.module_level_refs:
                resolved = resolve_reference(
                    ref, module_info.imports, module_info.module_name, all_symbols
                )
                if resolved:
                    resolved_refs.add(resolved)
            for name in module_info.symbols:
                qualified = f"{module_info.module_name}.{name}"
                if qualified in graph:
                    graph[qualified].update(resolved_refs)

    return all_symbols, graph


def resolve_reference(
    ref: str,
    imports: dict[str, tuple[str, str | None]],
    module_name: str,
    all_symbols: dict[str, Symbol],
) -> str | None:
    """Resolve a reference name to a qualified symbol name.

    Returns the qualified name if:
    - It's a defined symbol in all_symbols, OR
    - It's an import from a tracked module (shrinkray.* or tests.*)
    """
    # Check if it's an imported name
    if ref in imports:
        source_module, original_name = imports[ref]
        if original_name:
            qualified = f"{source_module}.{original_name}"
        else:
            qualified = source_module
        # Return if it's a symbol OR if it's from a tracked module
        # (module imports like 'from shrinkray.subprocess import worker'
        # may not be in all_symbols but still indicate a source reference)
        if qualified in all_symbols or is_tracked_module(source_module):
            return qualified
        return None

    # Check if it's a local symbol
    local_qualified = f"{module_name}.{ref}"
    if local_qualified in all_symbols:
        return local_qualified

    return None


def find_reachable(graph: dict[str, set[str]], roots: set[str]) -> set[str]:
    """Find all symbols reachable from the given roots using BFS."""
    reachable: set[str] = set()
    queue = deque(roots)

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)

        for dep in graph.get(current, set()):
            if dep not in reachable:
                queue.append(dep)

    return reachable


# === Caching ===


def load_cache() -> dict[str, Any]:
    """Load the dead code cache from disk.

    The cache is invalidated if the script itself has changed.
    """
    if not CACHE_FILE.exists():
        return {}
    try:
        cache = json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    # Check if script has changed - if so, invalidate entire cache
    script_key = str(SCRIPT_FILE)
    script_size, script_mtime = get_file_key(SCRIPT_FILE)
    cached_script = cache.get(script_key, {})
    if (
        cached_script.get("size") != script_size
        or cached_script.get("mtime") != script_mtime
    ):
        return {}

    return cache


def save_cache(cache: dict[str, Any]) -> None:
    """Save the cache to disk."""
    # Store script file info for cache invalidation
    script_key = str(SCRIPT_FILE)
    script_size, script_mtime = get_file_key(SCRIPT_FILE)
    cache[script_key] = {"size": script_size, "mtime": script_mtime}

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_file_key(path: Path) -> tuple[int, float]:
    """Get the cache key for a file (size, mtime)."""
    stat = path.stat()
    return (stat.st_size, stat.st_mtime)


def module_info_to_cache(info: ModuleInfo) -> dict[str, Any]:
    """Convert ModuleInfo to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "symbols": {
            name: {
                "kind": sym.kind,
                "line": sym.line,
                "is_fixture": sym.is_fixture,
                "is_test": sym.is_test,
                "is_pytest_hook": sym.is_pytest_hook,
                "is_pytestmark": sym.is_pytestmark,
                "uses_fixtures": sym.uses_fixtures,
                "invokes_shrinkray_subprocess": sym.invokes_shrinkray_subprocess,
                "marked_used": sym.marked_used,
            }
            for name, sym in info.symbols.items()
        },
        "references": {name: sorted(refs) for name, refs in info.references.items()},
        "imports": {
            name: {"module": mod, "original": orig}
            for name, (mod, orig) in info.imports.items()
        },
        "module_level_refs": sorted(info.module_level_refs),
    }

    # Include local imports if present
    if hasattr(info, "local_imports"):
        result["local_imports"] = {
            symbol_name: {
                name: {"module": mod, "original": orig}
                for name, (mod, orig) in imports.items()
            }
            for symbol_name, imports in info.local_imports.items()  # type: ignore[attr-defined]
        }

    return result


def module_info_from_cache(
    module_name: str, file: Path, cached: dict[str, Any]
) -> ModuleInfo:
    """Reconstruct ModuleInfo from cached data."""
    info = ModuleInfo(module_name=module_name, file=file)

    for name, sym_data in cached.get("symbols", {}).items():
        info.symbols[name] = Symbol(
            module=module_name,
            name=name,
            kind=sym_data["kind"],
            line=sym_data["line"],
            file=file,
            is_fixture=sym_data.get("is_fixture", False),
            is_test=sym_data.get("is_test", False),
            is_pytest_hook=sym_data.get("is_pytest_hook", False),
            is_pytestmark=sym_data.get("is_pytestmark", False),
            uses_fixtures=sym_data.get("uses_fixtures", False),
            invokes_shrinkray_subprocess=sym_data.get(
                "invokes_shrinkray_subprocess", False
            ),
            marked_used=sym_data.get("marked_used", False),
        )

    for name, refs in cached.get("references", {}).items():
        info.references[name] = set(refs)

    for name, imp_data in cached.get("imports", {}).items():
        info.imports[name] = (imp_data["module"], imp_data["original"])

    info.module_level_refs = set(cached.get("module_level_refs", []))

    # Restore local imports if present
    if "local_imports" in cached:
        info.local_imports = {}  # type: ignore[attr-defined]
        for symbol_name, imports in cached["local_imports"].items():
            info.local_imports[symbol_name] = {  # type: ignore[attr-defined]
                name: (imp_data["module"], imp_data["original"])
                for name, imp_data in imports.items()
            }

    return info


def analyze_with_cache(
    path: Path, src_dir: Path, cache: dict[str, Any]
) -> tuple[ModuleInfo, bool]:
    """Analyze a module, using cache if available.

    Returns (ModuleInfo, was_cached).
    """
    path_str = str(path)
    module_name = get_module_name(path, src_dir)
    size, mtime = get_file_key(path)

    if path_str in cache:
        cached = cache[path_str]
        if cached.get("size") == size and cached.get("mtime") == mtime:
            return module_info_from_cache(module_name, path, cached["data"]), True

    # Cache miss - parse the file
    info = analyze_module(path, src_dir)

    # Update cache
    cache[path_str] = {
        "size": size,
        "mtime": mtime,
        "data": module_info_to_cache(info),
    }

    return info, False


def references_source_symbols(
    qualified_name: str,
    graph: dict[str, set[str]],
    all_symbols: dict[str, Symbol],
) -> bool:
    """Check if a symbol references any source (shrinkray.*) symbols."""
    visited: set[str] = set()
    queue = deque([qualified_name])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        # Check if this is a source reference (not the starting symbol itself)
        if current != qualified_name:
            # Check if it's a defined symbol from shrinkray
            if current in all_symbols:
                sym = all_symbols[current]
                if sym.module.startswith(PACKAGE_PREFIX):
                    return True
            # Also check if it's a module reference (e.g., shrinkray.subprocess.worker)
            elif current.startswith(PACKAGE_PREFIX):
                return True

        # Add dependencies to queue
        for dep in graph.get(current, set()):
            if dep not in visited:
                queue.append(dep)

    return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Dead code detection for shrinkray")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore and don't update the cache"
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Only check source code, skip tests",
    )
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print("Error: src/ directory not found", file=sys.stderr)
        return 1

    cache = {} if args.no_cache else load_cache()
    modules: dict[str, ModuleInfo] = {}
    files_parsed = 0
    files_cached = 0

    # Analyze all Python files in src/shrinkray/
    shrinkray_dir = SRC_DIR / "shrinkray"
    for py_file in shrinkray_dir.rglob("*.py"):
        try:
            info, was_cached = analyze_with_cache(py_file, SRC_DIR, cache)
            modules[info.module_name] = info
            if was_cached:
                files_cached += 1
            else:
                files_parsed += 1
        except cst.ParserSyntaxError as e:
            print(f"Syntax error in {py_file}: {e}", file=sys.stderr)
            return 1

    # Analyze test files if not --source-only
    if not args.source_only and TEST_DIR.exists():
        for py_file in TEST_DIR.rglob("*.py"):
            try:
                info, was_cached = analyze_with_cache(py_file, Path("."), cache)
                modules[info.module_name] = info
                if was_cached:
                    files_cached += 1
                else:
                    files_parsed += 1
            except cst.ParserSyntaxError as e:
                print(f"Syntax error in {py_file}: {e}", file=sys.stderr)
                return 1

    # Save updated cache
    if not args.no_cache:
        save_cache(cache)

    # Build dependency graph
    all_symbols, graph = build_dependency_graph(modules)

    # Compute roots
    # 1. Source roots from entry points
    roots: set[str] = {r for r in SOURCE_ROOTS if r in all_symbols}
    if not roots:
        print("Warning: No source root symbols found!", file=sys.stderr)
        return 1

    # 2. Symbols marked with "# used" comment are always roots
    for qualified_name, sym in all_symbols.items():
        if sym.marked_used:
            roots.add(qualified_name)

    # 3. Test roots: fixtures and test functions that reference source symbols
    if not args.source_only:
        for qualified_name, sym in all_symbols.items():
            if not sym.module.startswith(TEST_PREFIX):
                continue

            # Fixtures are always roots
            if sym.is_fixture:
                roots.add(qualified_name)
            # Pytest hooks are always roots (called by pytest)
            elif sym.is_pytest_hook:
                roots.add(qualified_name)
            # pytestmark variables are always roots (used by pytest)
            elif sym.is_pytestmark:
                roots.add(qualified_name)
            # Test functions that use fixtures are roots (conservative)
            elif sym.is_test and sym.uses_fixtures:
                roots.add(qualified_name)
            # Test functions that invoke shrinkray subprocess are roots
            elif sym.is_test and sym.invokes_shrinkray_subprocess:
                roots.add(qualified_name)
            # Test functions that reference source symbols are roots
            elif sym.is_test and references_source_symbols(
                qualified_name, graph, all_symbols
            ):
                roots.add(qualified_name)

    reachable = find_reachable(graph, roots)

    # Find unreachable symbols
    unreachable = [
        sym for qualified, sym in all_symbols.items() if qualified not in reachable
    ]

    # Sort by file and line for consistent output
    unreachable.sort(key=lambda s: (str(s.file), s.line))

    if unreachable:
        for sym in unreachable:
            print(f"{sym.file}:{sym.line}: unused {sym.kind} {sym.name}")
        print(f"\nFound {len(unreachable)} unreachable symbol(s)")
        if files_cached > 0:
            print(f"({files_parsed} files parsed, {files_cached} cached)")
        return 1

    print("No dead code detected.")
    if files_cached > 0:
        print(f"({files_parsed} files parsed, {files_cached} cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
