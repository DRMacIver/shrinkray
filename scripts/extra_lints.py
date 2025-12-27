#!/usr/bin/env python3
"""Custom linting rules using libcst.

Rules:
1. No class-based tests in test files
2. No imports inside test functions
3. No use of async iterators without contextlib.aclosing
4. No swallowing exception stack traces (str(e) without traceback logging)
5. No mutable default arguments

Uses a cache in .cache/extra_lints_cache.json to skip unchanged files.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import libcst as cst
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
)


CACHE_FILE = Path(".cache/extra_lints_cache.json")


@dataclass
class LintError:
    file: Path
    line: int
    column: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.rule}: {self.message}"


# Names that indicate traceback is being preserved
TRACEBACK_FUNCTIONS = frozenset(
    {
        "print_exc",
        "format_exc",
        "print_exception",
        "format_exception",
        "print_tb",
        "format_tb",
        "exception",  # logging.exception
    }
)


class ExceptHandlerAnalyzer(cst.CSTVisitor):
    """Analyzes an except handler body to detect stack trace swallowing."""

    def __init__(self, exception_name: str) -> None:
        self.exception_name = exception_name
        self.has_traceback_call = False
        self.has_raise = False
        self.exception_stringified_at: cst.CSTNode | None = None
        self._nested_exception_depth = 0

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> bool:
        # Track nested exception handlers to avoid false positives
        self._nested_exception_depth += 1
        return True

    def leave_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        self._nested_exception_depth -= 1

    def visit_Raise(self, node: cst.Raise) -> bool:
        # Only count raises in the outermost handler
        if self._nested_exception_depth == 0:
            self.has_raise = True
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func

        # Check for traceback functions
        func_name = None
        if isinstance(func, cst.Name):
            func_name = func.value
        elif isinstance(func, cst.Attribute):
            func_name = func.attr.value

        if func_name in TRACEBACK_FUNCTIONS:
            self.has_traceback_call = True

        # Check for str(exception_var) or repr(exception_var)
        if isinstance(func, cst.Name) and func.value in ("str", "repr"):
            if len(node.args) >= 1:
                arg = node.args[0].value
                if self._is_exception_reference(arg):
                    if self.exception_stringified_at is None:
                        self.exception_stringified_at = node

        return True

    def visit_FormattedStringExpression(self, node: cst.FormattedStringExpression) -> bool:
        """Check for f-string with exception variable like f'{e}'."""
        if self._is_exception_reference(node.expression):
            if self.exception_stringified_at is None:
                self.exception_stringified_at = node
        return True

    def _is_exception_reference(self, node: cst.BaseExpression) -> bool:
        """Check if node references the exception variable."""
        if isinstance(node, cst.Name) and node.value == self.exception_name:
            return True
        # Also check for e.exceptions[0] pattern (ExceptionGroup)
        if isinstance(node, cst.Subscript):
            if isinstance(node.value, cst.Attribute):
                attr = node.value
                if isinstance(attr.value, cst.Name) and attr.value.value == self.exception_name:
                    return True
        return False


@dataclass
class LintVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    file: Path
    errors: list[LintError] = field(default_factory=list)
    _in_test_function: bool = field(default=False, init=False)
    _in_aclosing_context: bool = field(default=False, init=False)
    _is_test_file: bool = field(default=False, init=False)
    _nesting_depth: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._is_test_file = self.file.name.startswith("test_")

    def _get_position(self, node: cst.CSTNode) -> tuple[int, int]:
        pos = cast(CodeRange, self.metadata[PositionProvider][node])
        return pos.start.line, pos.start.column

    def _add_error(self, node: cst.CSTNode, rule: str, message: str) -> None:
        line, column = self._get_position(node)
        self.errors.append(LintError(self.file, line, column, rule, message))

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Rule 1: No class-based tests (only at module level)
        if (
            self._is_test_file
            and self._nesting_depth == 0
            and node.name.value.startswith("Test")
        ):
            self._add_error(
                node,
                "no-class-tests",
                f"Class-based test '{node.name.value}' found. Use module-level test functions instead.",
            )
        self._nesting_depth += 1
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._nesting_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Track if we're inside a test function
        if self._is_test_file and node.name.value.startswith("test_"):
            self._in_test_function = True
        self._nesting_depth += 1

        # Rule 5: No mutable default arguments
        self._check_mutable_defaults(node.params)

        return True

    def _check_mutable_defaults(self, params: cst.Parameters) -> None:
        """Check all parameters for mutable default values."""
        all_params: list[cst.Param] = [
            *params.params,
            *params.posonly_params,
            *params.kwonly_params,
        ]

        for param in all_params:
            if param.default is not None:
                if self._is_mutable_default(param.default):
                    self._add_error(
                        param.default,
                        "mutable-default-argument",
                        f"Mutable default argument for '{param.name.value}'. "
                        "Use None and initialize in function body.",
                    )

    def _is_mutable_default(self, node: cst.BaseExpression) -> bool:
        """Check if a default value is a mutable type."""
        # Empty list []
        if isinstance(node, cst.List):
            return True
        # Empty dict {}
        if isinstance(node, cst.Dict):
            return True
        # Empty set - set() call
        if isinstance(node, cst.Set):
            return True
        # set(), dict(), list() calls
        if isinstance(node, cst.Call):
            func = node.func
            if isinstance(func, cst.Name) and func.value in ("list", "dict", "set"):
                return True
        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._nesting_depth -= 1
        if self._is_test_file and node.name.value.startswith("test_"):
            self._in_test_function = False

    def visit_Import(self, node: cst.Import) -> bool:
        # Rule 2: No imports inside test functions
        if self._in_test_function:
            self._add_error(
                node,
                "no-import-in-test",
                "Import statement inside test function. Move imports to module level.",
            )
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        # Rule 2: No imports inside test functions
        if self._in_test_function:
            self._add_error(
                node,
                "no-import-in-test",
                "Import statement inside test function. Move imports to module level.",
            )
        return True

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> bool:
        # Rule 4: No swallowing exception stack traces (skip test files)
        if not self._is_test_file:
            self._check_exception_handler(node, node.name, node.body)
        return True

    def visit_ExceptStarHandler(self, node: cst.ExceptStarHandler) -> bool:
        # Rule 4: Also check except* handlers (skip test files)
        if not self._is_test_file:
            self._check_exception_handler(node, node.name, node.body)
        return True

    def _check_exception_handler(
        self,
        node: cst.ExceptHandler | cst.ExceptStarHandler,
        name: cst.AsName | None,
        body: cst.BaseSuite,
    ) -> None:
        """Check an exception handler for stack trace swallowing."""
        # Only check handlers that capture the exception with 'as name'
        if name is None:
            return

        # Get the exception variable name
        if not isinstance(name.name, cst.Name):
            return
        exception_name = name.name.value

        # Analyze the handler body by visiting it with the analyzer
        analyzer = ExceptHandlerAnalyzer(exception_name)
        # Walk the body statements using libcst's visit
        if isinstance(body, cst.IndentedBlock):
            for stmt in body.body:
                stmt.visit(analyzer)
        elif isinstance(body, cst.SimpleStatementSuite):
            for stmt in body.body:
                stmt.visit(analyzer)

        # If exception is stringified without traceback preservation, report error
        if analyzer.exception_stringified_at is not None:
            if not analyzer.has_traceback_call and not analyzer.has_raise:
                self._add_error(
                    analyzer.exception_stringified_at,
                    "swallowed-exception-traceback",
                    f"Exception '{exception_name}' converted to string without "
                    "preserving traceback. Call traceback.print_exc() or re-raise.",
                )

    def visit_With(self, node: cst.With) -> bool:
        # Check if any item is aclosing()
        for item in node.items:
            if self._is_aclosing_call(item.item):
                self._in_aclosing_context = True
        return True

    def leave_With(self, node: cst.With) -> None:
        # Check if we're leaving an aclosing context
        for item in node.items:
            if self._is_aclosing_call(item.item):
                self._in_aclosing_context = False

    def _is_aclosing_call(self, node: cst.BaseExpression) -> bool:
        """Check if node is a call to aclosing() or contextlib.aclosing()."""
        if not isinstance(node, cst.Call):
            return False
        func = node.func
        if isinstance(func, cst.Name) and func.value == "aclosing":
            return True
        if isinstance(func, cst.Attribute) and func.attr.value == "aclosing":
            return True
        return False

    def visit_For(self, node: cst.For) -> bool:
        # Rule 3: No async for without aclosing
        if node.asynchronous is not None and not self._in_aclosing_context:
            # Check if the iterator is wrapped in aclosing
            if not self._is_aclosing_call(node.iter):
                self._add_error(
                    node,
                    "async-iter-needs-aclosing",
                    "Async for loop without aclosing(). Wrap the iterator in 'async with aclosing(...):'",
                )
        return True

    def visit_ListComp(self, node: cst.ListComp) -> bool:
        self._check_comprehension_for(node.for_in)
        return True

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        self._check_comprehension_for(node.for_in)
        return True

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        self._check_comprehension_for(node.for_in)
        return True

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        self._check_comprehension_for(node.for_in)
        return True

    def _check_comprehension_for(self, for_in: cst.CompFor) -> None:
        # Check if async comprehension without aclosing
        if for_in.asynchronous is not None and not self._in_aclosing_context:
            if not self._is_aclosing_call(for_in.iter):
                self._add_error(
                    for_in,
                    "async-iter-needs-aclosing",
                    "Async comprehension without aclosing(). Wrap the iterator in aclosing().",
                )
        # Check inner fors (inner_for_in is a single CompFor or None, not a sequence)
        if for_in.inner_for_in is not None:
            self._check_comprehension_for(for_in.inner_for_in)


def lint_file(path: Path) -> list[LintError]:
    """Lint a single file and return any errors."""
    try:
        source = path.read_text()
        tree = cst.parse_module(source)
        wrapper = MetadataWrapper(tree)
        visitor = LintVisitor(file=path)
        wrapper.visit(visitor)
        return visitor.errors
    except cst.ParserSyntaxError as e:
        return [LintError(path, e.raw_line, e.raw_column, "parse-error", str(e))]


def load_cache() -> dict[str, Any]:
    """Load the lint cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, Any]) -> None:
    """Save the lint cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_file_key(path: Path) -> tuple[int, float]:
    """Get the cache key for a file (size, mtime)."""
    stat = path.stat()
    return (stat.st_size, stat.st_mtime)


def file_changed(path: Path, cache: dict[str, Any]) -> bool:
    """Check if a file has changed since it was last cached."""
    path_str = str(path)
    if path_str not in cache:
        return True
    cached = cache[path_str]
    size, mtime = get_file_key(path)
    return cached.get("size") != size or cached.get("mtime") != mtime


def main() -> int:
    """Run linting on all Python files in src and tests."""
    cache = load_cache()
    errors: list[LintError] = []
    files_checked = 0
    files_skipped = 0

    for directory in ["src", "tests"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            if not file_changed(py_file, cache):
                files_skipped += 1
                continue
            files_checked += 1
            file_errors = lint_file(py_file)
            errors.extend(file_errors)
            # Update cache for this file (even if it has errors, so we
            # don't re-check until it changes)
            size, mtime = get_file_key(py_file)
            cache[str(py_file)] = {"size": size, "mtime": mtime}

    # Save cache after checking all files
    save_cache(cache)

    if errors:
        for error in sorted(errors, key=lambda e: (str(e.file), e.line, e.column)):
            print(error)
        print(f"\nFound {len(errors)} error(s)")
        return 1

    if files_skipped > 0 and files_checked == 0:
        print("All custom lint checks passed! (all files cached)")
    elif files_skipped > 0:
        print(f"All custom lint checks passed! ({files_checked} checked, {files_skipped} cached)")
    else:
        print("All custom lint checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
