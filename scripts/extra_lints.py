#!/usr/bin/env python3
"""Custom linting rules using libcst.

Rules:
1. No class-based tests in test files
2. No imports inside test functions
3. No use of async iterators without contextlib.aclosing
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider


@dataclass
class LintError:
    file: Path
    line: int
    column: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.rule}: {self.message}"


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
        pos = self.metadata[PositionProvider][node]
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
        return True

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


def main() -> int:
    """Run linting on all Python files in src and tests."""
    errors: list[LintError] = []

    for directory in ["src", "tests"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            errors.extend(lint_file(py_file))

    if errors:
        for error in sorted(errors, key=lambda e: (str(e.file), e.line, e.column)):
            print(error)
        print(f"\nFound {len(errors)} error(s)")
        return 1

    print("All custom lint checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
