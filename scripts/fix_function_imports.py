#!/usr/bin/env python3
"""Fix function-level imports by moving them to module level.

Uses libcst to find import statements inside function bodies and moves them
to the top of the file, after existing imports. Relies on ruff to sort
and deduplicate imports afterward.

Respects # noqa: no-import-in-function suppression comments.
"""

import sys
from pathlib import Path

import libcst as cst


class ImportHoister(cst.CSTTransformer):
    """Moves import statements from inside functions to module level."""

    def __init__(self) -> None:
        self.function_depth = 0
        self.collected_imports: list[cst.Import | cst.ImportFrom] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.function_depth += 1
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.function_depth -= 1
        return updated_node

    def _has_noqa(self, stmt: cst.SimpleStatementLine) -> bool:
        """Check if the statement has a no-import-in-function noqa comment.

        Renders the statement to code to catch comments in any position,
        including inside parenthesized multi-line imports.
        """
        code = cst.Module(body=[stmt]).code
        return "no-import-in-function" in code

    def _is_import_line(self, stmt: cst.BaseStatement) -> bool:
        """Check if a statement line contains only imports and should be moved."""
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        if not all(isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body):
            return False
        if self._has_noqa(stmt):
            return False
        return True

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        if self.function_depth == 0:
            return updated_node

        new_body: list[cst.BaseStatement] = []
        changed = False
        for stmt in updated_node.body:
            if self._is_import_line(stmt):
                assert isinstance(stmt, cst.SimpleStatementLine)
                for small_stmt in stmt.body:
                    assert isinstance(small_stmt, (cst.Import, cst.ImportFrom))
                    clean = small_stmt.with_changes(
                        semicolon=cst.MaybeSentinel.DEFAULT
                    )
                    self.collected_imports.append(clean)
                changed = True
            else:
                new_body.append(stmt)

        if not changed:
            return updated_node

        if not new_body:
            new_body = [cst.SimpleStatementLine(body=[cst.Pass()])]

        return updated_node.with_changes(body=new_body)

    def _find_import_insertion_point(self, module: cst.Module) -> int:
        """Find the index after the last top-level import statement."""
        last_import_idx = -1
        for i, stmt in enumerate(module.body):
            if isinstance(stmt, cst.SimpleStatementLine) and any(
                isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
            ):
                last_import_idx = i
            elif (
                isinstance(stmt, cst.If)
                and isinstance(stmt.test, cst.Name)
                and stmt.test.value == "TYPE_CHECKING"
            ):
                last_import_idx = i

        if last_import_idx >= 0:
            return last_import_idx + 1

        # No imports found - insert after module docstring if present
        if module.body:
            first = module.body[0]
            if isinstance(first, cst.SimpleStatementLine) and len(first.body) == 1:
                expr = first.body[0]
                if isinstance(expr, cst.Expr) and isinstance(
                    expr.value,
                    (
                        cst.SimpleString,
                        cst.ConcatenatedString,
                        cst.FormattedString,
                    ),
                ):
                    return 1

        return 0

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        if not self.collected_imports:
            return updated_node

        insert_idx = self._find_import_insertion_point(updated_node)

        new_body = list(updated_node.body)
        for i, imp in enumerate(self.collected_imports):
            line = cst.SimpleStatementLine(body=[imp])
            new_body.insert(insert_idx + i, line)

        return updated_node.with_changes(body=new_body)


def fix_file(path: Path) -> bool:
    """Fix function-level imports in a single file. Returns True if changed."""
    source = path.read_text()
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return False

    transformer = ImportHoister()
    new_tree = tree.visit(transformer)

    if not transformer.collected_imports:
        return False

    new_source = new_tree.code
    if new_source == source:
        return False

    path.write_text(new_source)
    return True


def main() -> int:
    files_fixed = 0

    for directory in ["src", "tests"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for py_file in sorted(dir_path.rglob("*.py")):
            if fix_file(py_file):
                print(f"Fixed: {py_file}")
                files_fixed += 1

    if files_fixed:
        print(f"\nFixed imports in {files_fixed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
