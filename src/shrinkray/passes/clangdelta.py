import os
import subprocess
from glob import glob
from shutil import which
from tempfile import NamedTemporaryFile

import trio

from shrinkray.passes.definitions import ReductionPump
from shrinkray.problem import ReductionProblem
from shrinkray.work import NotFound


C_FILE_EXTENSIONS = (".c", ".cpp", ".h", ".hpp", ".cxx", ".cc")


def find_clang_delta():
    clang_delta = which("clang_delta") or ""
    if not clang_delta:
        possible_locations = glob(
            "/opt/homebrew//Cellar/creduce/*/libexec/clang_delta"
        ) + glob("/usr/libexec/clang_delta")
        if possible_locations:
            clang_delta = max(possible_locations)
    return clang_delta


TRANSFORMATIONS: list[str] = [
    "aggregate-to-scalar",
    "binop-simplification",
    "callexpr-to-value",
    "class-template-to-class",
    "combine-global-var",
    "combine-local-var",
    "copy-propagation",
    "empty-struct-to-int",
    "expression-detector",
    "instantiate-template-param",
    "instantiate-template-type-param-to-int",
    "lift-assignment-expr",
    "local-to-global",
    "move-function-body",
    "move-global-var",
    "param-to-global",
    "param-to-local",
    "reduce-array-dim",
    "reduce-array-size",
    "reduce-class-template-param",
    "reduce-pointer-level",
    "reduce-pointer-pairs",
    "remove-addr-taken",
    "remove-array",
    "remove-base-class",
    "remove-ctor-initializer",
    "remove-enum-member-value",
    "remove-namespace",
    "remove-nested-function",
    "remove-pointer",
    "remove-trivial-base-template",
    "remove-unresolved-base",
    "remove-unused-enum-member",
    "remove-unused-field",
    "remove-unused-function",
    "remove-unused-outer-class",
    "remove-unused-var",
    "rename-class",
    "rename-cxx-method",
    "rename-fun",
    "rename-param",
    "rename-var",
    "replace-array-access-with-index",
    "replace-array-index-var",
    "replace-callexpr",
    "replace-class-with-base-template-spec",
    "replace-dependent-name",
    "replace-dependent-typedef",
    "replace-derived-class",
    "replace-function-def-with-decl",
    "replace-one-level-typedef-type",
    "replace-simple-typedef",
    "replace-undefined-function",
    "return-void",
    "simple-inliner",
    "simplify-callexpr",
    "simplify-comma-expr",
    "simplify-dependent-typedef",
    "simplify-if",
    "simplify-nested-class",
    "simplify-recursive-template-instantiation",
    "simplify-struct",
    "simplify-struct-union-decl",
    "template-arg-to-int",
    "template-non-type-arg-to-int",
    "unify-function-decl",
    "union-to-struct",
    "vector-to-array",
]


class ClangDelta:
    def __init__(self, path: str):
        self.path_to_exec = path

        self.transformations: list[str] = TRANSFORMATIONS

    def __validate_transformation(self, transformation: str) -> None:
        if transformation not in self.transformations:
            raise ValueError(f"Invalid transformation {transformation}")

    async def query_instances(self, transformation: str, data: bytes) -> int:
        self.__validate_transformation(transformation)
        with NamedTemporaryFile(suffix=".cpp", delete_on_close=False) as tmp:
            tmp.write(data)
            tmp.close()

            try:
                results = (
                    await trio.run_process(
                        [
                            self.path_to_exec,
                            f"--query-instances={transformation}",
                            tmp.name,
                        ],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                ).stdout
            except subprocess.CalledProcessError as e:
                msg = (e.stdout + e.stderr).strip()
                if b"Assertion failed" in msg:
                    return 0
                else:
                    raise ClangDeltaError(msg)
            finally:
                os.unlink(tmp.name)

            prefix = b"Available transformation instances:"
            assert results.startswith(prefix)
            return int(results[len(prefix) :].strip().decode("ascii"))

    async def apply_transformation(
        self, transformation: str, counter: int, data: bytes
    ) -> bytes:
        self.__validate_transformation(transformation)
        with NamedTemporaryFile(suffix=".cpp", delete_on_close=False) as tmp:
            tmp.write(data)
            tmp.close()

            try:
                return (
                    await trio.run_process(
                        [
                            self.path_to_exec,
                            f"--transformation={transformation}",
                            f"--counter={int(counter)}",
                            tmp.name,
                        ],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                ).stdout
            except subprocess.CalledProcessError as e:
                if (
                    e.stdout.strip()
                    == b"Error: No modification to the transformed program!"
                ):
                    return data
                elif b"Assertion failed" in e.stderr.strip():
                    return data
                else:
                    raise ClangDeltaError(e.stdout + e.stderr)
            finally:
                os.unlink(tmp.name)


class ClangDeltaError(Exception):
    def __init__(self, message):
        assert b"Assertion failed" not in message, message
        super().__init__(message)


def clang_delta_pump(
    clang_delta: ClangDelta, transformation: str
) -> ReductionPump[bytes]:
    async def apply(problem: ReductionProblem[bytes]) -> bytes:
        target = problem.current_test_case
        assert target is not None
        try:
            n = await clang_delta.query_instances(transformation, target)
        except ClangDeltaError:
            return target
        i = 1
        while i <= n:

            async def can_apply(j: int) -> bool:
                attempt = await clang_delta.apply_transformation(
                    transformation, j, target
                )
                assert attempt is not None
                if attempt == target:
                    return False
                return await problem.is_interesting(attempt)

            try:
                i = await problem.work.find_first_value(range(i, n + 1), can_apply)
            except NotFound:
                break
            # Note: ClangDeltaError from can_apply would be wrapped in an ExceptionGroup
            # by trio's nursery, so we can't catch it here. apply_transformation already
            # handles assertion failures by returning original data, so errors will
            # propagate up and abort the pump.

            target = await clang_delta.apply_transformation(transformation, i, target)
            assert target is not None
            n = await clang_delta.query_instances(transformation, target)
        return target

    apply.__name__ = f"clang_delta({transformation})"

    return apply


def clang_delta_pumps(clang_delta: ClangDelta) -> list[ReductionPump[bytes]]:
    return [
        clang_delta_pump(clang_delta, transformation)
        for transformation in clang_delta.transformations
    ]
