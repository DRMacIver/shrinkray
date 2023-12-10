from lib2to3.fixes.fix_next import is_assign_target
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

from shrinkray.problem import ReductionProblem
from shrinkray.reducer import ReductionPass
from shrinkray.work import NotFound


class ClangDelta:
    def __init__(self, path):
        self.path_to_exec = path

        self.transformations = [
            line.strip()
            for line in subprocess.check_output(
                [path, "--transformations"], universal_newlines=True
            ).splitlines()
            if line.strip()
        ]

    def __validate_transformation(self, transformation):
        if transformation not in self.transformations:
            raise ValueError(f"Invalid transformation {transformation}")

    def query_instances(self, transformation: str, data: bytes) -> int:
        self.__validate_transformation(transformation)
        with NamedTemporaryFile(suffix=".cpp", delete_on_close=False) as tmp:
            tmp.write(data)
            tmp.close()

            try:
                results = subprocess.check_output(
                    [
                        self.path_to_exec,
                        f"--query-instances={transformation}",
                        tmp.name,
                    ],
                    stderr=subprocess.PIPE,
                )
            except subprocess.SubprocessError as e:
                msg = (e.stdout + e.stderr).strip()
                if msg == b"Error: Unsupported file type!":
                    raise ValueError("Not a C or C++ test case")
                elif b"Assertion failed" in msg:
                    return 0
                else:
                    raise ClangDeltaError(msg)
            finally:
                os.unlink(tmp.name)

            prefix = b"Available transformation instances:"
            assert results.startswith(prefix)
            return int(results[len(prefix) :].strip().decode("ascii"))

    def apply_transformation(
        self, transformation: str, counter: int, data: bytes
    ) -> bytes:
        self.__validate_transformation(transformation)
        with NamedTemporaryFile(suffix=".cpp", delete_on_close=False) as tmp:
            tmp.write(data)
            tmp.close()

            try:
                return subprocess.check_output(
                    [
                        self.path_to_exec,
                        f"--transformation={transformation}",
                        f"--counter={int(counter)}",
                        tmp.name,
                    ],
                    stderr=subprocess.PIPE,
                )
            except subprocess.SubprocessError as e:
                if e.stdout.strip() == b"Error: Unsupported file type!":
                    raise ValueError("Not a C or C++ test case")
                elif (
                    e.stdout.strip()
                    == b"Error: No modification to the transformed program!"
                ):
                    return data
                else:
                    raise ClangDeltaError(e.stdout + e.stderr)
            finally:
                os.unlink(tmp.name)


class ClangDeltaError(Exception):
    pass


def clang_delta_pump(clang_delta: ClangDelta, transformation: str) -> bytes:
    async def apply(problem: ReductionProblem[bytes]):
        target = problem.current_test_case
        try:
            n = clang_delta.query_instances(transformation, target)
        except ValueError:
            return
        i = 1
        while i <= n:

            async def can_apply(j):
                attempt = clang_delta.apply_transformation(transformation, j, target)
                if attempt == target:
                    return False
                return await problem.is_interesting(attempt)

            try:
                i = await problem.work.find_first_value(range(i, n + 1), can_apply)
            except NotFound:
                break
            except ClangDeltaError as e:
                # Clang delta has a large number of internal assertions that you can trigger
                # if you feed it bad enough C++. We solve this problem by ignoring it.
                if "Assertion failed" in e.args[0]:
                    return target

            target = clang_delta.apply_transformation(transformation, i, target)
            n = clang_delta.query_instances(transformation, target)
        return target

    apply.__name__ = f"clang_delta({transformation})"

    return apply


def clang_delta_pumps(clang_delta: ClangDelta) -> list[ReductionPass[bytes]]:
    return [
        clang_delta_pump(clang_delta, transformation)
        for transformation in clang_delta.transformations
    ]
