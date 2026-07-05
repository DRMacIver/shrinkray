"""Tests for wiring external reducers into ShrinkRay and the state layer."""

import os

import trio

from shrinkray.problem import BasicReductionProblem, sort_key_for_initial
from shrinkray.reducer import DirectoryShrinkRay, ShrinkRay
from shrinkray.work import WorkContext


def make_shrinkray(initial: bytes, **kwargs) -> ShrinkRay:
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return True

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=initial,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(initial),
    )
    return ShrinkRay(target=problem, **kwargs)


PYTHON_SOURCE = b"def f(x):\n    return x\n"
NON_PYTHON = b"this is (not python at all;;;\n"


# === build_external_reducer_passes ===


def test_python_reducer_added_for_python_input():
    reducer = make_shrinkray(PYTHON_SOURCE)
    passes = reducer.build_external_reducer_passes()
    assert [p.__name__ for p in passes] == ["python"]


def test_python_reducer_not_added_when_disabled():
    reducer = make_shrinkray(PYTHON_SOURCE, python_reducer=False)
    assert reducer.build_external_reducer_passes() == []


def test_python_reducer_not_added_for_non_python():
    reducer = make_shrinkray(NON_PYTHON)
    assert reducer.build_external_reducer_passes() == []


def test_user_reducers_added_in_order():
    reducer = make_shrinkray(
        NON_PYTHON, external_reducers=[["cmd-a"], ["cmd-b", "arg"]]
    )
    passes = reducer.build_external_reducer_passes()
    assert [p.__name__ for p in passes] == ["reduce-with-0", "reduce-with-1"]


def test_python_and_user_reducers_combined():
    reducer = make_shrinkray(PYTHON_SOURCE, external_reducers=[["cmd-a"]])
    passes = reducer.build_external_reducer_passes()
    assert [p.__name__ for p in passes] == ["python", "reduce-with-0"]


def test_external_passes_registered_in_pass_lists():
    reducer = make_shrinkray(PYTHON_SOURCE)
    names = [p.__name__ for p in reducer.great_passes]
    assert "python" in names
    initial_names = [p.__name__ for p in reducer.initial_cuts]
    assert "python" in initial_names


# === _log_file_for ===


def test_log_file_for_returns_none_without_dir():
    reducer = make_shrinkray(PYTHON_SOURCE, reducer_log_dir=None)
    assert reducer._log_file_for("python") is None


def test_log_file_for_creates_dir_and_path(tmp_path):
    log_dir = tmp_path / "logs"
    reducer = make_shrinkray(PYTHON_SOURCE, reducer_log_dir=str(log_dir))
    path = reducer._log_file_for("python")
    assert path == os.path.join(str(log_dir), "reducer-python.log")
    assert log_dir.is_dir()


# === DirectoryShrinkRay forwards a per-key log dir ===


def test_directory_reducer_uses_per_key_log_dir(tmp_path):
    initial = {"a.txt": b"KEEP\nremove me\n"}

    async def is_interesting(tc: dict[str, bytes]) -> bool:
        await trio.lowlevel.checkpoint()
        return b"KEEP" in tc.get("a.txt", b"")

    problem: BasicReductionProblem[dict[str, bytes]] = BasicReductionProblem(
        initial=initial,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(initial),
        size=lambda tc: sum(len(v) for v in tc.values()),
    )
    reducer = DirectoryShrinkRay(
        target=problem,
        python_reducer=False,
        reducer_log_dir=str(tmp_path / "logs"),
    )
    trio.run(reducer.run)
    # The interesting content survives; the rest is reduced away.
    assert problem.current_test_case == {"a.txt": b"KEEP"}
