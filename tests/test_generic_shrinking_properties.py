from random import Random

import hypothesmith
import trio
from hypothesis import Phase, assume, example, given, note, settings, strategies as st
from hypothesis.errors import Frozen, StopTest

from shrinkray.passes.python import is_python
from shrinkray.problem import BasicReductionProblem, default_sort_key
from shrinkray.reducer import ShrinkRay
from shrinkray.work import Volume, WorkContext

from tests.helpers import assert_no_blockers, assert_reduces_to, direct_reductions


def tidy_python_example(s):
    results = []
    for line in s.splitlines():
        line, *_ = line.split("#")
        line = line.strip()
        if line:
            results.append(line)
    output = "\n".join(results)
    if output.startswith('"""'):
        output = output[3:]
        i = output.index('"""')
        output = output[i + 3 :]
    return output.strip() + "\n"


POTENTIAL_BLOCKERS = [
    b"\n",
    b"s",
    b"0",
    b"()",
    b"[]",
    b"\t\t",
    b"#\x00",
    b"#\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    b"\t\t\t\t\t\t\t\t\t\t\t\t",
    # b"from\t\t\t.\t\t\t\t\t\timport\ta\t\t\t\t\t\t\t\t\t\t\nclass\t\to\t\t\t\t\t\t\t:\n\tdef\t\t\t\t\ta\t(\t\t\t\t\t\t\t\t\t)\t\t\t\t\t\t\t:...\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
    # "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
]

python_files = st.builds(
    lambda s, e: s.encode(e),
    hypothesmith.from_grammar(),
    st.sampled_from(("utf-8",)),
)
test_cases = (python_files).filter(lambda b: 1 < len(b) <= 1000)


common_settings = settings(deadline=None, max_examples=10, report_multiple_bugs=False)


@common_settings
@given(
    initial=test_cases,
    rnd=st.randoms(use_true_random=True),
    is_interesting_sync=st.functions(
        returns=st.booleans(), pure=True, like=lambda test_case: False
    ),
    data=st.data(),
    parallelism=st.integers(1, 1),
)
async def test_can_shrink_arbitrary_problems(
    initial, rnd, data, parallelism, is_interesting_sync
):
    is_interesting_cache = {}

    initial_is_python = is_python(initial)

    async def is_interesting(test_case: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        if test_case == initial:
            return True
        elif not test_case:
            return False
        try:
            return is_interesting_cache[test_case]
        except KeyError:
            pass
        if initial_is_python and not is_python(test_case):
            result = False
        else:
            try:
                result = is_interesting_sync(test_case)
            except (StopTest, Frozen):  # pragma: no cover
                result = False
        is_interesting_cache[test_case] = result
        if result:
            note(f"{test_case} is interesting")
        return result

    work = WorkContext(
        random=rnd,
        volume=Volume.quiet,
        parallelism=parallelism,
    )
    problem = BasicReductionProblem(
        initial=initial, is_interesting=is_interesting, work=work
    )

    reducer = ShrinkRay(problem)

    with trio.move_on_after(10) as cancel_scope:
        await reducer.run()
    assert not cancel_scope.cancelled_caught

    assert len(problem.current_test_case) <= len(initial)


@settings(common_settings, phases=[Phase.explicit])
@example(b":\x80", 1)
@example(b"#\x80", 1)
@example(
    initial=b"..........................................................................................................................................................................................................................................................................................................................................................................................",
    parallelism=2,
)
@given(
    initial=test_cases,
    parallelism=st.integers(1, 10),
)
async def test_can_fail_to_shrink_arbitrary_problems(initial, parallelism):
    async def is_interesting(test_case: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return test_case == initial

    work = WorkContext(
        random=Random(0),
        volume=Volume.quiet,
        parallelism=parallelism,
    )
    problem = BasicReductionProblem(
        initial=initial, is_interesting=is_interesting, work=work
    )

    reducer = ShrinkRay(problem)

    with trio.move_on_after(10) as cancel_scope:
        await reducer.run()
    assert not cancel_scope.cancelled_caught

    assert problem.current_test_case == initial


@example(b"from p import*", 1)
@example(
    b'from types import MethodType\ndef is_hypothesis_test(test):\nif isinstance(test, MethodType):\nreturn is_hypothesis_test(test.__func__)\nreturn getattr(test, "is_hypothesis_test", False)',
    1,
)
@example(b"''", 1)
@example(b"# Hello world", 1)
@example(b"\x00\x01", 1)
@common_settings
@given(
    initial=test_cases,
    parallelism=st.integers(1, 10),
)
async def test_can_succeed_at_shrinking_arbitrary_problems(initial, parallelism):
    initial_is_python = is_python(initial)

    async def is_interesting(test_case: bytes) -> bool:
        if initial_is_python and not is_python(test_case):
            return False
        await trio.lowlevel.checkpoint()
        return len(test_case) > 0

    work = WorkContext(
        random=Random(0),
        volume=Volume.quiet,
        parallelism=parallelism,
    )
    problem = BasicReductionProblem(
        initial=initial, is_interesting=is_interesting, work=work
    )

    reducer = ShrinkRay(problem)

    with trio.move_on_after(10) as cancel_scope:
        await reducer.run()
    assert not cancel_scope.cancelled_caught

    assert len(problem.current_test_case) == 1


def test_no_blockers():
    assert_no_blockers(
        potential_blockers=POTENTIAL_BLOCKERS,
        is_interesting=is_python,
        lower_bounds=[1, 2, 5, 10],
    )


@common_settings
@given(st.binary(), st.data())
def test_always_reduces_to_each_direct_reduction(origin, data):
    reductions = sorted(direct_reductions(origin), key=default_sort_key, reverse=True)

    assume(reductions)

    target = data.draw(st.sampled_from(reductions))

    assert_reduces_to(origin=origin, target=target, language_restrictions=False)


@common_settings
@given(st.binary(), st.integers(2, 8), st.data())
def test_parallelism_never_prevents_reduction(origin, parallelism, data):
    reductions = sorted(direct_reductions(origin), key=default_sort_key, reverse=True)

    assume(reductions)

    parallel_reductions = sorted(
        direct_reductions(origin, parallelism=parallelism),
        key=default_sort_key,
        reverse=True,
    )

    assert set(reductions).issubset(parallel_reductions)

    target = data.draw(st.sampled_from(reductions))

    assert_reduces_to(
        origin=origin,
        target=target,
        parallelism=parallelism,
        language_restrictions=False,
    )
