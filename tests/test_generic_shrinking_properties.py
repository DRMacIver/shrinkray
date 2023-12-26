from shrinkray.reducer import ShrinkRay
import trio
from hypothesis import strategies as st, given, settings, note
from shrinkray.work import WorkContext, Volume
from shrinkray.problem import BasicReductionProblem
from hypothesis.errors import StopTest

test_cases = st.binary(min_size=1)


@given(
    initial=test_cases,
    rnd=st.randoms(use_true_random=False),
    data=st.data(),
    parallelism=st.integers(1, 1),
)
def test_can_shrink_arbitrary_problems(initial, rnd, data, parallelism):
    is_interesting_cache = {}
    finished = False

    async def is_interesting(test_case: bytes) -> bool:
        nonlocal finished
        await trio.lowlevel.checkpoint()
        if test_case == initial:
            return True
        if not test_case:
            return False
        try:
            return is_interesting_cache[test_case]
        except KeyError:
            if finished:
                result = False
            else:
                try:
                    result = data.draw(st.booleans())
                except StopTest:
                    result = False
                    finished = True
            if result:
                note(f"{test_case} is interesting")
            is_interesting_cache[test_case] = result

    work = WorkContext(
        random=rnd,
        volume=Volume.quiet,
        parallelism=parallelism,
    )
    problem = BasicReductionProblem(
        initial=initial, is_interesting=is_interesting, work=work
    )

    reducer = ShrinkRay(problem)

    async def run_for_test():
        with trio.move_on_after(30) as cancel_scope:
            await reducer.run()
        assert not cancel_scope.cancelled_caught

    try:
        trio.run(run_for_test)
    except* StopTest as e:
        raise e

    assert len(reducer.current_test_case) <= len(initial)
