import pytest

from shrinkray.work import NotFound, Volume, WorkContext, parallel_map


async def identity(x: int) -> int:
    return x


@pytest.mark.parametrize("p", [1, 2, 3, 4])
async def test_parallel_map(p: int) -> None:
    input = [1, 2, 3]
    async with parallel_map(input, identity, parallelism=p) as mapped:
        values = [x async for x in mapped]
        print(values)
    assert values == input


@pytest.mark.parametrize("p", [1, 2, 3, 4])
async def test_worker_map(p: int) -> None:
    work = WorkContext(parallelism=p)

    input = range(1000)

    i = 0
    async with work.map(input, identity) as mapped:
        async for x in mapped:
            assert input[i] == x
            i += 1


@pytest.mark.parametrize("p", [1, 2])
async def test_worker_map_empty(p: int) -> None:
    """Test map with empty sequence."""
    work = WorkContext(parallelism=p)

    results = []
    async with work.map([], identity) as mapped:
        async for x in mapped:
            results.append(x)

    assert results == []


# =============================================================================
# find_large_integer tests
# =============================================================================


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(0, id="zero"),  # Edge case: f(0) assumed true, f(1) false
        pytest.param(2, id="small"),  # Small value within linear scan (< 4)
        pytest.param(4, id="boundary"),  # Boundary where linear scan ends
        pytest.param(
            5, id="just_past_boundary"
        ),  # First value requiring exponential probe
        pytest.param(64, id="power_of_two"),  # Power of 2, tests binary search
        pytest.param(100, id="large"),  # Larger value requiring binary search
    ],
)
async def test_find_large_integer(target: int) -> None:
    """Test find_large_integer finds the correct boundary value."""
    work = WorkContext(parallelism=1)

    async def f(n: int) -> bool:
        return n <= target

    result = await work.find_large_integer(f)
    assert result == target


# =============================================================================
# find_first_value tests
# =============================================================================


@pytest.mark.parametrize("p", [1, 2])
async def test_find_first_value_found(p: int) -> None:
    """Test find_first_value when value is found."""
    work = WorkContext(parallelism=p)

    async def is_even(n: int) -> bool:
        return n % 2 == 0

    result = await work.find_first_value([1, 3, 4, 5, 6], is_even)
    assert result == 4


@pytest.mark.parametrize("p", [1, 2])
async def test_find_first_value_not_found(p: int) -> None:
    """Test find_first_value raises NotFound when no match."""
    work = WorkContext(parallelism=p)

    async def always_false(n: int) -> bool:
        return False

    with pytest.raises(NotFound):
        await work.find_first_value([1, 2, 3], always_false)


@pytest.mark.parametrize("p", [1, 2])
async def test_find_first_value_empty(p: int) -> None:
    """Test find_first_value raises NotFound for empty list."""
    work = WorkContext(parallelism=p)

    async def always_true(n: int) -> bool:
        return True

    with pytest.raises(NotFound):
        await work.find_first_value([], always_true)


# =============================================================================
# filter tests
# =============================================================================


@pytest.mark.parametrize("p", [1, 2])
async def test_filter_basic(p: int) -> None:
    """Test filter returns matching items."""
    work = WorkContext(parallelism=p)

    async def is_even(n: int) -> bool:
        return n % 2 == 0

    async with work.filter([1, 2, 3, 4, 5], is_even) as filtered:
        results = [x async for x in filtered]

    assert results == [2, 4]


@pytest.mark.parametrize("p", [1, 2])
async def test_filter_none_match(p: int) -> None:
    """Test filter with no matching items."""
    work = WorkContext(parallelism=p)

    async def always_false(n: int) -> bool:
        return False

    async with work.filter([1, 2, 3], always_false) as filtered:
        results = [x async for x in filtered]

    assert results == []


# =============================================================================
# Volume enum tests
# =============================================================================


def test_volume_ordering():
    """Test Volume enum ordering for comparison operations."""
    assert Volume.quiet < Volume.normal < Volume.verbose < Volume.debug


# =============================================================================
# Logging method tests
# =============================================================================


def test_warn():
    """Test warn method calls report without raising."""
    work = WorkContext()
    work.warn("test warning")


def test_note():
    """Test note method calls report without raising."""
    work = WorkContext()
    work.note("test note")


def test_debug():
    """Test debug method calls report without raising."""
    work = WorkContext()
    work.debug("test debug")


def test_report():
    """Test report method is a no-op."""
    work = WorkContext()
    work.report("test", Volume.normal)


# =============================================================================
# WorkContext initialization tests
# =============================================================================


def test_workcontext_defaults():
    """Test WorkContext default values."""
    work = WorkContext()
    assert work.parallelism == 1
    assert work.volume == Volume.normal
    assert work.random is not None


def test_workcontext_custom_random():
    """Test WorkContext with custom random."""
    from random import Random

    rnd = Random(42)
    work = WorkContext(random=rnd)
    assert work.random is rnd


def test_workcontext_custom_parallelism():
    """Test WorkContext with custom parallelism."""
    work = WorkContext(parallelism=4)
    assert work.parallelism == 4


def test_workcontext_custom_volume():
    """Test WorkContext with custom volume."""
    work = WorkContext(volume=Volume.debug)
    assert work.volume == Volume.debug
