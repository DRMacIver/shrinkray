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


# =============================================================================
# find_large_integer tests
# =============================================================================


async def test_find_large_integer_zero():
    """Test find_large_integer when result is 0."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n == 0

    # f(0) is assumed True, f(1) is False, so result is 0
    result = await work.find_large_integer(f)
    assert result == 0


async def test_find_large_integer_small():
    """Test find_large_integer with small result (1-3)."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n <= 2

    result = await work.find_large_integer(f)
    assert result == 2


async def test_find_large_integer_at_boundary():
    """Test find_large_integer at boundary of linear scan (4)."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n <= 4

    result = await work.find_large_integer(f)
    assert result == 4


async def test_find_large_integer_large():
    """Test find_large_integer with large result (requires binary search)."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n <= 100

    result = await work.find_large_integer(f)
    assert result == 100


async def test_find_large_integer_very_large():
    """Test find_large_integer with very large result."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n <= 1000

    result = await work.find_large_integer(f)
    assert result == 1000


async def test_find_large_integer_power_of_two():
    """Test find_large_integer with power of 2 result."""
    work = WorkContext(parallelism=1)

    async def f(n):
        return n <= 64

    result = await work.find_large_integer(f)
    assert result == 64


# =============================================================================
# find_first_value tests
# =============================================================================


async def test_find_first_value_found():
    """Test find_first_value when value is found."""
    work = WorkContext(parallelism=1)

    async def is_even(n):
        return n % 2 == 0

    result = await work.find_first_value([1, 3, 4, 5, 6], is_even)
    assert result == 4


async def test_find_first_value_not_found():
    """Test find_first_value raises NotFound when no match."""
    work = WorkContext(parallelism=1)

    async def always_false(n):
        return False

    with pytest.raises(NotFound):
        await work.find_first_value([1, 2, 3], always_false)


async def test_find_first_value_empty():
    """Test find_first_value raises NotFound for empty list."""
    work = WorkContext(parallelism=1)

    async def always_true(n):
        return True

    with pytest.raises(NotFound):
        await work.find_first_value([], always_true)


async def test_find_first_value_parallel():
    """Test find_first_value with parallelism."""
    work = WorkContext(parallelism=2)

    async def is_big(n):
        return n > 5

    result = await work.find_first_value(list(range(10)), is_big)
    assert result == 6


# =============================================================================
# filter tests
# =============================================================================


async def test_filter_basic():
    """Test filter returns matching items."""
    work = WorkContext(parallelism=1)

    async def is_even(n):
        return n % 2 == 0

    async with work.filter([1, 2, 3, 4, 5], is_even) as filtered:
        results = [x async for x in filtered]

    assert results == [2, 4]


async def test_filter_none_match():
    """Test filter with no matching items."""
    work = WorkContext(parallelism=1)

    async def always_false(n):
        return False

    async with work.filter([1, 2, 3], always_false) as filtered:
        results = [x async for x in filtered]

    assert results == []


async def test_filter_parallel():
    """Test filter with parallelism."""
    work = WorkContext(parallelism=2)

    async def is_odd(n):
        return n % 2 == 1

    async with work.filter(list(range(10)), is_odd) as filtered:
        results = [x async for x in filtered]

    assert results == [1, 3, 5, 7, 9]


# =============================================================================
# map empty sequence test
# =============================================================================


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
# Volume enum tests
# =============================================================================


def test_volume_values():
    """Test Volume enum has expected values."""
    assert Volume.quiet == 0
    assert Volume.normal == 1
    assert Volume.verbose == 2
    assert Volume.debug == 3


def test_volume_ordering():
    """Test Volume enum ordering."""
    assert Volume.quiet < Volume.normal < Volume.verbose < Volume.debug


# =============================================================================
# Logging method tests
# =============================================================================


def test_warn():
    """Test warn method calls report."""
    work = WorkContext()
    # warn is a no-op in base implementation but should not raise
    work.warn("test warning")


def test_note():
    """Test note method calls report."""
    work = WorkContext()
    work.note("test note")


def test_debug():
    """Test debug method calls report."""
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
