from abc import ABC, abstractmethod, abstractproperty
import time
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar, cast
import attrs
from humanize import naturalsize, precisedelta
import trio
from attrs import define

from shrinkray.work import WorkContext

S = TypeVar("S")
T = TypeVar("T")


def shortlex(value: Any) -> Any:
    return (len(value), value)


def default_display(value: Any) -> str:
    r = repr(value)
    if len(r) < 50:
        return f"{r} (size {len(value)})"
    return f"value of size {len(value)}"


def default_size(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


@define
class ReductionStats:
    reductions: int = 0
    failed_reductions: int = 0

    calls: int = 0
    interesting_calls: int = 0
    wasted_interesting_calls: int = 0

    time_of_last_reduction: float = 0.0
    start_time: float = attrs.Factory(time.time)

    initial_test_case_size: int = 0
    current_test_case_size: int = 0

    def time_since_last_reduction(self) -> float:
        return time.time() - self.time_of_last_reduction

    def display_stats(self) -> str:
        runtime = time.time() - self.start_time
        if self.reductions > 0:
            reduction_percentage = (
                1.0 - self.current_test_case_size / self.initial_test_case_size
            ) * 100
            reduction_rate = (
                self.initial_test_case_size - self.current_test_case_size
            ) / runtime
            reduction_msg = (
                f"Current test case size: {naturalsize(self.current_test_case_size)} "
                f"({reduction_percentage:.2f}% reduction, {naturalsize(reduction_rate)} / second)"
            )
        else:
            reduction_msg = (
                f"Current test case size: {self.current_test_case_size} bytes"
            )

        return "\n".join(
            [
                reduction_msg,
                f"Total runtime: {precisedelta(runtime)}",
                (
                    f"Calls to interestingness test: {self.calls} ({self.calls / runtime:.2f} calls / second, "
                    f"{self.interesting_calls / self.calls * 100.0:.2f}% interesting, "
                    f"{self.wasted_interesting_calls / self.calls * 100:.2f}% wasted)"
                )
                if self.calls > 0
                else "Not yet called interestingness test",
                f"Time since last reduction: {self.time_since_last_reduction():.2f}s ({self.reductions / runtime:.2f} reductions / second)"
                if self.reductions
                else "No reductions yet",
            ]
        )


@define
class ReductionProblem(Generic[T], ABC):
    work: WorkContext

    def __attrs_post_init__(self) -> None:
        self.__view_cache: dict[Any, ReductionProblem[Any]] = {}

    def view(
        self, format: "Format[T, S] | type[Format[T, S]]"
    ) -> "ReductionProblem[S]":
        try:
            return cast(ReductionProblem[S], self.__view_cache[format])
        except KeyError:
            pass

        concrete_format: Format[T, S] = format() if isinstance(format, type) else format

        result: View[T, S] = View(
            problem=self,
            work=self.work,
            dump=concrete_format.dumps,
            parse=concrete_format.parse,
        )

        return cast(ReductionProblem[S], self.__view_cache.setdefault(format, result))

    async def setup(self):
        pass

    @abstractproperty
    def current_test_case(self) -> T:
        ...

    @abstractmethod
    async def is_interesting(self, test_case: T) -> bool:
        pass

    @abstractmethod
    def sort_key(self, test_case: T) -> Any:
        ...

    @abstractmethod
    def size(self, test_case: T) -> int:
        return len(test_case)  # type: ignore

    @property
    def current_size(self) -> int:
        return self.size(self.current_test_case)

    def backtrack(self, new_test_case: T) -> "ReductionProblem[T]":
        return BasicReductionProblem(
            initial=new_test_case,
            is_interesting=self.is_interesting,
            work=self.work,
            sort_key=self.sort_key,
            size=self.size,
            display=self.display,
        )


class InvalidInitialExample(ValueError):
    pass


class BasicReductionProblem(ReductionProblem[T]):
    def __init__(
        self,
        initial: T,
        is_interesting: Callable[[T], Awaitable[bool]],
        work: WorkContext,
        sort_key: Callable[[T], Any] = shortlex,
        size: Callable[[T], int] = default_size,
        display: Callable[[T], str] = default_display,
        stats: Optional[ReductionStats] = None,
    ):
        super().__init__(work=work)
        self.__current = initial
        self.__sort_key = sort_key
        self.__size = size
        self.display = display
        if stats is None:
            self.stats = ReductionStats()
            self.stats.initial_test_case_size = self.size(initial)
            self.stats.current_test_case_size = self.size(initial)
        else:
            self.stats = stats

        self.__is_interesting = is_interesting
        self.__on_reduce_callbacks: list[Callable[[T], Awaitable[None]]] = []
        self.__current = initial
        self.__has_set_up = False

    async def setup(self):
        if self.__has_set_up:
            return
        self.__has_set_up = True
        if not await self.__is_interesting(self.current_test_case):
            raise InvalidInitialExample(
                f"Initial example ({self.display(self.current_test_case)}) does not satisfy interestingness test."
            )

    def sort_key(self, test_case: T) -> Any:
        return self.__sort_key(test_case)

    def size(self, test_case: T) -> int:
        return self.__size(test_case)

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        """Every time `is_interesting` is called with a successful reduction,
        call `fn` with the new value. Note that these are called outside the lock."""
        self.__on_reduce_callbacks.append(callback)

    async def is_interesting(self, value: T) -> bool:
        """Returns true if this value is interesting."""
        await trio.lowlevel.checkpoint()
        if value == self.current_test_case:
            return True
        result = await self.__is_interesting(value)
        self.stats.failed_reductions += 1
        self.stats.calls += 1
        if result:
            self.stats.interesting_calls += 1
            if self.sort_key(value) < self.sort_key(self.current_test_case):
                self.stats.failed_reductions -= 1
                self.stats.reductions += 1
                self.stats.time_of_last_reduction = time.time()
                self.stats.current_test_case_size = self.size(value)
                self.__current = value
                for f in self.__on_reduce_callbacks:
                    await f(value)
            else:
                self.stats.wasted_interesting_calls += 1
        return result

    @property
    def current_test_case(self) -> T:
        return self.__current


class ParseError(Exception):
    pass


class Format(Generic[S, T], ABC):
    @property
    def name(self) -> str:
        return repr(self)

    @abstractmethod
    def parse(self, input: S) -> T:
        ...

    @abstractmethod
    def dumps(self, input: T) -> S:
        ...


class View(ReductionProblem[T], Generic[S, T]):
    def __init__(
        self,
        problem: ReductionProblem[S],
        parse: Callable[[S], T],
        dump: Callable[[T], S],
        work: Optional[WorkContext] = None,
        sort_key: Optional[Callable[[T], Any]] = None,
    ):
        super().__init__(work=work or problem.work)
        self.__problem = problem
        self.__parse = parse
        self.__dump = dump
        self.__sort_key = sort_key
        self.stats = problem.stats

        current = problem.current_test_case
        self.__prev = current
        self.__current = parse(current)

    @property
    def current_test_case(self) -> T:
        current = self.__problem.current_test_case
        if current != self.__prev:
            self.__prev = current
            new_value = self.__parse(current)
            if self.__sort_key is None or self.__sort_key(new_value) < self.__sort_key(
                self.__current
            ):
                self.__current = new_value
        return self.__current

    async def is_interesting(self, test_case: T) -> bool:
        return await self.__problem.is_interesting(self.__dump(test_case))

    def sort_key(self, test_case: T) -> Any:
        if self.__sort_key is not None:
            return self.__sort_key(test_case)
        return self.__problem.sort_key(self.__dump(test_case))

    def size(self, test_case: T) -> int:
        return self.__problem.size(self.__dump(test_case))
