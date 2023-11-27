import hashlib
from abc import ABC, abstractmethod, abstractproperty
import time
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar, cast

import trio
from attrs import define

from shrinkray.work import WorkContext

S = TypeVar("S")
T = TypeVar("T")


def default_cache_key(value: Any) -> str:
    if not isinstance(value, bytes):
        if not isinstance(value, str):
            value = repr(value)
        value = value.encode("utf-8")

    hex = hashlib.sha1(value).hexdigest()[:8]
    return f"{len(value)}:{hex}"


def shortlex(value: Any) -> Any:
    return (len(value), value)


def default_canonicalise(value: T) -> T:
    return value


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
    cache_hits: int = 0
    cache_misses: int = 0

    reductions: int = 0
    failed_reductions: int = 0

    time_of_last_reduction: float = 0.0

    initial_test_case_size: int = 0
    current_test_case_size: int = 0

    def time_since_last_reduction(self) -> float:
        return time.time() - self.time_of_last_reduction

    def display_stats(self) -> str:
        reduction_percentage = (
            1.0 - self.current_test_case_size / self.initial_test_case_size
        ) * 100

        calls = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / calls if calls > 0 else 0) * 100

        return "\n".join(
            [
                f"Current test case size: {self.current_test_case_size} bytes ({reduction_percentage:.2f}% reduction)",
                f"Time since last reduction: {self.time_since_last_reduction():.2f}s"
                if self.reductions
                else "No reductions yet",
                f"Cache Hit Rate: {cache_hit_rate:.2f}",
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
    def cached_is_interesting(self, test_case: T) -> Optional[bool]:
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

    def canonicalise(self, test_case: T) -> T:
        return test_case


class BasicReductionProblem(ReductionProblem[T]):
    def __init__(
        self,
        initial: T,
        is_interesting: Callable[[T], Awaitable[bool]],
        work: WorkContext,
        cache_key: Callable[[T], str] = default_cache_key,
        sort_key: Callable[[T], Any] = shortlex,
        size: Callable[[T], int] = default_size,
        canonicalise: Callable[[T], T] = default_canonicalise,
        display: Callable[[T], str] = default_display,
    ):
        super().__init__(work=work)
        self.__current = initial
        self.cache_key = cache_key
        self.__sort_key = sort_key
        self.__size = size
        self.__canonicalise = canonicalise
        self.display = display
        self.stats = ReductionStats()
        self.stats.initial_test_case_size = self.size(initial)
        self.stats.current_test_case_size = self.size(initial)

        self.__is_interesting = is_interesting
        self.__on_reduce_callbacks: list[Callable[[T], Awaitable[None]]] = []
        self.__cache = {}
        self.__current = initial
        self.__has_set_up = False

    async def setup(self):
        if self.__has_set_up:
            return
        self.__has_set_up = True
        if not await self.__is_interesting(self.current_test_case):
            raise ValueError(
                f"Initial example ({self.display(self.current_test_case)}) does not satisfy interestingness test."
            )

        self.__cache[self.cache_key(self.__current)] = True

        canonical = self.canonicalise(self.__current)

        if canonical != self.__current:
            if not await self.__is_interesting(canonical):
                self.work.warn(
                    f"Initial example ({self.display(self.__current)}) was interesting, but canonicalised version ({self.display(canonical)}) was not. Disabling canonicalisation."
                )
                self.__canonicalise = default_canonicalise
            else:
                self.__cache[self.cache_key(canonical)] = True

    def canonicalise(self, test_case: T) -> T:
        return self.__canonicalise(test_case)

    def sort_key(self, test_case: T) -> Any:
        return self.__sort_key(test_case)

    def size(self, test_case: T) -> int:
        return self.__size(test_case)

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        """Every time `is_interesting` is called with a successful reduction,
        call `fn` with the new value. Note that these are called outside the lock."""
        self.__on_reduce_callbacks.append(callback)

    def cached_is_interesting(self, value: T) -> Optional[bool]:
        return self.__cache.get(self.cache_key(value))

    async def is_interesting(self, value: T) -> bool:
        """Returns true if this value is interesting.

        Caches and maintains relevant state.

        Note: This function will lock while maintaining state, but
        will *not* lock around calling the underlying interestingness
        test.
        """
        await trio.lowlevel.checkpoint()
        keys = [self.cache_key(value)]
        try:
            self.stats.cache_hits += 1
            return self.__cache[keys[0]]
        except KeyError:
            pass

        value = self.canonicalise(value)
        keys.append(self.cache_key(value))
        try:
            result = self.__cache[keys[-1]]
        except KeyError:
            self.stats.cache_hits -= 1
            self.stats.cache_misses += 1
            result = await self.__is_interesting(value)
            self.stats.failed_reductions += 1
            if result:
                if self.sort_key(value) < self.sort_key(self.current_test_case):
                    self.stats.failed_reductions -= 1
                    self.stats.reductions += 1
                    self.stats.time_of_last_reduction = time.time()
                    self.stats.current_test_case_size = self.size(value)
                    self.__current = value
                    for f in self.__on_reduce_callbacks:
                        await f(value)
        for key in keys:
            self.__cache[key] = result
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

    def cached_is_interesting(self, test_case: T) -> None | bool:
        return self.__problem.cached_is_interesting(self.__dump(test_case))

    async def is_interesting(self, test_case: T) -> bool:
        return await self.__problem.is_interesting(self.__dump(test_case))

    def canonicalise(self, test_case: T) -> T:
        return self.__parse(self.__problem.canonicalise(self.__dump(test_case)))

    def sort_key(self, test_case: T) -> Any:
        if self.__sort_key is not None:
            return self.__sort_key(test_case)
        return self.__problem.sort_key(self.__dump(test_case))

    def size(self, test_case: T) -> int:
        return self.__problem.size(self.__dump(test_case))
