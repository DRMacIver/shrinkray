import hashlib
import os
import sys
import threading
from abc import ABC, abstractmethod, abstractproperty
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum, IntEnum, auto
from itertools import islice
from random import Random
from threading import Lock
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    cast,
)

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
    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        ...

    def canonicalise(self, test_case: T) -> T:
        return test_case


class BasicReductionProblem(ReductionProblem[T]):
    @classmethod
    async def __new__(self, cls, *args, **kwargs):  # type: ignore
        result = super().__new__(cls)
        await result.__init__(*args, **kwargs)
        return result

    async def __init__(  # type: ignore
        self,
        initial: T,
        is_interesting: Callable[[T], Awaitable[bool]],
        work: WorkContext,
        cache_key: Callable[[T], str] = default_cache_key,
        sort_key: Callable[[T], Any] = shortlex,
        canonicalise: Callable[[T], T] = default_canonicalise,
        display: Callable[[T], str] = default_display,
    ):
        super().__init__(work=work)
        self.__current = initial
        self.cache_key = cache_key
        self.__sort_key = sort_key
        self.__canonicalise = canonicalise
        self.display = display

        self.__lock = Lock()

        self.__is_interesting = is_interesting
        self.__on_reduce_callbacks: list[Callable[[T], Awaitable[None]]] = []
        self.__cache = {}

        if not await is_interesting(initial):
            raise ValueError(
                f"Initial example ({self.display(initial)}) does not satisfy interestingness test."
            )

        self.__cache[self.cache_key(initial)] = True

        canonical = self.canonicalise(initial)

        if not await is_interesting(canonical):
            self.work.warn(
                f"Initial example ({self.display(initial)}) was interesting, but canonicalised version ({self.display(canonical)}) was not. Disabling canonicalisation."
            )
            self.__canonicalise = default_canonicalise
        else:
            self.__cache[self.cache_key(canonical)] = True

    def canonicalise(self, test_case: T) -> T:
        return self.__canonicalise(test_case)

    def sort_key(self, test_case: T) -> Any:
        return self.__sort_key(test_case)

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        """Every time `is_interesting` is called with a successful reduction,
        call `fn` with the new value. Note that these are called outside the lock."""
        self.__on_reduce_callbacks.append(callback)

    @contextmanager
    def __locked(self) -> Iterator[None]:
        """Run the block of this context manager locked if parallelism
        is enabled."""
        try:
            self.__lock.acquire()
            yield
        finally:
            self.__lock.release()

    async def is_interesting(self, value: T) -> bool:
        """Returns true if this value is interesting.

        Caches and maintains relevant state.

        Note: This function will lock while maintaining state, but
        will *not* lock around calling the underlying interestingness
        test.
        """
        keys = [self.cache_key(value)]
        try:
            return self.__cache[keys[0]]
        except KeyError:
            pass

        value = self.canonicalise(value)
        keys.append(self.cache_key(value))
        try:
            result = self.__cache[keys[-1]]
        except KeyError:
            result = await self.__is_interesting(value)
            with self.__locked():
                if result:
                    if self.sort_key(value) < self.sort_key(self.current_test_case):
                        self.__current = value
                        for f in self.__on_reduce_callbacks:
                            await f(value)
                self.work.tick()
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

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        async def composed(value: S) -> None:
            await callback(self.__parse(value))

        self.__problem.on_reduce(composed)

    def canonicalise(self, test_case: T) -> T:
        return self.__parse(self.__problem.canonicalise(self.__dump(test_case)))

    def sort_key(self, test_case: T) -> Any:
        if self.__sort_key is not None:
            return self.__sort_key(test_case)
        return self.__problem.sort_key(self.__dump(test_case))
