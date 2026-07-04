import itertools
import re
from typing import Iterable, Iterator


TOKEN = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> Iterator[str]:
    for match in TOKEN.finditer(text):
        yield match.group(0).lower()


def word_counts(documents: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        for token in tokenize(document):
            counts[token] = counts.get(token, 0) + 1
    return counts


def top_n(counts: dict[str, int], n: int = 3) -> list[tuple[str, int]]:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return list(itertools.islice(ordered, n))


def running_average(values: Iterable[float]) -> Iterator[float]:
    total = 0.0
    for index, value in enumerate(values, start=1):
        total += value
        yield total / index


if __name__ == "__main__":
    docs = ["the quick brown fox", "the lazy dog", "the fox jumps"]
    print(top_n(word_counts(docs)))
    print(list(running_average([1.0, 2.0, 3.0, 4.0])))
