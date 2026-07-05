"""Byte-level reduction passes.

This module provides reduction passes that operate on raw bytes.
These are the foundation of Shrink Ray's reduction strategy, as
all file formats ultimately reduce to bytes.

Key passes:
- hollow: Keeps only start/end of bracketed regions
- lift_braces: Replaces {...} with its content
- debracket: Removes matching bracket pairs
- delete_byte_spans: Deletes contiguous byte ranges
- short_deletions: Deletes small (1-10 byte) sequences
- remove_indents/remove_whitespace: Whitespace normalization
- lower_bytes/lower_individual_bytes: Replace bytes with lower-sorting ones
- lexeme_based_deletions: Deletes between repeated patterns

Formats:
- Split(delimiter): Parses bytes into list of segments
- Tokenize(): Parses bytes into tokens (identifiers, numbers, etc.)
"""

from collections import defaultdict, deque
from collections.abc import Sequence

from attrs import define

from shrinkray.passes.patching import Cuts, Patches, apply_patches
from shrinkray.problem import Format, ReductionProblem


@define(frozen=True)
class Split(Format[bytes, list[bytes]]):
    """Format that splits bytes by a delimiter.

    This enables sequence-based passes to work on lines, statements, etc.

    Example:
        # Delete duplicate lines
        compose(Split(b"\\n"), delete_duplicates)

        # Delete blocks of 1-10 semicolon-separated statements
        compose(Split(b";"), block_deletion(1, 10))
    """

    splitter: bytes

    def __repr__(self) -> str:
        return f"Split({repr(self.splitter)})"

    @property
    def name(self) -> str:
        return f"split({repr(self.splitter)})"

    def parse(self, input: bytes) -> list[bytes]:
        return input.split(self.splitter)

    def dumps(self, input: list[bytes]) -> bytes:
        return self.splitter.join(input)


def find_ngram_endpoints(value: bytes) -> list[tuple[int, list[int]]]:
    """Find repeated byte patterns and their positions.

    This is used by lexeme_based_deletions to identify regions between
    repeated patterns that might be deletable. For example, in code like:
        print("hello"); print("world"); print("test")
    The repeated "print" patterns suggest the semicolon-separated regions
    might be independently deletable.

    Returns a list of (ngram_length, [positions]) tuples.
    """
    if len(set(value)) <= 1:
        return []
    queue: deque[tuple[int, Sequence[int]]] = deque([(0, range(len(value)))])
    results: list[tuple[int, list[int]]] = []

    while queue and len(results) < 10000:
        k, indices = queue.popleft()

        if k > 1:
            normalized: list[int] = []
            for i in indices:
                if not normalized or i >= normalized[-1] + k:
                    normalized.append(i)
            indices = normalized

        while (
            indices[-1] + k < len(value) and len({value[i + k] for i in indices}) == 1
        ):
            k += 1

        if k > 0 and (indices[0] == 0 or len({value[i - 1] for i in indices}) > 1):
            assert isinstance(indices, list), value
            results.append((k, indices))

        split: dict[int, list[int]] = defaultdict(list)
        for i in indices:
            try:
                split[value[i + k]].append(i)
            except IndexError:
                pass
        queue.extend([(k + 1, v) for v in split.values() if len(v) > 1])

    return results


def _is_identifier_char(c: int) -> bool:
    return (
        b"A"[0] <= c <= b"Z"[0]
        or b"a"[0] <= c <= b"z"[0]
        or b"0"[0] <= c <= b"9"[0]
        or c == b"_"[0]
    )


def tokenize(text: bytes) -> list[bytes]:
    """Split bytes into tokens: identifiers, numbers, and other characters.

    This is a simple tokenizer that groups:
    - Identifiers: [A-Za-z_][A-Za-z0-9_]*
    - Numbers: [0-9]+ (with optional decimal point)
    - Spaces: runs of spaces
    - Everything else: individual characters

    Example:
        tokenize(b"foo = 123") -> [b"foo", b" ", b"=", b" ", b"123"]
    """
    result: list[bytes] = []
    i = 0
    while i < len(text):
        c = bytes([text[i]])
        j = i + 1
        if c.isalpha() or c == b"_":
            # Identifier: consume alphanumeric and underscore
            while j < len(text) and _is_identifier_char(text[j]):
                j += 1
        elif b"0" <= c <= b"9":
            # Number: consume digits and decimal point
            while j < len(text) and (
                text[j] == b"."[0] or b"0"[0] <= text[j] <= b"9"[0]
            ):
                j += 1
        elif c == b" ":
            # Space run: consume consecutive spaces
            while j < len(text) and (text[j] == b" "[0]):
                j += 1
        result.append(text[i:j])
        i = j
    assert b"".join(result) == text
    return result


async def lexeme_based_deletions(
    problem: ReductionProblem[bytes], min_size: int = 8
) -> None:
    """Delete regions between repeated byte patterns.

    This pass finds repeated patterns (like repeated keywords or punctuation)
    and tries to delete the regions between them. For code like:

        print("a"); print("b"); print("c")

    The repeated "print(" pattern suggests each print statement might be
    independently deletable. This pass identifies such regions and tries
    to delete them.

    Only regions >= min_size bytes are considered to avoid tiny deletions.
    """
    intervals_by_k: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for k, endpoints in find_ngram_endpoints(problem.current_test_case):
        intervals_by_k[k].update(zip(endpoints, endpoints[1:], strict=False))

    # Sort by ngram length (longer patterns first) then by interval size
    intervals_to_delete = [
        t
        for _, intervals in sorted(intervals_by_k.items(), reverse=True)
        for t in sorted(intervals, key=lambda t: (t[1] - t[0], t[0]), reverse=True)
        if t[1] - t[0] >= min_size
    ]

    await delete_intervals(problem, intervals_to_delete, shuffle=True)


async def delete_intervals(
    problem: ReductionProblem[bytes],
    intervals_to_delete: list[tuple[int, int]],
    shuffle: bool = False,
) -> None:
    """Try to delete each of the given byte intervals.

    Each interval (start, end) represents a contiguous region to try deleting.
    The patch applier will find which intervals can be deleted independently
    and combine compatible deletions.
    """
    await apply_patches(problem, Cuts(), [[t] for t in intervals_to_delete])


def brace_intervals(target: bytes, brace: bytes) -> list[tuple[int, int]]:
    """Find all intervals enclosed by matching brace pairs.

    Given a two-byte brace string like b"{}", returns intervals for content
    between each matched open/close pair. Handles nesting correctly.
    """
    open, close = brace
    intervals: list[tuple[int, int]] = []
    stack: list[int] = []
    for i, c in enumerate(target):
        if c == open:
            stack.append(i)
        elif c == close and stack:
            start = stack.pop() + 1
            end = i
            if end > start:
                intervals.append((start, end))
    return intervals


async def debracket(problem: ReductionProblem[bytes]) -> None:
    """Remove matching bracket pairs, keeping their content.

    Example transformations:
        "(x + y)" -> "x + y"
        "[1, 2]" -> "1, 2"
        "{foo}" -> "foo"

    This is useful when brackets become unnecessary after other reductions,
    e.g., if a function call was simplified to just its first argument.
    """
    cuts = [
        [(u - 1, u), (v, v + 1)]
        for brackets in [b"{}", b"()", b"[]"]
        for u, v in brace_intervals(problem.current_test_case, brackets)
    ]
    await apply_patches(
        problem,
        Cuts(),
        cuts,
    )


def quote_intervals(target: bytes) -> list[tuple[int, int]]:
    """Find all intervals enclosed by matching quote pairs.

    Returns intervals between consecutive single or double quotes.
    """
    indices: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(target):
        indices[c].append(i)

    intervals: list[tuple[int, int]] = []
    for quote in b"\"'":
        xs = indices[quote]
        for u, v in zip(xs, xs[1:], strict=False):
            if u + 1 < v:
                intervals.append((u + 1, v))
    return intervals


async def hollow(problem: ReductionProblem[bytes]) -> None:
    """Delete the contents of bracketed and quoted regions.

    Example transformations:
        '{"lots": "of json"}' -> '{}'
        "[1, 2, 3, 4, 5]" -> "[]"
        '"long string here"' -> '""'

    This is one of the most effective early passes: it quickly removes
    large chunks of content from structured data, keeping only the
    "skeleton" of brackets and quotes.

    Intervals are sorted by size (smallest first) to maximize the chance
    of finding independent deletions that can be combined.
    """
    target = problem.current_test_case
    intervals: list[tuple[int, int]] = []
    for b in [
        quote_intervals(target),
        brace_intervals(target, b"[]"),
        brace_intervals(target, b"{}"),
    ]:
        b.sort(key=lambda t: (t[1] - t[0], t[0]))
        intervals.extend(b)
    await delete_intervals(
        problem,
        intervals,
    )


async def short_deletions(problem: ReductionProblem[bytes]) -> None:
    """Try deleting every small (1-10 byte) substring.

    This is a brute-force pass that tries all possible small deletions.
    It's expensive but effective for cleaning up small syntax elements
    that other passes miss.

    Example: After other passes simplify "foo(x, y)" to "foo(x)", this
    pass might find that deleting ", y" or "x, " works.
    """
    target = problem.current_test_case
    await delete_intervals(
        problem,
        [
            (i, j)
            for i in range(len(target))
            for j in range(i + 1, min(i + 11, len(target) + 1))
        ],
    )


async def lift_braces(problem: ReductionProblem[bytes]) -> None:
    """Replace outer braces with inner braces' content.

    For nested braces like {A{B}C}, this tries to replace the outer
    braces with just the inner content: {A{B}C} -> {B}

    Example transformations:
        "if (x) { if (y) { z } }" -> "if (x) { z }"
        "{ outer { inner } more }" -> "{ inner }"

    This is useful for eliminating wrapper blocks while keeping the
    essential nested structure. It complements debracket (which removes
    brackets entirely) and hollow (which empties brackets).
    """
    target = problem.current_test_case

    open_brace, close_brace = b"{}"
    start_stack: list[int] = []
    child_stack: list[list[tuple[int, int]]] = []

    results: list[tuple[int, int, list[tuple[int, int]]]] = []

    # Track brace nesting and record parent-child relationships
    for i, c in enumerate(target):
        if c == open_brace:
            start_stack.append(i)
            child_stack.append([])
        elif c == close_brace and start_stack:
            start = start_stack.pop() + 1
            end = i
            children = child_stack.pop()
            if child_stack:
                child_stack[-1].append((start, end))
            if end > start:
                results.append((start, end, children))

    # For each parent-child pair, try deleting parent content around child
    cuts: list[list[tuple[int, int]]] = []
    for start, end, children in results:
        for child_start, child_end in children:
            cuts.append([(start, child_start), (child_end, end)])

    await apply_patches(problem, Cuts(), cuts)


@define(frozen=True)
class Tokenize(Format[bytes, list[bytes]]):
    def __repr__(self) -> str:
        return "tokenize"

    @property
    def name(self) -> str:
        return "tokenize"

    def parse(self, input: bytes) -> list[bytes]:
        return tokenize(input)

    def dumps(self, input: list[bytes]) -> bytes:
        return b"".join(input)


async def delete_byte_spans(problem: ReductionProblem[bytes]) -> None:
    """Delete spans between occurrences of the same byte value.

    For each byte value that appears multiple times, tries to delete
    regions from the start to the first occurrence, between consecutive
    occurrences, and from the last occurrence to the end.
    """
    indices: dict[int, list[int]] = defaultdict(list)
    target = problem.current_test_case
    for i, c in enumerate(target):
        indices[c].append(i)

    spans: list[tuple[int, int]] = []

    for c, ix in sorted(indices.items()):
        if len(ix) > 1:
            spans.append((0, ix[0] + 1))
            spans.extend(zip(ix, ix[1:], strict=False))
            spans.append((ix[-1], len(target)))

    await apply_patches(problem, Cuts(), [[s] for s in spans])


async def remove_indents(problem: ReductionProblem[bytes]) -> None:
    """Remove leading spaces from lines.

    Finds runs of spaces following newlines and tries to delete them.
    Useful for normalizing indentation in code.
    """
    target = problem.current_test_case
    spans: list[list[tuple[int, int]]] = []

    newline = ord(b"\n")
    space = ord(b" ")

    for i, c in enumerate(target):
        if c == newline:
            j = i + 1
            while j < len(target) and target[j] == space:
                j += 1

            if j > i + 1:
                spans.append([(i + 1, j)])

    await apply_patches(problem, Cuts(), spans)


async def remove_whitespace(problem: ReductionProblem[bytes]) -> None:
    """Collapse runs of whitespace.

    Finds consecutive whitespace characters and tries to remove all but
    the first, or all but the first two. Complements remove_indents by
    handling whitespace anywhere in the file.
    """
    target = problem.current_test_case
    spans: list[list[tuple[int, int]]] = []

    for i, c in enumerate(target):
        char = bytes([c])
        if char.isspace():
            j = i + 1
            while j < len(target) and target[j : j + 1].isspace():
                j += 1

            if j > i + 1:
                spans.append([(i, j)])
            if j > i + 2:
                spans.append([(i + 1, j)])

    await apply_patches(problem, Cuts(), spans)


class NewlineReplacer(Patches[frozenset[int], bytes]):
    @property
    def empty(self) -> frozenset[int]:
        return frozenset()

    def combine(self, *patches: frozenset[int]) -> frozenset[int]:
        result: set[int] = set()
        for p in patches:
            result.update(p)
        return frozenset(result)

    def apply(self, patch: frozenset[int], target: bytes) -> bytes:
        result = bytearray()

        for i, c in enumerate(target):
            if i in patch:
                result.extend(b"\n")
            else:
                result.append(c)
        return bytes(result)

    def size(self, patch: frozenset[int]) -> int:
        return len(patch)


async def replace_space_with_newlines(problem: ReductionProblem[bytes]) -> None:
    """Replace spaces and tabs with newlines.

    Tries replacing each space or tab with a newline. This can help
    normalize formatting and may enable other line-based reductions.
    """
    await apply_patches(
        problem,
        NewlineReplacer(),
        [
            frozenset({i})
            for i, c in enumerate(problem.current_test_case)
            if c in b" \t"
        ],
    )


ReplacementPatch = dict[int, int]


class ByteReplacement(Patches[ReplacementPatch, bytes]):
    @property
    def empty(self) -> ReplacementPatch:
        return {}

    def combine(self, *patches: ReplacementPatch) -> ReplacementPatch:
        result = {}
        for p in patches:
            for k, v in p.items():
                if k not in result:
                    result[k] = v
                else:
                    result[k] = min(result[k], v)
        return result

    def apply(self, patch: ReplacementPatch, target: bytes) -> bytes:
        table = bytearray(range(256))
        for source, replacement in patch.items():
            table[source] = replacement
        return target.translate(bytes(table))

    def size(self, patch: ReplacementPatch) -> int:
        return 0


class ByteRanks:
    """Every byte value ranked by the problem's ordering of single bytes.

    Rank 0 is the byte whose one-byte test case sorts lowest. Numeric byte
    order and the problem's ordering can disagree (natural text ordering
    ranks b"z" below b"\\x00"), so lowering passes descend in this rank
    space rather than assuming smaller byte values are better.
    """

    def __init__(self, problem: ReductionProblem[bytes]):
        self.order = sorted(range(256), key=lambda i: problem.sort_key(bytes([i])))
        self.rank = [0] * 256
        for rank, byte in enumerate(self.order):
            self.rank[byte] = rank

    def is_lower(self, replacement: int, c: int) -> bool:
        """Whether `replacement` is plausibly lower than `c`.

        A candidate counts if it is lower in either ordering: single-byte
        ranks are only a proxy for how a byte sorts in context (b"\\n" is
        the lowest byte on its own but the highest whitespace within a
        line), so numerically lower bytes stay in play too.
        """
        return self.rank[replacement] < self.rank[c] or replacement < c

    def lowering_candidates(self, c: int) -> list[int]:
        """Candidate replacements for byte `c`.

        The classic numeric candidates (0, 1, half, predecessor) plus a
        small set of preferred characters (whitespace, "0", "a", "z") that
        the natural text ordering ranks below most bytes even when they are
        numerically larger, so e.g. a control character can become a letter.
        """
        candidates = {0, 1, c // 2, c - 1} | set(b" \t\r\n0az")
        return sorted(x for x in candidates if 0 <= x and self.is_lower(x, c))


async def lower_bytes(problem: ReductionProblem[bytes]) -> None:
    """Globally replace byte values with lower-sorting ones.

    For each distinct byte value in the input, tries replacing all
    occurrences with values that sort below it (see
    ByteRanks.lowering_candidates). Also tries replacing pairs of bytes
    with the same value.
    """
    ranks = ByteRanks(problem)
    sources = sorted(set(problem.current_test_case))

    patches = [{c: r} for c in sources for r in ranks.lowering_candidates(c)] + [
        {c: r, d: r}
        for c in sources
        for d in sources
        if c != d
        for r in sorted(
            set(ranks.lowering_candidates(c)) | set(ranks.lowering_candidates(d))
        )
        if ranks.is_lower(r, c) or ranks.is_lower(r, d)
    ]

    await apply_patches(problem, ByteReplacement(), patches, early_abort=True)


class IndividualByteReplacement(Patches[ReplacementPatch, bytes]):
    @property
    def empty(self) -> ReplacementPatch:
        return {}

    def combine(self, *patches: ReplacementPatch) -> ReplacementPatch:
        result = {}
        for p in patches:
            for k, v in p.items():
                if k not in result:
                    result[k] = v
                else:
                    result[k] = min(result[k], v)
        return result

    def apply(self, patch: ReplacementPatch, target: bytes) -> bytes:
        # Patches are generated from and applied to the same test case, so
        # every position is in range.
        result = bytearray(target)
        for i, replacement in patch.items():
            result[i] = replacement
        return bytes(result)

    def size(self, patch: ReplacementPatch) -> int:
        return 0


async def lower_individual_bytes(problem: ReductionProblem[bytes]) -> None:
    """Replace individual bytes at specific positions with lower-sorting
    values.

    Unlike lower_bytes (which replaces all occurrences of a byte value),
    this tries reducing individual byte positions. Also handles carry-like
    patterns where decrementing one byte allows the next to become 255.
    """
    ranks = ByteRanks(problem)
    initial = problem.current_test_case
    patches: list[dict[int, int]] = [
        {i: r} for i, c in enumerate(initial) for r in ranks.lowering_candidates(c)
    ] + [
        {i - 1: initial[i - 1] - 1, i: 255}
        for i, c in enumerate(initial)
        if i > 0 and initial[i - 1] > 0 and c == 0
    ]
    await apply_patches(problem, IndividualByteReplacement(), patches, early_abort=True)


# These are some cheat substitutions that are sometimes helpful, but mostly
# for passing stupid tests.
STANDARD_SUBSTITUTIONS = [(b"\0\0", b"\1"), (b"\0\0", b"\xff")]


async def standard_substitutions(problem: ReductionProblem[bytes]):
    """Apply standard byte sequence substitutions.

    Tries some specific byte sequence replacements that are sometimes
    helpful, primarily for handling edge cases in artificial test inputs.
    """
    i = 0
    while i < len(problem.current_test_case):
        for k, v in STANDARD_SUBSTITUTIONS:
            x = problem.current_test_case
            if i + len(k) <= len(x) and x[i : i + len(k)] == k:
                attempt = x[:i] + v + x[i + len(k) :]
                # An interesting attempt is only adopted if it sorts below
                # the current test case; a non-adopted attempt must not
                # stop the scan or we'd retry it forever.
                if (
                    await problem.is_interesting(attempt)
                    and problem.current_test_case == attempt
                ):
                    break
        else:
            i += 1


async def line_sorter(problem: ReductionProblem[bytes]):
    """Sort lines into a more canonical order.

    Uses insertion sort to reorder lines, swapping adjacent lines when
    doing so maintains interestingness and produces a lexicographically
    smaller result. This normalizes line order for reproducibility.
    """
    lines = problem.current_test_case.split(b"\n")
    i = 1
    while i < len(lines):
        j = i
        while j > 0:
            attempt = list(lines)
            attempt[j - 1], attempt[j] = attempt[j], attempt[j - 1]
            new_test_case = b"\n".join(attempt)
            if problem.sort_key(new_test_case) < problem.sort_key(
                problem.current_test_case
            ):
                if not await problem.is_interesting(new_test_case):
                    break
                else:
                    j -= 1
                    lines = attempt
            else:
                break
        i += 1
