# Sort-key tuning corpus

Shrink ray's very-reduced outputs often look *bad* to humans. Part of that is
its aggressive transformations (good), and part may be that its **sort key** —
the ordering that decides which of two valid test cases is "more reduced" — is
not well adapted to human preference. This directory builds a curated corpus for
investigating and (eventually) tuning the sort key.

The signal we collect: places where the current sort key and a language's
**formatter** disagree. Concretely, pairs of valid programs `(x, y)` where

```
sort_key(x) < sort_key(y)   but   sort_key(format(x)) > sort_key(format(y))
```

i.e. the sort key prefers `x`, but once you format both, `x` is the *larger* of
the two.

These are **starting points for investigation, not proven defects.** A formatter
is only one proxy for human preference, and in many of these pairs the sort key
may be perfectly reasonable — the point is that a disagreement is a place worth
looking at when deciding how a new sort key should behave.

## Pipeline

1. **`langs.py`** — per-language registry. For C, C++, Python, SQL, JSON, XML,
   HTML it defines a strict "full front end" parse check (the validity gate) and
   a canonical formatter. Run it directly for a self-test over the seeds.

2. **`seeds/<lang>/`** — starting test cases. Each language has synthetic seeds
   plus `wild_*` seeds pulled from real open-source projects by
   **`fetch_wild_seeds.py`** (kept only if they pass the strict parse check).

3. **`gather.py`** — for each seed, runs shrink ray several times with an
   interestingness test that requires the candidate to *parse* and otherwise
   accepts a random fraction `p` (keyed by `Random(hash(salt + candidate))`).
   Every parsing candidate the reducer visits is recorded, so each run sweeps a
   neighbourhood of valid programs. Output: `corpus/<lang>/<seed>.jsonl`
   (base64, deduplicated). **Gitignored** — regenerable, potentially large.

4. **`inversions.py`** — for each seed's candidates, formats them all and finds
   the sort-key/formatter inversions, tallying which sort-key criterion causes
   each flip. Output: `corpus/_inversions/<lang>.json` (stats + a diverse set of
   example pairs).

5. **`report.py`** — renders `corpus/_inversions/REPORT.md`: formatter choices,
   disagreement rates, and illustrative example pairs.

6. **`deletion_shrink.py`** — a sharper property test: for each formatted
   instance, delete a single byte or line, and (if still valid) reformat; the
   result *should* be a strict shrink under the sort key. It records every
   violation (`sort_key(format(delete(f))) > sort_key(f)`), split by byte vs
   line deletion, grouped by instance, to `corpus/_deletion_shrink/<lang>.json`.
   `SR_DEL_KINDS=line` isolates one deletion kind. Finding: violated by every
   formatter except JSON's `indent=2` (which is monotonic); black and ruff
   behave identically (magic trailing comma + line wrapping).

## Formatters chosen

| Language | Parser (validity gate)      | Formatter                |
| -------- | --------------------------- | ------------------------ |
| C        | `clang -fsyntax-only` (C11) | `clang-format`           |
| C++      | `clang -fsyntax-only` (C++17) | `clang-format`         |
| Python   | `ast.parse`                 | `black`                  |
| SQL      | `sqlglot` (multi-dialect)   | `sqlglot` pretty-print   |
| JSON     | `json.loads`                | `json.dumps(indent=2)`   |
| XML      | `lxml` (strict)             | `lxml` `pretty_print`    |
| HTML     | `html5lib` (strict)         | `BeautifulSoup.prettify` |

## Ordering corpus (labelled pairs)

`ordering_pairs.json` is a hand-curated set of pairs of similar test cases, each
labelled with which member a *good* shrink order should treat as simpler (rank
strictly smaller). It is the target a sort key should match, and the concrete
testbed for tuning one. Pairs are tagged by `kind` (`formatting`, `content`,
`cosmetic`, `quirk`) and `confidence`, and include both cases the current key
gets right and wrong — the real cramped-vs-readable C/C++ examples, the
magic-trailing-comma content cases, natural-order cosmetics, formatter/parser
corruptions (as guards), and an `avg_sq_line` helps/hurts set (see below). Built
by `build_ordering_pairs.py` (which pulls the real C/C++ pairs from
`evaluation/corpus/*/shrinkray_reduced.{c,cpp}`).

The `avgsq-helps-*` / `avgsq-hurts-*` pairs are same-program, **equal byte
length** pairs, so `avg_sq_line` is the deciding criterion. They capture where
that criterion (which always prefers more, shorter lines) is *right* — a
line-break at a statement/element boundary — and where it is *wrong* — a break
mid-construct or a blank line inside a body. A good order should get both; the
current key gets the four helps right and the four hurts wrong.

`ordering_eval.py` scores a sort order against it — by default shrink ray's
current key, or any `str -> comparable` function via `evaluate(key)`. The current
key scores **20/33**: `cosmetic` 5/5, `quirk` 4/4, `content` 5/7 (misses the
magic-comma cases), `formatting` 6/17 (misses every readable-code case and the
four `avg_sq` hurts). The `length`-decided misses want a cheaper-whitespace
primary length; the `avg_sq`-decided misses want a structure signal that is not
fooled by blank lines / mid-construct splits.

## Running

Everything needs the extra parsers/formatters, injected with `uv run --with`:

```bash
DEPS="--with sqlglot --with lxml --with html5lib --with beautifulsoup4 --with black"

uv run $DEPS python evaluation/sortkey/langs.py            # self-test seeds
uv run $DEPS python evaluation/sortkey/fetch_wild_seeds.py # refresh wild seeds
uv run $DEPS python evaluation/sortkey/gather.py           # gather all corpora
uv run $DEPS python evaluation/sortkey/inversions.py       # find inversions
uv run $DEPS python evaluation/sortkey/report.py           # render REPORT.md

uv run python evaluation/sortkey/build_ordering_pairs.py   # rebuild labelled pairs
uv run python evaluation/sortkey/ordering_eval.py          # score sort key vs pairs
```

Each stage takes an optional list of languages, e.g. `... gather.py python json`.

## Findings & open direction (2026-07-04)

The `deletion_shrink` audit (delete a byte/line from a formatted instance,
reformat, check it shrinks) found the **current sort key is essentially never
badly wrong**: >99% of the ~6000 violations are formatter/parser quirks where
the reformatted deleted version genuinely *is* worse — escaping (`>`→`&gt;`),
`html5lib` injecting empty elements, `sqlglot` re-inserting an implied
`SELECT *`, `black`'s magic trailing comma exploding an argument list, two
`#define`s merged onto one line, etc. In all of these the sort key is *right*
to reject the result.

The one genuine ordering tension found is the **inline-comment merge**: deleting
the newline before a standalone comment (`x\n# c` → `x  # c`) yields −1 line but
+1 byte — a cleaner, tidier form the key nonetheless rejects. The cause is
structural: the sort key (`LazyChainedSortKey` over `NATURAL_ORDERING_FUNCTIONS`
in `problem.py`) is a **strict lexicographic chain** that short-circuits on the
first differing criterion, and **length is criterion #1** — so one extra byte is
decisive and the lower criteria (line count, balance, …) only break *exact-length*
ties, which almost never occur.

A sharper, genuinely-bad symptom of the same root cause: the sort key
**systematically rates formatted code as worse than cramped code**, because
formatting only adds whitespace (newlines, indentation) and whitespace counts
full price under length-first ordering. Measured on the committed
`shrinkray_reduced.{c,cpp}` evaluation outputs — with comments stripped first,
since clang-format's `FixNamespaceComments` adds `} // namespace x` annotations
that shouldn't count — clang-format's readable version is still rated worse in
**5/5** cases, always on the length criterion (e.g. a 79-byte one-liner vs its
106-byte / 10-line formatted form). Comment-stripping barely moves the numbers
(the whitespace, not the comments, is what inflates length), so this is *why*
the reduced C/C++ examples come out as dense unreadable blobs: the reducer is
actively driven toward them. NB for future work: comparisons that involve a
formatter should strip comments first.

**Deferred idea (not yet actioned — wants a bigger/better corpus first):**
replace strict byte-length with some **per-character weighting** (let different
characters cost different amounts). This directly targets the symptom above: if
newlines and indentation spaces are cheap, formatting barely changes the score,
so the ordering stops preferring cramped over readable. Other candidate levers
noted: blend `bytes + k·lines` into one scalar; or normalize away
formatter-discretion noise (trailing commas, comment placement) before comparing.
**Do not change the sort key until the corpus is more complete.**
