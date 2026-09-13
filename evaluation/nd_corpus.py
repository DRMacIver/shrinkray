#!/usr/bin/env python3
"""Drive the corpus of real nondeterministic bugs in evaluation/nd-corpus/.

Each entry there is a directory holding a pinned tool (installed by its
`setup.sh` into `<entry>/.tool/`), the triggering input (`original.<ext>`)
and an interestingness test (`test.sh`) that passes only on runs where the
bug actually manifests, so the same candidate can pass one run and fail the
next. Some entries also ship `test-deterministic.sh`, a variant of the same
bug pinned to a fixed seed or environment, for comparing a reduction under
nondeterminism against deterministic ground truth.

For every selected entry the driver:

1. runs `setup.sh` (idempotent),
2. measures how often `test.sh` passes on the original input (N runs),
3. reduces a copy of the original with the installed `shrinkray` CLI in
   basic UI mode, recording wall time, the final size, Shrink Ray's own
   "interesting on X of Y replays" summary line when it printed one, and
   its exit status,
4. independently re-measures how often `test.sh` passes on the reduced file
   (N runs), which is the quality signal that matters: a reduction that
   lost the bug shows up as a low rate here.

Wall time and rates are what a real user experiences, so unlike
`benchmark.py` this is not a call-count benchmark; treat the numbers as
observations on one machine, not as reproducible metrics.

    python3 evaluation/nd_corpus.py --list
    python3 evaluation/nd_corpus.py                    # everything
    python3 evaluation/nd_corpus.py jq-1.5-string-repeat-uninitialised
    python3 evaluation/nd_corpus.py --deterministic    # test-deterministic.sh
    python3 evaluation/nd_corpus.py --no-reduce        # rates only
    python3 evaluation/nd_corpus.py --json out.json
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "evaluation" / "nd-corpus"

REPLAY_SUMMARY = re.compile(
    r"(?:interesting on|reproduced) (\d+) (?:of|/) (\d+) (?:replays|runs)"
)


# --- entries ----------------------------------------------------------------


class Entry:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        originals = sorted(p for p in path.glob("original.*") if p.is_file())
        if len(originals) != 1:
            raise ValueError(f"{path}: expected exactly one original.* file")
        self.original = originals[0]
        self.setup = path / "setup.sh"
        self.test = path / "test.sh"
        self.deterministic_test = path / "test-deterministic.sh"
        for script in (self.setup, self.test):
            if not script.is_file():
                raise ValueError(f"{path}: missing {script.name}")

    @property
    def has_deterministic(self) -> bool:
        return self.deterministic_test.is_file()

    def test_script(self, deterministic: bool) -> Path:
        if deterministic:
            if not self.has_deterministic:
                raise ValueError(f"{self.name} has no test-deterministic.sh")
            return self.deterministic_test
        return self.test


def load_entries() -> dict[str, Entry]:
    entries = {}
    for path in sorted(CORPUS.iterdir()):
        if path.is_dir() and (path / "test.sh").is_file():
            entries[path.name] = Entry(path)
    return entries


# --- measurement ------------------------------------------------------------


def run_setup(entry: Entry) -> None:
    subprocess.run([str(entry.setup)], check=True, cwd=entry.path)


def measure_rate(script: Path, candidate: Path, runs: int) -> int:
    """How many of `runs` invocations of the interestingness test pass on
    `candidate`. Mirrors how Shrink Ray invokes a test: the candidate is
    copied under its original basename into a fresh working directory and
    that path is also passed as the first argument."""
    hits = 0
    with tempfile.TemporaryDirectory() as workdir:
        target = Path(workdir) / candidate.name
        shutil.copyfile(candidate, target)
        for _ in range(runs):
            result = subprocess.run(
                [str(script), str(target)],
                cwd=workdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                hits += 1
    return hits


def reduce_entry(
    entry: Entry,
    script: Path,
    *,
    deterministic: bool,
    timeout: float,
    parallelism: int | None,
    max_seconds: float | None,
    workdir: Path,
) -> dict:
    """Reduce a copy of the entry's original with the shrinkray CLI. The
    copy is left in `workdir` (Shrink Ray reduces in place, keeping a .bak
    of the original next to it)."""
    target = workdir / entry.original.name
    shutil.copyfile(entry.original, target)
    command = [
        "uv",
        "run",
        "shrinkray",
        "--ui=basic",
        "--no-history",
        "--no-llm",
        # A real bug can have a one-byte reproducer (Pygments #852 does):
        # the trivial-result guard would otherwise exit 1 and skip the
        # final replay summary.
        "--trivial-is-not-error",
        "--seed",
        "0",
        "--timeout",
        str(timeout),
    ]
    if deterministic:
        command.append("--assume-deterministic")
    if parallelism is not None:
        command += ["--parallelism", str(parallelism)]
    command += [str(script), str(target)]

    start = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max_seconds,
        )
        output = completed.stdout + completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(
            errors="replace"
        )
        returncode = None
    seconds = time.monotonic() - start

    match = None
    for match in REPLAY_SUMMARY.finditer(output):
        pass  # keep the last summary line
    reported = (
        {"interesting": int(match.group(1)), "runs": int(match.group(2))}
        if match
        else None
    )
    return {
        "command": command,
        "seconds": round(seconds, 1),
        "returncode": returncode,
        "timed_out": timed_out,
        "reported_replays": reported,
        "output": output,
        "result_path": str(target),
    }


def run_entry(entry: Entry, args: argparse.Namespace) -> dict:
    script = entry.test_script(args.deterministic)
    print(f"== {entry.name} ({script.name})", file=sys.stderr, flush=True)
    run_setup(entry)

    initial_size = entry.original.stat().st_size
    original_hits = measure_rate(script, entry.original, args.runs)
    print(
        f"   original: {original_hits}/{args.runs} interesting, {initial_size} bytes",
        file=sys.stderr,
        flush=True,
    )
    record = {
        "name": entry.name,
        "test": script.name,
        "deterministic": args.deterministic,
        "runs": args.runs,
        "initial_size": initial_size,
        "original_hits": original_hits,
        "final_size": None,
        "reduced_hits": None,
        "seconds": None,
        "returncode": None,
        "timed_out": False,
        "reported_replays": None,
    }
    if args.no_reduce:
        return record

    keep_dir = Path(args.keep_results) / entry.name if args.keep_results else None
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = keep_dir if keep_dir is not None else Path(tmp)
        reduction = reduce_entry(
            entry,
            script,
            deterministic=args.deterministic,
            timeout=args.timeout,
            parallelism=args.parallelism,
            max_seconds=args.max_seconds,
            workdir=workdir,
        )
        result_path = Path(reduction["result_path"])
        final_size = result_path.stat().st_size if result_path.is_file() else None
        reduced_hits = (
            measure_rate(script, result_path, args.runs)
            if result_path.is_file()
            else None
        )
        if keep_dir is not None:
            (keep_dir / "shrinkray.log").write_text(reduction["output"])
    record.update(
        {
            "final_size": final_size,
            "reduced_hits": reduced_hits,
            "seconds": reduction["seconds"],
            "returncode": reduction["returncode"],
            "timed_out": reduction["timed_out"],
            "reported_replays": reduction["reported_replays"],
        }
    )
    print(
        f"   reduced: {final_size} bytes in {reduction['seconds']}s "
        f"(exit {reduction['returncode']}), "
        f"{reduced_hits}/{args.runs} interesting",
        file=sys.stderr,
        flush=True,
    )
    return record


# --- output -----------------------------------------------------------------


def fmt_rate(hits, runs: int) -> str:
    return "-" if hits is None else f"{hits}/{runs}"


def fmt_replays(reported) -> str:
    if reported is None:
        return "-"
    return f"{reported['interesting']}/{reported['runs']}"


def print_table(results: list[dict]) -> None:
    header = (
        f"{'entry':<44} {'orig':>7} {'initial':>8} {'final':>7} "
        f"{'secs':>8} {'exit':>5} {'reported':>9} {'reduced':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        exit_code = "T/O" if r["timed_out"] else r["returncode"]
        print(
            f"{r['name']:<44} {fmt_rate(r['original_hits'], r['runs']):>7} "
            f"{r['initial_size']:>8} {r['final_size'] if r['final_size'] is not None else '-':>7} "
            f"{r['seconds'] if r['seconds'] is not None else '-':>8} "
            f"{exit_code if exit_code is not None else '-':>5} "
            f"{fmt_replays(r['reported_replays']):>9} "
            f"{fmt_rate(r['reduced_hits'], r['runs']):>8}"
        )
    print("-" * len(header))
    print(
        "orig/reduced: test.sh passes out of N runs on the original / reduced "
        "file; reported: Shrink Ray's own final replay summary"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="entries to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list entries and exit")
    parser.add_argument(
        "--runs", type=int, default=20, help="test runs per rate measurement"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use each entry's test-deterministic.sh and pass "
        "--assume-deterministic to shrinkray (entries without one are skipped)",
    )
    parser.add_argument(
        "--no-reduce", action="store_true", help="only set up and measure rates"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="shrinkray --timeout for each interestingness-test run (seconds)",
    )
    parser.add_argument(
        "--parallelism", type=int, help="shrinkray --parallelism (default: its own)"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="kill a reduction that runs longer than this (default: no limit)",
    )
    parser.add_argument(
        "--keep-results",
        help="directory to keep each entry's reduced file and shrinkray log in",
    )
    parser.add_argument("--json", help="write results to this file")
    args = parser.parse_args()

    entries = load_entries()
    if args.list:
        for entry in entries.values():
            det = " (+deterministic)" if entry.has_deterministic else ""
            print(f"{entry.name:<44} {entry.original.name}{det}")
        return 0

    names = args.names or list(entries)
    unknown = [n for n in names if n not in entries]
    if unknown:
        parser.error(
            f"unknown entries: {', '.join(unknown)}. Available: {', '.join(entries)}"
        )
    if args.deterministic:
        skipped = [n for n in names if not entries[n].has_deterministic]
        for n in skipped:
            print(f"skipping {n}: no test-deterministic.sh", file=sys.stderr)
        names = [n for n in names if n not in skipped]

    results = [run_entry(entries[name], args) for name in names]

    print()
    print_table(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
