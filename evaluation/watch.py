#!/usr/bin/env python3
"""Run one corpus entry interactively in shrink ray's normal interface.

Unlike run.py — which runs headless with `--ui=basic`, resumes any
in-progress reduction, and records results — this is for *watching*:
it always resets the starting point to the entry's original input,
launches shrink ray attached to your terminal (so you get the TUI),
and records nothing. It reduces a separate work file
(<entry>/work/watch<ext>), so it never disturbs run.py's resumable
state or the committed shrinkray_reduced.* / result.json.

Any unrecognised arguments are forwarded to shrinkray, e.g.:

    python3 evaluation/watch.py ruff-0.0.277-isort-skip-block-panic
    python3 evaluation/watch.py corpus-entry-id --parallelism 4
    python3 evaluation/watch.py corpus-entry-id --ui=basic
"""

import argparse
import shlex
import shutil
import subprocess
import sys

from run import (
    CORPUS_DIR,
    REPO_ROOT,
    is_interesting,
    load_entries,
    original_path,
    prepare,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id", help="corpus entry id")
    parser.add_argument(
        "--parallelism", type=int, default=None,
        help="override the entry's parallelism setting",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="set up the oracle and print the shrinkray command instead of running it",
    )
    args, extra = parser.parse_known_args()

    entries = load_entries([args.id])
    if not entries:
        available = ", ".join(p.parent.name for p in sorted(CORPUS_DIR.glob("*/meta.json")))
        print(f"no corpus entry named {args.id!r}", file=sys.stderr)
        print(f"available entries: {available}", file=sys.stderr)
        return 2
    entry = entries[0]

    if entry["oracle"]["type"] == "docker" and shutil.which("docker") is None:
        print("docker not found on PATH", file=sys.stderr)
        return 2

    print(f"Setting up oracle for {entry['id']} ({entry['tool']}) ...", flush=True)
    oracle, check = prepare(entry)
    try:
        # Always start over from the original input.
        source = entry["dir"] / "work" / ("watch" + entry["extension"])
        shutil.copy(original_path(entry), source)

        if not is_interesting(check, source):
            print(
                f"ERROR: {original_path(entry)} does not reproduce the failure "
                f"(try `python3 evaluation/run.py --check {entry['id']}`)",
                file=sys.stderr,
            )
            return 1

        command = [
            "uv", "run", "shrinkray",
            f"--timeout={entry.get('timeout', 10)}",
        ]
        if entry["oracle"]["type"] == "docker":
            command.append("--input-type=stdin")
        parallelism = (
            args.parallelism
            if args.parallelism is not None
            else entry.get("parallelism")
        )
        if parallelism is not None:
            command.append(f"--parallelism={parallelism}")
        command += entry.get("shrinkray_args", [])
        command += extra
        command += [str(check), str(source)]

        if args.dry_run:
            print(shlex.join(command))
            return 0

        proc = subprocess.run(command, cwd=REPO_ROOT)
        print(f"\nReduced file left in {source}")
        print("(watch.py does not update shrinkray_reduced.* or result.json;")
        print(" use run.py for recorded reductions)")
        return proc.returncode
    finally:
        oracle.teardown()


if __name__ == "__main__":
    sys.exit(main())
