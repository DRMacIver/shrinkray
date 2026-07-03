#!/usr/bin/env python3
"""Reduce the compiler-bug corpus with shrink ray against real compilers.

Each corpus entry is a program that makes a specific old version of gcc
or clang crash. This script runs shrink ray's C/C++ passes on each one,
using the real compiler (in a Docker container) as the interestingness
oracle, and reports how far each was reduced.

The old compilers are amd64-only, so they run under emulation on Apple
Silicon; this is slow but works. Each entry gets a persistent container
(started once, reused across the thousands of test calls a reduction
makes) to avoid per-call container-startup overhead.

Usage:
    python3 run.py                 # reduce every entry
    python3 run.py <entry-id> ...  # reduce specific entries
    python3 run.py --check         # just verify each entry still crashes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


CORPUS_DIR = Path(__file__).parent
REPO_ROOT = CORPUS_DIR.parent
CHECK_SCRIPT = CORPUS_DIR / "check.sh"


def load_entries(names: list[str]) -> list[dict]:
    entries = []
    for meta_path in sorted(CORPUS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        meta["dir"] = meta_path.parent
        if not names or meta["id"] in names:
            entries.append(meta)
    return entries


def container_name(entry: dict) -> str:
    return "srbug-" + entry["id"]


def compiler_exe(entry: dict) -> str:
    return "g++" if entry["compiler"] == "gcc" else "clang++"


def start_container(entry: dict, workdir: Path) -> str:
    """Start (or restart) a persistent compiler container bind-mounting
    workdir at /w, and return its name."""
    name = container_name(entry)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d", "--platform", "linux/amd64",
            "--name", name,
            "-v", f"{workdir}:/w",
            entry["image"], "sleep", "infinity",
        ],
        check=True, capture_output=True,
    )
    return name


def stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def env_for(entry: dict, container: str, workdir: Path) -> dict:
    env = dict(os.environ)
    env.update(
        SR_CONTAINER=container,
        SR_WORKDIR=str(workdir),
        SR_COMPILER=compiler_exe(entry),
        SR_STD=entry["std"],
        SR_SIGNATURE=entry["signature"],
    )
    return env


def crashes(entry: dict, source: Path, container: str, workdir: Path) -> bool:
    result = subprocess.run(
        ["bash", str(CHECK_SCRIPT), str(source)],
        env=env_for(entry, container, workdir),
    )
    return result.returncode == 0


def reduce_entry(entry: dict) -> dict:
    """Reduce a single entry and return a result record."""
    workdir = entry["dir"] / "work"
    workdir.mkdir(exist_ok=True)
    source = workdir / "reduced.cpp"
    # Resume from an in-progress reduction if one is present (shrink ray
    # reduces in place), so a run interrupted by the slow emulated
    # oracle can be restarted without losing work.
    if not source.exists():
        shutil.copy(entry["dir"] / "original.cpp", source)
    original_size = (entry["dir"] / "original.cpp").stat().st_size

    container = start_container(entry, workdir)
    try:
        if not crashes(entry, source, container, workdir):
            return {
                "id": entry["id"],
                "status": "ERROR: original does not reproduce the crash",
            }

        env = env_for(entry, container, workdir)
        start = time.time()
        # The old compilers only run under emulation here, and emulation
        # penalises concurrency (parallel emulated compiles thrash and
        # run slower than sequential ones), so we reduce single-threaded.
        # An explicit timeout avoids shrink ray's dynamic-timeout
        # calibration, which would otherwise probe a slow first call.
        proc = subprocess.run(
            [
                "uv", "run", "shrinkray",
                "--ui=basic", "--no-history", "--in-place",
                "--parallelism=1", "--timeout=10",
                str(CHECK_SCRIPT), str(source),
            ],
            cwd=REPO_ROOT, env=env,
            capture_output=True, text=True,
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            return {
                "id": entry["id"],
                "status": f"ERROR: shrinkray exited {proc.returncode}",
                "stderr": proc.stderr[-2000:],
            }

        reduced_size = source.stat().st_size
        still = crashes(entry, source, container, workdir)
        return {
            "id": entry["id"],
            "status": "ok" if still else "ERROR: reduced file no longer crashes",
            "original_bytes": original_size,
            "reduced_bytes": reduced_size,
            "ratio": round(100 * (1 - reduced_size / original_size), 1),
            "seconds": round(elapsed, 1),
        }
    finally:
        stop_container(container)


def check_entry(entry: dict) -> dict:
    workdir = entry["dir"] / "work"
    workdir.mkdir(exist_ok=True)
    container = start_container(entry, workdir)
    try:
        ok = crashes(entry, entry["dir"] / "original.cpp", container, workdir)
        return {"id": entry["id"], "reproduces": ok}
    finally:
        stop_container(container)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="entry ids to run (default: all)")
    parser.add_argument(
        "--check", action="store_true",
        help="only verify each entry still crashes its compiler",
    )
    args = parser.parse_args()

    if shutil.which("docker") is None:
        print("docker not found on PATH", file=sys.stderr)
        return 2

    entries = load_entries(args.ids)
    if not entries:
        print("no matching corpus entries", file=sys.stderr)
        return 2

    results = []
    for entry in entries:
        action = "Checking" if args.check else "Reducing"
        print(f"{action} {entry['id']} ({entry['image']}) ...", flush=True)
        record = check_entry(entry) if args.check else reduce_entry(entry)
        results.append(record)
        print("   " + json.dumps(record), flush=True)

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    ok = all(
        r.get("reproduces") if args.check else r.get("status") == "ok"
        for r in results
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
