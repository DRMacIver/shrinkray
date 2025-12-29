#!/usr/bin/env python3
"""Update gallery GIFs from VHS tape files.

This script regenerates gallery GIFs when:
- The tape file has changed since the GIF was last generated
- Any file in src/ has changed since the GIF was last generated
- Any source file in the gallery subdirectory (excluding .gif/.png) has changed

It uses git to efficiently check for changes, falling back to file
modification times if there are uncommitted changes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def git_checkout(paths: list[str]) -> None:
    """Checkout files from git, restoring them to their committed state.

    Silently ignores paths that aren't tracked or don't exist in git.
    """
    if not paths:
        return

    # Filter to only paths that exist in git
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--"] + paths,
        capture_output=True,
    )
    # Even if some files aren't tracked, try to checkout the ones that are
    subprocess.run(
        ["git", "checkout", "--"] + paths,
        capture_output=True,
    )


def get_resettable_gallery_paths(tapes: list[Path], include_outputs: bool) -> list[str]:
    """Get paths to gallery files that should be reset to git version.

    Args:
        tapes: List of tape file paths
        include_outputs: If True, include generated outputs (gif, png).
                        If False, only include source files that may be modified
                        during demos (like hello.py being reduced by shrinkray).

    Returns all non-executable files except .tape files.
    """
    paths = []
    output_extensions = {".gif", ".png"}

    for tape in tapes:
        gallery_dir = tape.parent
        for f in gallery_dir.iterdir():
            if not f.is_file():
                continue
            # Skip tape files (the source of truth for regeneration)
            if f.suffix.lower() == ".tape":
                continue
            # Skip executable files (test scripts)
            if os.access(f, os.X_OK):
                continue
            # Skip outputs if not including them
            if not include_outputs and f.suffix.lower() in output_extensions:
                continue
            paths.append(str(f))

    return paths


def get_git_commit_time(path: str) -> int | None:
    """Get the Unix timestamp of the last commit that touched a path.

    Returns None if the path has uncommitted changes or isn't tracked.
    """
    # Check if the path has uncommitted changes
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None  # Has uncommitted changes

    # Check if the path has staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None  # Has staged changes

    # Get the commit time
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    return int(result.stdout.strip())


def get_file_mtime(path: str) -> int:
    """Get the modification time of a file."""
    return int(os.path.getmtime(path))


def get_latest_src_time() -> tuple[int, bool]:
    """Get the latest modification time of any file in src/.

    Returns (timestamp, used_git) where used_git indicates whether
    git commit times were used (True) or file mtimes (False).
    """
    # Check if src/ has any uncommitted changes
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "src/"],
        capture_output=True,
    )
    has_uncommitted = result.returncode != 0

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "src/"],
        capture_output=True,
    )
    has_staged = result.returncode != 0

    if has_uncommitted or has_staged:
        # Fall back to file modification times
        latest = 0
        for root, _, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    mtime = get_file_mtime(path)
                    latest = max(latest, mtime)
        return latest, False

    # Use git commit time for src/
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", "src/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        # No commits touching src/ yet, use file times
        latest = 0
        for root, _, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    mtime = get_file_mtime(path)
                    latest = max(latest, mtime)
        return latest, False

    return int(result.stdout.strip()), True


def get_gallery_dir_time(gallery_dir: Path, use_git: bool) -> int:
    """Get the latest modification time of source files in a gallery subdirectory.

    Source files are any files that are not .gif or .png (the generated outputs).
    """
    source_files = [
        f
        for f in gallery_dir.iterdir()
        if f.is_file() and f.suffix.lower() not in (".gif", ".png")
    ]

    if not source_files:
        return 0

    latest = 0
    for f in source_files:
        if use_git:
            t = get_git_commit_time(str(f))
            if t is None:
                t = get_file_mtime(str(f))
        else:
            t = get_file_mtime(str(f))
        latest = max(latest, t)

    return latest


def needs_update(tape_path: Path, gif_path: Path, src_time: int, use_git: bool) -> bool:
    """Check if a GIF needs to be regenerated."""
    if not gif_path.exists():
        return True

    # Get gif modification time
    if use_git:
        gif_time = get_git_commit_time(str(gif_path))
        if gif_time is None:
            # GIF has uncommitted changes or isn't tracked, use mtime
            gif_time = get_file_mtime(str(gif_path))
    else:
        gif_time = get_file_mtime(str(gif_path))

    # Get tape modification time
    if use_git:
        tape_time = get_git_commit_time(str(tape_path))
        if tape_time is None:
            tape_time = get_file_mtime(str(tape_path))
    else:
        tape_time = get_file_mtime(str(tape_path))

    # Get gallery directory source files time
    gallery_dir_time = get_gallery_dir_time(tape_path.parent, use_git)

    # Check if tape, src, or gallery source files are newer than gif
    return tape_time > gif_time or src_time > gif_time or gallery_dir_time > gif_time


def find_tapes() -> list[Path]:
    """Find all tape files in the gallery directory."""
    gallery = Path("gallery")
    if not gallery.exists():
        return []
    return sorted(gallery.rglob("*.tape"))


def run_vhs(tape_path: Path) -> bool:
    """Run VHS on a tape file. Returns True on success."""
    print(f"Generating: {tape_path}")

    # VHS generates output files relative to the tape's directory
    tape_dir = tape_path.parent

    result = subprocess.run(
        ["vhs", tape_path.name],
        cwd=tape_dir,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update gallery GIFs from VHS tape files"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if updates are needed without running VHS",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all GIFs",
    )
    args = parser.parse_args()

    tapes = find_tapes()
    if not tapes:
        print("No tape files found in gallery/")
        return 0

    # Checkout gallery files from git before checking timestamps
    # This ensures we compare against committed versions, not local modifications
    all_resettable = get_resettable_gallery_paths(tapes, include_outputs=True)

    print("Restoring gallery files from git...")
    git_checkout(all_resettable)

    src_time, used_git = get_latest_src_time()
    if used_git:
        print("Using git commit times for change detection")
    else:
        print("Using file modification times (src/ has uncommitted changes)")

    updates_needed = []
    for tape in tapes:
        # Derive gif path from tape path
        gif_path = tape.with_suffix(".gif")

        if args.force or needs_update(tape, gif_path, src_time, used_git):
            updates_needed.append(tape)

    if not updates_needed:
        print("All gallery GIFs are up to date")
        return 0

    print(f"\nGIFs needing update: {len(updates_needed)}")
    for tape in updates_needed:
        print(f"  - {tape}")

    if args.check:
        return 1  # Indicate updates are needed

    # Run VHS on each tape
    print()
    failed = []
    for tape in updates_needed:
        if not run_vhs(tape):
            failed.append(tape)

    if failed:
        print(f"\nFailed to generate {len(failed)} GIF(s):")
        for tape in failed:
            print(f"  - {tape}")
        # Restore all files before returning (including outputs, since generation failed)
        print("Restoring gallery files from git...")
        git_checkout(all_resettable)
        return 1

    # Restore source files that may have been modified during demos
    # (e.g., shrinkray reducing hello.py), but NOT outputs which we just generated
    source_paths = get_resettable_gallery_paths(tapes, include_outputs=False)
    print("Restoring gallery source files from git...")
    git_checkout(source_paths)

    print(f"\nSuccessfully updated {len(updates_needed)} GIF(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
