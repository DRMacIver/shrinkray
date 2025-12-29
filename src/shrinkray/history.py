"""History management for shrink ray reduction sessions.

Records initial state and all successful reductions to a .shrinkray directory,
allowing analysis and reproduction of reduction runs.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from attrs import define


if TYPE_CHECKING:
    pass


def sanitize_for_filename(s: str) -> str:
    """Replace unsafe characters with underscores, limit length.

    Keeps alphanumeric characters, dashes, underscores, and dots.
    Collapses multiple underscores and limits to 50 characters.
    """
    # Replace unsafe characters with underscore
    safe = re.sub(r"[^\w\-.]", "_", s)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing underscores and limit length
    return safe[:50].strip("_")


@define
class HistoryManager:
    """Manages history directory structure for reduction sessions.

    Creates and maintains a directory structure like:
        .shrinkray/<run_id>/
            initial/
                <target_file>     - Original file
                <test_file>       - Copy of interestingness test (if local)
                run.sh            - Wrapper script with exact original args
            reductions/
                0001/
                    <target_file> - Reduced file
                    <target_file>.out - stdout+stderr output
                0002/
                    ...
    """

    run_id: str
    history_dir: str  # Path to .shrinkray/<run_id>
    target_basename: str
    reduction_counter: int = 0
    also_interesting_counter: int = 0
    initialized: bool = False

    @classmethod
    def create(cls, test: list[str], filename: str) -> HistoryManager:
        """Create a new HistoryManager with a unique run ID.

        Args:
            test: The interestingness test command (list of strings)
            filename: Path to the target file being reduced
        """
        # Generate run ID: (test-basename)-(filename)-(datetime)-(random hex)
        test_name = sanitize_for_filename(os.path.basename(test[0]))
        file_name = sanitize_for_filename(os.path.basename(filename))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_hex = os.urandom(4).hex()
        run_id = f"{test_name}-{file_name}-{timestamp}-{random_hex}"

        history_dir = os.path.join(os.getcwd(), ".shrinkray", run_id)
        target_basename = os.path.basename(filename)

        return cls(
            run_id=run_id,
            history_dir=history_dir,
            target_basename=target_basename,
        )

    def initialize(
        self, initial_content: bytes, test: list[str], filename: str
    ) -> None:
        """Create initial directory structure and copy files.

        Args:
            initial_content: The original file content
            test: The interestingness test command
            filename: Path to the original target file
        """
        if self.initialized:
            return

        # Create directories
        initial_dir = os.path.join(self.history_dir, "initial")
        os.makedirs(initial_dir, exist_ok=True)
        os.makedirs(os.path.join(self.history_dir, "reductions"), exist_ok=True)

        # Copy original target file
        target_path = os.path.join(initial_dir, self.target_basename)
        with open(target_path, "wb") as f:
            f.write(initial_content)

        # Copy interestingness test if it's a local file
        test_path = test[0]
        copied_test_basename: str | None = None
        if os.path.isfile(test_path):
            copied_test_basename = os.path.basename(test_path)
            shutil.copy2(test_path, os.path.join(initial_dir, copied_test_basename))

        # Create wrapper script
        self._create_wrapper_script(test, copied_test_basename, initial_dir)

        self.initialized = True

    def _create_wrapper_script(
        self,
        test: list[str],
        copied_test_basename: str | None,
        initial_dir: str,
    ) -> None:
        """Create run.sh wrapper script.

        Args:
            test: The original test command
            copied_test_basename: Basename of copied test file, or None if not copied
            initial_dir: Path to the initial directory
        """
        script_path = os.path.join(initial_dir, "run.sh")

        # Build command, referencing test file via $(dirname "$0") if it was copied
        if copied_test_basename is not None:
            test_ref = f'"$(dirname "$0")/{copied_test_basename}"'
        else:
            test_ref = shlex.quote(test[0])

        # Quote remaining arguments
        args = " ".join(shlex.quote(arg) for arg in test[1:])
        if args:
            command = f"{test_ref} {args}"
        else:
            command = test_ref

        script_content = f'''#!/bin/bash
# Shrink Ray interestingness test wrapper
# Run with: ./run.sh [target_file]
# If no target_file specified, uses the original

DIR="$(dirname "$0")"
TARGET="${{1:-"$DIR/{self.target_basename}"}}"

{command} "$TARGET"
'''
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

    def record_reduction(
        self, test_case: bytes, output: bytes | None = None
    ) -> None:
        """Record a successful reduction.

        Args:
            test_case: The reduced file content
            output: Combined stdout/stderr from the test, or None if not captured
        """
        self.reduction_counter += 1
        subdir = os.path.join(
            self.history_dir,
            "reductions",
            f"{self.reduction_counter:04d}",
        )
        os.makedirs(subdir, exist_ok=True)

        # Write reduced file
        with open(os.path.join(subdir, self.target_basename), "wb") as f:
            f.write(test_case)

        # Write output if available
        if output is not None:
            output_name = f"{self.target_basename}.out"
            with open(os.path.join(subdir, output_name), "wb") as f:
                f.write(output)

    def record_also_interesting(
        self, test_case: bytes, output: bytes | None = None
    ) -> None:
        """Record an also-interesting test case.

        These are test cases that don't satisfy the main interestingness test
        but have some other interesting property indicated by a special exit code.

        Args:
            test_case: The file content
            output: Combined stdout/stderr from the test, or None if not captured
        """
        self.also_interesting_counter += 1
        subdir = os.path.join(
            self.history_dir,
            "also-interesting",
            f"{self.also_interesting_counter:04d}",
        )
        os.makedirs(subdir, exist_ok=True)

        # Write file
        with open(os.path.join(subdir, self.target_basename), "wb") as f:
            f.write(test_case)

        # Write output if available
        if output is not None:
            output_name = f"{self.target_basename}.out"
            with open(os.path.join(subdir, output_name), "wb") as f:
                f.write(output)
