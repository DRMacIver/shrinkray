#!/usr/bin/env python3
"""Regenerate the results tables in evaluation/RESULTS.md.

Reads each corpus entry's meta.json, committed reduced outputs, and
result.json (written by run.py), and rewrites the block of RESULTS.md
between the BEGIN/END GENERATED markers. Everything outside the markers
is hand-written analysis and is left untouched.

Usage:
    python3 report.py
"""

import json
import re
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).parent
CORPUS_DIR = EVAL_DIR / "corpus"
RESULTS_MD = EVAL_DIR / "RESULTS.md"

BEGIN = "<!-- BEGIN GENERATED RESULTS (run report.py; do not edit by hand) -->"
END = "<!-- END GENERATED RESULTS -->"


def nows_size(path: Path) -> int:
    return len(re.sub(rb"\s+", b"", path.read_bytes()))


def load_rows() -> list[dict]:
    rows = []
    for meta_path in sorted(CORPUS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        entry_dir = meta_path.parent
        ext = meta["extension"]
        row = {
            "id": meta["id"],
            "format": meta["format"],
            "tool": meta["tool"],
            "original": entry_dir / ("original" + ext),
            "reduced": entry_dir / ("shrinkray_reduced" + ext),
            "creduce": entry_dir / ("creduce_reduced" + ext),
            "result": None,
        }
        result_path = entry_dir / "result.json"
        if result_path.exists():
            row["result"] = json.loads(result_path.read_text())
        rows.append(row)
    return rows


def size_cells(row: dict) -> dict:
    original = row["original"].stat().st_size
    cells = {"original": original}
    if row["reduced"].exists():
        reduced = row["reduced"].stat().st_size
        cells["reduced"] = reduced
        cells["nows"] = nows_size(row["reduced"])
        cells["ratio"] = f"{100 * (1 - reduced / original):.1f}%"
    if row["result"] is not None and row["result"].get("seconds") is not None:
        cells["seconds"] = row["result"]["seconds"]
    return cells


def main_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Entry | Format | Tool | Original | Reduced | Reduced `nows` | Ratio | Seconds |",
        "|-------|--------|------|---------:|--------:|---------------:|------:|--------:|",
    ]
    for row in rows:
        cells = size_cells(row)
        lines.append(
            "| {id} | {format} | {tool} | {original} | {reduced} | {nows} | {ratio} | {seconds} |".format(
                id=row["id"],
                format=row["format"],
                tool=row["tool"],
                original=cells["original"],
                reduced=cells.get("reduced", "—"),
                nows=cells.get("nows", "—"),
                ratio=cells.get("ratio", "—"),
                seconds=cells.get("seconds", "—"),
            )
        )
    return lines


def creduce_table(rows: list[dict]) -> list[str]:
    rows = [r for r in rows if r["creduce"].exists()]
    if not rows:
        return []
    lines = [
        "",
        "### c-reduce comparison (C/C++ entries)",
        "",
        "| Entry | Original | shrink ray | c-reduce | shrink ray `nows` | c-reduce `nows` |",
        "|-------|---------:|-----------:|---------:|------------------:|----------------:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {original} | {sr} | {cr} | {sr_nows} | {cr_nows} |".format(
                id=row["id"],
                original=row["original"].stat().st_size,
                sr=row["reduced"].stat().st_size if row["reduced"].exists() else "—",
                cr=row["creduce"].stat().st_size,
                sr_nows=nows_size(row["reduced"]) if row["reduced"].exists() else "—",
                cr_nows=nows_size(row["creduce"]),
            )
        )
    return lines


def main() -> int:
    rows = load_rows()
    generated = [BEGIN, ""]
    generated += main_table(rows)
    generated += creduce_table(rows)
    generated += ["", END]

    text = RESULTS_MD.read_text()
    if BEGIN not in text or END not in text:
        print(f"markers not found in {RESULTS_MD}", file=sys.stderr)
        return 1
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    RESULTS_MD.write_text(pre + "\n".join(generated) + post)
    print(f"updated {RESULTS_MD} ({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
