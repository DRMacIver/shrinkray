"""Download real-world files from open-source projects and keep the ones that
pass our strict front-end parse check, saving them as ``wild_*`` seeds.

Idempotent: downloads are cached under a scratch dir and re-filtered each run.
Run under the same env as langs.py (uv run --with ...).

    python evaluation/sortkey/fetch_wild_seeds.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from langs import LANGUAGES, seeds_dir  # noqa: E402

CACHE = Path("/private/tmp/claude-501/wild-seed-cache")

# Candidate real-world sources. TheAlgorithms/{C,C-Plus-Plus} give genuinely
# self-contained programs (standard headers only) that pass -fsyntax-only.
CANDIDATES: dict[str, list[str]] = {
    "c": [
        "https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/merge_sort.c",
        "https://raw.githubusercontent.com/TheAlgorithms/C/master/searching/binary_search.c",
        "https://raw.githubusercontent.com/TheAlgorithms/C/master/misc/quartile.c",
        "https://raw.githubusercontent.com/zserge/jsmn/25647e692c7906b96ffd2b05ca54c097948e879c/jsmn.h",
    ],
    "cpp": [
        "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/math/sieve_of_eratosthenes.cpp",
        "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/sorting/quick_sort.cpp",
        "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/others/happy_number.cpp",
        "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/data_structures/queue_using_two_stacks.cpp",
    ],
    "python": [
        "https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/colorsys.py",
        "https://raw.githubusercontent.com/pallets/click/8.1.7/src/click/globals.py",
        "https://raw.githubusercontent.com/pallets/jinja/3.1.2/src/jinja2/loaders.py",
        "https://raw.githubusercontent.com/psf/requests/v2.31.0/src/requests/hooks.py",
    ],
    "sql": [
        "https://raw.githubusercontent.com/prisma/database-schema-examples/main/postgres/basic-blog/schema.sql",
    ],
    "json": [
        "https://raw.githubusercontent.com/expressjs/express/4.18.2/package.json",
        "https://raw.githubusercontent.com/tsconfig/bases/main/bases/node18.json",
        "https://raw.githubusercontent.com/SchemaStore/schemastore/master/src/schemas/json/prettierrc.json",
    ],
    "xml": [
        "https://raw.githubusercontent.com/junit-team/junit4/r4.13.2/pom.xml",
        "https://raw.githubusercontent.com/feathericons/feather/v4.29.0/icons/anchor.svg",
        "https://raw.githubusercontent.com/simple-icons/simple-icons/13.0.0/icons/git.svg",
        "https://raw.githubusercontent.com/apache/maven/maven-3.9.6/pom.xml",
    ],
    "html": [
        "https://raw.githubusercontent.com/h5bp/html5-boilerplate/v9.0.0/dist/index.html",
        "https://raw.githubusercontent.com/mdn/beginner-html-site/gh-pages/index.html",
        "https://raw.githubusercontent.com/mdn/learning-area/main/html/introduction-to-html/getting-started/index.html",
    ],
}

SIZE_CAP = 20000  # bytes; keep seeds modest so reductions finish quickly
KEEP_PER_LANG = 2


def name_token(url: str) -> str:
    """A short owner_basename token to disambiguate same-named files."""
    parts = url.split("/")
    owner = parts[3] if len(parts) > 3 else "src"
    basename = parts[-1].rsplit(".", 1)[0]
    return f"{owner}_{basename}"


def download(url: str) -> bytes | None:
    if url.strip().endswith("..."):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    key = url.replace("/", "_").replace(":", "_")
    cached = CACHE / key
    if cached.exists():
        return cached.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {url}: {exc}")
        return None
    cached.write_bytes(data)
    return data


def main() -> int:
    for lang_name, urls in CANDIDATES.items():
        lang = LANGUAGES[lang_name]
        print(f"== {lang_name} ==")
        passing: list[tuple[int, str, bytes]] = []
        for url in urls:
            data = download(url)
            if data is None:
                continue
            if len(data) > SIZE_CAP:
                print(f"  skip (too big {len(data)}B): {url.rsplit('/', 1)[-1]}")
                continue
            ok = lang.parse(data)
            print(f"  parse={ok!s:5} {len(data):>6}B  {url.rsplit('/', 1)[-1]}")
            if ok:
                passing.append((len(data), name_token(url), data))
        passing.sort()
        dest = seeds_dir() / lang_name
        for _size, token, data in passing[:KEEP_PER_LANG]:
            out = dest / f"wild_{token}.{lang.ext}"
            out.write_bytes(data)
            print(f"  -> saved {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
