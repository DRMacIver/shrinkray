"""Natively build tree-sitter grammars on platforms without prebuilt bundles.

tree-sitter-language-pack publishes prebuilt grammar bundles only for
Linux, macOS and Windows, and its runtime fallback downloads the *Linux*
bundle on every other OS (its platform key maps any non-mac, non-windows
system to "linux"). Many grammar libraries happen to load anyway because
they are dependency-free ELF objects, but that is an accident, and any
grammar with a C++ scanner links against Linux's libc/libstdc++ and
cannot be loaded at all (on OpenBSD, nim was failing this way).

On platforms the pack has no bundles for, this script compiles every
grammar shrinkray maps from the upstream parser-sources archive straight
into the pack's cache, replacing whatever the Linux bundle left there.
It exits nonzero if any mapped language still fails to load afterwards,
so CI keeps covering shrinkray's full language set.

Run it with the interpreter that has shrinkray installed, after setting
XDG_CACHE_HOME if you want the cache somewhere specific. Requires cc,
c++, and zstd.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import tree_sitter_language_pack

from shrinkray.passes.treesitter import EXTENSION_LANGUAGES


def try_load(language: str) -> Exception | None:
    try:
        tree_sitter_language_pack.get_language(language)
        return None
    except Exception as e:
        return e


def cache_version_dir() -> Path:
    """The pack's versioned cache directory (created by load attempts)."""
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    candidates = sorted((cache_home / "tree-sitter-language-pack").glob("v*"))
    if not candidates:
        sys.exit(
            "No tree-sitter-language-pack cache directory found; "
            "expected the failed load attempts to have created one."
        )
    return candidates[-1]


def parser_sources(version_dir: Path) -> Path:
    """Download and unpack the upstream parser-sources archive."""
    manifest = json.loads((version_dir / "manifest.json").read_text())
    version = manifest["version"]
    some_bundle_url = next(iter(manifest["platforms"].values()))["url"]
    base_url = some_bundle_url.rsplit("/", 1)[0]
    url = f"{base_url}/parser-sources-{version}.tar.zst"

    workdir = Path(tempfile.mkdtemp(prefix="ts-parser-sources-"))
    archive = workdir / "parser-sources.tar.zst"
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)

    tar_path = workdir / "parser-sources.tar"
    with open(tar_path, "wb") as out:
        subprocess.run(["zstd", "-dc", str(archive)], stdout=out, check=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(workdir, filter="data")
    return workdir / "parsers"


def build_grammar(name: str, source_dir: Path, libs_dir: Path) -> None:
    """Compile one grammar's parser (and optional scanner) into libs_dir."""
    src = source_dir / name / "src"
    objects = []
    uses_cxx = False
    for filename, compiler in [
        ("parser.c", "cc"),
        ("scanner.c", "cc"),
        ("scanner.cc", "c++"),
    ]:
        source = src / filename
        if not source.exists():
            continue
        uses_cxx = uses_cxx or compiler == "c++"
        obj = src / (filename + ".o")
        subprocess.run(
            [compiler, "-O2", "-fPIC", "-I", str(src), "-c", str(source), "-o", str(obj)],
            check=True,
        )
        objects.append(str(obj))
    linker = "c++" if uses_cxx else "cc"
    # The library must be named after the symbol parser.c exports, not
    # the source directory: the registry resolves aliases first (e.g.
    # "csharp" -> "c_sharp") and looks for libtree_sitter_c_sharp.so.
    # The parameter list may be `(void)` or `()` depending on the
    # tree-sitter version that generated the parser (scss uses `()`).
    symbols = re.findall(
        r"const TSLanguage \*\s*tree_sitter_(\w+)\s*\(\s*(?:void)?\s*\)",
        (src / "parser.c").read_text(errors="replace"),
    )
    if not symbols:
        sys.exit(f"Could not find the exported grammar symbol in {src}/parser.c")
    out = libs_dir / f"libtree_sitter_{symbols[-1]}.so"
    subprocess.run([linker, "-shared", "-o", str(out), *objects], check=True)


def verify(languages: list[str]) -> int:
    failures = {lang: e for lang in languages if (e := try_load(lang)) is not None}
    for lang, e in failures.items():
        print(f"FAIL {lang}: {type(e).__name__}: {e}", flush=True)
    if failures:
        return 1
    print(f"All {len(languages)} mapped grammars load.", flush=True)
    return 0


def main() -> int:
    languages = sorted(set(EXTENSION_LANGUAGES.values()))

    if "--verify" in sys.argv:
        return verify(languages)

    if sys.platform.startswith(("linux", "darwin", "win")):
        # The pack has real prebuilt bundles here; nothing to repair, but
        # still verify so CI notices if a grammar goes missing upstream.
        return verify(languages)

    # No prebuilt bundles for this platform: whatever the runtime fallback
    # put in the cache is a Linux binary. Load attempts also make the pack
    # download its manifest, which tells us where the sources live.
    for lang in languages:
        try_load(lang)

    version_dir = cache_version_dir()
    libs_dir = version_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)
    sources = parser_sources(version_dir)

    for lang in languages:
        print(f"Building {lang}", flush=True)
        build_grammar(lang, sources, libs_dir)

    # Loaded libraries are cached in-process, so the post-build check must
    # run in a fresh interpreter to actually exercise the new builds.
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--verify"], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
