# Compiler-bug corpus

A small corpus of **real** C++ compiler bugs — inputs that make a
specific old version of gcc or clang crash with an internal compiler
error — used to test shrink ray's pure-Python C/C++ reduction passes
(the `clang_delta` replacement in `src/shrinkray/passes/cpp.py`) on
realistic material, and to compare its output against `c-reduce`.

Each entry is a plausible, pre-reduction-sized program (roughly 0.8–1.4
kB of application-shaped scaffolding — namespaces, helper templates,
unrelated classes) with a genuine bug trigger buried inside. Reducing
it should strip the scaffolding back down to the essential trigger,
exercising exactly the transformations the C/C++ passes implement:
namespace splicing, function/definition deletion, base-class removal,
template stripping, call simplification, and so on.

## Provenance

The triggers were found by screening gcc's own `g++.dg/cpp1y`
testsuite and a battery of C++11/14 dark-corner snippets against old
compiler Docker images (`gcc:4.9`, `silkeh/clang:3.5`), then embedding
each confirmed crash in a larger realistic program and re-verifying the
crash. `csmith` was also run against `gcc:4.9` but found no
wrong-code/ICE cases (gcc 4.9 is robust on csmith's UB-free output);
the C++ front-end/codegen bugs were far higher-yield.

| Entry | Compiler | Crash signature |
|-------|----------|-----------------|
| `gcc49-pr61636-genlambda-member` | gcc 4.9 | front-end segfault: generic lambda names an enclosing-class member |
| `gcc49-pr64382-genlambda-template-member` | gcc 4.9 | front-end segfault: generic lambda in a class-template member |
| `gcc49-pr77739-variadic-auto-lambda` | gcc 4.9 | ICE in `create_tmp_var`: auto variadic member returns a pack-capturing lambda |
| `gcc49-udlit-char-pack-template` | gcc 4.9 | ICE in `type_dependent_expression_p`: char-pack UDL in a default template argument |
| `clang35-variadic-callback-blockdecl` | clang 3.5 | codegen crash in `GetAddrOfBlockDecl`: variadic generic lambda over a captured pointer |

Each entry directory holds:
- `original.cpp` — the realistic pre-reduction program.
- `meta.json` — compiler image, `-std`, the crash signature substring, and a description.
- `shrinkray_reduced.cpp` / `creduce_reduced.cpp` — the reduced outputs (committed for comparison).

## Running

The old compilers are amd64-only and run under emulation on Apple
Silicon (slow but functional). Each reduction uses a persistent
compiler container as the interestingness oracle. **Emulation penalises
concurrency**, so reductions run single-threaded (`--parallelism=1`).

```bash
# Verify every entry still crashes its compiler
python3 compiler-bug-corpus/run.py --check

# Reduce every entry with shrink ray (writes work/reduced.cpp, resumable)
python3 compiler-bug-corpus/run.py

# Reduce one entry
python3 compiler-bug-corpus/run.py gcc49-udlit-char-pack-template
```

`check.sh` is the interestingness test: it compiles a candidate in the
compiler container and reports "interesting" iff the specific crash
signature is still present.

## Comparing against c-reduce

`creduce/` holds a driver that reduces the same entries with c-reduce
2.11.0 (and its bundled `clang_delta`) against the identical
compiler-in-Docker oracle, so the two tools are compared on equal
footing. Build the c-reduce host image once, then run the driver:

```bash
docker build --platform linux/amd64 -t creduce-host \
    -f compiler-bug-corpus/creduce/Dockerfile.creduce compiler-bug-corpus/creduce

# one entry (writes <entry>/creduce_reduced.cpp)
compiler-bug-corpus/creduce/run_creduce.sh gcc49-udlit-char-pack-template
```

c-reduce runs inside `creduce-host` and reaches the compiler container
via the mounted Docker socket. Because c-reduce's long tail is
prohibitively slow under emulation, each run is capped by
`CREDUCE_BUDGET` seconds (default 900) and the in-place result is taken.

## Results

See `RESULTS.md` for the size comparison against c-reduce and the
analysis of what c-reduce removes that shrink ray currently can't
(type replacement and namespace-qualifier rewriting).
