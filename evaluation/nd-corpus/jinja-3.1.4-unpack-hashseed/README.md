# Jinja2 3.1.4: tuple-unpacking `set` exports its names in hash-seed order

- **Source issue:** https://github.com/pallets/jinja/issues/2021
- **Tool:** `jinja2==3.1.4` (with `markupsafe==3.0.2`), run under Python
  3.12 from a uv venv.
- **Input under reduction:** `original.jinja`, the two `set` statements
  from the issue's inline reproducer (the issue has no attachment).

## The bug

Compiling a template is meant to be reproducible, but for a `set` that
unpacks a tuple (`{% set a, b = ... %}`) Jinja 3.1.4 emits the
`context.vars.update({...})` / `context.exported_vars.update(...)` lines
in the order it iterates a *set* of the assigned names. Which order that
is depends on the process's hash seed, so the same template compiles to
different Python source from one run to the next. Fixed in Jinja 3.1.5
(pallets/jinja#2022 sorts the names).

## Nondeterminism source

Python hash randomisation (set iteration order over identifier strings).
Fixing `PYTHONHASHSEED` makes any given run reproducible, so a
deterministic variant exists.

## Detection

`compile.py` compiles the template with `Environment().compile(raw=True)`,
checks that the generated module is itself valid Python, and prints it.
`test.sh` runs it twice on the candidate: once under the default
(randomised) hash seed and once with `PYTHONHASHSEED=0`, and is
interesting iff both compile and the generated sources differ. A template
whose `set` statements assign a single name each compiles identically
under every seed, so reduction cannot "succeed" by deleting the unpacking.

`test-deterministic.sh` pins the candidate run to `PYTHONHASHSEED=1`,
which on the original template orders both statements' names the other
way round from seed 0, and is interesting iff the seed-1 and seed-0
sources differ.

## Measured reproduction rate (this machine, macOS arm64)

- `test.sh` on `original.jinja`: **15 / 20** runs interesting. With two
  unpacking statements of two names each there are four possible
  outputs, so a randomised run differs from seed 0 about three times in
  four.
- A single `{% set a, b = c %}` statement: 6 / 10, as expected for two
  possible outputs.
- `test-deterministic.sh`: 3 / 3 (deterministic, as expected).
- An empty template: 0 / 1.

## Setup notes

`setup.sh` creates `.tool/` as a uv venv with `jinja2==3.1.4`. Nothing is
installed globally.
