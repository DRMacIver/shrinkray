"""Compile the template in argv[1] to Python source and print that source.

Exit 0 with the generated module on stdout if Jinja compiles the template
and the generated code is itself valid Python, exit 1 otherwise. The
interestingness test compares this output between a hash-randomised run
and a PYTHONHASHSEED=0 run.
"""

import sys

from jinja2 import Environment


def main() -> int:
    with open(sys.argv[1]) as f:
        source = f.read()
    try:
        generated = Environment().compile(source, raw=True)
        # raw=True stops before compiling the generated Python; require that
        # step too, so two malformed code-generation results are not a hit.
        compile(generated, sys.argv[1], "exec")
    except Exception:
        return 1
    sys.stdout.write(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
