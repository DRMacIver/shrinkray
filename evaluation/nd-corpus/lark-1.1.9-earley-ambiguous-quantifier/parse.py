"""Parse a fixed input with the grammar in argv[1] and print the tree.

Exit 0 with the pretty-printed tree on stdout if parsing succeeds, exit 1
otherwise (grammar error, parse error). The interestingness test compares
this output between a hash-randomised run and a PYTHONHASHSEED=0 run.
"""

import sys

from lark import Lark


INPUT = "a.?"


def main() -> int:
    with open(sys.argv[1]) as f:
        grammar = f.read()
    try:
        parser = Lark(grammar)
        tree = parser.parse(INPUT)
    except Exception:
        return 1
    sys.stdout.write(tree.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
