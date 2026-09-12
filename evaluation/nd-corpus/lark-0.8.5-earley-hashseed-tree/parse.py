"""Parse input.txt (next to this script) with the grammar in argv[1] using
the Earley parser and print the tree.

Exit 0 with the pretty-printed tree on stdout if parsing succeeds, exit 1
otherwise. The interestingness test compares this output between a
hash-randomised run and a PYTHONHASHSEED=0 run.
"""

import os
import sys

import lark


HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    with open(sys.argv[1]) as f:
        grammar = f.read()
    with open(os.path.join(HERE, "input.txt")) as f:
        text = f.read()
    try:
        parser = lark.Lark(grammar, parser="earley")
        tree = parser.parse(text)
    except Exception:
        return 1
    sys.stdout.write(tree.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
