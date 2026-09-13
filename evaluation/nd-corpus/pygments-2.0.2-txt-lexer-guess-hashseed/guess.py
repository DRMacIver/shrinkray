"""Print the name of the lexer Pygments 2.0.2 picks for the content of
argv[1] when it is named test.txt (as `pygmentize test.txt` would).

The filename is part of the bug (several lexers claim *.txt), so it is fixed
here rather than taken from the candidate's path.
"""

import sys

from pygments.lexers import get_lexer_for_filename


with open(sys.argv[1]) as f:
    code = f.read()
print(get_lexer_for_filename("test.txt", code).name)
