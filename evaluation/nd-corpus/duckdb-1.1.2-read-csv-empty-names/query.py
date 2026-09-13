"""Read the CSV in argv[1] with duckdb 1.1.2 using two empty column names
(duckdb/duckdb#14428) and print the rows as repr.

Exit 0 and print the rows if the query ran; exit 1 if duckdb raised a Python
exception. A crash of the duckdb library kills the interpreter with a signal.
"""

import sys

import duckdb


def main() -> int:
    path = sys.argv[1].replace("'", "''")
    try:
        rows = duckdb.sql(
            f"from read_csv('{path}', header=false, names=['', ''])"
        ).fetchall()
    except duckdb.Error:
        return 1
    print(repr(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
