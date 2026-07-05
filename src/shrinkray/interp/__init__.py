"""Running the reducer and the TUI in one process on separate interpreters.

The trio-based reducer runs in the main interpreter (its native
dependencies cannot be imported in subinterpreters) while the
asyncio-based textual TUI runs in an isolated subinterpreter with its
own GIL. The two sides speak a line-oriented JSON protocol over a
socketpair.
"""
