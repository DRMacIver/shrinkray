# External Reducers

An **external reducer** is a subprocess that reduces a test case by talking to
shrink ray over a small line-based protocol. It lets reduction logic live
outside the main process — used for the built-in libcst Python reducer, and
available to users via `--reduce-with`.

## Why

- **Isolation**: heavyweight or crash-prone reducers (libcst, third-party tools)
  run in their own process instead of the reducer's trio event loop.
- **Extensibility**: users can plug in their own reducers in any language
  without touching shrink ray, by implementing the protocol.

## The protocol

Newline-delimited JSON on the reducer's stdin/stdout; stderr is logged to a file
(under `.shrinkray/<run>/reducers/` when history is enabled). Test-case content
is base64-encoded so arbitrary bytes survive JSON.

- **query** (reducer → shrink ray): `{"content": <base64>}` — a candidate the
  reducer wants evaluated.
- **feedback** (shrink ray → reducer): `{"content": <base64>, "interesting": <bool>}`.

Sequence:

1. On launch shrink ray sends one feedback message: the initial test case, with
   `interesting` true. This is the reducer's starting point.
2. The reducer emits queries. Shrink ray runs its interestingness test on each
   (up to `parallelism` concurrently) and replies with a feedback message
   echoing the query's `content` plus the boolean result.
3. When the current test case changes (a reduction is adopted, possibly
   reformatted, or another pass advances it), shrink ray sends an unsolicited
   feedback message with the new content. The reducer distinguishes a reply
   from an update by whether the `content` matches an outstanding query.
4. The reducer exits when it has nothing left to try. Shrink ray also terminates
   it if it emits no query for a timeout (default 60s).

Because queries can be pipelined and answered concurrently, a reducer keeps its
own parallelism to keep shrink ray busy. Shrink ray passes its parallelism to
the subprocess in `SHRINKRAY_REDUCER_PARALLELISM`.

## Code layout

- `shrinkray/reducers/protocol.py` — message encode/decode and a line reader.
- `shrinkray/reducers/driver.py` — `RemoteReductionProblem` (a
  `ReductionProblem` whose `is_interesting` is answered over the protocol) and
  `run_reducer`, which drives ordinary reduction passes against it. This is how
  a pass written against `ReductionProblem` runs unchanged inside a reducer
  subprocess: only the problem it runs against changes.
- `shrinkray/reducers/python.py` — the built-in Python reducer's entry point.
- `shrinkray/passes/external.py` — the shrink-ray side: `drive_external_reducer`
  (the protocol loop) and `external_reducer`, which wraps a command line as a
  `ReductionPass`, managing the subprocess, its log file, and the timeout.

`ShrinkRay` builds its external reducer passes in `build_external_reducer_passes`:
the built-in Python reducer (when enabled and the input looks like Python)
followed by any user `--reduce-with` reducers.
