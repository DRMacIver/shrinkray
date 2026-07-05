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

Shrink ray → reducer:

- **reduce** `{"reduce": <base64>}` — "reduce this test case to a fixpoint and
  tell me when you are idle".
- **feedback** `{"content": <base64>, "interesting": <bool>}` — the result for a
  query, or an unsolicited update to the current test case.

Reducer → shrink ray:

- **query** `{"content": <base64>}` — a candidate it wants evaluated.
- **idle** `{"idle": true}` — "I have reached a fixpoint for the current reduce
  request".

Sequence:

1. Shrink ray sends a reduce request with the current test case. (The first
   message the reducer receives is always a reduce request.)
2. The reducer emits queries. Shrink ray runs its interestingness test on each
   (up to `parallelism` concurrently) and replies with a feedback message
   echoing the query's `content` plus the boolean result. The reducer adopts a
   candidate as its new current when the reply says it is interesting.
3. When the reducer can make no more progress it sends idle; shrink ray's pass
   returns, leaving the reducer alive.
4. On a later invocation shrink ray sends another reduce request (with the
   possibly-changed current) and the reducer works again. It exits only when
   shrink ray closes its stdin.

Keeping the reducer alive across reduce requests means expensive startup (such
as importing libcst) is paid once. A reducer that instead **exits** at a
fixpoint (closing its stdout) is also supported: shrink ray relaunches it next
time. Shrink ray also terminates a reducer that produces no output for a timeout
(default 60s).

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
