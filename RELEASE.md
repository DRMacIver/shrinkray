- The timeout for interestingness tests now adapts to measured test runtimes
  over the course of the run, instead of staying fixed at 10x the first run's
  time. This speeds up reduction when tests get faster as the test case
  shrinks, while staying robust to variable timings under parallel load.
- When reduction stalls while tests are timing out, Shrink Ray now temporarily
  raises the timeout (up to `--timeout`, or 5 minutes if unset) to check
  whether slower test runs would unlock further reductions, and lowers it
  again if they don't. Reduction no longer ends while a raised timeout might
  still make progress.
- `--timeout` now sets the maximum the adaptive timeout may reach rather than
  a fixed timeout. With `--timeout` <= 0 the adaptive timeout has no upper
  bound: tests still get killed once they run well past recent runtimes, but
  the timeout can always be raised again, so no candidate is permanently lost.
- The TUI now shows the current test timeout and the fraction of recent test
  runs that timed out.
