- Added `--restart/--no-restart` (default on). The restart phase re-reduces
  from the original input once a fixpoint is reached, which can find smaller
  results greedy reduction misses but costs extra work; `--no-restart` skips
  it.
