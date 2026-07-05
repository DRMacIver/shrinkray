- Reduction passes are now scheduled adaptively: expensive passes that
  rarely find anything run only after everything else has converged, and a
  pass that recently made no progress is given only a short trial before
  Shrink Ray moves on to more promising work (it is still re-run in full
  before finishing, so final results are unaffected). Reductions make
  progress sooner and typically need fewer runs of the interestingness test.
