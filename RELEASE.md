- When the LLM is enabled and the input's language has a tree-sitter grammar,
  Shrink Ray now also uses the model for targeted transformations: the grammar
  finds the site (a call to a function defined in the file, a use of a
  variable/type alias/macro, a function body, a loop, a constant expression)
  and the model rewrites just that site — inlining the call or definition,
  stubbing the body, unrolling the loop's first iteration, or folding the
  constant — leaving the rest of the file untouched. These unlock reductions
  (like deleting a function once its calls are inlined) that were previously
  only found for C and C++.
