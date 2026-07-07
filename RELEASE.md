- When the LLM is enabled and the input's language has a tree-sitter grammar,
  Shrink Ray now also uses the model to inline function calls: the grammar
  finds calls to functions defined in the file, and the model rewrites just
  that call, leaving the rest of the file untouched. Inlining a call often
  lets the function definition be deleted afterwards, unlocking reductions
  that were previously only found for C and C++.
