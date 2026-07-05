- Shrink Ray now has grammar-aware reduction passes for any language with a
  tree-sitter grammar (Go, Rust, JavaScript, Java, Haskell, and many others,
  selected by file extension). These delete whole syntactic constructs, lift
  nested structure into parent positions, replace constructs with smaller
  same-kind ones found elsewhere in the file, and delete dead declarations
  together with the imports only they used — the latter unblocks reduction in
  languages like Go whose compilers reject unused imports.
