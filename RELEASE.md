- Reduced memory use on inputs containing a very large number of integer
  literals or identifiers: the passes that rewrite them now process them with
  bounded concurrency instead of all at once.
- Grammar-aware reduction now applies to more file types, recognised by
  extension: SCSS/Less, GraphQL, Protobuf, Terraform/HCL, Vue, Svelte, Solidity,
  Gradle/Groovy, Clojure, Fortran, PowerShell, and CMake. Files of these types
  now benefit from structure-aware deletions rather than byte-level cuts alone.
