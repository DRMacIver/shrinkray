# Reduction Passes Overview

This document provides a high-level overview of the reduction pass modules. For detailed information about specific passes, see the docstrings in each module.

## Module Overview

### bytes.py

Byte-level reduction passes that operate on raw bytes. These are the foundation of Shrink Ray's reduction strategy, as all file formats ultimately reduce to bytes. Includes passes for bracket manipulation (`hollow`, `lift_braces`, `debracket`), byte span deletion, whitespace normalisation, and byte value reduction. Also provides the `Split` and `Tokenize` formats for viewing bytes as sequences.

### sequences.py

Generic operations on sequences (lists, tuples, etc.). Provides `block_deletion` for removing contiguous blocks, `delete_elements` for single-element removal, and `delete_duplicates` for removing duplicate elements. These are typically used via `compose()` with a format like `Split(b"\n")`.

### genericlanguages.py

Reduction passes for "things that look like programming languages" - text with comments, identifiers, brackets, and integer literals. Includes comment removal (`cut_comment_like_things`), integer literal reduction (`reduce_integer_literals`), identifier normalisation (`normalize_identifiers`), and bracket simplification. Language-agnostic, working on any text that uses common programming conventions.

### python.py

Python-specific AST-aware reductions using libcst. Includes passes for lifting indented constructs (replacing `if`/`while`/`with` blocks with their bodies), replacing function bodies with `...`, stripping type annotations, and deleting statements. These understand Python syntax and produce valid Python output.

### json.py

JSON-specific passes. Provides `DeleteIdentifiers` for recursively removing keys from JSON objects throughout the structure.

### sat.py

Passes for DIMACS CNF files (SAT solver input format). Includes clause deletion, literal deletion, unit propagation, and literal sign flipping. The `DimacsCNF` format parses bytes into a list of clauses (each clause being a list of integers).

### clangdelta.py

C/C++ support via creduce's `clang_delta` tool. Unlike other passes, these are **pumps** - they may temporarily increase code size. For example, inlining a function makes code larger but may enable further reductions. Wraps clang_delta transformations like `simple-inliner`, `remove-unused-function`, and `rename-var`.

## Pass Ordering in ShrinkRay

The `ShrinkRay` reducer organises passes into stages:

1. **initial_cuts**: Fast, high-value passes (comments, hollow, large blocks) with timeout-based cancellation
2. **great_passes**: Core loop (line deletion, token deletion, lift_braces) - runs until no progress
3. **ok_passes**: Run when great_passes plateau (smaller blocks, normalisation)
4. **last_ditch_passes**: Expensive or low-yield passes (byte lowering, brackets)

Great passes loop until no progress, tracking which passes succeeded to prioritise them on subsequent iterations.
