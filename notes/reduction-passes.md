# Reduction Passes Guide

This document catalogs the reduction passes and their strategies.

## Byte-Level Passes (bytes.py)

### Deletion Passes

**delete_byte_spans**
Tries deleting contiguous byte ranges. Generates spans of various sizes and positions.
```
Input:  "hello world goodbye"
Tries:  "world goodbye", "hello goodbye", "hello world", etc.
```

**short_deletions**
Deletes small (1-8 byte) sequences throughout the file.
```
Input:  "x = 1 + 2"
Tries:  " = 1 + 2", "x= 1 + 2", "x  1 + 2", etc.
```

**hollow**
Keeps only the first and last portions of the file.
```
Input:  "int main() { lots of code here }"
Result: "int main() { }"  (if interesting)
```

### Structural Passes

**Split(delimiter)** (Format)
Parses bytes into list of segments, enables sequence operations.
```python
compose(Split(b"\n"), block_deletion(1, 10))  # Delete 1-10 lines
compose(Split(b";"), delete_duplicates)       # Delete duplicate statements
```

**Tokenize()** (Format)
Parses into tokens respecting quotes and brackets.
```python
compose(Tokenize(), block_deletion(1, 20))  # Delete 1-20 tokens
```

**lift_braces**
Replaces `{...}` or `(...)` with just the content.
```
Input:  "if (x) { return 1; }"
Tries:  "if (x)  return 1; " or "if x { return 1; }"
```

**debracket**
Removes matching bracket pairs.
```
Input:  "(x + y)"
Result: "x + y"
```

### Whitespace Passes

**remove_indents**
Reduces or removes indentation.
```
Input:  "    x = 1"
Result: "x = 1"
```

**remove_whitespace**
Deletes whitespace characters.
```
Input:  "x = 1 + 2"
Tries:  "x= 1 + 2", "x =1 + 2", etc.
```

**replace_space_with_newlines**
Replaces spaces with newlines (can help line-based passes).

### Normalization Passes

**lower_bytes** / **lower_individual_bytes**
Reduces byte values (e.g., 'Z' -> 'A' -> '0' -> '\0').

**standard_substitutions**
Common replacements: "true" -> "1", "false" -> "0", etc.

**line_sorter**
Sorts lines to make duplicates adjacent (helps delete_duplicates).

## Sequence Passes (sequences.py)

**block_deletion(min, max)**
Deletes contiguous blocks of elements.
```
[a, b, c, d, e] with block_deletion(2, 3)
Tries: [c, d, e], [a, d, e], [a, b, e], [a, b, c], etc.
```

**delete_duplicates**
Removes all but one occurrence of duplicate elements.
```
[a, b, a, c, a] -> [a, b, c]
```

**delete_elements**
Deletes individual elements one at a time.

## Generic Language Passes (genericlanguages.py)

**cut_comment_like_things**
Removes comments in various syntaxes: `#...`, `//...`, `/*...*/`, `"""..."""`

**reduce_integer_literals**
Binary-searches integer literals toward 0.
```
"x = 12345" -> "x = 0" (if works) or smallest working value
```

**combine_expressions**
Evaluates simple arithmetic.
```
"1 + 2" -> "3"
```

**normalize_identifiers**
Renames identifiers to shorter versions.
```
"longVariableName" -> "a"
```

**merge_adjacent_strings**
Removes whitespace between adjacent string quotes.
```
"'hello' 'world'" -> "'hello''world'" -> (later) "'helloworld'"
```

**simplify_brackets**
Replaces bracket types: `{}` -> `[]` -> `()` (lexicographically smaller).

## Python Passes (python.py)

Uses libcst for AST-aware transformations.

**lift_indented_constructs**
Replaces `if/while/with` blocks with their body.
```python
if True:
    x = 1
# becomes:
x = 1
```

**replace_bodies_with_ellipsis**
Replaces block bodies with `...`.
```python
def foo():
    complex_stuff()
# becomes:
def foo():
    ...
```

**strip_annotations**
Removes type annotations.
```python
def foo(x: int) -> str:
# becomes:
def foo(x):
```

**delete_statements** / **replace_statements_with_pass**
Removes or replaces statements.

## JSON Passes (json.py)

**delete_identifiers**
Removes dictionary keys throughout the JSON structure.

## SAT Passes (sat.py)

For DIMACS CNF files (SAT solver input).

**delete_clauses**
Removes clauses from the formula.

**delete_variables**
Removes variables, simplifying clauses that contain them.

**unit_propagate**
Applies unit propagation to simplify the formula.
```
[[1], [-1, 2]] -> [[1], [2]]  (since 1 is forced true, -1 is false)
```

## C/C++ Pumps (clangdelta.py)

These use creduce's clang_delta tool for semantic transformations.

Notable transformations:
- **simple-inliner**: Inline function calls (may increase size temporarily)
- **remove-unused-function**: Delete unused functions
- **rename-var**: Shorten variable names
- **aggregate-to-scalar**: Decompose structs

These are "pumps" because some (like inlining) can increase code size, enabling further reductions.

## Pass Ordering in ShrinkRay

1. **initial_cuts**: Fast, high-value (comments, hollow, large blocks)
2. **great_passes**: Core loop (line deletion, token deletion, lift_braces)
3. **ok_passes**: When great_passes plateau (smaller blocks, normalization)
4. **last_ditch_passes**: Expensive/low-yield (byte lowering, brackets)

Great passes loop until no progress, tracking successful passes to prioritize them.
