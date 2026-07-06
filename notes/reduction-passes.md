# Reduction Passes Overview

This document provides a high-level overview of the reduction pass modules. For detailed information about specific passes, see the docstrings in each module.

## Module Overview

### bytes.py

Byte-level reduction passes that operate on raw bytes. These are the foundation of Shrink Ray's reduction strategy, as all file formats ultimately reduce to bytes. Includes passes for bracket manipulation (`hollow`, `lift_braces`, `debracket`), byte span deletion (`delete_byte_spans`, `short_deletions`, `lexeme_based_deletions`), whitespace normalisation (`remove_indents`, `remove_whitespace`, `replace_space_with_newlines`), byte value reduction (`lower_bytes`, `lower_individual_bytes`), common substitutions (`standard_substitutions`), and line sorting (`line_sorter`). Also provides the `Split` and `Tokenize` formats for viewing bytes as sequences.

### sequences.py

Generic operations on sequences (lists, tuples, etc.). Provides `block_deletion` for removing contiguous blocks, `delete_elements` for single-element removal, and `delete_duplicates` for removing duplicate elements. These are typically used via `compose()` with a format like `Split(b"\n")`.

### genericlanguages.py

Reduction passes for "things that look like programming languages" - text with comments, identifiers, brackets, and integer literals. Includes comment removal (`cut_comment_like_things`), integer literal reduction (`reduce_integer_literals`), expression combining (`combine_expressions`), adjacent string merging (`merge_adjacent_strings`), replacing falsey values with zero (`replace_falsey_with_zero`), identifier normalisation (`normalize_identifiers`), and bracket simplification. Language-agnostic, working on any text that uses common programming conventions.

### python.py

Python-specific AST-aware reductions using libcst. Includes passes for lifting indented constructs (replacing `if`/`while`/`try`/`with` blocks with their bodies), replacing indented block bodies with `...`, replacing statements with `pass`, stripping type annotations, and deleting statements. These understand Python syntax and produce valid Python output.

These passes are not run in-process. Instead they run inside a subprocess as an **external reducer** (see `notes/external-reducers.md`): `ShrinkRay` launches `python -m shrinkray.reducers.python`, which runs `PYTHON_PASSES` against a `RemoteReductionProblem` that answers `is_interesting` over the external-reducer protocol. This keeps libcst (and any future heavyweight reducers) out of the main process while preserving parallelism.

### json.py

JSON-specific passes. Provides `delete_identifiers` for recursively removing keys from JSON objects throughout the structure (built on the `DeleteIdentifiers` patch type).

### sat.py

Passes for DIMACS CNF files (SAT solver input format). Includes clause deletion, literal deletion, unit propagation, literal sign flipping, forced literal assignment, clause combining and sorting, literal merging, variable renumbering, and restriction to a single connected component. The `DimacsCNF` format parses bytes into a list of clauses (each clause being a list of integers).

### cpp.py

C/C++ passes built on a sloppy lexer plus bracket matching (a pure Python replacement for creduce's `clang_delta`, which shrink ray previously shelled out to). Rather than parsing properly, these find things that look like functions, namespaces, class heads, templates, call expressions and typedefs, then overgenerate candidate edits and let the interestingness test reject the wrong ones. Includes `replace_function_bodies` (function def -> declaration), `delete_function_definitions`, `remove_namespaces` (including `extern "C"`), `remove_base_classes`, `remove_constructor_initializers`, `remove_template_parts`, `replace_type_with_int`, and `simplify_call_expressions`, plus two **pumps** (which may temporarily increase code size): `inline_typedefs` and `inline_function_calls`.

### downloads.py

The startup download coordinator. Two resources a reduction may need are
fetched lazily: the LLM model (multi-gigabyte) and tree-sitter grammars
for the input's language. Neither downloads silently. `ShrinkRayState`
builds a `DownloadCoordinator` describing what's missing
(`missing_grammars` diffs the input's languages against
`downloaded_languages()`; the LLM model via
`LlamaCppClient.model_needs_download`), the UI presents it (the TUI as a
modal over the already-running reduction, the basic UI as a printed
list), and only then does `DownloadCoordinator.start(disabled)` begin the
approved downloads. An already-cached model still loads immediately
(`start_immediate`) since there's nothing to consent to.

Passes that depend on a download join the running reduction when it
completes: the LLM passes already wait on `LLMClient.wait_until_ready`,
and the reducer carries a `pending_treesitter_language` whose passes it
registers (via `_register_pending_treesitter`) once
`grammar_available` turns true, waiting at its fixpoint for the outcome
rather than terminating without them (`grammar_plan` decides per file
whether a grammar is usable now or pending). A declined or failed
download simply never adds its passes.

### llm.py

The LLM mode (on by default; `--no-llm` or `SHRINKRAY_LLM=0` disables). `llm_rewrite` feeds the whole current
test case to a language model, prompted with the file name, the text of
the user's interestingness script, and the output the test produced for
the current test case, and asks for several progressively
smaller rewrites in fenced code blocks; every block that sorts below the
current test case is offered to `is_interesting`, so a wrong or
hallucinating model wastes time but can't hurt correctness. The pass runs
in the last-ditch tier (a generation costs seconds to minutes, so it only
runs when the cheap passes stall) unless `--llm-only` strips every other
pass. The model download/load starts on a background thread as soon as
the reduction begins (`start_loading`), overlapping with the cheap
passes; the pass waits for readiness (`wait_until_ready`) only when it's
actually scheduled, and skips the wait entirely for inputs it could
never prompt with. Inference is in-process through llama-cpp-python
(`llm_client.py`), one generation at a time behind a thread lock;
the `LLMClient` ABC is the seam for pointing at other completion sources
(e.g. an OpenAI-compatible endpoint) later. Prompt-shape decisions were
measured with `evaluation/llm_prompt_experiment.py` against the benchmark
problems: including the interestingness script in the prompt roughly
doubled the valid-candidate rate, whole-file rewrites beat line-deletion
lists by a wide margin, and the model frequently found reductions below
shrink ray's own fixpoint on already-reduced corpus entries.

### treesitter.py

Grammar-aware passes for any language with a grammar in
tree-sitter-language-pack, selected by file extension (Go, Rust,
JavaScript, Java, and many more; JSON and DIMACS CNF keep their dedicated
passes only). tree-sitter is a parser rather than a generator, but nodes
carry byte offsets, so every transformation is expressed through the
ordinary `Cuts`/`Replacements` patch machinery: `delete_children` (runs of
named children, eating adjacent separators), `lift_nodes` (replace a node
with a same-type descendant from any depth, or any named descendant within
two levels), `delete_orphaned_declarations` (delete a node together with
top-level declarations/imports left textually unreferenced — the pass that
makes dead code deletable in languages like Go that reject unused
imports), and `substitute_nodes` (replace a node's text with the smallest
same-type text in the file). tree-sitter parses broken input tolerantly,
so these passes keep working on syntactically-invalid intermediate states.
See `evaluation/RESULTS.md` for the evaluation that motivated them.

## Pass Ordering in ShrinkRay

The `ShrinkRay` reducer organises passes into stages:

1. **initial_cuts**: Fast, high-value passes (comments, hollow, large blocks) with timeout-based cancellation
2. **great_passes**: Core loop (line and semicolon-split deletion, hollow, lift_braces, debracket) - runs until no progress
3. **ok_passes**: Run when great_passes plateau (smaller blocks, normalisation)
4. **last_ditch_passes**: Expensive or low-yield passes (token block deletion, brackets)
5. **polish_passes**: Very expensive, very low-yield passes (short_deletions, byte lowering) that mostly normalise rather than shrink. They only run once every other stage has converged.

Great passes loop until no progress, tracking which passes succeeded to prioritise them on subsequent iterations.

### Adaptive scheduling

`run_pass` adapts to each pass's recent record (all of this was tuned
against `evaluation/benchmark.py`, which measures interestingness calls on
a fixed problem suite):

- **Fingerprints**: a pass that ran to completion without making progress
  is skipped for free until the test case changes. Pass candidate
  generation is deterministic given the test case (randomness only affects
  order), so re-running it would be a no-op.
- **Probation budgets**: a pass whose previous completed run was fruitless
  gets only `probation_budget` consecutive failed calls on its next run
  before being abandoned for now.
- **Targeted verification**: abandoned passes are recorded in
  `incomplete_passes` and re-run without a budget before the reducer
  finishes, so the final result is a fixpoint of every pass, exactly as if
  no budgets existed. Budgets only move work later (usually onto much
  smaller test cases); they never skip it entirely.
