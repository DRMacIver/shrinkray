- Shrink Ray now has an LLM mode, enabled by default: alongside the ordinary
  reduction passes, it asks a language model (running locally, in-process) to
  propose smaller test cases once the other passes stop making progress. The
  model is shown your interestingness test and the output it produced for the
  current test case. Model suggestions are candidates like any others —
  they're only accepted if your interestingness test passes — so a bad model
  costs time but never correctness.
- The first use downloads the default model (Qwen3.5-4B, about 2.7GB) from
  Hugging Face in the background while the ordinary passes reduce; the LLM
  passes join in once it's ready.
- `--no-llm` (or `SHRINKRAY_LLM=0` in the environment) disables the LLM
  passes, `--llm-model` picks a different model (a local `.gguf` file or a
  Hugging Face `repo:filename` reference), and `--llm-only` runs only the LLM
  passes. On platforms where the bundled llama-cpp-python cannot load, shrink
  ray warns and reduces without the LLM passes.
- Shrink Ray no longer downloads anything silently. When a reduction would
  fetch the LLM model or a tree-sitter grammar for the input's language, it now
  says so up front: the interactive UI shows a startup dialog listing each
  download with a checkbox to skip it, and the basic UI prints the list. The
  reduction starts immediately behind the dialog on the ordinary passes, and
  each download's extra passes join in as it completes — so declining a
  download, or dismissing the dialog, never blocks reduction.
