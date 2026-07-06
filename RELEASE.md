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
