- Added an experimental LLM mode: `--llm` enables reduction passes that ask a
  language model, running locally in-process, to propose smaller test cases.
  This needs the optional `llm` extra (e.g. `uv tool install 'shrinkray[llm]'`).
  The first use downloads the default model (Qwen3.5-4B, about 2.7GB) from
  Hugging Face; the download runs in the background while the ordinary passes
  reduce, and the LLM passes join in once it's ready.
- `--llm-model` picks the model: either a path to a local `.gguf` file or a
  Hugging Face `repo:filename` reference.
- `--llm-only` runs only the LLM passes, disabling all of shrink ray's other
  reduction passes.
