# Code Analyzer

A Dockerized command-line tool that scans C/C++ source files for security vulnerabilities and suggests fixes, using a locally running LLM. Code never leaves your machine: no external API calls are made during analysis.

Default model: [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), run through Hugging Face Transformers. Any compatible Hugging Face model can be used instead.

## How it works

1. **Parse.** The source file is parsed into a syntax tree with [Tree-sitter](https://tree-sitter.github.io/).
2. **Chunk.** The code is split into chunks at syntactic boundaries (for example, never in the middle of a function), each under a configurable size limit.
3. **Add context.** The file's header lines (includes, declarations, class definitions) are prepended to every chunk, so the model sees the declarations the code depends on.
4. **Analyze.** Each chunk is sent to the local model with a configurable prompt, and the findings are printed per chunk.

The project is split into small modules: model downloading, the model pipeline, the syntax-tree chunk parser, and the CLI. This makes it straightforward to swap the model, change the prompt, or add more languages.

```
code_analyzer/
├── analyzer.py                  # CLI entry point and analysis flow
├── syntax_tree_chunk_parser.py  # Tree-sitter based chunking
├── hf_model_pipeline.py         # Loads the model and runs prompts
├── hf_model_downloader.py       # Downloads models from Hugging Face
└── install_hook.py              # Pre-downloads the model into the cache
```

## Usage

### 1. Build the image

```bash
docker build -t code-analyzer:dev .
```

### 2. (Optional) Pre-download the model

The model is not bundled in the image, to keep it small and because Gemma requires accepting its license on Hugging Face. Set a Hugging Face **read** token for an account that has been granted access to `google/gemma-3-1b-it`.

Use a Docker volume so the model is downloaded only once:

```bash
docker volume create code-analyzer-hf

docker run --rm \
  --entrypoint python \
  -v code-analyzer-hf:/opt/hf_cache \
  -e HF_HOME=/opt/hf_cache \
  -e HF_TOKEN="<your-hf-read-token>" \
  code-analyzer:dev -m code_analyzer.install_hook
```

### 3. Analyze a file

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -v code-analyzer-hf:/opt/hf_cache \
  -e HF_HOME=/opt/hf_cache \
  code-analyzer:dev /workspace/your_file.c
```

The first `-v` mounts your current directory at `/workspace` inside the container. If you skipped step 2, pass `-e HF_TOKEN="<your-hf-read-token>"` instead and the model is downloaded on the first run.

Sample files to try are in `Tests/`.

## Options

| Option | Description |
|---|---|
| `code_file_path` | Path to the C/C++ file to analyze |
| `--model-id` | Hugging Face model ID (default: `google/gemma-3-1b-it`) |
| `--model-path` | Path to a model that is already downloaded, instead of a model ID |
| `--max-chars` | Maximum characters per code chunk (one chunk per prompt) |
| `--prompt` | Custom prompt template; must contain a `{code}` placeholder |
| `--treat-h-as-c` | Treat `.h` files as C (by default they are treated as C++) |
| `--no-header-context` | Don't prepend the file's header context to each chunk |
| `--debug` | Print debug output, including the full prompt |

Run `docker run --rm code-analyzer:dev --help` for the full list.

## Limitations and next steps

Small local models are fast and private but limited: findings can be imprecise, and reported line numbers are not always accurate. Next steps:

- Benchmark detection accuracy against a labeled dataset such as the NIST Juliet Test Suite
- Compare results across model sizes
- Output findings in SARIF format, for use in CI pipelines and code review tools

## Troubleshooting

**401 Unauthorized when downloading the model:** make sure your Hugging Face account has been granted access to `google/gemma-3-1b-it` and that you're using a token with read access.
