# Code Analyzer – Docker CLI

Code Analyzer is a Dockerized CLI tool that analyzes your source C/C++ code using local running LLMs.

This repository packages the analyzer as a Dockerized CLI.  

Note: my computer had a had time running with the Microsoft phi4 model, so I switched to the gemma-3-1b-it model, as suggested.

## Usage

1. Build the image  
2. (Optional) Run the installation hook to download the model  
3. Run the tool

---

## 1. Build the image

```bash
sudo docker build -t code-analyzer:dev .
```

---

## 2. (Optional) Run the installation hook (model download)

The hook downloads the model from Hugging Face and stores it in the cache.  
You must set your Hugging Face **READ** token as an environment variable (gemma3-1b-it authorized).

*Note: I did not include the model in the Docker image to keep it lightweight, and for legal reasons as gemma3-1b-it requires user authorization.
```bash
sudo docker run --rm \
  --entrypoint python \
  -e HF_TOKEN="<your-hf-read-token>" \
  code-analyzer:dev -m code_analyzer.install_hook
```

**Tip:** Use a persistent Docker volume to avoid re-downloading models:

```bash
sudo docker volume create code-analyzer-hf
sudo docker run --rm \
  --entrypoint python \
  -v code-analyzer-hf:/opt/hf_cache \
  -e HF_HOME=/opt/hf_cache \
  -e HF_TOKEN="<your-hf-read-token>" \
  code-analyzer:dev -m code_analyzer.install_hook
```

---

## 3. Run the tool

Show help (and learn about analyze arguments):
```bash
sudo docker run --rm code-analyzer:dev --help
```

## Run the Tool on Your Files
```bash
sudo docker run --rm \
  -v $(pwd):/workspace \
  -e HF_TOKEN="<your-hf-read-token>" \
  code-analyzer:dev /workspace/your_file.c\cpp
```
*Notes:*

* The `-v` flag mounts your local directory to `/workspace` in the container.

* if you downloaded the model in step 2, you can skip the `HF_TOKEN` environment variable.

---

## Troubleshooting

- **401 Unauthorized (model access denied):**  
  Ensure your Hugging Face account has been granted access to `google/gemma-3-1b-it` and you are using a **READ** token.