# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install Python dependencies first to maximize layer caching
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

# Copy your package source to a stable path and install it
# (At runtime, you'll mount your host project to /workspace)
COPY code_analyzer /opt/app/code_analyzer
COPY pyproject.toml /opt/app/
RUN pip install /opt/app

# Optional: set a shared model cache inside the image so downloads are cached in a layer
ENV HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache
# Create non-root user and prepare writable dirs for mounting & caches
RUN useradd -m -u 1000 appuser && \
    mkdir -p /opt/hf_cache /workspace && \
    chown -R appuser:appuser /opt/hf_cache /workspace
USER appuser

ENTRYPOINT ["code-analyzer"]
CMD ["--help"]