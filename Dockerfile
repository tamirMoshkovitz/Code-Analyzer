FROM python:3.11-slim

# Prevents Python from writing .pyc files & enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create and switch to a non-root user for security
RUN useradd -m appuser
USER appuser

# System deps (add any you need for compilation/parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirement pins first for better Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the package source
COPY code_analyzer /app/code_analyzer
COPY pyproject.toml /app/pyproject.toml
# (Optional) include README if you referenced it in pyproject
# COPY README.md /app/README.md

# Install as an editable wheel so 'code-analyzer' becomes available
RUN pip install .

# --- ONE-TIME INSTALL STEP (runs only while building the image) ---
# Anything heavy (model downloads, grammar compile, etc.) goes here.
RUN python -m code_analyzer.install_hook

# Default entrypoint is the CLI; args passed to `docker run` go to the tool
ENTRYPOINT ["code-analyzer"]
# Optionally, a sane default command (shows help if no args are provided)
CMD ["--help"]