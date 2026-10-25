# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ARG TARGETARCH
ARG LOCAL_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ARG LOCAL_MODEL_SHA256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    LOCAL_MODEL_PATH=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    OPENROUTER_CONFIG_DIR=/config \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt .
RUN arch="${TARGETARCH:-$(uname -m)}"; \
    case "$arch" in \
        arm64|aarch64) \
            wheel="llama_cpp_python-0.3.36-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl" ;; \
        amd64|x86_64) \
            wheel="llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" ;; \
        *) \
            echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fL --retry 3 --retry-delay 5 \
        -o "/tmp/$wheel" \
        "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/$wheel"; \
    pip install --find-links /tmp --only-binary=llama-cpp-python -r requirements.txt; \
    rm -f "/tmp/$wheel"

RUN mkdir -p /models /app/.streamlit \
    && curl -fL --retry 3 --retry-delay 5 --retry-connrefused \
        -o "$LOCAL_MODEL_PATH" "$LOCAL_MODEL_URL" \
    && echo "${LOCAL_MODEL_SHA256}  ${LOCAL_MODEL_PATH}" | sha256sum -c - \
    && chmod 644 "$LOCAL_MODEL_PATH"

COPY --chown=appuser:appuser app.py classifier.py ./
COPY --chown=appuser:appuser entrypoint.sh /usr/local/bin/openrouter-entrypoint.sh
RUN chmod 755 /usr/local/bin/openrouter-entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/usr/local/bin/openrouter-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
