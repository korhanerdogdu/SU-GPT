FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

COPY server/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --prefer-binary -r /tmp/requirements.txt


FROM python:3.12-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

RUN addgroup --system advisu \
    && adduser --system --ingroup advisu --home /app advisu

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=advisu:advisu server/ /app/
COPY --chown=advisu:advisu data/ /app/data/
COPY --chown=advisu:advisu server/docker-entrypoint.sh /usr/local/bin/advisu-entrypoint

RUN chmod 0755 /usr/local/bin/advisu-entrypoint \
    && mkdir -p /app/chroma_store /app/uploaded_documents /app/sources /app/.cache/huggingface \
    && chown -R advisu:advisu /app

USER advisu

EXPOSE 8000 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/test' % os.getenv('PORT', '8000'), timeout=3)" || exit 1

ENTRYPOINT ["advisu-entrypoint"]
# Shell expansion is intentional: Render injects PORT (normally 10000), while
# docker compose leaves it at 8000 for local development.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
