FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server /app
COPY data /app/data
COPY server/docker-entrypoint.sh /usr/local/bin/advisu-entrypoint
RUN chmod +x /usr/local/bin/advisu-entrypoint \
    && mkdir -p /app/chroma_store /app/uploaded_documents /app/sources /app/.cache/huggingface

EXPOSE 8000

ENTRYPOINT ["advisu-entrypoint"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
