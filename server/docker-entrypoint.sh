#!/bin/sh
set -eu

if [ "${AUTO_BUILD_VECTOR_INDEX:-true}" = "true" ] && [ ! -f "${CHROMA_PERSIST_DIR:-/app/chroma_store}/chroma.sqlite3" ]; then
  echo "adviSU: vector index is missing; building it from the bundled official corpus."
  python /app/scripts/ingest_degree_requirements.py \
    --data-dir "${DEGREE_DATA_DIR:-/app/data}" \
    --persist-dir "${CHROMA_PERSIST_DIR:-/app/chroma_store}"
fi

exec "$@"
