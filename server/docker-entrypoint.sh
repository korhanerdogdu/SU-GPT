#!/bin/sh
set -eu

chroma_dir="${CHROMA_PERSIST_DIR:-/app/chroma_store}"
mkdir -p "$chroma_dir" "${DOCUMENT_STORAGE_DIR:-/app/uploaded_documents}" "${SOURCES_DIR:-/app/sources}"

# Indexing ~30k records downloads an embedding model and can exceed the 512 MB
# Render Free limit. It is therefore opt-in for a raw container. docker-compose
# explicitly enables it for local development, where the result is persisted.
case "${AUTO_BUILD_VECTOR_INDEX:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    if [ ! -f "$chroma_dir/chroma.sqlite3" ]; then
      echo "adviSU: vector index is missing; building it from the bundled official corpus."
      python /app/scripts/ingest_degree_requirements.py \
        --data-dir "${DEGREE_DATA_DIR:-/app/data}" \
        --persist-dir "$chroma_dir"
    else
      echo "adviSU: using the existing vector index at $chroma_dir."
    fi
    ;;
  *)
    if [ ! -f "$chroma_dir/chroma.sqlite3" ]; then
      echo "adviSU: vector index is absent and AUTO_BUILD_VECTOR_INDEX is disabled; starting without startup ingestion."
    fi
    ;;
esac

exec "$@"
