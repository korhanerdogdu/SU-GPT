#!/bin/sh
set -eu

chroma_dir="${CHROMA_PERSIST_DIR:-/app/chroma_store}"
mkdir -p "$chroma_dir" "${DOCUMENT_STORAGE_DIR:-/app/uploaded_documents}" "${SOURCES_DIR:-/app/sources}"

# Indexing ~30k records downloads an embedding model and can exceed the 512 MB
# Render Free limit. It is therefore opt-in for a raw container. docker-compose
# explicitly enables it for local development, where the result is persisted.
#
# The presence check is on the *collection's document count*, not just whether the sqlite file
# exists: Chroma creates that file as soon as a client opens the store, before anything is
# upserted into it, so a container that was ever started once with an empty/interrupted
# ingestion leaves behind a chroma.sqlite3 that looks "already built" forever after -- every
# later restart would see the file, skip ingestion, and silently keep serving an empty index.
doc_count() {
  python -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='$chroma_dir')
    print(client.get_collection('${CHROMA_COLLECTION_NAME:-su_knowledge}').count())
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

case "${AUTO_BUILD_VECTOR_INDEX:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    count="$(doc_count)"
    if [ "$count" -eq 0 ] 2>/dev/null; then
      echo "adviSU: vector index is missing or empty (0 documents); building it from the bundled official corpus."
      python /app/scripts/ingest_degree_requirements.py \
        --data-dir "${DEGREE_DATA_DIR:-/app/data}" \
        --persist-dir "$chroma_dir"
    else
      echo "adviSU: using the existing vector index at $chroma_dir ($count documents)."
    fi
    ;;
  *)
    if [ "$(doc_count)" -eq 0 ] 2>/dev/null; then
      echo "adviSU: vector index is absent and AUTO_BUILD_VECTOR_INDEX is disabled; starting without startup ingestion."
    fi
    ;;
esac

# The retrieval_lab dense half (modules/lab_retriever.py) embeds the whole corpus (~30k chunks)
# on first use and memory-maps the cached matrix on every call after that. That first encode is
# CPU-bound Python/BLAS work with no async yield points; run inside a live request it holds the
# GIL long enough to freeze the entire event loop -- every other request, including the
# healthcheck and /auth/login, times out until it finishes. Warming the cache here, in its own
# short-lived process before uvicorn starts accepting connections, means the cost is paid once
# per fresh volume instead of once per live request. Best-effort: lab_retriever already degrades
# to BM25F-only if the dense half is unavailable, so a warm-up failure here must not block startup.
case "${ADVISU_LAB_DENSE:-true}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "adviSU: warming the dense retrieval cache (one-time cost per persisted volume)..."
    # The image pins OMP_NUM_THREADS=MKL_NUM_THREADS=1 (Dockerfile) so concurrent live requests
    # never oversubscribe the CPU against each other. That same limit applied to this one-time,
    # single-process warm-up just makes an otherwise-parallelizable embedding pass take far
    # longer than it needs to. Override it for this subprocess only -- uvicorn, started below via
    # exec, still inherits the image's single-thread default from its own environment.
    # `nproc` itself is unreliable in this image (observed reporting 1 on an 8-core host with no
    # cgroup CPU limit set) -- /proc/cpuinfo's own processor count is what Python's
    # os.cpu_count()/sched_getaffinity() actually agree with here, so read it directly instead.
    warm_threads="$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)"
    OMP_NUM_THREADS="$warm_threads" MKL_NUM_THREADS="$warm_threads" python -c "
from retrieval_lab.corpus import load_corpus
from retrieval_lab.dense import DENSE_CONFIGS, DenseIndex
corpus = load_corpus()
DenseIndex(DENSE_CONFIGS['e5_small'], corpus)
print('adviSU: dense retrieval cache ready (%d chunks).' % len(corpus))
" || echo "adviSU: dense cache warm-up failed; requests will fall back to BM25F+metadata only."
    ;;
esac

exec "$@"
