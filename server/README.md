- Create Virtual Environment: python -m venv myenv
- Activate myenv : myenv/scripts/activate
- pip install -r requirements.txt

## Containers and Render

Build the backend from the **repository root**, not from `server/`, because the image bundles both
the application and the official `data/` corpus:

```bash
docker build -f server/Dockerfile -t advisu-backend .
docker run --rm -p 8000:8000 -e PORT=8000 --env-file .env advisu-backend
```

The container binds to `0.0.0.0:${PORT:-8000}`. On Render use Dockerfile Path
`server/Dockerfile`, Docker Build Context Directory `.`, and Health Check Path `/test`.
`AUTO_BUILD_VECTOR_INDEX` is opt-in in a raw container because startup embedding can exceed a
512 MB instance; local `docker compose` explicitly enables it and persists the result.

See [`../docs/deployment-render.md`](../docs/deployment-render.md) for production settings.

## Catalog JSONL ingest

The university catalog data under `~/data` is structured JSONL, not PDF-like
free text. Convert it into self-contained RAG knowledge cards and write vectors
to the local Chroma store with:

```bash
python scripts/ingest_catalog_data.py --data-dir ~/data --reset
```

Use the same command after the yearly data refresh. The command does not need a
Groq API key because embedding is done locally with SentenceTransformers. Asking
questions through `/ask/` still needs `GROQ_API_KEY` in `server/.env`.
