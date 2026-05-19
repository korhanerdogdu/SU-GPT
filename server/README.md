- Create Virtual Environment: python -m venv myenv
- Activate myenv : myenv/scripts/activate
- pip install -r requirements.txt

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
