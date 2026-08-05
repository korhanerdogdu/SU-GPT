from logger import logger


def _format_source(metadata: dict) -> str:
    if not metadata:
        return ""
    source = metadata.get("source") or metadata.get("file_name") or ""
    parts = [source] if source else []
    if metadata.get("page") is not None:
        parts.append(f"page {metadata['page']}")
    if metadata.get("slide") is not None:
        parts.append(f"slide {metadata['slide']}")
    if metadata.get("section"):
        parts.append(f"section '{metadata['section']}'")
    return ", ".join(p for p in parts if p)


def query_chain(chain, user_input: str):
    try:
        logger.debug("Running retrieval chain")
        result = chain({"query": user_input})
        sources = []
        source_chunk_ids = []
        seen = set()
        seen_chunks = set()
        for doc in result.get("source_documents", []):
            label = _format_source(doc.metadata)
            if label and label not in seen:
                sources.append(label)
                seen.add(label)
            chunk_id = (doc.metadata or {}).get("chunk_id") or (doc.metadata or {}).get("chunkId")
            if chunk_id and chunk_id not in seen_chunks:
                source_chunk_ids.append(chunk_id)
                seen_chunks.add(chunk_id)
        response = {
            "response": result["result"],
            "sources": sources,
            "source_chunk_ids": source_chunk_ids,
        }
        # RetrievalQA does not expose LangChain message metadata directly. Read only the
        # provider adapter's already-sanitized operational record; never retain prompts,
        # answers, authorization data, or reasoning fields.
        llm = getattr(
            getattr(getattr(chain, "combine_documents_chain", None), "llm_chain", None),
            "llm",
            None,
        )
        telemetry = llm.get_last_telemetry() if hasattr(llm, "get_last_telemetry") else None
        if telemetry:
            response["_provider_telemetry"] = telemetry
        logger.debug("Retrieval chain completed")
        return response
    except Exception as e:
        # Provider exceptions can embed raw response bodies.  Persist only the exception class.
        logger.error("retrieval chain failed (error_class=%s)", type(e).__name__)
        raise
