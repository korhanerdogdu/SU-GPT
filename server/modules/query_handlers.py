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
        logger.debug(f"Running chain for input: {user_input}")
        result = chain({"query": user_input})
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            label = _format_source(doc.metadata)
            if label and label not in seen:
                sources.append(label)
                seen.add(label)
        response = {
            "response": result["result"],
            "sources": sources,
        }
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.exception("Error in query_chain")
        raise
