from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from modules.catalog_retriever import structured_metadata_score
from modules.config import CROSS_ENCODER_MODEL_NAME


@lru_cache(maxsize=1)
def get_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank_documents(query: str, documents: List[Document], top_k: int) -> List[Document]:
    if not documents:
        return []

    model = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]) + structured_metadata_score(query, item[0].metadata or {}),
        reverse=True,
    )

    reranked_documents = []
    for document, score in ranked[:top_k]:
        combined_score = float(score) + structured_metadata_score(query, document.metadata or {})
        document.metadata = {
            **document.metadata,
            "rerank_score": float(score),
            "combined_rerank_score": combined_score,
        }
        reranked_documents.append(document)

    return reranked_documents
