from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

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
        key=lambda item: float(item[1]),
        reverse=True,
    )

    reranked_documents = []
    for document, score in ranked[:top_k]:
        document.metadata = {
            **document.metadata,
            "rerank_score": float(score),
        }
        reranked_documents.append(document)

    return reranked_documents
