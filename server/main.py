from fastapi import FastAPI,UploadFile,File,Form,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from langchain_core.documents import Document
from modules.load_vectorstore import (
    get_vectorstore,
    load_vectorstore,
    load_vectorstore_multi,
)
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.config import RERANK_TOP_K, RETRIEVAL_CANDIDATE_K
from modules.reranker import rerank_documents
from logger import logger

app = FastAPI(title="SU-GPT — Course-Aware RAG Assistant")


class StaticRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


def _format_for_context(docs: List[Document]) -> List[Document]:
    """Prepend a [Source: ...] header to each chunk so the LLM can ground & cite."""
    formatted: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_name") or "unknown"
        location_parts = []
        if meta.get("page") is not None:
            location_parts.append(f"page {meta['page']}")
        if meta.get("slide") is not None:
            location_parts.append(f"slide {meta['slide']}")
        if meta.get("section"):
            location_parts.append(f"section '{meta['section']}'")
        location = ", ".join(location_parts)
        header = f"[Source: {source}" + (f", {location}" if location else "") + "]"
        formatted.append(
            Document(
                page_content=f"{header}\n{doc.page_content}",
                metadata=meta,
            )
        )
    return formatted

# allow frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500,content={"error":str(exc)})
    
@app.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"recieved {len(files)} files")
        chunk_count = load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message":"Files processed and vectorstore updated","chunks":chunk_count}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500,content={"error":str(e)})


@app.post("/upload_documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Multi-format upload endpoint (Section 2).

    Accepts PDF, PPTX, DOCX, MD, and TXT files. Unsupported types are
    skipped and reported in the response.
    """
    try:
        logger.info(f"received {len(files)} document(s) for multi-format ingest")
        result = load_vectorstore_multi(files)
        logger.info(
            "multi-format ingest complete: %d chunks, %d accepted, %d skipped",
            result["chunks"],
            len(result["accepted_files"]),
            len(result["skipped_files"]),
        )
        return {
            "message": "Files processed and vectorstore updated",
            **result,
        }
    except Exception as e:
        logger.exception("Error during multi-format document upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        vectorstore = get_vectorstore()
        candidate_docs = vectorstore.similarity_search(question, k=RETRIEVAL_CANDIDATE_K)
        reranked_docs = rerank_documents(question, candidate_docs, top_k=RERANK_TOP_K)
        context_docs = _format_for_context(reranked_docs)
        retriever = StaticRetriever(documents=context_docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}
