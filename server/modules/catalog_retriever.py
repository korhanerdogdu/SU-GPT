from __future__ import annotations

import re
from typing import Any, Iterable, List

from langchain_core.documents import Document


CATALOG_DOCUMENT_TYPES = {
    "available_term",
    "available_terms_summary",
    "basic_science_credit",
    "course_catalog_overview",
    "course_offering_history",
    "degree_requirement_category_pool",
    "degree_requirement_pool_course",
    "degree_requirement_profile",
    "degree_requirement_record",
    "graduation_requirement_category",
    "graduation_requirement_course",
    "graduation_requirement_course_list",
    "graduation_requirement_summary",
    "official_degree_evaluation_category",
    "official_degree_evaluation_course_assignment",
    "official_degree_evaluation_summary",
    "program_requirement",
    "program_requirement_list",
    "schedule_course_list",
    "schedule_section",
}

MAJOR_PROGRAMS = {
    "BIO",
    "CS",
    "DSA",
    "ECON",
    "EE",
    "IE",
    "MAN",
    "MAT",
    "ME",
    "PSIR",
    "PSY",
    "VACD",
}

REQUIREMENT_TYPES = {
    "required": {"required", "zorunlu", "must"},
    "core": {"core", "cekirdek"},
    "area": {"area", "alan"},
    "free": {"free", "serbest"},
    "university": {"university", "universite", "common"},
}

COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[A-Z]?)\b")
TERM_CODE_RE = re.compile(r"\b(20\d{2})(0[123])\b")
ACADEMIC_TERM_RE = re.compile(
    r"\b(fall|spring|summer|guz|bahar|yaz)\s+(20\d{2})\s*[-/]\s*(20\d{2})\b",
    re.IGNORECASE,
)


def retrieve_documents(vectorstore: Any, query: str, k: int) -> List[Document]:
    """Retrieve with light structured filters before falling back to dense search.

    The catalog corpus has many near-identical rows. Pure dense search can
    over-rank generic "SU credit" or "schedule" wording. This retriever keeps the
    existing Chroma dense path, but first constrains obvious catalog questions by
    document type, course code, term, program, and requirement category.
    """

    filters = _candidate_filters(query)
    documents: list[Document] = []

    for where in filters:
        documents.extend(_exact_get(vectorstore, where, limit=max(k * 3, 12)))
        try:
            documents.extend(vectorstore.similarity_search(query, k=max(k, 8), filter=where))
        except Exception:
            continue

    if not documents:
        documents.extend(vectorstore.similarity_search(query, k=k))

    ranked = _lexical_rank(query, _dedupe(documents))
    return ranked[: max(k, 1)]


def structured_metadata_score(query: str, metadata: dict[str, Any]) -> float:
    course_ids = _extract_course_ids(query)
    term_codes = set(_extract_term_codes(query))
    programs = set(_extract_programs(query))
    requirement_types = set(_extract_requirement_types(query))

    value = 0.0
    if metadata.get("course_id") in course_ids:
        value += 20
    if metadata.get("term_code") in term_codes:
        value += 8
    if metadata.get("program") in programs:
        value += 6
    if metadata.get("requirement_type") in requirement_types:
        value += 4
    if _is_graduation_query(query) and metadata.get("source_authority") == "official_degree_evaluation":
        value += 14
    if _is_graduation_query(query) and metadata.get("source_authority") == "official_degree_requirements":
        value += 6
    if metadata.get("aggregate") and not course_ids:
        value += 1
    return value


def _candidate_filters(query: str) -> list[dict[str, Any]]:
    course_ids = _extract_course_ids(query)
    term_codes = _extract_term_codes(query)
    programs = _extract_programs(query)
    requirement_types = _extract_requirement_types(query)
    type_filter = _document_type_filter(query)
    source_filter = _source_collection_filter(query)

    filters: list[dict[str, Any]] = []

    base_parts = []
    if source_filter:
        base_parts.append({"source_collection": source_filter})
    if type_filter:
        base_parts.append({"document_type": {"$in": type_filter}})
    if term_codes:
        base_parts.append({"term_code": {"$in": term_codes}})
    if programs:
        base_parts.append({"program": {"$in": programs}})

    base_parts_without_requirement = list(base_parts)
    if requirement_types:
        base_parts.append({"requirement_type": {"$in": requirement_types}})

    if course_ids:
        for course_id in course_ids:
            filters.append(_where([*base_parts, {"course_id": course_id}]))
            if requirement_types:
                filters.append(_where([*base_parts_without_requirement, {"course_id": course_id}]))

    if base_parts:
        if not course_ids and (
            "program_requirement_list" in type_filter
            or "schedule_course_list" in type_filter
            or "graduation_requirement_course_list" in type_filter
        ):
            filters.append(_where([*base_parts, {"aggregate": True}]))
        filters.append(_where(base_parts))

    return filters


def _document_type_filter(query: str) -> list[str]:
    q = query.lower()
    if _has_any(q, ["schedule", "time", "meeting", "meet", "crn", "section", "instructor", "saat", "program", "sube"]):
        return ["schedule_section", "schedule_course_list"]
    if _has_any(
        q,
        [
            "graduation",
            "mezuniyet",
            "bscs",
            "degree evaluation",
            "audit",
            "graduate",
            "total su credits",
            "minimum total",
            "kredi",
            "dağılım",
            "dagilim",
        ],
    ):
        return [
            "degree_requirement_profile",
            "degree_requirement_category_pool",
            "degree_requirement_pool_course",
            "graduation_requirement_summary",
            "graduation_requirement_category",
            "graduation_requirement_course",
            "graduation_requirement_course_list",
            "official_degree_evaluation_summary",
            "official_degree_evaluation_category",
            "official_degree_evaluation_course_assignment",
        ]
    if _has_any(q, ["basic science", "temel bilim", "engineering ects", "muhendislik", "credit breakdown"]):
        return ["basic_science_credit", "course_catalog_overview"]
    if _has_any(q, ["description", "prerequisite", "prereq", "corequisite", "course catalog", "ders icerigi", "on kosul"]):
        return ["course_catalog_overview", "course_offering_history"]
    if _has_any(q, ["count", "sayilir", "sayılır", "elective", "secmeli", "required", "zorunlu", "curriculum", "requirement"]):
        return ["program_requirement", "program_requirement_list"]
    if _has_any(q, ["available terms", "which terms", "term list", "semester list", "donemler", "hangi donem"]):
        return ["available_term", "available_terms_summary"]
    return []


def _source_collection_filter(query: str) -> str:
    q = query.lower()
    if _has_any(
        q,
        [
            "embedding",
            "embeddings",
            "word2vec",
            "bert",
            "gpt",
            "transformer",
            "semantic search",
            "rag",
            "retrieval augmented",
            "dense retrieval",
            "reranking",
            "crossencoder",
            "cross-encoder",
            "prompt engineering",
            "kv cache",
            "fine-tuning",
            "finetuning",
        ],
    ):
        return "sources"
    return ""


def _extract_course_ids(query: str) -> list[str]:
    found = []
    for subject, number in COURSE_CODE_RE.findall(query.upper()):
        course_id = f"{subject} {number}"
        if course_id not in found:
            found.append(course_id)
    return found


def _is_graduation_query(query: str) -> bool:
    q = query.lower()
    return _has_any(
        q,
        [
            "graduation",
            "mezuniyet",
            "degree evaluation",
            "audit",
            "graduate",
            "total su credits",
            "kredi",
            "dağılım",
            "dagilim",
            "requirement",
        ],
    )


def _extract_term_codes(query: str) -> list[str]:
    found = []
    for year, suffix in TERM_CODE_RE.findall(query):
        code = f"{year}{suffix}"
        if code not in found:
            found.append(code)

    season_to_suffix = {
        "fall": "01",
        "guz": "01",
        "spring": "02",
        "bahar": "02",
        "summer": "03",
        "yaz": "03",
    }
    for season, start_year, _end_year in ACADEMIC_TERM_RE.findall(query):
        suffix = season_to_suffix.get(season.lower())
        if suffix:
            code = f"{start_year}{suffix}"
            if code not in found:
                found.append(code)
    return found


def _extract_programs(query: str) -> list[str]:
    upper = query.upper()
    found = []
    for program in sorted(MAJOR_PROGRAMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(program)}\b", upper):
            found.append(program)
    return found


def _extract_requirement_types(query: str) -> list[str]:
    q = query.lower()
    found = []
    for requirement_type, needles in REQUIREMENT_TYPES.items():
        if any(needle in q for needle in needles):
            found.append(requirement_type)
    return found


def _where(parts: list[dict[str, Any]]) -> dict[str, Any]:
    parts = [part for part in parts if part]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _exact_get(vectorstore: Any, where: dict[str, Any], limit: int) -> list[Document]:
    if not where:
        return []
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return []
    try:
        result = collection.get(where=where, limit=limit, include=["documents", "metadatas"])
    except Exception:
        return []

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        Document(page_content=page_content, metadata=metadata or {})
        for page_content, metadata in zip(documents, metadatas)
        if page_content
    ]


def _lexical_rank(query: str, documents: Iterable[Document]) -> list[Document]:
    q = query.lower()
    terms = set(re.findall(r"[a-zA-Z0-9]+", q))
    course_ids = _extract_course_ids(query)
    term_codes = set(_extract_term_codes(query))
    programs = set(_extract_programs(query))
    requirement_types = set(_extract_requirement_types(query))

    def score(doc: Document) -> float:
        metadata = doc.metadata or {}
        text = doc.page_content.lower()
        value = structured_metadata_score(query, metadata)
        for term in terms:
            if len(term) > 1 and term in text:
                value += 0.25
        return value

    return sorted(documents, key=score, reverse=True)


def _dedupe(documents: Iterable[Document]) -> list[Document]:
    seen = set()
    deduped = []
    for doc in documents:
        metadata = doc.metadata or {}
        key = metadata.get("chunk_id") or (metadata.get("source"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)
