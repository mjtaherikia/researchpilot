

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_PATH = Path("data/documents.json")


class CorpusError(Exception):
    """Raised when the document corpus cannot be read or validated."""


def load_documents(
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load and validate documents from a JSON file."""
    corpus_path = Path(
        path or os.getenv("RESEARCHPILOT_CORPUS", DEFAULT_CORPUS_PATH)
    )

    try:
        raw: Any = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(
            f"Could not read corpus at {corpus_path}"
        ) from exc

    if not isinstance(raw, list):
        raise CorpusError("Corpus must be a JSON list.")

    documents: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CorpusError(f"Document at index {index} must be an object.")

        required = ("id", "title", "source", "text")
        if any(
            not isinstance(item.get(field), str)
            or not item[field].strip()
            for field in required
        ):
            raise CorpusError(
                f"Document at index {index} has missing or invalid fields."
            )

        doc = {field: item[field].strip() for field in required}

        if doc["id"] in seen_ids:
            raise CorpusError(f"Duplicate document ID: {doc['id']}")

        seen_ids.add(doc["id"])
        documents.append(doc)

    return documents


def tokenize(text: str) -> list[str]:
    """Extract case-insensitive alphanumeric tokens."""
    return re.findall(r"\b[\w-]+\b", text.casefold(), flags=re.UNICODE)


def search_documents(
    query: str,
    top_k: int = 5,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve documents using a simple lexical baseline.

    This is not semantic search. It ranks documents by the number
    of distinct query terms appearing in their combined text.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    documents = load_documents(path)
    ranked: list[dict[str, Any]] = []

    for doc in documents:
        searchable_text = f"{doc['title']} {doc['text']}"
        doc_terms = set(tokenize(searchable_text))
        matched_terms = query_terms & doc_terms

        if not matched_terms:
            continue

        ranked.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "score": round(len(matched_terms) / len(query_terms), 4),
                "matched_terms": sorted(matched_terms),
                "excerpt": doc["text"][:500],
            }
        )

    ranked.sort(
        key=lambda result: (
            -result["score"],
            result["id"],
        )
    )

    return ranked[:top_k]


def get_document(
    document_id: str,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Return one document by its exact ID."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    for doc in load_documents(path):
        if doc["id"] == document_id:
            return doc

    raise KeyError(f"Document not found: {document_id}")
