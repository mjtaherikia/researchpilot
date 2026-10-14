
import json

import pytest

from researchpilot.corpus import (
    CorpusError,
    get_document,
    load_documents,
    search_documents,
)


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "doc-a",
                    "title": "Retrieval systems",
                    "source": "test",
                    "text": "Retrieval systems find relevant documents.",
                },
                {
                    "id": "doc-b",
                    "title": "Model training",
                    "source": "test",
                    "text": "Training updates model parameters.",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_search_returns_relevant_document_first(corpus_file):
    results = search_documents("retrieval documents", path=corpus_file)

    assert results
    assert results[0]["id"] == "doc-a"
    assert "excerpt" in results[0]


def test_search_returns_empty_for_no_match(corpus_file):
    results = search_documents("volcano astronomy", path=corpus_file)

    assert results == []


def test_search_rejects_empty_query(corpus_file):
    with pytest.raises(ValueError):
        search_documents("  ", path=corpus_file)


@pytest.mark.parametrize("top_k", [0, 11, -1, True, 1.5])
def test_search_rejects_invalid_top_k(corpus_file, top_k):
    with pytest.raises(ValueError):
        search_documents("retrieval", top_k=top_k, path=corpus_file)


def test_get_document_by_exact_id(corpus_file):
    document = get_document("doc-b", path=corpus_file)

    assert document["title"] == "Model training"


def test_get_document_rejects_unknown_id(corpus_file):
    with pytest.raises(KeyError):
        get_document("missing-doc", path=corpus_file)


def test_load_documents_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicates.json"
    doc = {
        "id": "same-id",
        "title": "Title",
        "source": "test",
        "text": "Body",
    }
    path.write_text(json.dumps([doc, doc]), encoding="utf-8")

    with pytest.raises(CorpusError, match="Duplicate"):
        load_documents(path)


def test_load_documents_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(CorpusError):
        load_documents(path)
