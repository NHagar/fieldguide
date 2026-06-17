from __future__ import annotations

import json
import math

import pytest

from fieldguide import extractors
from fieldguide.api import CANONICAL_TEXT_VERSION, build_index
from fieldguide.hybrid import (
    HYBRID_METADATA_FILENAME,
    HybridFieldguideIndex,
    build_hybrid_from_json,
    build_hybrid_from_source,
)
from fieldguide import hybrid_mcp_server


class FakeEmbedder:
    model_name = "fake-local-embedder"
    dimension = 6

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        return [_normalize(_vector_for(text)) for text in texts]


def _vector_for(text: str) -> list[float]:
    lowered = text.lower()
    features = [
        lowered.count("approval") + lowered.count("procurement"),
        lowered.count("payroll") + lowered.count("timesheet"),
        lowered.count("incident") + lowered.count("complaint"),
        lowered.count("legal@example.com") + lowered.count("acme"),
        len([word for word in lowered.split() if len(word) > 7]),
        1.0,
    ]
    return [float(value) for value in features]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def test_hybrid_build_source_uses_docling_and_searches(monkeypatch, tmp_path):
    def fake_docling(_path):
        return (
            "Approval memo for Acme Medical. Competitive bidding was waived after quote review.",
            [],
            {"pdf_backend": "docling", "topic_text_available": True},
            ["Approval memo for Acme Medical"],
        )

    monkeypatch.setattr(extractors, "_extract_pdf_docling", fake_docling)
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.pdf").write_bytes(b"%PDF fake")

    index_dir = tmp_path / "hybrid"
    embedder = FakeEmbedder()
    stats = build_hybrid_from_source(source, index_dir, embedder=embedder)

    assert stats.doc_count == 1
    assert stats.chunk_count == 1
    metadata = json.loads((index_dir / HYBRID_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["source"]["pdf_backend"] == "docling"
    assert metadata["embedding"]["model"] == "fake-local-embedder"

    index = HybridFieldguideIndex(index_dir, embedder=embedder)
    response = index.search("approval", mode="hybrid", max_results=5)

    result = response["result"]["results"][0]
    assert response["tool_name"] == "fieldguide_hybrid_search"
    assert result["doc_type"] == "pdf"
    assert result["source_metadata"]["original_filename"] == "sample.pdf"
    assert result["page_number"] == 1
    assert result["char_start"] is not None
    assert "canonical_text" not in result
    assert "Approval memo" in result["snippet"]


def test_hybrid_build_from_existing_json_index(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "procurement.txt").write_text(
        "legal@example.com approval procurement quote review",
        encoding="utf-8",
    )
    (source / "payroll.txt").write_text(
        "Payroll timesheet controls and benefits approval",
        encoding="utf-8",
    )
    json_index = tmp_path / "json-index"
    build_index(source, json_index)

    index_dir = tmp_path / "hybrid"
    embedder = FakeEmbedder()
    stats = build_hybrid_from_json(json_index, index_dir, embedder=embedder)

    assert stats.doc_count == 2
    assert stats.chunk_count == 2
    response = HybridFieldguideIndex(index_dir, embedder=embedder).search(
        "approval",
        mode="keyword",
        max_results=1,
        doc_types=["text"],
        entities=["legal@example.com"],
    )

    assert response["result"]["result_count"] == 1
    result = response["result"]["results"][0]
    assert result["doc_id"].startswith("D-")
    assert result["chunk_id"].startswith("CH-")
    assert result["matched_entities"][0]["text"] == "legal@example.com"
    assert result["source_metadata"]["source_batch"] == "default"


def test_hybrid_json_import_warns_for_non_docling_pdfs(tmp_path):
    json_index = tmp_path / "index.json"
    json_index.write_text(
        json.dumps(
            {
                "corpus_id": "C-TEST",
                "corpus_version": "v1",
                "canonical_text_version": CANONICAL_TEXT_VERSION,
                "documents": [
                    {
                        "corpus_id": "C-TEST",
                        "corpus_version": "v1",
                        "doc_id": "D-PDF",
                        "source_uri": "sample.pdf",
                        "source_batch": "default",
                        "original_filename": "sample.pdf",
                        "title": "Legacy PDF",
                        "doc_type": "pdf",
                        "page_count": 1,
                        "metadata": {"pdf_backend": "legacy"},
                        "extraction": {"extraction_method": "native_text", "warnings": [], "text_available": True},
                        "pages": [],
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "CH-PDF",
                        "doc_id": "D-PDF",
                        "page_number": 1,
                        "char_start": 0,
                        "char_end": 25,
                        "text": "Legacy PDF approval text",
                        "token_estimate": 7,
                        "entities": [],
                    }
                ],
                "entities": [],
            }
        ),
        encoding="utf-8",
    )

    stats = build_hybrid_from_json(json_index, tmp_path / "hybrid", embedder=FakeEmbedder())

    assert stats.warnings
    assert "not docling-processed" in stats.warnings[0]


def test_hybrid_source_filters_caps_and_empty_docs(tmp_path):
    source = tmp_path / "source"
    (source / "batch_a").mkdir(parents=True)
    (source / "batch_b").mkdir()
    (source / "batch_a" / "procurement.txt").write_text(
        "legal@example.com approval procurement quote review",
        encoding="utf-8",
    )
    (source / "batch_b" / "payroll.txt").write_text(
        "Payroll timesheet controls and benefits enrollment",
        encoding="utf-8",
    )
    (source / "empty.txt").write_text("", encoding="utf-8")

    embedder = FakeEmbedder()
    index_dir = tmp_path / "hybrid"
    stats = build_hybrid_from_source(source, index_dir, embedder=embedder)
    assert stats.doc_count == 3
    assert stats.chunk_count == 2

    index = HybridFieldguideIndex(index_dir, embedder=embedder)
    capped = index.search("approval procurement payroll", mode="hybrid", max_results=1)
    assert capped["result"]["result_count"] == 1

    source_filtered = index.search("payroll", mode="keyword", sources=["batch_b"])
    assert source_filtered["result"]["result_count"] == 1
    assert source_filtered["result"]["results"][0]["source_metadata"]["source_batch"] == "batch_b"

    entity_filtered = index.search("approval", mode="keyword", entities=["legal@example.com"])
    assert entity_filtered["result"]["result_count"] == 1
    assert entity_filtered["result"]["results"][0]["source_metadata"]["source_batch"] == "batch_a"


def test_hybrid_empty_corpus_builds_and_searches(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty.txt").write_text("", encoding="utf-8")

    embedder = FakeEmbedder()
    index_dir = tmp_path / "hybrid"
    stats = build_hybrid_from_source(source, index_dir, embedder=embedder)

    assert stats.doc_count == 1
    assert stats.chunk_count == 0
    response = HybridFieldguideIndex(index_dir, embedder=embedder).search("anything", mode="keyword")
    assert response["result"]["result_count"] == 0


def test_hybrid_mcp_requires_server_side_index_path(monkeypatch):
    monkeypatch.delenv("FIELDGUIDE_HYBRID_INDEX", raising=False)
    monkeypatch.setattr(hybrid_mcp_server, "_INDEX_CACHE", None)

    with pytest.raises(RuntimeError, match="FIELDGUIDE_HYBRID_INDEX"):
        hybrid_mcp_server.hybrid_search("approval")


def test_hybrid_mcp_queries_configured_index(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "procurement.txt").write_text("Approval procurement quote review", encoding="utf-8")
    index_dir = tmp_path / "hybrid"
    build_hybrid_from_source(source, index_dir, embedder=FakeEmbedder())

    monkeypatch.setenv("FIELDGUIDE_HYBRID_INDEX", str(index_dir))
    monkeypatch.setattr(hybrid_mcp_server, "_INDEX_CACHE", None)

    response = hybrid_mcp_server.hybrid_search("approval", mode="keyword", max_results=1)

    assert response["tool_name"] == "fieldguide_hybrid_search"
    assert response["result"]["result_count"] == 1
