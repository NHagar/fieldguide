from __future__ import annotations

import pytest

from fieldguide.api import FieldguideIndex, build_index
from fieldguide.extractors import extract_entities, extraction_quality
from fieldguide.render import render_markdown


def _build_sample_index(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    procurement = [
        "2020-03-01 Emergency procurement waiver for Acme Medical. The memo discusses competitive bidding, sole source justification, vendor quote review, and purchase order approval.",
        "2020-04-03 Acme Medical vendor quote packet. Competitive bidding was waived for emergency purchase order materials and procurement approval.",
        "2020-05-15 Procurement approval notes for emergency purchase order. Sole source waiver and vendor quote comparison for Acme Medical supplies.",
    ]
    payroll = [
        "2021-01-02 Payroll adjustment report. Timesheet corrections, overtime approval, employee benefits, and payroll reconciliation.",
        "2021-02-10 Hiring and payroll memo. Employee onboarding, salary grade, timesheet audit, and benefits enrollment.",
        "2021-03-18 Payroll exception log. Overtime approval, employee payroll correction, and supervisor signoff.",
    ]
    incidents = [
        "2022-07-11 Incident report. Complaint investigation, witness interview, safety review, and corrective action.",
        "2022-08-19 Investigation follow up. Incident complaint, witness statement, safety review, and case number CASE-7781.",
        "2022-09-21 Corrective action report. Safety incident investigation, complaint findings, and witness notes.",
    ]
    for idx, text in enumerate(procurement + payroll + incidents, start=1):
        (source / f"doc_{idx:02d}.txt").write_text(text, encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(source, index_dir)
    return FieldguideIndex(index_dir)


def test_snippet_provenance_and_bounded_evidence(tmp_path):
    index = _build_sample_index(tmp_path)
    snippet = next(
        snippet
        for topic in index.topics
        for snippet in topic.get("representative_snippets", [])
    )
    doc = index.docs_by_id[snippet["doc_id"]]

    assert doc["canonical_text"][snippet["char_start"] : snippet["char_end"]] == snippet["text"]

    window = index.read_window(snippet_id=snippet["snippet_id"], max_chars=900)
    assert window["response_kind"] == "evidence"
    assert window["result"]["provenance"]["doc_id"] == snippet["doc_id"]
    assert snippet["text"] in window["result"]["text"]

    pages = index.read_pages(doc_id=snippet["doc_id"], pages=[snippet["page_number"]])
    assert pages["response_kind"] == "evidence"
    assert pages["result"]["provenance"]["page_numbers"] == [snippet["page_number"]]
    assert pages["result"]["pages"][0]["text"]


def test_budget_and_access_limits(tmp_path):
    index = _build_sample_index(tmp_path)

    overview = index.orient_corpus(token_budget=1000)
    assert overview["token_estimate"] <= 1000

    search = index.search_within(scope={"scope_id": "S-root"}, query="approval", max_results=2)
    assert len(search["result"]["results"]) <= 2

    snippet = next(
        snippet
        for topic in index.topics
        for snippet in topic.get("representative_snippets", [])
    )
    window = index.read_window(snippet_id=snippet["snippet_id"], max_chars=200)
    assert len(window["result"]["text"]) <= 200

    with pytest.raises(ValueError):
        index.read_pages(doc_id=snippet["doc_id"], pages=list(range(1, 10)))


def test_topic_scope_contains_search_results(tmp_path):
    index = _build_sample_index(tmp_path)
    topic = next(topic for topic in index.topics if topic["depth"] > 0)
    allowed_doc_ids = set(topic["doc_ids"])

    results = index.search_within(
        scope={"kind": "topic", "topic_ids": [topic["topic_id"]], "include_descendants": False},
        query="approval incident payroll procurement",
        max_results=50,
    )

    returned_doc_ids = {item["doc_id"] for item in results["result"]["results"]}
    assert returned_doc_ids <= allowed_doc_ids
    assert results["result"]["scope"]["topic_ids"] == [topic["topic_id"]]


def test_topic_quality_basics(tmp_path):
    index = _build_sample_index(tmp_path)
    topic_ids = {topic["topic_id"] for topic in index.topics}

    for topic in index.topics:
        assert topic["label_terms"]
        assert topic["doc_count"] >= 1
        if topic["parent_topic_id"]:
            assert topic["parent_topic_id"] in topic_ids
        for child_id in topic["child_topic_ids"]:
            assert child_id in topic_ids


def test_empty_text_docs_are_not_available_or_near_duplicates(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "empty_a.txt").write_text("", encoding="utf-8")
    (source / "empty_b.txt").write_text("", encoding="utf-8")
    (source / "normal.txt").write_text("body text without inferred chronology metadata", encoding="utf-8")

    index_dir = tmp_path / "index"
    build_index(source, index_dir)
    index = FieldguideIndex(index_dir)

    empty_docs = [doc for doc in index.documents if doc["original_filename"].startswith("empty_")]
    assert len(empty_docs) == 2
    assert all(not doc["extraction"]["text_available"] for doc in empty_docs)
    assert all(not doc["relationships"] for doc in empty_docs)
    normal_doc = index.docs_by_id[next(doc["doc_id"] for doc in index.documents if doc["original_filename"] == "normal.txt")]
    assert "document_date" not in normal_doc
    assert "modified_date" not in normal_doc["metadata"]

    overview = index.orient_corpus(token_budget=10_000)
    assert "date_span" not in overview["result"]["corpus_summary"]
    assert "date_histogram" not in overview["result"]["global_facets"]
    assert "documents_without_dates" not in overview["result"]["quality"]


def test_statistical_ner_and_case_regex():
    entities = extract_entities(
        (
            "Officer Jane Smith emailed Acme Medical in Chicago on March 1, 2020. "
            "The police report included PO 12345, a request, a Case Report, and P.O. Alejandro."
        ),
        doc_id="D-test",
        page_number=1,
    )
    by_type = {(entity["text"], entity["type"]) for entity in entities}

    assert ("Jane Smith", "person") in by_type
    assert ("Acme Medical", "organization") in by_type
    assert ("Chicago", "location") in by_type
    assert ("PO 12345", "case_number") in by_type
    assert not any(entity["type"] == "date" for entity in entities)
    assert not any(entity["text"].lower() == "police" and entity["type"] == "case_number" for entity in entities)
    assert not any(entity["text"].lower() in {"request", "case report", "p.o. alejandro"} and entity["type"] == "case_number" for entity in entities)
    assert extract_entities("\u200b", doc_id="D-test") == []


def test_entity_refinement_counts_are_distinct_documents(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("legal@example.com legal@example.com approval review", encoding="utf-8")
    (source / "b.txt").write_text("legal@example.com procurement approval", encoding="utf-8")
    (source / "c.txt").write_text("other@example.com unrelated", encoding="utf-8")

    index_dir = tmp_path / "index"
    build_index(source, index_dir)
    index = FieldguideIndex(index_dir)

    response = index.search_within(scope={"scope_id": "S-root"}, query="approval", max_results=10)
    refinement = next(item for item in response["result"]["suggested_refinements"] if item.get("value") == "legal@example.com")

    assert refinement["estimated_result_count"] == 2
    assert refinement["observed_result_count"] == 2


def test_extraction_quality_carries_ocr_confidence():
    quality = extraction_quality("ocr", [], text_available=True, ocr_confidence=0.8734)

    assert quality["ocr_confidence"] == 0.8734
    assert quality["layout_confidence"] is None


def test_markdown_renderer_for_orientation(tmp_path):
    index = _build_sample_index(tmp_path)
    response = index.orient_corpus(token_budget=10_000)
    rendered = render_markdown(response)

    assert rendered.startswith("# orient_corpus")
    assert "## Corpus" in rendered
    assert "## Top Topics" in rendered
    assert "| topic | docs | pages | terms | scope | warnings |" in rendered
