"""Core index builder and bounded agent-facing API."""

from __future__ import annotations

import json
import math
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .extractors import extract_entities, extract_file, extraction_quality, file_metadata
from .text import (
    average_vector,
    cosine,
    estimate_tokens,
    lexical_terms,
    make_excerpt_range,
    normalized_vector,
    stable_id,
    term_counts,
    tokenize,
    top_terms,
    trim_to_range,
)


INDEX_FILENAME = "index.json"
CANONICAL_TEXT_VERSION = "fieldguide-canonical-v1"

DEFAULT_CONFIG: dict[str, Any] = {
    "corpus": {"max_top_topics": 10, "max_topic_depth": 4},
    "extraction": {
        "pdf_backend": "legacy",
    },
    "chunking": {
        "target_chunk_chars": 1800,
        "min_chunk_chars": 500,
        "max_chunk_chars": 3500,
        "overlap_chars": 150,
    },
    "clustering": {
        "algorithm": "deterministic_tfidf_kmeans_top_down",
        "target_branch_factor": 8,
        "min_branch_factor": 2,
        "max_branch_factor": 8,
        "min_docs_to_split": 6,
        "min_docs_per_topic": 2,
        "max_depth": 4,
        "max_memberships_per_doc": 5,
        "secondary_membership_threshold": 0.20,
    },
    "topic_cards": {
        "representative_docs": 8,
        "representative_snippets": 5,
        "snippet_target_chars": 450,
        "snippet_max_chars": 700,
        "max_snippets_from_same_doc": 1,
    },
    "search": {
        "default_result_level": "passage",
        "default_max_results": 15,
        "max_results_hard": 50,
        "diversify_results": True,
    },
    "reading": {
        "read_window_default_max_chars": 3000,
        "read_window_hard_max_chars": 6000,
        "read_pages_default_max_chars": 6000,
        "read_pages_hard_max_chars": 10000,
        "read_pages_max_pages": 5,
        "read_pages_hard_max_pages": 8,
    },
    "budgets": {
        "orient_corpus_tokens": 12000,
        "expand_topic_tokens": 10000,
        "topic_card_tokens": 9000,
        "sample_topic_tokens": 3500,
        "search_within_tokens": 10000,
        "doc_card_tokens": 7000,
    },
}


@dataclass
class BuildStats:
    corpus_id: str
    corpus_version: str
    doc_count: int
    page_count: int
    chunk_count: int
    topic_count: int
    index_path: Path


def build_index(
    source_dir: str | Path,
    index_dir: str | Path,
    *,
    corpus_id: str = "C-FIELDGUIDE",
    corpus_version: str = "v1",
    config: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> BuildStats:
    """Build a local JSON index from a directory of source files."""

    source_root = Path(source_dir).expanduser().resolve()
    output_dir = Path(index_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_config = _deep_merge(DEFAULT_CONFIG, config or {})

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    source_files = list(_iter_source_files(source_root, output_dir))
    started_at = time.monotonic()
    if progress:
        progress(
            {
                "event": "build_start",
                "total_files": len(source_files),
                "source_dir": str(source_root),
                "index_dir": str(output_dir),
                "pdf_backend": merged_config["extraction"].get("pdf_backend", "legacy"),
            }
        )

    for file_index, path in enumerate(source_files, start=1):
        relative_path = str(path.relative_to(source_root))
        doc_started_at = time.monotonic()
        if progress:
            progress(
                {
                    "event": "document_start",
                    "index": file_index,
                    "total_files": len(source_files),
                    "source_uri": relative_path,
                    "size_bytes": path.stat().st_size,
                    "elapsed_seconds": doc_started_at - started_at,
                }
            )
        document, doc_chunks, doc_entities = _document_from_file(
            path,
            source_root=source_root,
            corpus_id=corpus_id,
            corpus_version=corpus_version,
            config=merged_config,
        )
        documents.append(document)
        chunks.extend(doc_chunks)
        entities.extend(doc_entities)
        total_elapsed = time.monotonic() - started_at
        doc_elapsed = time.monotonic() - doc_started_at
        average_seconds = total_elapsed / file_index if file_index else 0.0
        remaining = max(0, len(source_files) - file_index)
        if progress:
            progress(
                {
                    "event": "document_done",
                    "index": file_index,
                    "total_files": len(source_files),
                    "source_uri": relative_path,
                    "doc_id": document["doc_id"],
                    "doc_type": document["doc_type"],
                    "extraction_method": document["extraction"]["extraction_method"],
                    "page_count": document["page_count"],
                    "chunk_count": len(doc_chunks),
                    "duration_seconds": doc_elapsed,
                    "elapsed_seconds": total_elapsed,
                    "average_seconds_per_file": average_seconds,
                    "eta_seconds": average_seconds * remaining,
                }
            )

    documents.sort(key=lambda item: item["doc_id"])
    chunks.sort(key=lambda item: item["chunk_id"])
    entities.sort(key=lambda item: (item["doc_id"], item.get("char_start") or 0, item["text"]))
    _add_duplicate_relationships(documents)

    if progress:
        progress(
            {
                "event": "topicing_start",
                "doc_count": len(documents),
                "page_count": sum(len(doc.get("pages", [])) for doc in documents),
                "chunk_count": len(chunks),
                "entity_count": len(entities),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    vectors, idf = _build_document_vectors(documents)
    topics = _build_topics(documents, chunks, entities, vectors, idf, corpus_id, corpus_version, merged_config)
    scopes = _build_scopes(topics, documents, corpus_id, corpus_version)
    _attach_topic_memberships(documents, topics, vectors, merged_config)
    _attach_topic_representatives(topics, documents, chunks, entities, vectors, merged_config)
    if progress:
        progress(
            {
                "event": "topicing_done",
                "topic_count": len(topics),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )

    index = {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "canonical_text_version": CANONICAL_TEXT_VERSION,
        "config": merged_config,
        "documents": documents,
        "chunks": chunks,
        "topics": topics,
        "entities": entities,
        "scopes": scopes,
        "idf": idf,
    }

    index_path = output_dir / INDEX_FILENAME
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    if progress:
        progress(
            {
                "event": "build_done",
                "doc_count": len(documents),
                "page_count": sum(len(doc["pages"]) for doc in documents),
                "chunk_count": len(chunks),
                "topic_count": len(topics),
                "elapsed_seconds": time.monotonic() - started_at,
                "index_path": str(index_path),
            }
        )
    return BuildStats(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        doc_count=len(documents),
        page_count=sum(len(doc["pages"]) for doc in documents),
        chunk_count=len(chunks),
        topic_count=len(topics),
        index_path=index_path,
    )


class FieldguideIndex:
    """Bounded API over a built Fieldguide corpus index."""

    def __init__(self, index_dir: str | Path):
        index_path = Path(index_dir).expanduser().resolve()
        if index_path.is_dir():
            index_path = index_path / INDEX_FILENAME
        self.index_path = index_path
        self.data = json.loads(index_path.read_text(encoding="utf-8"))
        self.corpus_id = self.data["corpus_id"]
        self.corpus_version = self.data["corpus_version"]
        self.config = self.data.get("config", DEFAULT_CONFIG)
        self.documents = self.data.get("documents", [])
        self.chunks = self.data.get("chunks", [])
        self.topics = self.data.get("topics", [])
        self.entities = self.data.get("entities", [])
        self.scopes = self.data.get("scopes", [])
        self.docs_by_id = {doc["doc_id"]: doc for doc in self.documents}
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.topics_by_id = {topic["topic_id"]: topic for topic in self.topics}
        self.scopes_by_id = {scope["scope_id"]: scope for scope in self.scopes}
        self.entities_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in self.entities:
            self.entities_by_doc[entity["doc_id"]].append(entity)
        self.snippets_by_id: dict[str, dict[str, Any]] = {}
        for topic in self.topics:
            for snippet in topic.get("representative_snippets", []):
                self.snippets_by_id[snippet["snippet_id"]] = snippet

    def orient_corpus(
        self,
        *,
        max_top_topics: int = 10,
        include_facets: bool = True,
        include_quality: bool = True,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = token_budget or self.config["budgets"]["orient_corpus_tokens"]
        root = self._root_topic()
        top_topics = [
            self._topic_summary(topic, snippets=0, docs=0)
            for topic in self._children(root["topic_id"])[: max(0, max_top_topics)]
        ]
        if not top_topics and root:
            top_topics = [self._topic_summary(root, snippets=0, docs=0)]
        result = {
            "corpus_summary": self._corpus_summary(),
            "top_topics": top_topics,
            "root_scope": self.scopes_by_id.get("S-root", self._scope_for_docs(self.documents, kind="corpus")),
        }
        if include_facets:
            result["global_facets"] = self._global_facets()
        if include_quality:
            result["quality"] = self._corpus_quality()
        return self._envelope(
            "orient_corpus",
            result,
            response_kind="orientation",
            token_budget=token_budget,
            coverage={"topics_seen": [topic["topic_id"] for topic in top_topics], "docs_opened": 0, "pages_read": 0},
            actions=[
                self._action(
                    "search_within",
                    {"scope_id": "S-root", "query": ""},
                    "low",
                    "Search the explicit full-corpus scope.",
                    token_range=[800, 2200],
                )
            ],
        )

    def expand_topic(
        self,
        topic_id: str,
        *,
        max_children: int = 8,
        snippets_per_child: int = 2,
        docs_per_child: int = 3,
        include_neighbors: bool = True,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = token_budget or self.config["budgets"]["expand_topic_tokens"]
        topic = self._require_topic(topic_id)
        children = [
            self._topic_summary(child, snippets=snippets_per_child, docs=docs_per_child)
            for child in self._children(topic_id)[: max(0, max_children)]
        ]
        result = {
            "topic": self._topic_summary(topic, snippets=snippets_per_child, docs=docs_per_child),
            "children": children,
        }
        if include_neighbors:
            result["neighboring_topics"] = topic.get("neighboring_topics", [])
        return self._envelope(
            "expand_topic",
            result,
            response_kind="navigation",
            token_budget=token_budget,
            coverage={"topics_seen": [topic_id, *[child["topic_id"] for child in children]], "docs_opened": 0},
            actions=topic.get("actions", []),
        )

    def topic_card(
        self,
        topic_id: str,
        *,
        snippets: int = 5,
        representative_docs: int = 8,
        include_facets: bool = True,
        include_neighbors: bool = True,
        include_quality: bool = True,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = token_budget or self.config["budgets"]["topic_card_tokens"]
        topic = dict(self._require_topic(topic_id))
        topic["representative_snippets"] = topic.get("representative_snippets", [])[:snippets]
        topic["representative_docs"] = topic.get("representative_docs", [])[:representative_docs]
        if not include_neighbors:
            topic.pop("neighboring_topics", None)
        if not include_quality:
            topic.pop("quality", None)
        if include_facets:
            docs = [self.docs_by_id[doc_id] for doc_id in topic.get("doc_ids", []) if doc_id in self.docs_by_id]
            topic["facets"] = self._facets_for_docs(docs)
        topic["interpretation_warning"] = (
            "This topic card is an extractive navigation aid. It is not a factual summary of all documents in the topic."
        )
        return self._envelope(
            "topic_card",
            topic,
            response_kind="navigation",
            token_budget=token_budget,
            warnings=topic.get("quality", {}).get("warnings", []),
            coverage={"topics_seen": [topic_id], "docs_referenced": len(topic.get("representative_docs", []))},
            actions=topic.get("actions", []),
        )

    def search_within(
        self,
        *,
        scope: dict[str, Any],
        query: str,
        result_level: str = "passage",
        max_results: int = 15,
        filters: dict[str, Any] | None = None,
        diversify: bool = True,
        include_snippets: bool = True,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = token_budget or self.config["budgets"]["search_within_tokens"]
        max_results = min(max(0, max_results), self.config["search"]["max_results_hard"])
        resolved_scope = self._resolve_scope(scope)
        scoped_doc_ids = self._doc_ids_for_scope(resolved_scope, extra_filters=filters)
        query_terms = tokenize(query, keep_stopwords=False)
        if result_level not in {"document", "page", "passage"}:
            raise ValueError("result_level must be one of: document, page, passage")

        if result_level == "document":
            raw_results = self._search_documents(scoped_doc_ids, query_terms, include_snippets)
        elif result_level == "page":
            raw_results = self._search_pages(scoped_doc_ids, query_terms, include_snippets)
        else:
            raw_results = self._search_chunks(scoped_doc_ids, query_terms, include_snippets)

        if diversify:
            raw_results = self._diversify(raw_results)
        results = raw_results[:max_results]
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        result = {
            "scope": resolved_scope,
            "query": query,
            "result_level": result_level,
            "total_estimated_matches": len(raw_results),
            "results": results,
            "suggested_refinements": self._suggest_refinements(raw_results, resolved_scope, query),
        }
        warnings = []
        if resolved_scope.get("kind") == "corpus":
            warnings.append("Full-corpus search is explicit; consider narrowing to a topic or facet-filtered scope.")
        return self._envelope(
            "search_within",
            result,
            response_kind="search_results",
            token_budget=token_budget,
            warnings=warnings,
            coverage={
                "docs_referenced": len({item["doc_id"] for item in results}),
                "docs_opened": 0,
                "estimated_scope_doc_coverage": _safe_ratio(len({item["doc_id"] for item in results}), len(scoped_doc_ids)),
            },
            actions=[self._action("search_within", {"scope_id": resolved_scope["scope_id"], "query": query}, "low", "Refine this scoped search.")],
        )

    def doc_card(
        self,
        doc_id: str,
        *,
        include_page_map: bool = True,
        include_best_passages: bool = True,
        include_related_docs: bool = True,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        token_budget = token_budget or self.config["budgets"]["doc_card_tokens"]
        doc = self._require_doc(doc_id)
        result: dict[str, Any] = {
            "doc": self._document_summary(doc),
            "topic_memberships": doc.get("topic_memberships", []),
            "quality": doc["extraction"],
            "actions": [
                self._action(
                    "read_pages",
                    {"doc_id": doc_id, "pages": [1]},
                    "medium",
                    "Read the first page as bounded evidence.",
                    token_range=[400, 1800],
                )
            ],
        }
        if include_page_map:
            result["page_map"] = [self._page_map_item(doc, page) for page in doc.get("pages", [])]
        if include_best_passages:
            result["best_passages"] = self._best_passages_for_doc(doc)[:5]
        if include_related_docs:
            result["related_docs"] = self._related_docs(doc)
        return self._envelope(
            "doc_card",
            result,
            response_kind="triage",
            token_budget=token_budget,
            warnings=doc["extraction"].get("warnings", []),
            coverage={"docs_referenced": 1, "docs_opened": 0, "pages_read": 0},
            actions=result["actions"],
        )

    def read_window(
        self,
        *,
        doc_id: str | None = None,
        page_number: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        snippet_id: str | None = None,
        context_chars_before: int = 700,
        context_chars_after: int = 700,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        hard_max = self.config["reading"]["read_window_hard_max_chars"]
        max_chars = min(max_chars or self.config["reading"]["read_window_default_max_chars"], hard_max)
        if snippet_id:
            snippet = self.snippets_by_id.get(snippet_id)
            if not snippet:
                raise KeyError(f"unknown snippet_id: {snippet_id}")
            doc_id = snippet["doc_id"]
            page_number = snippet.get("page_number")
            char_start = snippet["char_start"]
            char_end = snippet["char_end"]
        if doc_id is None or char_start is None:
            raise ValueError("read_window requires snippet_id or doc_id with char_start")
        doc = self._require_doc(doc_id)
        text = doc["canonical_text"]
        char_end = char_start if char_end is None else char_end
        start = max(0, char_start - max(0, context_chars_before))
        end = min(len(text), char_end + max(0, context_chars_after))
        if end - start > max_chars:
            needed = max(0, char_end - char_start)
            extra = max(0, max_chars - needed)
            start = max(0, char_start - extra // 2)
            end = min(len(text), start + max_chars)
            if char_end > end:
                end = min(len(text), char_end)
                start = max(0, end - max_chars)
        start, end, window_text = trim_to_range(text, start, end)
        if page_number is None:
            page_number = _page_for_range(doc, char_start, char_end)
        result = {
            "doc_id": doc_id,
            "title": doc.get("title"),
            "doc_type": doc["doc_type"],
            "source_batch": doc.get("source_batch"),
            "page_number": page_number,
            "char_start": start,
            "char_end": end,
            "text": window_text,
            "provenance": self._provenance(doc, page_numbers=[page_number] if page_number else None, char_start=start, char_end=end),
            "extraction_quality": doc["extraction"],
            "neighboring_actions": self._neighboring_read_actions(doc, start, end),
        }
        return self._envelope(
            "read_window",
            result,
            response_kind="evidence",
            token_budget=None,
            coverage={"docs_opened": 1, "pages_read": 1 if page_number else 0},
            actions=result["neighboring_actions"],
        )

    def read_pages(self, *, doc_id: str, pages: list[int], max_chars: int | None = None) -> dict[str, Any]:
        hard_pages = self.config["reading"]["read_pages_hard_max_pages"]
        if len(pages) > hard_pages:
            raise ValueError(f"read_pages may read at most {hard_pages} pages per call")
        hard_chars = self.config["reading"]["read_pages_hard_max_chars"]
        max_chars = min(max_chars or self.config["reading"]["read_pages_default_max_chars"], hard_chars)
        doc = self._require_doc(doc_id)
        by_number = {page["page_number"]: page for page in doc.get("pages", [])}
        page_evidence: list[dict[str, Any]] = []
        total_chars = 0
        truncated = False
        for number in pages:
            page = by_number.get(number)
            if page is None:
                raise KeyError(f"{doc_id} has no page {number}")
            text = page["text"]
            remaining = max_chars - total_chars
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                truncated = True
            total_chars += len(text)
            page_evidence.append(
                {
                    "page_number": number,
                    "char_start": page["char_start"],
                    "char_end": min(page["char_end"], page["char_start"] + len(text)),
                    "text": text,
                    "extraction_warnings": page["extraction_quality"].get("warnings", []),
                }
            )
        result = {
            "doc_id": doc_id,
            "title": doc.get("title"),
            "doc_type": doc["doc_type"],
            "source_batch": doc.get("source_batch"),
            "pages": page_evidence,
            "provenance": self._provenance(doc, page_numbers=pages),
            "extraction_quality": doc["extraction"],
            "suggested_next_pages": self._next_page_actions(doc, pages),
        }
        response = self._envelope(
            "read_pages",
            result,
            response_kind="evidence",
            token_budget=None,
            warnings=["Result truncated to max_chars."] if truncated else [],
            coverage={"docs_opened": 1, "pages_read": len(page_evidence)},
            actions=result["suggested_next_pages"],
        )
        response["truncated"] = response["truncated"] or truncated
        return response

    def _require_doc(self, doc_id: str) -> dict[str, Any]:
        try:
            return self.docs_by_id[doc_id]
        except KeyError as exc:
            raise KeyError(f"unknown doc_id: {doc_id}") from exc

    def _require_topic(self, topic_id: str) -> dict[str, Any]:
        try:
            return self.topics_by_id[topic_id]
        except KeyError as exc:
            raise KeyError(f"unknown topic_id: {topic_id}") from exc

    def _root_topic(self) -> dict[str, Any]:
        for topic in self.topics:
            if topic["depth"] == 0:
                return topic
        raise KeyError("index has no root topic")

    def _children(self, topic_id: str) -> list[dict[str, Any]]:
        topic = self._require_topic(topic_id)
        return [self.topics_by_id[child_id] for child_id in topic.get("child_topic_ids", []) if child_id in self.topics_by_id]

    def _topic_summary(self, topic: dict[str, Any], *, snippets: int, docs: int) -> dict[str, Any]:
        summary = {
            "topic_id": topic["topic_id"],
            "parent_topic_id": topic.get("parent_topic_id"),
            "child_topic_ids": topic.get("child_topic_ids", []),
            "depth": topic["depth"],
            "doc_count": topic["doc_count"],
            "page_count": topic["page_count"],
            "chunk_count": topic["chunk_count"],
            "label_terms": topic.get("label_terms", []),
            "label_phrases": topic.get("label_phrases", []),
            "top_entities": topic.get("top_entities", []),
            "dominant_doc_types": topic.get("dominant_doc_types", []),
            "contrast_terms": topic.get("contrast_terms", []),
            "scope": topic.get("scope"),
            "quality": topic.get("quality"),
            "actions": topic.get("actions", []),
        }
        if snippets:
            summary["representative_snippets"] = topic.get("representative_snippets", [])[:snippets]
        if docs:
            summary["representative_docs"] = topic.get("representative_docs", [])[:docs]
        return summary

    def _corpus_summary(self) -> dict[str, Any]:
        return {
            "doc_count": len(self.documents),
            "page_count": sum(len(doc.get("pages", [])) for doc in self.documents),
            "chunk_count": len(self.chunks),
            "dominant_doc_types": _count_summary(doc["doc_type"] for doc in self.documents),
            "dominant_sources": _count_summary(doc.get("source_batch", "default") for doc in self.documents),
            "languages": _count_summary(doc.get("language", "unknown") for doc in self.documents),
        }

    def _global_facets(self) -> dict[str, Any]:
        return self._facets_for_docs(self.documents)

    def _facets_for_docs(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        doc_ids = {doc["doc_id"] for doc in docs}
        entities = [entity for entity in self.entities if entity["doc_id"] in doc_ids]
        return {
            "doc_types": _count_summary(doc["doc_type"] for doc in docs),
            "sources": _count_summary(doc.get("source_batch", "default") for doc in docs),
            "people": _count_summary(entity["text"] for entity in entities if entity["type"] == "person"),
            "organizations": _count_summary(entity["text"] for entity in entities if entity["type"] == "organization"),
            "locations": _count_summary(entity["text"] for entity in entities if entity["type"] == "location"),
            "extraction_quality": _count_summary(doc["extraction"]["extraction_method"] for doc in docs),
        }

    def _corpus_quality(self) -> dict[str, Any]:
        total = len(self.documents) or 1
        text_count = sum(1 for doc in self.documents if doc["extraction"].get("text_available"))
        ocr_count = sum(1 for doc in self.documents if doc["extraction"].get("extraction_method") == "ocr")
        known_ocr_confidences = [
            doc["extraction"]["ocr_confidence"]
            for doc in self.documents
            if isinstance(doc["extraction"].get("ocr_confidence"), (int, float))
        ]
        low_ocr_confidences = [confidence for confidence in known_ocr_confidences if confidence < 0.75]
        duplicates = sum(1 for doc in self.documents if doc.get("relationships"))
        warnings: list[str] = []
        if text_count < len(self.documents):
            warnings.append("Some documents have no extracted text.")
        return {
            "percent_text_extracted": round(text_count / total, 4),
            "percent_ocr": round(ocr_count / total, 4),
            "mean_ocr_confidence": round(sum(known_ocr_confidences) / len(known_ocr_confidences), 4) if known_ocr_confidences else None,
            "percent_low_ocr_confidence": round(len(low_ocr_confidences) / len(known_ocr_confidences), 4) if known_ocr_confidences else 0.0,
            "duplicate_rate": round(duplicates / total, 4),
            "language_distribution": _count_summary(doc.get("language", "unknown") for doc in self.documents),
            "warnings": warnings,
        }

    def _resolve_scope(self, scope: dict[str, Any]) -> dict[str, Any]:
        if "scope_id" in scope and len(scope) == 1:
            scope_id = scope["scope_id"]
            if scope_id not in self.scopes_by_id:
                raise KeyError(f"unknown scope_id: {scope_id}")
            return self.scopes_by_id[scope_id]
        if "scope_id" in scope and scope["scope_id"] in self.scopes_by_id:
            resolved = dict(self.scopes_by_id[scope["scope_id"]])
            resolved.update({key: value for key, value in scope.items() if key != "scope_id"})
            return resolved
        return {
            "scope_id": scope.get("scope_id", stable_id("S", json.dumps(scope, sort_keys=True), length=8)),
            "kind": scope.get("kind", "compound"),
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            **scope,
        }

    def _doc_ids_for_scope(self, scope: dict[str, Any], extra_filters: dict[str, Any] | None = None) -> set[str]:
        if scope.get("kind") == "corpus":
            doc_ids = {doc["doc_id"] for doc in self.documents}
        elif scope.get("doc_ids"):
            doc_ids = set(scope["doc_ids"])
        elif scope.get("topic_ids"):
            topic_ids: set[str] = set(scope["topic_ids"])
            if scope.get("include_descendants", True):
                for topic_id in list(topic_ids):
                    topic_ids.update(self._descendant_topic_ids(topic_id))
            doc_ids = set()
            for topic_id in topic_ids:
                topic = self.topics_by_id.get(topic_id)
                if topic:
                    doc_ids.update(topic.get("doc_ids", []))
        else:
            doc_ids = {doc["doc_id"] for doc in self.documents}

        filters = dict(scope.get("filters") or {})
        if extra_filters:
            filters.update(extra_filters)
        return self._apply_filters(doc_ids, filters)

    def _descendant_topic_ids(self, topic_id: str) -> set[str]:
        descendants: set[str] = set()
        stack = list(self._require_topic(topic_id).get("child_topic_ids", []))
        while stack:
            current = stack.pop()
            descendants.add(current)
            stack.extend(self.topics_by_id.get(current, {}).get("child_topic_ids", []))
        return descendants

    def _apply_filters(self, doc_ids: set[str], filters: dict[str, Any]) -> set[str]:
        if not filters:
            return doc_ids
        output: set[str] = set()
        entity_filters = set(_lower_list(filters.get("entities", [])))
        entity_filters.update(_lower_list(filters.get("organizations", [])))
        entity_filters.update(_lower_list(filters.get("people", [])))
        entity_filters.update(_lower_list(filters.get("locations", [])))
        docs_with_entities = set()
        if entity_filters:
            for entity in self.entities:
                if entity["doc_id"] in doc_ids and entity["normalized_text"] in entity_filters:
                    docs_with_entities.add(entity["doc_id"])
        for doc_id in doc_ids:
            doc = self.docs_by_id[doc_id]
            if filters.get("doc_types") and doc["doc_type"] not in filters["doc_types"]:
                continue
            if filters.get("sources") and doc.get("source_batch") not in filters["sources"]:
                continue
            if filters.get("extraction_quality") and doc["extraction"]["extraction_method"] not in filters["extraction_quality"]:
                continue
            if filters.get("languages") and doc.get("language", "unknown") not in filters["languages"]:
                continue
            if filters.get("has_tables") is not None:
                has_tables = any(page.get("has_table") for page in doc.get("pages", []))
                if has_tables != filters["has_tables"]:
                    continue
            if entity_filters and doc_id not in docs_with_entities:
                continue
            output.add(doc_id)
        return output

    def _search_chunks(self, doc_ids: set[str], query_terms: list[str], include_snippets: bool) -> list[dict[str, Any]]:
        chunks = [chunk for chunk in self.chunks if chunk["doc_id"] in doc_ids]
        scores = _bm25_scores(chunks, query_terms, "text")
        results: list[dict[str, Any]] = []
        for chunk, score in scores:
            if score <= 0:
                continue
            doc = self.docs_by_id[chunk["doc_id"]]
            matched_terms = _matched_terms(query_terms, chunk["text"])
            char_start, char_end = chunk["char_start"], chunk["char_end"]
            snippet_text = None
            if include_snippets:
                hit_start, hit_end = _first_hit_range(doc["canonical_text"], matched_terms, char_start, char_end)
                _, _, snippet_text = make_excerpt_range(doc["canonical_text"], hit_start, hit_end, 420, 650)
            results.append(self._search_result(doc, score, matched_terms, page_number=chunk.get("page_number"), char_start=char_start, char_end=char_end, snippet=snippet_text))
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def _search_pages(self, doc_ids: set[str], query_terms: list[str], include_snippets: bool) -> list[dict[str, Any]]:
        page_records: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            doc = self.docs_by_id[doc_id]
            for page in doc.get("pages", []):
                page_records.append({"doc_id": doc_id, **page})
        scores = _bm25_scores(page_records, query_terms, "text")
        results: list[dict[str, Any]] = []
        for page, score in scores:
            if score <= 0:
                continue
            doc = self.docs_by_id[page["doc_id"]]
            matched_terms = _matched_terms(query_terms, page["text"])
            snippet = None
            if include_snippets:
                hit_start, hit_end = _first_hit_range(doc["canonical_text"], matched_terms, page["char_start"], page["char_end"])
                _, _, snippet = make_excerpt_range(doc["canonical_text"], hit_start, hit_end, 420, 650)
            results.append(
                self._search_result(
                    doc,
                    score,
                    matched_terms,
                    page_number=page["page_number"],
                    char_start=page["char_start"],
                    char_end=page["char_end"],
                    snippet=snippet,
                )
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def _search_documents(self, doc_ids: set[str], query_terms: list[str], include_snippets: bool) -> list[dict[str, Any]]:
        docs = [self.docs_by_id[doc_id] for doc_id in doc_ids]
        scores = _bm25_scores(docs, query_terms, "canonical_text")
        results: list[dict[str, Any]] = []
        for doc, score in scores:
            if score <= 0:
                continue
            matched_terms = _matched_terms(query_terms, doc["canonical_text"])
            snippet = None
            char_start = char_end = None
            page_number = None
            if include_snippets and matched_terms:
                char_start, char_end = _first_hit_range(doc["canonical_text"], matched_terms, 0, len(doc["canonical_text"]))
                page_number = _page_for_range(doc, char_start, char_end)
                _, _, snippet = make_excerpt_range(doc["canonical_text"], char_start, char_end, 420, 650)
            results.append(self._search_result(doc, score, matched_terms, page_number=page_number, char_start=char_start, char_end=char_end, snippet=snippet))
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def _search_result(
        self,
        doc: dict[str, Any],
        score: float,
        matched_terms: list[str],
        *,
        page_number: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        snippet: str | None = None,
    ) -> dict[str, Any]:
        matched_entity_terms = {term.lower() for term in matched_terms}
        matched_entity_mentions = [
            entity
            for entity in self.entities_by_doc.get(doc["doc_id"], [])
            if entity["normalized_text"] in matched_entity_terms
        ]
        matched_entities = _dedupe_entities(matched_entity_mentions, limit=8)
        result = {
            "rank": 0,
            "score": round(score, 4),
            "doc_id": doc["doc_id"],
            "title": doc.get("title"),
            "doc_type": doc["doc_type"],
            "source_batch": doc.get("source_batch"),
            "page_number": page_number,
            "char_start": char_start,
            "char_end": char_end,
            "snippet": snippet,
            "matched_terms": matched_terms,
            "matched_entities": matched_entities,
            "matched_entity_mention_count": len(matched_entity_mentions),
            "topic_memberships": doc.get("topic_memberships", []),
            "actions": [
                self._action("doc_card", {"doc_id": doc["doc_id"]}, "low", "Open a document triage card."),
            ],
        }
        if char_start is not None:
            result["actions"].append(
                self._action(
                    "read_window",
                    {"doc_id": doc["doc_id"], "page_number": page_number, "char_start": char_start, "char_end": char_end},
                    "medium",
                    "Read a bounded evidence window around this hit.",
                    token_range=[400, 1800],
                )
            )
        if page_number is not None:
            result["actions"].append(
                self._action(
                    "read_pages",
                    {"doc_id": doc["doc_id"], "pages": [page_number]},
                    "medium",
                    "Read this page as bounded evidence.",
                    token_range=[500, 2000],
                )
            )
        return result

    def _diversify(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        seen_docs: set[str] = set()
        seen_sources: set[str] = set()
        for result in results:
            key = result["doc_id"]
            source = result.get("source_batch")
            if key in seen_docs or (source and source in seen_sources and len(selected) >= 3):
                deferred.append(result)
                continue
            selected.append(result)
            seen_docs.add(key)
            if source:
                seen_sources.add(source)
        return selected + deferred

    def _suggest_refinements(self, results: list[dict[str, Any]], scope: dict[str, Any], query: str) -> list[dict[str, Any]]:
        result_doc_ids = {result["doc_id"] for result in results}
        scope_doc_ids = self._doc_ids_for_scope(scope)
        docs = [self.docs_by_id[doc_id] for doc_id in result_doc_ids]
        suggestions: list[dict[str, Any]] = []
        for item in _count_summary(doc["doc_type"] for doc in docs)[:3]:
            suggestions.append(
                {
                    "type": "doc_type",
                    "value": item["value"],
                    "estimated_result_count": item["count"],
                    "action": self._action(
                        "search_within",
                        {"scope_id": scope["scope_id"], "query": query, "filters": {"doc_types": [item["value"]]}},
                        "low",
                        f"Narrow results to {item['value']} documents.",
                    ),
                }
            )
        observed_docs_by_entity: dict[str, set[str]] = defaultdict(set)
        scoped_docs_by_entity: dict[str, set[str]] = defaultdict(set)
        labels_by_entity: dict[str, Counter[str]] = defaultdict(Counter)
        refinement_entity_types = {"organization", "person", "location", "email", "case_number"}
        for entity in self.entities:
            if entity["type"] not in refinement_entity_types:
                continue
            entity_key = entity["normalized_text"]
            if entity["doc_id"] in scope_doc_ids:
                scoped_docs_by_entity[entity_key].add(entity["doc_id"])
            if entity["doc_id"] in result_doc_ids:
                observed_docs_by_entity[entity_key].add(entity["doc_id"])
                labels_by_entity[entity_key][entity["text"]] += 1

        ranked_entities = sorted(
            observed_docs_by_entity,
            key=lambda entity_key: (
                len(observed_docs_by_entity[entity_key]),
                len(scoped_docs_by_entity[entity_key]),
                sum(labels_by_entity[entity_key].values()),
            ),
            reverse=True,
        )
        for entity_key in ranked_entities[:3]:
            value = labels_by_entity[entity_key].most_common(1)[0][0]
            observed_count = len(observed_docs_by_entity[entity_key])
            scope_count = len(scoped_docs_by_entity[entity_key])
            suggestions.append(
                {
                    "type": "entity",
                    "value": value,
                    "estimated_result_count": observed_count,
                    "observed_result_count": observed_count,
                    "scope_document_count": scope_count,
                    "action": self._action(
                        "search_within",
                        {"scope_id": scope["scope_id"], "query": query, "filters": {"entities": [entity_key]}},
                        "low",
                        f"Narrow results to entity {value}.",
                    ),
                }
            )
        return suggestions[:5]

    def _document_summary(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "doc_id": doc["doc_id"],
            "source_uri": doc.get("source_uri"),
            "source_batch": doc.get("source_batch"),
            "original_filename": doc["original_filename"],
            "title": doc.get("title"),
            "doc_type": doc["doc_type"],
            "language": doc.get("language", "unknown"),
            "page_count": doc.get("page_count"),
            "char_count": doc.get("char_count"),
            "token_estimate": doc.get("token_estimate"),
            "extraction": doc.get("extraction"),
            "metadata": doc.get("metadata", {}),
            "relationships": doc.get("relationships", []),
            "topic_memberships": doc.get("topic_memberships", []),
            "facets": doc.get("facets", {}),
        }

    def _page_map_item(self, doc: dict[str, Any], page: dict[str, Any]) -> dict[str, Any]:
        entities = [
            entity
            for entity in self.entities_by_doc.get(doc["doc_id"], [])
            if entity.get("page_number") == page["page_number"]
        ]
        return {
            "page_number": page["page_number"],
            "token_estimate": page["token_estimate"],
            "top_terms": page.get("top_terms", []),
            "top_entities": _entity_summary(entities),
            "signals": _page_signals(page, entities),
            "has_table": page.get("has_table", False),
            "extraction_warnings": page["extraction_quality"].get("warnings", []),
            "read_action": self._action(
                "read_pages",
                {"doc_id": doc["doc_id"], "pages": [page["page_number"]]},
                "medium",
                f"Read page {page['page_number']}.",
                token_range=[500, 2000],
            ),
        }

    def _best_passages_for_doc(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        for topic in self.topics:
            for snippet in topic.get("representative_snippets", []):
                if snippet["doc_id"] == doc["doc_id"]:
                    snippets.append(snippet)
        if snippets:
            return snippets
        chunks = [chunk for chunk in self.chunks if chunk["doc_id"] == doc["doc_id"]]
        passages: list[dict[str, Any]] = []
        for chunk in chunks[:5]:
            start, end, text = make_excerpt_range(doc["canonical_text"], chunk["char_start"], chunk["char_end"], 420, 650)
            passages.append(
                {
                    "snippet_id": stable_id("SNIP-DOC", doc["doc_id"], start, end, length=12),
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title"),
                    "doc_type": doc["doc_type"],
                    "source_batch": doc.get("source_batch"),
                    "page_number": chunk.get("page_number"),
                    "char_start": start,
                    "char_end": end,
                    "text": text,
                    "selection_reason": ["contains_representative_terms"],
                    "matched_terms": chunk.get("top_terms", [])[:5],
                    "entities": [],
                    "snippet_warning": "Snippet is selected as representative of a topic signal. Read the evidence window before making claims.",
                    "read_action": self._action(
                        "read_window",
                        {"doc_id": doc["doc_id"], "char_start": start, "char_end": end},
                        "medium",
                        "Read the evidence window for this passage.",
                    ),
                }
            )
        return passages

    def _related_docs(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        related: list[dict[str, Any]] = []
        for relation in doc.get("relationships", []):
            target = self.docs_by_id.get(relation["target_doc_id"])
            if not target:
                continue
            related.append(
                {
                    "doc": self._document_summary(target),
                    "relation": relation["relation"],
                    "confidence": relation["confidence"],
                    "action": self._action("doc_card", {"doc_id": target["doc_id"]}, "low", "Open related document card."),
                }
            )
        return related

    def _scope_for_docs(self, docs: list[dict[str, Any]], *, kind: str = "document_set") -> dict[str, Any]:
        doc_ids = [doc["doc_id"] for doc in docs]
        return {
            "scope_id": stable_id("S", kind, ",".join(doc_ids), length=8) if kind != "corpus" else "S-root",
            "kind": kind,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "doc_ids": doc_ids,
            "estimated_doc_count": len(doc_ids),
            "estimated_page_count": sum(len(doc.get("pages", [])) for doc in docs),
        }

    def _provenance(
        self,
        doc: dict[str, Any],
        *,
        page_numbers: list[int] | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> dict[str, Any]:
        return {
            "doc_id": doc["doc_id"],
            "original_filename": doc["original_filename"],
            "source_uri": doc.get("source_uri"),
            "source_batch": doc.get("source_batch"),
            "page_numbers": page_numbers,
            "char_start": char_start,
            "char_end": char_end,
            "extraction_method": doc["extraction"]["extraction_method"],
            "extraction_warnings": doc["extraction"].get("warnings", []),
            "canonical_text_version": self.data.get("canonical_text_version", CANONICAL_TEXT_VERSION),
        }

    def _neighboring_read_actions(self, doc: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        before = max(0, start - 1800)
        after = min(len(doc["canonical_text"]), end + 1800)
        if before < start:
            actions.append(self._action("read_window", {"doc_id": doc["doc_id"], "char_start": before, "char_end": start}, "medium", "Read the preceding bounded window."))
        if end < after:
            actions.append(self._action("read_window", {"doc_id": doc["doc_id"], "char_start": end, "char_end": after}, "medium", "Read the following bounded window."))
        return actions

    def _next_page_actions(self, doc: dict[str, Any], pages: list[int]) -> list[dict[str, Any]]:
        page_numbers = {page["page_number"] for page in doc.get("pages", [])}
        suggestions: list[int] = []
        for page in pages:
            for neighbor in (page - 1, page + 1):
                if neighbor in page_numbers and neighbor not in pages and neighbor not in suggestions:
                    suggestions.append(neighbor)
        return [
            self._action("read_pages", {"doc_id": doc["doc_id"], "pages": [page]}, "medium", f"Read neighboring page {page}.")
            for page in suggestions[:4]
        ]

    def _action(
        self,
        action: str,
        args: dict[str, Any],
        estimated_cost: str,
        description: str,
        *,
        token_range: list[int] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "action": action,
            "args": args,
            "estimated_cost": estimated_cost,
            "description": description,
        }
        if token_range:
            payload["estimated_token_range"] = token_range
        return payload

    def _envelope(
        self,
        tool_name: str,
        result: Any,
        *,
        response_kind: str,
        token_budget: int | None,
        warnings: list[str] | None = None,
        actions: list[dict[str, Any]] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "tool_name": tool_name,
            "result": result,
            "token_estimate": 0,
            "truncated": False,
            "warnings": list(warnings or []),
            "actions": list(actions or []),
            "response_kind": response_kind,
            "cost_ledger": {
                "result_token_estimate": 0,
                "coverage": coverage or {},
                "cheaper_next_actions": list(actions or [])[:3],
            },
        }
        if token_budget:
            _apply_budget(response, token_budget)
        token_estimate = estimate_tokens(json.dumps(response, ensure_ascii=False, sort_keys=True))
        response["token_estimate"] = token_estimate
        response["cost_ledger"]["result_token_estimate"] = estimate_tokens(json.dumps(response["result"], ensure_ascii=False, sort_keys=True))
        return response


def _iter_source_files(source_root: Path, output_dir: Path) -> Iterable[Path]:
    skip_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build"}
    skip_suffixes = {".pyc", ".pyo"}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if output_dir in path.parents or path.name.startswith("."):
            continue
        relative_parts = path.relative_to(source_root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if any(part in skip_dirs for part in relative_parts):
            continue
        if path.suffix.lower() in skip_suffixes:
            continue
        yield path


def _document_from_file(
    path: Path,
    *,
    source_root: Path,
    corpus_id: str,
    corpus_version: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    extracted = extract_file(path, pdf_backend=config["extraction"].get("pdf_backend", "legacy"))
    file_meta = file_metadata(path, source_root)
    doc_id = stable_id("D", file_meta["source_uri"], sha256, length=12)
    canonical_text, pages = _canonical_pages(extracted.pages)
    topic_text, _topic_pages = _canonical_pages(extracted.topic_pages or extracted.pages)
    quality = extraction_quality(
        extracted.extraction_method,
        extracted.warnings,
        text_available=bool(canonical_text.strip()),
        ocr_confidence=extracted.metadata.get("ocr_confidence"),
    )
    title = _title_from_text(canonical_text) or path.stem
    page_entities: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    for page_number, page_text, char_start, char_end in pages:
        mentions = extract_entities(page_text, doc_id=doc_id, page_number=page_number, char_offset=char_start)
        page_entities.extend(mentions)
        page_records.append(
            {
                "doc_id": doc_id,
                "page_number": page_number,
                "text": page_text,
                "char_start": char_start,
                "char_end": char_end,
                "token_estimate": estimate_tokens(page_text),
                "top_terms": top_terms([page_text], limit=8),
                "top_entities": _entity_summary(mentions),
                "has_table": _looks_tabular(page_text),
                "has_image": False,
                "has_handwriting": False,
                "extraction_quality": quality,
            }
        )
    chunks = _chunks_for_document(doc_id, canonical_text, page_records, page_entities, config)
    document = {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "doc_id": doc_id,
        "source_uri": file_meta["source_uri"],
        "source_batch": path.parent.name if path.parent != source_root else "default",
        "original_filename": path.name,
        "title": title,
        "doc_type": extracted.doc_type,
        "language": "unknown",
        "page_count": len(page_records),
        "char_count": len(canonical_text),
        "token_estimate": estimate_tokens(canonical_text),
        "topic_token_estimate": estimate_tokens(topic_text),
        "extraction": quality,
        "metadata": {**file_meta, **extracted.metadata, "sha256": sha256},
        "relationships": [],
        "topic_memberships": [],
        "facets": {
            "doc_type": extracted.doc_type,
            "source_batch": path.parent.name if path.parent != source_root else "default",
            "has_tables": any(page.get("has_table") for page in page_records),
        },
        "pages": page_records,
        "canonical_text": canonical_text,
        "topic_text": topic_text,
    }
    return document, chunks, page_entities


def _canonical_pages(raw_pages: list[str]) -> tuple[str, list[tuple[int, str, int, int]]]:
    normalized_pages = [page.strip() for page in raw_pages if page is not None]
    if not normalized_pages:
        normalized_pages = [""]
    parts: list[str] = []
    page_records: list[tuple[int, str, int, int]] = []
    cursor = 0
    for index, page_text in enumerate(normalized_pages, start=1):
        if index > 1:
            separator = "\n\f\n"
            parts.append(separator)
            cursor += len(separator)
        char_start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        char_end = cursor
        page_records.append((index, page_text, char_start, char_end))
    return "".join(parts), page_records


def _chunks_for_document(
    doc_id: str,
    canonical_text: str,
    pages: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    chunking = config["chunking"]
    target = chunking["target_chunk_chars"]
    minimum = chunking["min_chunk_chars"]
    maximum = chunking["max_chunk_chars"]
    chunks: list[dict[str, Any]] = []
    for page in pages:
        start = page["char_start"]
        page_end = page["char_end"]
        if start == page_end:
            continue
        while start < page_end:
            proposed = min(page_end, start + target)
            if proposed < page_end:
                search_start = min(page_end, start + minimum)
                search_end = min(page_end, start + maximum)
                boundary = canonical_text.rfind("\n\n", search_start, search_end)
                if boundary == -1:
                    boundary = canonical_text.rfind("\n", search_start, search_end)
                if boundary == -1:
                    boundary = canonical_text.rfind(" ", search_start, search_end)
                if boundary != -1 and boundary > start:
                    proposed = boundary
            char_start, char_end, text = trim_to_range(canonical_text, start, proposed)
            if text:
                chunk_entities = [entity for entity in entities if char_start <= (entity.get("char_start") or -1) < char_end]
                chunks.append(
                    {
                        "chunk_id": stable_id("CH", doc_id, page["page_number"], char_start, char_end, length=14),
                        "doc_id": doc_id,
                        "page_number": page["page_number"],
                        "char_start": char_start,
                        "char_end": char_end,
                        "text": text,
                        "token_estimate": estimate_tokens(text),
                        "chunk_type": "page_segment",
                        "top_terms": top_terms([text], limit=8),
                        "entities": chunk_entities,
                        "topic_scores": [],
                    }
                )
            if proposed >= page_end:
                break
            start = max(proposed, start + 1)
    return chunks


def _add_duplicate_relationships(documents: list[dict[str, Any]]) -> None:
    by_file_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_text_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        if doc["metadata"].get("size_bytes", 0) > 0:
            by_file_hash[doc["metadata"].get("sha256", "")].append(doc)
        canonical_text = doc.get("canonical_text", "")
        if canonical_text.strip():
            by_text_hash[stable_id("TXT", canonical_text, length=16)].append(doc)
    for groups, relation in ((by_file_hash, "exact_duplicate"), (by_text_hash, "near_duplicate")):
        for group in groups.values():
            if len(group) < 2:
                continue
            for doc in group:
                for target in group:
                    if target["doc_id"] != doc["doc_id"]:
                        doc["relationships"].append({"relation": relation, "target_doc_id": target["doc_id"], "confidence": 1.0})


def _build_document_vectors(documents: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    doc_counts: dict[str, Counter[str]] = {}
    df: Counter[str] = Counter()
    for doc in documents:
        counts = term_counts(" ".join([doc.get("title") or "", doc["original_filename"], _topic_source_text(doc)]))
        doc_counts[doc["doc_id"]] = counts
        df.update(counts.keys())
    total_docs = len(documents) or 1
    idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1 for term, freq in df.items()}
    vectors = {doc_id: normalized_vector(counts, idf) for doc_id, counts in doc_counts.items()}
    return vectors, idf


def _build_topics(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    vectors: dict[str, dict[str, float]],
    idf: dict[str, float],
    corpus_id: str,
    corpus_version: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)
    entities_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entities_by_doc[entity["doc_id"]].append(entity)

    topics: list[dict[str, Any]] = []

    def create_topic(doc_ids: list[str], parent_id: str | None, depth: int, ordinal: int | None) -> dict[str, Any]:
        if parent_id is None:
            topic_id = f"T.{corpus_version}"
        else:
            topic_id = f"{parent_id}.{ordinal}"
        topic_docs = [docs_by_id[doc_id] for doc_id in doc_ids]
        centroid = average_vector(vectors[doc_id] for doc_id in doc_ids)
        label_terms = _topic_label_terms(topic_docs, idf)
        topic_entities = [entity for doc_id in doc_ids for entity in entities_by_doc.get(doc_id, [])]
        topic = {
            "corpus_id": corpus_id,
            "corpus_version": corpus_version,
            "topic_id": topic_id,
            "parent_topic_id": parent_id,
            "child_topic_ids": [],
            "depth": depth,
            "doc_ids": doc_ids,
            "doc_count": len(doc_ids),
            "page_count": sum(len(doc.get("pages", [])) for doc in topic_docs),
            "chunk_count": sum(len(chunks_by_doc.get(doc_id, [])) for doc_id in doc_ids),
            "label_terms": label_terms[:8] or ["low text signal"],
            "label_phrases": [term for term in label_terms if " " in term][:5],
            "top_entities": _entity_summary(topic_entities),
            "dominant_doc_types": _count_summary(doc["doc_type"] for doc in topic_docs),
            "dominant_sources": _count_summary(doc.get("source_batch", "default") for doc in topic_docs),
            "contrast_terms": [],
            "representative_docs": [],
            "representative_snippets": [],
            "neighboring_topics": [],
            "scope": {},
            "quality": _topic_quality(topic_docs, vectors, centroid),
            "actions": [],
            "_centroid": centroid,
        }
        topics.append(topic)

        clustering = config["clustering"]
        if depth >= clustering["max_depth"] or len(doc_ids) < clustering["min_docs_to_split"]:
            if len(doc_ids) >= clustering["min_docs_to_split"] and topic["quality"]["coherence"] < 0.12:
                topic["quality"]["warnings"].append("low-coherence topic; left unsplit")
            return topic

        k = min(clustering["max_branch_factor"], max(clustering["min_branch_factor"], round(math.sqrt(len(doc_ids)))))
        clusters = _kmeans_doc_ids(doc_ids, vectors, k=k)
        clusters = [cluster for cluster in clusters if len(cluster) >= clustering["min_docs_per_topic"]]
        if len(clusters) < 2:
            topic["quality"]["warnings"].append("topic did not split cleanly")
            return topic

        for child_ordinal, cluster in enumerate(clusters, start=1):
            child = create_topic(cluster, topic_id, depth + 1, child_ordinal)
            topic["child_topic_ids"].append(child["topic_id"])
        _add_sibling_contrast(topic, topics)
        return topic

    create_topic([doc["doc_id"] for doc in documents], None, 0, None)
    topic_by_id = {topic["topic_id"]: topic for topic in topics}
    for topic in topics:
        topic["neighboring_topics"] = _neighboring_topics(topic, topics)
        scope_id = "S-root" if topic["depth"] == 0 else stable_id("S", topic["topic_id"], "topic_subtree", length=8)
        topic["scope"] = {
            "scope_id": scope_id,
            "kind": "corpus" if topic["depth"] == 0 else "topic_subtree",
            "corpus_id": corpus_id,
            "corpus_version": corpus_version,
            "topic_ids": [topic["topic_id"]],
            "include_descendants": True,
            "estimated_doc_count": topic["doc_count"],
            "estimated_page_count": topic["page_count"],
        }
        topic["actions"] = _topic_actions(topic)
        topic.pop("_centroid", None)
    return sorted(topics, key=lambda item: (_topic_sort_key(item["topic_id"]), item["topic_id"]))


def _attach_topic_memberships(
    documents: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    vectors: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> None:
    topic_centroids = {topic["topic_id"]: average_vector(vectors[doc_id] for doc_id in topic.get("doc_ids", [])) for topic in topics}
    non_root_topics = [topic for topic in topics if topic["depth"] > 0] or topics
    max_memberships = config["clustering"]["max_memberships_per_doc"]
    threshold = config["clustering"]["secondary_membership_threshold"]
    for doc in documents:
        scored = []
        vector = vectors.get(doc["doc_id"], {})
        for topic in non_root_topics:
            if doc["doc_id"] in set(topic.get("doc_ids", [])):
                score = max(0.35, cosine(vector, topic_centroids[topic["topic_id"]]))
            else:
                score = cosine(vector, topic_centroids[topic["topic_id"]])
            if score >= threshold or doc["doc_id"] in topic.get("doc_ids", []):
                scored.append((topic["topic_id"], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        doc["topic_memberships"] = [
            {"topic_id": topic_id, "score": round(score, 4), "rank": rank, "is_primary": rank == 1}
            for rank, (topic_id, score) in enumerate(scored[:max_memberships], start=1)
        ]


def _attach_topic_representatives(
    topics: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    vectors: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> None:
    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entities_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)
    for entity in entities:
        entities_by_doc[entity["doc_id"]].append(entity)

    for topic in topics:
        centroid = average_vector(vectors[doc_id] for doc_id in topic.get("doc_ids", []))
        doc_scores = [
            (docs_by_id[doc_id], cosine(vectors.get(doc_id, {}), centroid))
            for doc_id in topic.get("doc_ids", [])
            if doc_id in docs_by_id
        ]
        doc_scores.sort(key=lambda item: item[1], reverse=True)
        topic["representative_docs"] = _representative_docs(topic, doc_scores, config)
        topic["representative_snippets"] = _representative_snippets(
            topic,
            doc_scores,
            docs_by_id,
            chunks_by_doc,
            entities_by_doc,
            config,
        )
        for doc_id in topic.get("doc_ids", []):
            for chunk in chunks_by_doc.get(doc_id, []):
                score = cosine(vectors.get(doc_id, {}), centroid)
                chunk.setdefault("topic_scores", []).append(
                    {"topic_id": topic["topic_id"], "score": round(score, 4), "rank": 1, "is_primary": False}
                )


def _build_scopes(
    topics: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    corpus_id: str,
    corpus_version: str,
) -> list[dict[str, Any]]:
    scopes = [
        {
            "scope_id": "S-root",
            "kind": "corpus",
            "corpus_id": corpus_id,
            "corpus_version": corpus_version,
            "doc_ids": [doc["doc_id"] for doc in documents],
            "estimated_doc_count": len(documents),
            "estimated_page_count": sum(len(doc.get("pages", [])) for doc in documents),
        }
    ]
    for topic in topics:
        if topic["depth"] == 0:
            continue
        scopes.append(
            {
                "scope_id": stable_id("S", topic["topic_id"], "topic_subtree", length=8),
                "kind": "topic_subtree",
                "corpus_id": corpus_id,
                "corpus_version": corpus_version,
                "topic_ids": [topic["topic_id"]],
                "include_descendants": True,
                "doc_ids": topic["doc_ids"],
                "estimated_doc_count": topic["doc_count"],
                "estimated_page_count": topic["page_count"],
            }
        )
    return scopes


def _topic_label_terms(topic_docs: list[dict[str, Any]], idf: dict[str, float]) -> list[str]:
    counts: Counter[str] = Counter()
    for doc in topic_docs:
        counts.update(term_counts(" ".join([doc.get("title") or "", doc["original_filename"], _topic_source_text(doc)])))
    scored = [(term, count * idf.get(term, 1.0)) for term, count in counts.items() if len(term) >= 3]
    scored.sort(key=lambda item: item[1], reverse=True)
    terms: list[str] = []
    for term, _ in scored:
        if any(term in existing or existing in term for existing in terms):
            continue
        terms.append(term)
        if len(terms) >= 12:
            break
    return terms


def _topic_source_text(doc: dict[str, Any]) -> str:
    return doc.get("topic_text") or doc.get("canonical_text", "")


def _topic_quality(topic_docs: list[dict[str, Any]], vectors: dict[str, dict[str, float]], centroid: dict[str, float]) -> dict[str, Any]:
    similarities = [cosine(vectors.get(doc["doc_id"], {}), centroid) for doc in topic_docs]
    coherence = sum(similarities) / len(similarities) if similarities else 0.0
    source_counts = Counter(doc.get("source_batch", "default") for doc in topic_docs)
    source_concentration = max(source_counts.values()) / len(topic_docs) if topic_docs else 0.0
    duplicate_docs = sum(1 for doc in topic_docs if doc.get("relationships"))
    warnings: list[str] = []
    if coherence < 0.2 and len(topic_docs) > 1:
        warnings.append("low-coherence topic")
    if source_concentration > 0.85 and len(source_counts) > 1:
        warnings.append("representative documents are concentrated in one source batch")
    return {
        "coherence": round(coherence, 4),
        "assignment_confidence": round(coherence, 4),
        "sibling_overlap": 0.0,
        "multi_topic_doc_rate": 0.0,
        "duplicate_rate": round(_safe_ratio(duplicate_docs, len(topic_docs)), 4),
        "ocr_risk": "low",
        "source_concentration": round(source_concentration, 4),
        "warnings": warnings,
    }


def _kmeans_doc_ids(doc_ids: list[str], vectors: dict[str, dict[str, float]], *, k: int) -> list[list[str]]:
    doc_ids = sorted(doc_ids)
    if len(doc_ids) <= k:
        return [[doc_id] for doc_id in doc_ids]
    seeds = _initial_seeds(doc_ids, vectors, k)
    centroids = [dict(vectors[doc_id]) for doc_id in seeds]
    assignments: dict[str, int] = {}
    for _ in range(12):
        changed = False
        clusters = [[] for _ in centroids]
        for doc_id in doc_ids:
            similarities = [cosine(vectors.get(doc_id, {}), centroid) for centroid in centroids]
            cluster_index = max(range(len(centroids)), key=lambda idx: (similarities[idx], -idx))
            clusters[cluster_index].append(doc_id)
            if assignments.get(doc_id) != cluster_index:
                changed = True
                assignments[doc_id] = cluster_index
        for index, cluster in enumerate(clusters):
            if cluster:
                centroids[index] = average_vector(vectors[doc_id] for doc_id in cluster)
        if not changed:
            break
    clusters = [[] for _ in centroids]
    for doc_id, cluster_index in assignments.items():
        clusters[cluster_index].append(doc_id)
    return [sorted(cluster) for cluster in clusters if cluster]


def _initial_seeds(doc_ids: list[str], vectors: dict[str, dict[str, float]], k: int) -> list[str]:
    seeds = [max(doc_ids, key=lambda doc_id: (len(vectors.get(doc_id, {})), doc_id))]
    while len(seeds) < k:
        candidates = [doc_id for doc_id in doc_ids if doc_id not in seeds]
        if not candidates:
            break
        next_seed = min(
            candidates,
            key=lambda doc_id: (
                max(cosine(vectors.get(doc_id, {}), vectors.get(seed, {})) for seed in seeds),
                doc_id,
            ),
        )
        seeds.append(next_seed)
    return seeds


def _representative_docs(topic: dict[str, Any], doc_scores: list[tuple[dict[str, Any], float]], config: dict[str, Any]) -> list[dict[str, Any]]:
    limit = config["topic_cards"]["representative_docs"]
    selected: list[tuple[dict[str, Any], float, list[str]]] = []
    seen: set[str] = set()

    def add(doc: dict[str, Any], score: float, reason: str) -> None:
        if doc["doc_id"] in seen or len(selected) >= limit:
            return
        selected.append((doc, score, [reason]))
        seen.add(doc["doc_id"])

    for doc, score in doc_scores[:3]:
        add(doc, score, "central_to_topic")
    seen_types: set[str] = set()
    seen_sources: set[str] = set()
    for doc, score in doc_scores:
        if doc["doc_type"] not in seen_types or doc.get("source_batch") not in seen_sources:
            add(doc, score, "diverse_example")
            seen_types.add(doc["doc_type"])
            seen_sources.add(doc.get("source_batch"))
    for doc, score in reversed(doc_scores):
        add(doc, score, "outlier_within_topic")
    return [
        {
            "doc_id": doc["doc_id"],
            "title": doc.get("title"),
            "doc_type": doc["doc_type"],
            "source_batch": doc.get("source_batch"),
            "topic_membership_score": round(score, 4),
            "selection_reason": reasons,
            "open_action": {
                "action": "doc_card",
                "args": {"doc_id": doc["doc_id"]},
                "estimated_cost": "low",
                "description": "Open a document triage card.",
            },
        }
        for doc, score, reasons in selected
    ]


def _representative_snippets(
    topic: dict[str, Any],
    doc_scores: list[tuple[dict[str, Any], float]],
    docs_by_id: dict[str, dict[str, Any]],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    entities_by_doc: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    limit = config["topic_cards"]["representative_snippets"]
    target = config["topic_cards"]["snippet_target_chars"]
    max_chars = config["topic_cards"]["snippet_max_chars"]
    max_per_doc = config["topic_cards"]["max_snippets_from_same_doc"]
    label_terms = topic.get("label_terms", [])
    selected: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    candidates: list[tuple[float, dict[str, Any], dict[str, Any], list[str]]] = []
    doc_score_map = {doc["doc_id"]: score for doc, score in doc_scores}
    for doc, doc_score in doc_scores:
        for chunk in chunks_by_doc.get(doc["doc_id"], []):
            matched = _matched_terms(label_terms, chunk["text"])
            if not matched and len(selected) >= limit:
                continue
            entity_density = len(chunk.get("entities", [])) / max(1, chunk["token_estimate"])
            score = 0.50 * doc_score + 0.35 * min(1.0, len(matched) / 4) + 0.15 * min(1.0, entity_density * 20)
            candidates.append((score, doc, chunk, matched))
    candidates.sort(key=lambda item: item[0], reverse=True)
    seen_texts: set[str] = set()
    for score, doc, chunk, matched in candidates:
        if len(selected) >= limit:
            break
        if per_doc[doc["doc_id"]] >= max_per_doc:
            continue
        full_text = doc["canonical_text"]
        hit_start, hit_end = _first_hit_range(full_text, matched, chunk["char_start"], chunk["char_end"])
        start, end, text = make_excerpt_range(full_text, hit_start, hit_end, target, max_chars)
        fingerprint = " ".join(tokenize(text)[:30])
        if not text or fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        per_doc[doc["doc_id"]] += 1
        snippet_entities = [
            entity for entity in entities_by_doc.get(doc["doc_id"], []) if start <= (entity.get("char_start") or -1) < end
        ]
        snippet_id = f"SNIP-{topic['topic_id']}-{doc['doc_id']}-p{chunk['page_number']:04d}-{start:07d}"
        selected.append(
            {
                "snippet_id": snippet_id,
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "doc_type": doc["doc_type"],
                "source_batch": doc.get("source_batch"),
                "page_number": chunk.get("page_number"),
                "char_start": start,
                "char_end": end,
                "text": text,
                "selection_reason": ["contains_representative_terms", "central_to_topic" if doc_score_map.get(doc["doc_id"], 0) >= 0.5 else "diverse_example"],
                "matched_terms": matched[:8],
                "entities": snippet_entities[:8],
                "snippet_warning": "Snippet is selected as representative of a topic signal. Read the evidence window before making claims.",
                "read_action": {
                    "action": "read_window",
                    "args": {"snippet_id": snippet_id},
                    "estimated_cost": "medium",
                    "estimated_token_range": [500, 1800],
                    "description": "Read the bounded evidence window for this snippet.",
                },
            }
        )
    return selected


def _topic_actions(topic: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {
            "action": "topic_card",
            "args": {"topic_id": topic["topic_id"]},
            "estimated_cost": "low",
            "description": "Open a compact topic card.",
        },
        {
            "action": "search_within",
            "args": {"scope_id": topic["scope"].get("scope_id"), "query": ""},
            "estimated_cost": "low",
            "estimated_token_range": [800, 2200],
            "description": "Search within this topic subtree.",
        },
    ]
    if topic.get("child_topic_ids"):
        actions.insert(
            0,
            {
                "action": "expand_topic",
                "args": {"topic_id": topic["topic_id"]},
                "estimated_cost": "low",
                "description": "Show child topics under this branch.",
            },
        )
    return actions


def _add_sibling_contrast(parent: dict[str, Any], topics: list[dict[str, Any]]) -> None:
    topic_by_id = {topic["topic_id"]: topic for topic in topics}
    children = [topic_by_id[child_id] for child_id in parent.get("child_topic_ids", []) if child_id in topic_by_id]
    for child in children:
        sibling_terms = set()
        for sibling in children:
            if sibling["topic_id"] != child["topic_id"]:
                sibling_terms.update(sibling.get("label_terms", []))
        contrast = [term for term in child.get("label_terms", []) if term not in sibling_terms]
        child["contrast_terms"] = [
            {
                "term": term,
                "distinguishes_from_topic_ids": [sibling["topic_id"] for sibling in children if sibling["topic_id"] != child["topic_id"]],
                "score": 0.75,
            }
            for term in contrast[:5]
        ]


def _neighboring_topics(topic: dict[str, Any], topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if topic["depth"] == 0:
        return []
    centroid = average_vector({term: 1.0} for term in topic.get("label_terms", []))
    neighbors: list[tuple[float, dict[str, Any]]] = []
    for other in topics:
        if other["topic_id"] == topic["topic_id"] or other["depth"] != topic["depth"]:
            continue
        other_centroid = average_vector({term: 1.0} for term in other.get("label_terms", []))
        score = cosine(centroid, other_centroid)
        if score > 0:
            neighbors.append((score, other))
    neighbors.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "topic_id": other["topic_id"],
            "label_terms": other.get("label_terms", [])[:5],
            "score": round(score, 4),
            "action": {
                "action": "topic_card",
                "args": {"topic_id": other["topic_id"]},
                "estimated_cost": "low",
                "description": "Open neighboring topic card.",
            },
        }
        for score, other in neighbors[:5]
    ]


def _bm25_scores(records: list[dict[str, Any]], query_terms: list[str], text_key: str) -> list[tuple[dict[str, Any], float]]:
    if not records or not query_terms:
        return []
    query_terms = list(dict.fromkeys(query_terms))
    tokenized = [tokenize(record.get(text_key, "")) for record in records]
    avgdl = sum(len(tokens) for tokens in tokenized) / len(tokenized) if tokenized else 1
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    scores: list[tuple[dict[str, Any], float]] = []
    k1 = 1.5
    b = 0.75
    total = len(records)
    for record, tokens in zip(records, tokenized):
        counts = Counter(tokens)
        dl = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append((record, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _matched_terms(terms: list[str], text: str) -> list[str]:
    lowered = text.lower()
    matches = []
    for term in terms:
        if term and term.lower() in lowered:
            matches.append(term)
    return list(dict.fromkeys(matches))


def _first_hit_range(text: str, terms: list[str], start: int, end: int) -> tuple[int, int]:
    lowered = text.lower()
    for term in terms:
        position = lowered.find(term.lower(), start, end)
        if position != -1:
            return position, position + len(term)
    return start, min(end, start + 1)


def _apply_budget(response: dict[str, Any], token_budget: int) -> None:
    while estimate_tokens(json.dumps(response, ensure_ascii=False, sort_keys=True)) > token_budget:
        if not _trim_once(response.get("result")):
            if response.get("actions"):
                response["actions"].pop()
            elif response["cost_ledger"].get("cheaper_next_actions"):
                response["cost_ledger"]["cheaper_next_actions"].pop()
            else:
                break
        response["truncated"] = True
        if "Result truncated to token budget." not in response["warnings"]:
            response["warnings"].append("Result truncated to token budget.")


def _trim_once(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key in ("global_facets", "facets"):
            if key in obj:
                del obj[key]
                return True
        root_scope = obj.get("root_scope")
        if isinstance(root_scope, dict) and root_scope.get("doc_ids"):
            del root_scope["doc_ids"]
            return True
        for key in (
            "representative_snippets",
            "representative_docs",
            "best_passages",
            "page_map",
            "suggested_refinements",
            "neighboring_topics",
            "top_entities",
            "contrast_terms",
            "actions",
        ):
            value = obj.get(key)
            if isinstance(value, list) and value:
                value.pop()
                return True
        for key in ("text", "snippet"):
            value = obj.get(key)
            if isinstance(value, str) and len(value) > 500:
                obj[key] = value[:500]
                return True
        for value in obj.values():
            if _trim_once(value):
                return True
        for key in ("results", "children", "top_topics"):
            value = obj.get(key)
            if isinstance(value, list) and value:
                value.pop()
                return True
    elif isinstance(obj, list):
        if obj:
            last = obj[-1]
            if _trim_once(last):
                return True
            obj.pop()
            return True
    return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_ratio(left: int | float, right: int | float) -> float:
    return float(left) / float(right) if right else 0.0


def _count_summary(values: Iterable[Any], limit: int = 10) -> list[dict[str, Any]]:
    counts = Counter(str(value) for value in values if value is not None and str(value) != "")
    return [{"value": value, "count": count} for value, count in counts.most_common(limit)]


def _entity_summary(entities: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter((entity["text"], entity["type"]) for entity in entities)
    return [
        {"text": text, "type": entity_type, "count": count}
        for (text, entity_type), count in counts.most_common(limit)
    ]


def _dedupe_entities(entities: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in entities:
        key = (entity["normalized_text"], entity["type"], entity["text"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(entity)
        if len(selected) >= limit:
            break
    return selected


def _lower_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).lower() for value in values]


def _page_for_range(doc: dict[str, Any], char_start: int, char_end: int | None = None) -> int | None:
    char_end = char_start if char_end is None else char_end
    for page in doc.get("pages", []):
        if page["char_start"] <= char_start <= page["char_end"] or page["char_start"] <= char_end <= page["char_end"]:
            return page["page_number"]
    return None


def _looks_tabular(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    tabular = sum(1 for line in lines if "\t" in line or line.count("|") >= 2 or line.count(",") >= 4)
    return tabular >= 2


def _title_from_text(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip()
        if 4 <= len(clean) <= 120:
            return clean
    return None


def _page_signals(page: dict[str, Any], entities: list[dict[str, Any]]) -> list[str]:
    signals: list[str] = []
    text = page["text"].lower()
    if page.get("has_table"):
        signals.append("possible table")
    if any(entity["type"] == "money" for entity in entities):
        signals.append("money amounts")
    if any(entity["type"] == "email" for entity in entities):
        signals.append("email addresses")
    if "subject:" in text and "from:" in text:
        signals.append("email header")
    if page.get("top_terms"):
        signals.append("top terms: " + " / ".join(page["top_terms"][:3]))
    return signals[:5]


def _topic_sort_key(topic_id: str) -> tuple[int, ...]:
    parts = []
    for part in topic_id.split("."):
        if part.startswith("v") and part[1:].isdigit():
            parts.append(int(part[1:]))
        elif part.isdigit():
            parts.append(int(part))
    return tuple(parts)
