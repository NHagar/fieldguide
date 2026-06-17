"""LanceDB-backed hybrid search index for Fieldguide chunks."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

import lancedb
import pyarrow as pa

from .api import (
    CANONICAL_TEXT_VERSION,
    DEFAULT_CONFIG,
    INDEX_FILENAME,
    _deep_merge,
    _document_from_file,
    _iter_source_files,
    _matched_terms,
)
from .text import estimate_tokens, make_excerpt_range, tokenize


HYBRID_METADATA_FILENAME = "metadata.json"
HYBRID_TABLE_NAME = "chunks"
HYBRID_INDEX_FORMAT = "fieldguide-hybrid-lancedb-v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SNIPPET_CHARS = 650
MAX_SNIPPET_CHARS = 2000
MAX_RESULTS_HARD = 50
ENTITY_SEARCH_OVERSAMPLE = 8
VECTOR_INDEX_MIN_ROWS = 4096
VECTOR_DISTANCE_METRIC = "cosine"


class Embedder(Protocol):
    """Small protocol for local embedding providers."""

    model_name: str
    dimension: int

    def embed(self, texts: list[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Return one normalized vector per input text."""


@dataclass
class HybridBuildStats:
    corpus_id: str
    corpus_version: str
    doc_count: int
    chunk_count: int
    vector_dimension: int
    index_path: Path
    metadata_path: Path
    warnings: list[str]


class SentenceTransformerEmbedder:
    """Lazy local SentenceTransformer embedder."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, trust_remote_code=False)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                dimension = len(self.embed(["dimension probe"])[0])
            self._dimension = int(dimension)
        return self._dimension

    def embed(self, texts: list[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_float_vector(vector) for vector in vectors]


def build_hybrid_from_source(
    source_dir: str | Path,
    index_dir: str | Path,
    *,
    corpus_id: str = "C-FIELDGUIDE",
    corpus_version: str = "v1",
    pdf_backend: str = "docling",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Embedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    vector_index_min_rows: int = VECTOR_INDEX_MIN_ROWS,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> HybridBuildStats:
    """Build a LanceDB hybrid search index directly from a source directory."""

    source_root = Path(source_dir).expanduser().resolve()
    output_dir = Path(index_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_config = _deep_merge(DEFAULT_CONFIG, {"extraction": {"pdf_backend": pdf_backend}})

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    source_files = list(_iter_source_files(source_root, output_dir))
    started_at = time.monotonic()
    if progress:
        progress(
            {
                "event": "hybrid_build_start",
                "source_mode": "source",
                "total_files": len(source_files),
                "pdf_backend": pdf_backend,
            }
        )

    for file_index, path in enumerate(source_files, start=1):
        if progress:
            progress(
                {
                    "event": "document_start",
                    "index": file_index,
                    "total_files": len(source_files),
                    "source_uri": str(path.relative_to(source_root)),
                    "size_bytes": path.stat().st_size,
                }
            )
        doc_started_at = time.monotonic()
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
        if progress:
            progress(
                {
                    "event": "document_done",
                    "index": file_index,
                    "total_files": len(source_files),
                    "source_uri": document["source_uri"],
                    "doc_id": document["doc_id"],
                    "chunk_count": len(doc_chunks),
                    "page_count": document["page_count"],
                    "extraction_method": document["extraction"]["extraction_method"],
                    "duration_seconds": time.monotonic() - doc_started_at,
                }
            )

    return _write_hybrid_index(
        output_dir,
        documents=documents,
        chunks=chunks,
        entities=entities,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        source_mode="source",
        source_metadata={"pdf_backend": pdf_backend},
        embedding_model=embedding_model,
        embedder=embedder,
        batch_size=batch_size,
        vector_index_min_rows=vector_index_min_rows,
        started_at=started_at,
        progress=progress,
    )


def build_hybrid_from_json(
    json_index: str | Path,
    index_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Embedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    vector_index_min_rows: int = VECTOR_INDEX_MIN_ROWS,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> HybridBuildStats:
    """Build a LanceDB hybrid search index from an existing Fieldguide JSON index."""

    input_path = Path(json_index).expanduser().resolve()
    if input_path.is_dir():
        input_path = input_path / INDEX_FILENAME
    data = json.loads(input_path.read_text(encoding="utf-8"))
    documents = list(data.get("documents", []))
    chunks = list(data.get("chunks", []))
    entities = list(data.get("entities", []))
    corpus_id = data.get("corpus_id", "C-FIELDGUIDE")
    corpus_version = data.get("corpus_version", "v1")

    warnings: list[str] = []
    non_docling_pdfs = [
        doc
        for doc in documents
        if doc.get("doc_type") == "pdf"
        and doc.get("metadata", {}).get("pdf_backend") != "docling"
        and doc.get("extraction", {}).get("extraction_method") != "docling"
    ]
    if non_docling_pdfs:
        warnings.append(
            "JSON import preserves existing extraction; "
            f"{len(non_docling_pdfs)} PDF document(s) were not docling-processed. "
            "Rebuild from source with --pdf-backend docling to guarantee docling preprocessing."
        )
    if progress:
        progress(
            {
                "event": "hybrid_build_start",
                "source_mode": "json",
                "doc_count": len(documents),
                "chunk_count": len(chunks),
            }
        )

    return _write_hybrid_index(
        Path(index_dir).expanduser().resolve(),
        documents=documents,
        chunks=chunks,
        entities=entities,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        source_mode="json",
        source_metadata={"json_canonical_text_version": data.get("canonical_text_version")},
        embedding_model=embedding_model,
        embedder=embedder,
        batch_size=batch_size,
        vector_index_min_rows=vector_index_min_rows,
        build_warnings=warnings,
        started_at=time.monotonic(),
        progress=progress,
    )


class HybridFieldguideIndex:
    """Search API over a built LanceDB hybrid chunk index."""

    def __init__(self, index_dir: str | Path, *, embedder: Embedder | None = None):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.metadata_path = self.index_dir / HYBRID_METADATA_FILENAME
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.corpus_id = self.metadata["corpus_id"]
        self.corpus_version = self.metadata["corpus_version"]
        embedding = self.metadata.get("embedding", {})
        model_name = embedding.get("model") or DEFAULT_EMBEDDING_MODEL
        self.embedder = embedder or SentenceTransformerEmbedder(model_name)
        self.db = lancedb.connect(str(self.index_dir))
        self.table = self.db.open_table(HYBRID_TABLE_NAME)

    def search(
        self,
        query: str,
        *,
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        max_results: int = 10,
        doc_types: list[str] | None = None,
        sources: list[str] | None = None,
        entities: list[str] | None = None,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> dict[str, Any]:
        """Run bounded chunk search against the hybrid index."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if mode not in {"hybrid", "vector", "keyword"}:
            raise ValueError("mode must be one of: hybrid, vector, keyword")

        max_results = min(max(0, max_results), MAX_RESULTS_HARD)
        snippet_chars = min(max(80, snippet_chars), MAX_SNIPPET_CHARS)
        normalized_entity_filters = set(_normalized_values(entities))
        fetch_limit = max_results
        if normalized_entity_filters and max_results:
            fetch_limit = min(MAX_RESULTS_HARD * ENTITY_SEARCH_OVERSAMPLE, max_results * ENTITY_SEARCH_OVERSAMPLE)
        fetch_limit = max(1, fetch_limit)

        where_clause = _where_clause(doc_types=doc_types, sources=sources)
        query_builder = self._query_builder(query, mode)
        if where_clause:
            query_builder = query_builder.where(where_clause)
        rows = query_builder.select(_search_columns()).limit(fetch_limit).to_list()

        formatted: list[dict[str, Any]] = []
        for row in rows:
            if normalized_entity_filters and not _row_matches_entities(row, normalized_entity_filters):
                continue
            formatted.append(
                _format_search_result(
                    row,
                    query=query,
                    mode=mode,
                    snippet_chars=snippet_chars,
                    entity_filters=normalized_entity_filters,
                )
            )
            if len(formatted) >= max_results:
                break
        for rank, result in enumerate(formatted, start=1):
            result["rank"] = rank

        warnings = list(self.metadata.get("warnings", []))
        if normalized_entity_filters and len(formatted) < max_results and len(rows) == fetch_limit:
            warnings.append("Entity filters are applied after LanceDB candidate retrieval; broaden the query if results look thin.")

        return {
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "tool_name": "fieldguide_hybrid_search",
            "response_kind": "search_results",
            "result": {
                "query": query,
                "mode": mode,
                "max_results": max_results,
                "result_count": len(formatted),
                "filters": {
                    key: value
                    for key, value in {
                        "doc_types": doc_types,
                        "sources": sources,
                        "entities": entities,
                    }.items()
                    if value
                },
                "results": formatted,
            },
            "warnings": warnings,
            "token_estimate": estimate_tokens(json.dumps(formatted, ensure_ascii=False, sort_keys=True)),
        "truncated": max_results > 0 and len(formatted) == max_results and len(rows) > len(formatted),
            "index_summary": {
                "format": self.metadata.get("format"),
                "embedding_model": self.metadata.get("embedding", {}).get("model"),
                "chunk_count": self.metadata.get("chunk_count"),
                "doc_count": self.metadata.get("doc_count"),
                "source_mode": self.metadata.get("source", {}).get("mode"),
            },
        }

    def _query_builder(self, query: str, mode: str) -> Any:
        if mode == "keyword":
            return self.table.search(query, query_type="fts", fts_columns="text")

        vector = self.embedder.embed([query], batch_size=1)[0]
        if mode == "vector":
            return self.table.search(vector, query_type="vector", vector_column_name="vector").distance_type(VECTOR_DISTANCE_METRIC)
        return (
            self.table.search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
            .vector(vector)
            .text(query)
            .distance_type(VECTOR_DISTANCE_METRIC)
        )


def _write_hybrid_index(
    output_dir: Path,
    *,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    corpus_id: str,
    corpus_version: str,
    source_mode: str,
    source_metadata: dict[str, Any],
    embedding_model: str,
    embedder: Embedder | None,
    batch_size: int,
    vector_index_min_rows: int,
    started_at: float,
    progress: Callable[[dict[str, Any]], None] | None,
    build_warnings: list[str] | None = None,
) -> HybridBuildStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or SentenceTransformerEmbedder(embedding_model)
    dimension = int(embedder.dimension)
    rows = _rows_for_chunks(documents, chunks, entities)

    if progress:
        progress({"event": "embedding_start", "chunk_count": len(rows), "embedding_model": embedder.model_name})
    texts = [row["text"] for row in rows]
    vectors = embedder.embed(texts, batch_size=batch_size)
    if len(vectors) != len(rows):
        raise RuntimeError("embedder returned a different number of vectors than input texts")
    for row, vector in zip(rows, vectors):
        if len(vector) != dimension:
            raise RuntimeError(f"embedder returned vector dimension {len(vector)}; expected {dimension}")
        row["vector"] = vector

    db = lancedb.connect(str(output_dir))
    table = db.create_table(HYBRID_TABLE_NAME, data=rows or None, schema=_chunk_schema(dimension), mode="overwrite")
    table.create_fts_index("text", replace=True)

    warnings = list(build_warnings or [])
    vector_index_created = False
    if len(rows) >= vector_index_min_rows:
        try:
            table.create_index(
                metric=VECTOR_DISTANCE_METRIC,
                vector_column_name="vector",
                index_type="IVF_PQ",
                replace=True,
            )
            vector_index_created = True
        except Exception as exc:  # noqa: BLE001 - exact vector scan still works without ANN.
            warnings.append(f"Vector ANN index was not created; exact vector scan remains available: {exc}")

    extraction_warnings = _extraction_warnings(documents)
    metadata = {
        "format": HYBRID_INDEX_FORMAT,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "canonical_text_version": CANONICAL_TEXT_VERSION,
        "source": {"mode": source_mode, **source_metadata},
        "embedding": {
            "model": embedder.model_name,
            "dimension": dimension,
            "normalize_embeddings": True,
            "trust_remote_code": False,
            "distance_metric": VECTOR_DISTANCE_METRIC,
        },
        "doc_count": len(documents),
        "chunk_count": len(rows),
        "page_count": sum(len(doc.get("pages", [])) for doc in documents),
        "source_doc_type_counts": _count_summary(doc.get("doc_type") for doc in documents),
        "source_batch_counts": _count_summary(doc.get("source_batch") for doc in documents),
        "table": HYBRID_TABLE_NAME,
        "fts_index": {"column": "text", "created": True},
        "vector_index": {
            "created": vector_index_created,
            "min_rows": vector_index_min_rows,
            "metric": VECTOR_DISTANCE_METRIC,
        },
        "extraction_warnings": extraction_warnings,
        "warnings": warnings,
        "build_seconds": round(time.monotonic() - started_at, 4),
    }
    metadata_path = output_dir / HYBRID_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if progress:
        progress(
            {
                "event": "hybrid_build_done",
                "doc_count": len(documents),
                "chunk_count": len(rows),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    return HybridBuildStats(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        doc_count=len(documents),
        chunk_count=len(rows),
        vector_dimension=dimension,
        index_path=output_dir,
        metadata_path=metadata_path,
        warnings=warnings,
    )


def _rows_for_chunks(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    entities_by_doc: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        entities_by_doc.setdefault(entity["doc_id"], []).append(entity)

    rows: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda item: item.get("chunk_id", "")):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        doc = docs_by_id.get(chunk.get("doc_id"))
        if not doc:
            continue
        chunk_entities = chunk.get("entities") or []
        if not chunk_entities:
            chunk_start = chunk.get("char_start", 0)
            chunk_end = chunk.get("char_end", 0)
            chunk_entities = [
                entity
                for entity in entities_by_doc.get(doc["doc_id"], [])
                if chunk_start <= _entity_char_start(entity) < chunk_end
            ]
        entity_keys = sorted({str(entity.get("normalized_text") or "").lower() for entity in chunk_entities if entity.get("normalized_text")})
        extraction = doc.get("extraction", {})
        rows.append(
            {
                "corpus_id": doc.get("corpus_id"),
                "corpus_version": doc.get("corpus_version"),
                "chunk_id": chunk["chunk_id"],
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "source_uri": doc.get("source_uri"),
                "source_batch": doc.get("source_batch"),
                "original_filename": doc.get("original_filename"),
                "doc_type": doc.get("doc_type"),
                "page_number": chunk.get("page_number"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "text": text,
                "token_estimate": chunk.get("token_estimate") or estimate_tokens(text),
                "extraction_method": extraction.get("extraction_method"),
                "extraction_quality_json": _json_dumps(extraction),
                "extraction_warnings_json": _json_dumps(extraction.get("warnings", [])),
                "entities_json": _json_dumps(chunk_entities),
                "entity_keys": "|" + "|".join(entity_keys) + "|" if entity_keys else "",
            }
        )
    return rows


def _chunk_schema(dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("corpus_id", pa.string()),
            pa.field("corpus_version", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("source_uri", pa.string()),
            pa.field("source_batch", pa.string()),
            pa.field("original_filename", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("page_number", pa.int64()),
            pa.field("char_start", pa.int64()),
            pa.field("char_end", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("token_estimate", pa.int64()),
            pa.field("extraction_method", pa.string()),
            pa.field("extraction_quality_json", pa.string()),
            pa.field("extraction_warnings_json", pa.string()),
            pa.field("entities_json", pa.string()),
            pa.field("entity_keys", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ]
    )


def _search_columns() -> list[str]:
    return [
        "chunk_id",
        "doc_id",
        "title",
        "source_uri",
        "source_batch",
        "original_filename",
        "doc_type",
        "page_number",
        "char_start",
        "char_end",
        "text",
        "token_estimate",
        "extraction_method",
        "extraction_quality_json",
        "extraction_warnings_json",
        "entities_json",
        "entity_keys",
    ]


def _format_search_result(
    row: dict[str, Any],
    *,
    query: str,
    mode: str,
    snippet_chars: int,
    entity_filters: set[str],
) -> dict[str, Any]:
    query_terms = tokenize(query, keep_stopwords=False)
    matched_terms = _matched_terms(query_terms, row.get("text") or "")
    snippet_start, snippet_end, snippet = _snippet(row, matched_terms, snippet_chars)
    entities = _json_loads(row.get("entities_json"), default=[])
    matched_entities = _matched_entities(entities, query, entity_filters)
    scores = {
        key: round(float(row[key]), 6)
        for key in ("_relevance_score", "_score", "_distance")
        if key in row and isinstance(row[key], (int, float))
    }
    if "_relevance_score" in scores:
        score = scores["_relevance_score"]
    elif "_score" in scores:
        score = scores["_score"]
    elif "_distance" in scores:
        score = round(1.0 / (1.0 + max(0.0, scores["_distance"])), 6)
    else:
        score = 0.0
    extraction_quality = _json_loads(row.get("extraction_quality_json"), default={})
    return {
        "rank": 0,
        "score": score,
        "scores": scores,
        "mode": mode,
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "source_metadata": {
            "source_uri": row.get("source_uri"),
            "source_batch": row.get("source_batch"),
            "original_filename": row.get("original_filename"),
        },
        "doc_type": row.get("doc_type"),
        "page_number": row.get("page_number"),
        "char_start": snippet_start,
        "char_end": snippet_end,
        "chunk_char_start": row.get("char_start"),
        "chunk_char_end": row.get("char_end"),
        "snippet": snippet,
        "matched_terms": matched_terms,
        "matched_entities": matched_entities,
        "extraction_quality": extraction_quality,
        "extraction_warnings": _json_loads(row.get("extraction_warnings_json"), default=[]),
    }


def _snippet(row: dict[str, Any], matched_terms: list[str], snippet_chars: int) -> tuple[int | None, int | None, str]:
    text = row.get("text") or ""
    if not text:
        return row.get("char_start"), row.get("char_start"), ""
    local_start = 0
    local_end = min(1, len(text))
    lowered = text.lower()
    for term in matched_terms:
        position = lowered.find(term.lower())
        if position != -1:
            local_start = position
            local_end = position + len(term)
            break
    excerpt_start, excerpt_end, snippet = make_excerpt_range(text, local_start, local_end, snippet_chars, snippet_chars)
    chunk_start = row.get("char_start")
    if isinstance(chunk_start, int):
        return chunk_start + excerpt_start, chunk_start + excerpt_end, snippet
    return None, None, snippet


def _matched_entities(entities: list[dict[str, Any]], query: str, entity_filters: set[str]) -> list[dict[str, Any]]:
    lowered = query.lower()
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        normalized = str(entity.get("normalized_text") or "").lower()
        text = str(entity.get("text") or "")
        if not normalized or (normalized not in entity_filters and normalized not in lowered and text.lower() not in lowered):
            continue
        key = (normalized, str(entity.get("type")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "text": entity.get("text"),
                "type": entity.get("type"),
                "page_number": entity.get("page_number"),
                "char_start": entity.get("char_start"),
                "char_end": entity.get("char_end"),
            }
        )
        if len(selected) >= 8:
            break
    return selected


def _where_clause(*, doc_types: list[str] | None, sources: list[str] | None) -> str | None:
    clauses: list[str] = []
    if doc_types:
        clauses.append(_in_clause("doc_type", doc_types))
    if sources:
        clauses.append(_in_clause("source_batch", sources))
    return " AND ".join(clauses) if clauses else None


def _in_clause(field: str, values: list[str]) -> str:
    unique_values = list(dict.fromkeys(str(value) for value in values if str(value)))
    if not unique_values:
        return "true"
    if len(unique_values) == 1:
        return f"{field} = {_sql_string(unique_values[0])}"
    return f"{field} IN ({','.join(_sql_string(value) for value in unique_values)})"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _row_matches_entities(row: dict[str, Any], entity_filters: set[str]) -> bool:
    entity_keys = str(row.get("entity_keys") or "")
    return any(f"|{entity}|" in entity_keys for entity in entity_filters)


def _entity_char_start(entity: dict[str, Any]) -> int:
    value = entity.get("char_start")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _normalized_values(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _extraction_warnings(documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for doc in documents:
        for warning in doc.get("extraction", {}).get("warnings", []):
            warnings.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "source_uri": doc.get("source_uri"),
                    "warning": str(warning),
                }
            )
    return warnings


def _count_summary(values: Iterable[Any], limit: int = 10) -> list[dict[str, Any]]:
    counts = Counter(str(value) for value in values if value is not None and str(value) != "")
    return [{"value": value, "count": count} for value, count in counts.most_common(limit)]
