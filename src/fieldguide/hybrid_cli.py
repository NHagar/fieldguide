"""Command-line interface for the LanceDB hybrid Fieldguide index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .hybrid import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SNIPPET_CHARS,
    HybridFieldguideIndex,
    build_hybrid_from_json,
    build_hybrid_from_source,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldguide-hybrid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("--json", action="store_true", help="Print the raw JSON response")

    build_source = subparsers.add_parser(
        "build-source",
        help="Build a LanceDB hybrid index from a source directory",
        parents=[output_parent],
    )
    build_source.add_argument("source", type=Path)
    build_source.add_argument("--index", type=Path, default=Path(".fieldguide_lancedb"))
    build_source.add_argument("--corpus-id", default="C-FIELDGUIDE")
    build_source.add_argument("--corpus-version", default="v1")
    build_source.add_argument("--pdf-backend", choices=["legacy", "docling"], default="docling")
    _add_embedding_args(build_source)
    build_source.add_argument("--progress", action="store_true", help="Print build progress to stderr")

    build_json = subparsers.add_parser(
        "build-from-json",
        help="Build a LanceDB hybrid index from an existing Fieldguide JSON index",
        parents=[output_parent],
    )
    build_json.add_argument("json_index", type=Path)
    build_json.add_argument("--index", type=Path, default=Path(".fieldguide_lancedb"))
    _add_embedding_args(build_json)
    build_json.add_argument("--progress", action="store_true", help="Print build progress to stderr")

    search = subparsers.add_parser("search", help="Search a LanceDB hybrid index", parents=[output_parent])
    search.add_argument("--index", type=Path, default=Path(".fieldguide_lancedb"))
    search.add_argument("--query", required=True)
    search.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    search.add_argument("--max-results", type=int, default=10)
    search.add_argument("--doc-type", action="append", dest="doc_types")
    search.add_argument("--source", action="append", dest="sources")
    search.add_argument("--entity", action="append", dest="entities")
    search.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)

    args = parser.parse_args(argv)
    try:
        payload = _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return structured errors for tool callers.
        error = {"error": type(exc).__name__, "message": str(exc)}
        if getattr(args, "json", False):
            print(json.dumps(error, indent=2), file=sys.stderr)
        else:
            print(f"# error\n\n{error['error']}: {error['message']}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render(payload), end="")
    return 0


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "build-source":
        stats = build_hybrid_from_source(
            args.source,
            args.index,
            corpus_id=args.corpus_id,
            corpus_version=args.corpus_version,
            pdf_backend=args.pdf_backend,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            progress=_progress_printer if args.progress else None,
        )
        return _stats_payload(stats)

    if args.command == "build-from-json":
        stats = build_hybrid_from_json(
            args.json_index,
            args.index,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            progress=_progress_printer if args.progress else None,
        )
        return _stats_payload(stats)

    if args.command == "search":
        index = HybridFieldguideIndex(args.index)
        return index.search(
            args.query,
            mode=args.mode,
            max_results=args.max_results,
            doc_types=args.doc_types,
            sources=args.sources,
            entities=args.entities,
            snippet_chars=args.snippet_chars,
        )

    raise ValueError(f"unknown command: {args.command}")


def _stats_payload(stats: Any) -> dict[str, Any]:
    return {
        "tool_name": "fieldguide_hybrid_build",
        "corpus_id": stats.corpus_id,
        "corpus_version": stats.corpus_version,
        "doc_count": stats.doc_count,
        "chunk_count": stats.chunk_count,
        "vector_dimension": stats.vector_dimension,
        "index_path": str(stats.index_path),
        "metadata_path": str(stats.metadata_path),
        "warnings": stats.warnings,
    }


def _render(payload: dict[str, Any]) -> str:
    if payload.get("tool_name") == "fieldguide_hybrid_build":
        lines = [
            "# fieldguide_hybrid_build",
            "",
            f"corpus: `{payload.get('corpus_id')}` / `{payload.get('corpus_version')}`",
            f"docs: {payload.get('doc_count')}  chunks: {payload.get('chunk_count')}",
            f"vector dimension: {payload.get('vector_dimension')}",
            f"index: `{payload.get('index_path')}`",
            "",
        ]
        for warning in payload.get("warnings", []):
            lines.append(f"- warning: {warning}")
        return "\n".join(lines).rstrip() + "\n"

    result = payload.get("result", {})
    lines = [
        "# fieldguide_hybrid_search",
        "",
        f"query: `{result.get('query')}`  mode: `{result.get('mode')}`  results: {result.get('result_count')}",
        "",
    ]
    for item in result.get("results", []):
        location = f"{item.get('doc_id')} p{item.get('page_number')} chars {item.get('char_start')}-{item.get('char_end')}"
        lines.extend(
            [
                f"## {item.get('rank')}. {item.get('title') or item.get('doc_id')}",
                "",
                f"- score: {item.get('score')}  source: `{item.get('source_metadata', {}).get('source_uri')}`",
                f"- location: {location}",
                "",
                item.get("snippet") or "",
                "",
            ]
        )
    for warning in payload.get("warnings", []):
        lines.append(f"- warning: {warning}")
    return "\n".join(lines).rstrip() + "\n"


def _progress_printer(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "hybrid_build_start":
        print(
            f"[hybrid] start mode={event.get('source_mode')} "
            f"files={event.get('total_files', '?')} chunks={event.get('chunk_count', '?')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "document_start":
        print(
            f"[{event.get('index'):>3}/{event.get('total_files'):<3}] extracting {event.get('source_uri')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "document_done":
        print(
            f"[{event.get('index'):>3}/{event.get('total_files'):<3}] done "
            f"chunks={event.get('chunk_count')} method={event.get('extraction_method')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "embedding_start":
        print(
            f"[hybrid] embedding chunks={event.get('chunk_count')} model={event.get('embedding_model')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "hybrid_build_done":
        print(
            f"[hybrid] done docs={event.get('doc_count')} chunks={event.get('chunk_count')} "
            f"elapsed={event.get('elapsed_seconds'):.1f}s",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())

