"""Command-line interface for Fieldguide."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .api import FieldguideIndex, build_index
from .render import render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldguide")
    subparsers = parser.add_subparsers(dest="command", required=True)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("--json", action="store_true", help="Print the raw JSON response instead of Markdown")

    build_parser = subparsers.add_parser("build", help="Build an index from a source directory", parents=[output_parent])
    build_parser.add_argument("source", type=Path)
    build_parser.add_argument("--index", type=Path, default=Path(".fieldguide_index"))
    build_parser.add_argument("--corpus-id", default="C-FIELDGUIDE")
    build_parser.add_argument("--corpus-version", default="v1")
    build_parser.add_argument("--pdf-backend", choices=["legacy", "docling"], default="legacy")
    build_parser.add_argument("--progress", action="store_true", help="Print build progress and ETA to stderr")

    orient_parser = subparsers.add_parser("orient", help="Orient to the indexed corpus", parents=[output_parent])
    _add_index_arg(orient_parser)
    orient_parser.add_argument("--max-top-topics", type=int, default=10)
    orient_parser.add_argument("--token-budget", type=int)

    expand_parser = subparsers.add_parser("expand-topic", help="Show child topics", parents=[output_parent])
    _add_index_arg(expand_parser)
    expand_parser.add_argument("topic_id")
    expand_parser.add_argument("--max-children", type=int, default=8)

    topic_parser = subparsers.add_parser("topic-card", help="Open a topic card", parents=[output_parent])
    _add_index_arg(topic_parser)
    topic_parser.add_argument("topic_id")
    topic_parser.add_argument("--snippets", type=int, default=5)
    topic_parser.add_argument("--representative-docs", type=int, default=8)

    search_parser = subparsers.add_parser("search", help="Search within an explicit scope", parents=[output_parent])
    _add_index_arg(search_parser)
    search_parser.add_argument("--query", required=True)
    search_scope = search_parser.add_mutually_exclusive_group(required=True)
    search_scope.add_argument("--scope-id")
    search_scope.add_argument("--topic-id")
    search_scope.add_argument("--corpus", action="store_true")
    search_parser.add_argument("--level", choices=["document", "page", "passage"], default="passage")
    search_parser.add_argument("--max-results", type=int, default=15)
    search_parser.add_argument("--doc-type", action="append", dest="doc_types")
    search_parser.add_argument("--source", action="append", dest="sources")
    search_parser.add_argument("--entity", action="append", dest="entities")

    doc_parser = subparsers.add_parser("doc-card", help="Open a document card", parents=[output_parent])
    _add_index_arg(doc_parser)
    doc_parser.add_argument("doc_id")

    window_parser = subparsers.add_parser("read-window", help="Read a bounded evidence window", parents=[output_parent])
    _add_index_arg(window_parser)
    window_parser.add_argument("--snippet-id")
    window_parser.add_argument("--doc-id")
    window_parser.add_argument("--page-number", type=int)
    window_parser.add_argument("--char-start", type=int)
    window_parser.add_argument("--char-end", type=int)
    window_parser.add_argument("--max-chars", type=int)

    pages_parser = subparsers.add_parser("read-pages", help="Read selected pages", parents=[output_parent])
    _add_index_arg(pages_parser)
    pages_parser.add_argument("doc_id")
    pages_parser.add_argument("--pages", required=True, help="Comma-separated page numbers, e.g. 1,2,5")
    pages_parser.add_argument("--max-chars", type=int)

    args = parser.parse_args(argv)
    try:
        payload = _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return JSON errors for tool callers.
        error = {"error": type(exc).__name__, "message": str(exc)}
        if getattr(args, "json", False):
            print(json.dumps(error, indent=2), file=sys.stderr)
        else:
            print(f"# error\n\n{error['error']}: {error['message']}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_markdown(payload), end="")
    return 0


def _add_index_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=Path(".fieldguide_index"))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "build":
        stats = build_index(
            args.source,
            args.index,
            corpus_id=args.corpus_id,
            corpus_version=args.corpus_version,
            config={"extraction": {"pdf_backend": args.pdf_backend}},
            progress=_progress_printer if args.progress else None,
        )
        return {
            "corpus_id": stats.corpus_id,
            "corpus_version": stats.corpus_version,
            "doc_count": stats.doc_count,
            "page_count": stats.page_count,
            "chunk_count": stats.chunk_count,
            "topic_count": stats.topic_count,
            "index_path": str(stats.index_path),
        }

    index = FieldguideIndex(args.index)
    if args.command == "orient":
        return index.orient_corpus(max_top_topics=args.max_top_topics, token_budget=args.token_budget)
    if args.command == "expand-topic":
        return index.expand_topic(args.topic_id, max_children=args.max_children)
    if args.command == "topic-card":
        return index.topic_card(args.topic_id, snippets=args.snippets, representative_docs=args.representative_docs)
    if args.command == "search":
        if args.scope_id:
            scope = {"scope_id": args.scope_id}
        elif args.topic_id:
            scope = {"kind": "topic_subtree", "topic_ids": [args.topic_id], "include_descendants": True}
        else:
            scope = {"kind": "corpus", "corpus_id": index.corpus_id, "corpus_version": index.corpus_version}
        filters = {
            key: value
            for key, value in {
                "doc_types": args.doc_types,
                "sources": args.sources,
                "entities": args.entities,
            }.items()
            if value
        }
        return index.search_within(
            scope=scope,
            query=args.query,
            result_level=args.level,
            max_results=args.max_results,
            filters=filters or None,
        )
    if args.command == "doc-card":
        return index.doc_card(args.doc_id)
    if args.command == "read-window":
        return index.read_window(
            snippet_id=args.snippet_id,
            doc_id=args.doc_id,
            page_number=args.page_number,
            char_start=args.char_start,
            char_end=args.char_end,
            max_chars=args.max_chars,
        )
    if args.command == "read-pages":
        pages = [int(part.strip()) for part in args.pages.split(",") if part.strip()]
        return index.read_pages(doc_id=args.doc_id, pages=pages, max_chars=args.max_chars)
    raise ValueError(f"unknown command: {args.command}")


def _progress_printer(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "build_start":
        print(
            f"[build] files={event.get('total_files')} pdf_backend={event.get('pdf_backend')} index={event.get('index_dir')}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "document_start":
        print(
            f"[{event.get('index'):>3}/{event.get('total_files'):<3}] extracting "
            f"{event.get('source_uri')} size={_format_size(event.get('size_bytes'))}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "document_done":
        print(
            f"[{event.get('index'):>3}/{event.get('total_files'):<3}] done "
            f"{_format_duration(event.get('duration_seconds'))} "
            f"pages={event.get('page_count')} chunks={event.get('chunk_count')} "
            f"method={event.get('extraction_method')} "
            f"avg={_format_duration(event.get('average_seconds_per_file'))}/file "
            f"eta={_format_duration(event.get('eta_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "topicing_start":
        print(
            f"[topics] docs={event.get('doc_count')} pages={event.get('page_count')} "
            f"chunks={event.get('chunk_count')} entities={event.get('entity_count')} "
            f"elapsed={_format_duration(event.get('elapsed_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "topicing_done":
        print(
            f"[topics] done topics={event.get('topic_count')} elapsed={_format_duration(event.get('elapsed_seconds'))}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "build_done":
        print(
            f"[build] done docs={event.get('doc_count')} pages={event.get('page_count')} "
            f"chunks={event.get('chunk_count')} topics={event.get('topic_count')} "
            f"elapsed={_format_duration(event.get('elapsed_seconds'))}",
            file=sys.stderr,
            flush=True,
        )


def _format_duration(value: Any) -> str:
    if value is None:
        return "?"
    seconds = max(0, int(round(float(value))))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_size(value: Any) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


if __name__ == "__main__":
    raise SystemExit(main())
