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


if __name__ == "__main__":
    raise SystemExit(main())
