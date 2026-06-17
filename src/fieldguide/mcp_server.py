"""MCP server exposing the bounded Fieldguide API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .api import FieldguideIndex


mcp = FastMCP(
    "fieldguide",
    instructions=(
        "Use these tools to orient to and query a Fieldguide corpus. "
        "The raw index is intentionally not exposed as a resource or tool result."
    ),
)

_INDEX_CACHE: tuple[Path, int, FieldguideIndex] | None = None


def _configured_index_path() -> Path:
    value = os.environ.get("FIELDGUIDE_INDEX")
    if not value:
        raise RuntimeError("FIELDGUIDE_INDEX must be set for the Fieldguide MCP server.")
    return Path(value).expanduser()


def _index_file_state(index_path: Path) -> tuple[Path, int]:
    index_file = index_path / "index.json" if index_path.is_dir() else index_path
    try:
        return index_file.resolve(), index_file.stat().st_mtime_ns
    except OSError as exc:
        raise RuntimeError("Configured Fieldguide index cannot be loaded.") from exc


def _index() -> FieldguideIndex:
    global _INDEX_CACHE
    configured_path = _configured_index_path()
    index_file, mtime_ns = _index_file_state(configured_path)
    if _INDEX_CACHE is None or _INDEX_CACHE[0] != index_file or _INDEX_CACHE[1] != mtime_ns:
        try:
            _INDEX_CACHE = (index_file, mtime_ns, FieldguideIndex(configured_path))
        except Exception as exc:  # noqa: BLE001 - hide server-side index path details.
            raise RuntimeError("Configured Fieldguide index cannot be loaded.") from exc
    return _INDEX_CACHE[2]


@mcp.tool(name="fieldguide_orient")
def orient(max_top_topics: int = 10, token_budget: int | None = None) -> dict[str, Any]:
    """Orient to the corpus and get top topics, facets, quality, and suggested next actions."""
    return _index().orient_corpus(max_top_topics=max_top_topics, token_budget=token_budget)


@mcp.tool(name="fieldguide_expand_topic")
def expand_topic(topic_id: str, max_children: int = 8) -> dict[str, Any]:
    """Show child topics and neighboring topics for a Fieldguide topic ID."""
    return _index().expand_topic(topic_id, max_children=max_children)


@mcp.tool(name="fieldguide_topic_card")
def topic_card(topic_id: str, snippets: int = 5, representative_docs: int = 8) -> dict[str, Any]:
    """Open a topic card with representative docs, snippets, facets, quality, and actions."""
    return _index().topic_card(topic_id, snippets=snippets, representative_docs=representative_docs)


@mcp.tool(name="fieldguide_search")
def search(
    query: str,
    scope_id: str | None = None,
    topic_id: str | None = None,
    corpus: bool = False,
    include_descendants: bool = True,
    level: Literal["document", "page", "passage"] = "passage",
    max_results: int = 15,
    doc_types: list[str] | None = None,
    sources: list[str] | None = None,
    entities: list[str] | None = None,
) -> dict[str, Any]:
    """Search within the corpus, a scope ID, or a topic subtree."""
    selected_scopes = sum(1 for value in (scope_id, topic_id, corpus) if value)
    if selected_scopes > 1:
        raise ValueError("Use only one of scope_id, topic_id, or corpus.")

    index = _index()
    if scope_id:
        scope = {"scope_id": scope_id}
    elif topic_id:
        scope = {"kind": "topic_subtree", "topic_ids": [topic_id], "include_descendants": include_descendants}
    else:
        scope = {"kind": "corpus", "corpus_id": index.corpus_id, "corpus_version": index.corpus_version}

    filters = {
        key: value
        for key, value in {
            "doc_types": doc_types,
            "sources": sources,
            "entities": entities,
        }.items()
        if value
    }
    return index.search_within(
        scope=scope,
        query=query,
        result_level=level,
        max_results=max_results,
        filters=filters or None,
    )


@mcp.tool(name="fieldguide_doc_card")
def doc_card(doc_id: str) -> dict[str, Any]:
    """Open a document triage card with page map, best passages, related docs, and actions."""
    return _index().doc_card(doc_id)


@mcp.tool(name="fieldguide_read_window")
def read_window(
    snippet_id: str | None = None,
    doc_id: str | None = None,
    page_number: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded evidence window by snippet ID or document character range."""
    return _index().read_window(
        snippet_id=snippet_id,
        doc_id=doc_id,
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
        max_chars=max_chars,
    )


@mcp.tool(name="fieldguide_read_pages")
def read_pages(doc_id: str, pages: list[int], max_chars: int | None = None) -> dict[str, Any]:
    """Read selected pages from a document, bounded by Fieldguide's page and character limits."""
    return _index().read_pages(doc_id=doc_id, pages=pages, max_chars=max_chars)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
