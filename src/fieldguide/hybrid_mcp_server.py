"""MCP server exposing the LanceDB hybrid Fieldguide search API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .hybrid import DEFAULT_SNIPPET_CHARS, HYBRID_METADATA_FILENAME, HybridFieldguideIndex


mcp = FastMCP(
    "fieldguide-hybrid",
    instructions=(
        "Use Fieldguide Hybrid as a bounded LanceDB-backed search tool. "
        "Do not inspect raw index files or ask for the index path; use only this MCP tool. "
        "Use fieldguide_hybrid_search for semantic, keyword, or hybrid chunk search. "
        "Treat returned snippets as bounded evidence previews, not full documents. "
        "Cite doc_id, title, source URI, page number, and character range when using results. "
        "If extraction warnings are returned, surface them when they affect confidence."
    ),
)

_INDEX_CACHE: tuple[Path, int, HybridFieldguideIndex] | None = None


def _configured_index_path() -> Path:
    value = os.environ.get("FIELDGUIDE_HYBRID_INDEX")
    if not value:
        raise RuntimeError("FIELDGUIDE_HYBRID_INDEX must be set for the Fieldguide Hybrid MCP server.")
    return Path(value).expanduser()


def _metadata_file_state(index_path: Path) -> tuple[Path, int]:
    metadata_file = index_path / HYBRID_METADATA_FILENAME if index_path.is_dir() else index_path
    try:
        return metadata_file.resolve(), metadata_file.stat().st_mtime_ns
    except OSError as exc:
        raise RuntimeError("Configured Fieldguide Hybrid index cannot be loaded.") from exc


def _new_index(index_path: Path) -> HybridFieldguideIndex:
    return HybridFieldguideIndex(index_path)


def _index() -> HybridFieldguideIndex:
    global _INDEX_CACHE
    configured_path = _configured_index_path()
    metadata_file, mtime_ns = _metadata_file_state(configured_path)
    if _INDEX_CACHE is None or _INDEX_CACHE[0] != metadata_file or _INDEX_CACHE[1] != mtime_ns:
        try:
            index_path = metadata_file.parent if metadata_file.name == HYBRID_METADATA_FILENAME else configured_path
            _INDEX_CACHE = (metadata_file, mtime_ns, _new_index(index_path))
        except Exception as exc:  # noqa: BLE001 - hide server-side index path details.
            raise RuntimeError("Configured Fieldguide Hybrid index cannot be loaded.") from exc
    return _INDEX_CACHE[2]


@mcp.tool(name="fieldguide_hybrid_search")
def hybrid_search(
    query: str,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    max_results: int = 10,
    doc_types: list[str] | None = None,
    sources: list[str] | None = None,
    entities: list[str] | None = None,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> dict[str, Any]:
    """Search the configured LanceDB hybrid Fieldguide index."""
    return _index().search(
        query,
        mode=mode,
        max_results=max_results,
        doc_types=doc_types,
        sources=sources,
        entities=entities,
        snippet_chars=snippet_chars,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

