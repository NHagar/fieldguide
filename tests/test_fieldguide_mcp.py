from __future__ import annotations

import pytest

from fieldguide import mcp_server
from fieldguide.api import build_index


def _build_sample_index(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "procurement.txt").write_text(
        "Emergency procurement approval for Acme Medical. "
        "The memo discusses competitive bidding and vendor quote review.",
        encoding="utf-8",
    )
    (source / "payroll.txt").write_text(
        "Payroll review notes discuss overtime approval and timesheet controls.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    build_index(source, index_dir)
    return index_dir


def test_mcp_requires_server_side_index_path(monkeypatch):
    monkeypatch.delenv("FIELDGUIDE_INDEX", raising=False)
    monkeypatch.setattr(mcp_server, "_INDEX_CACHE", None)

    with pytest.raises(RuntimeError, match="FIELDGUIDE_INDEX"):
        mcp_server.orient()


def test_mcp_tools_query_configured_index(monkeypatch, tmp_path):
    index_dir = _build_sample_index(tmp_path)
    monkeypatch.setenv("FIELDGUIDE_INDEX", str(index_dir))
    monkeypatch.setattr(mcp_server, "_INDEX_CACHE", None)

    overview = mcp_server.orient(max_top_topics=1)
    assert overview["tool_name"] == "orient_corpus"
    assert overview["result"]["corpus_summary"]["doc_count"] == 2

    results = mcp_server.search(query="approval", corpus=True, max_results=1)
    assert results["tool_name"] == "search_within"
    assert results["result"]["results"]
    assert results["result"]["results"][0]["doc_id"].startswith("D-")
