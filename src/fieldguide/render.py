"""Markdown rendering for agent-facing Fieldguide responses."""

from __future__ import annotations

from typing import Any


def render_markdown(payload: dict[str, Any]) -> str:
    if "tool_name" not in payload:
        return _render_build(payload)

    tool_name = payload["tool_name"]
    result = payload.get("result", {})
    lines = [
        f"# {tool_name}",
        _kv_line(
            corpus=f"{payload.get('corpus_id')} / {payload.get('corpus_version')}",
            kind=payload.get("response_kind"),
            tokens=payload.get("token_estimate"),
            truncated=payload.get("truncated"),
        ),
    ]
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", *[f"- {_clean(warning)}" for warning in payload["warnings"]]])

    renderer = {
        "orient_corpus": _render_orient,
        "expand_topic": _render_expand_topic,
        "topic_card": _render_topic_card,
        "search_within": _render_search,
        "doc_card": _render_doc_card,
        "read_window": _render_read_window,
        "read_pages": _render_read_pages,
    }.get(tool_name, _render_generic)
    lines.extend(["", *renderer(result)])
    actions = payload.get("actions", [])
    if actions:
        lines.extend(["", "## Actions", *_render_actions(actions[:6])])
    return "\n".join(line.rstrip() for line in lines if line is not None).strip() + "\n"


def _render_build(payload: dict[str, Any]) -> str:
    lines = [
        "# build",
        _kv_line(
            corpus=f"{payload.get('corpus_id')} / {payload.get('corpus_version')}",
            docs=payload.get("doc_count"),
            pages=payload.get("page_count"),
            chunks=payload.get("chunk_count"),
            topics=payload.get("topic_count"),
        ),
        "",
        f"index: `{payload.get('index_path')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def _render_orient(result: dict[str, Any]) -> list[str]:
    summary = result.get("corpus_summary", {})
    lines = [
        "## Corpus",
        _kv_line(
            docs=summary.get("doc_count"),
            pages=summary.get("page_count"),
            chunks=summary.get("chunk_count"),
            root_scope=result.get("root_scope", {}).get("scope_id"),
        ),
        "",
        "## Top Topics",
        _topic_table(result.get("top_topics", [])),
    ]
    facets = result.get("global_facets") or {}
    if facets:
        lines.extend(
            [
                "",
                "## Facets",
                f"- doc_types: {_counts_inline(facets.get('doc_types', []))}",
                f"- extraction: {_counts_inline(facets.get('extraction_quality', []))}",
                f"- sources: {_counts_inline(facets.get('sources', [])[:5])}",
                f"- organizations: {_entity_counts_inline(facets.get('organizations', [])[:5])}",
            ]
        )
    quality = result.get("quality") or {}
    if quality:
        lines.extend(
            [
                "",
                "## Quality",
                _kv_line(
                    text_extracted=quality.get("percent_text_extracted"),
                    ocr=quality.get("percent_ocr"),
                    mean_ocr_confidence=quality.get("mean_ocr_confidence"),
                    low_ocr_confidence=quality.get("percent_low_ocr_confidence"),
                    duplicates=quality.get("duplicate_rate"),
                ),
            ]
        )
    return lines


def _render_expand_topic(result: dict[str, Any]) -> list[str]:
    topic = result.get("topic", {})
    lines = ["## Topic", *_topic_block(topic), "", "## Children", _topic_table(result.get("children", []))]
    if result.get("neighboring_topics"):
        lines.extend(["", "## Neighbor Topics", *_neighbor_lines(result["neighboring_topics"])])
    return lines


def _render_topic_card(result: dict[str, Any]) -> list[str]:
    lines = ["## Topic", *_topic_block(result)]
    if result.get("interpretation_warning"):
        lines.extend(["", f"> {_clean(result['interpretation_warning'])}"])
    if result.get("representative_docs"):
        lines.extend(["", "## Representative Docs", *_doc_ref_lines(result["representative_docs"])])
    if result.get("representative_snippets"):
        lines.extend(["", "## Representative Snippets", *_snippet_lines(result["representative_snippets"])])
    if result.get("facets"):
        facets = result["facets"]
        lines.extend(
            [
                "",
                "## Facets",
                f"- doc_types: {_counts_inline(facets.get('doc_types', []))}",
                f"- extraction: {_counts_inline(facets.get('extraction_quality', []))}",
                f"- sources: {_counts_inline(facets.get('sources', []))}",
            ]
        )
    return lines


def _render_search(result: dict[str, Any]) -> list[str]:
    scope = result.get("scope", {})
    lines = [
        "## Search",
        _kv_line(
            query=result.get("query"),
            level=result.get("result_level"),
            matches=result.get("total_estimated_matches"),
            scope=scope.get("scope_id"),
        ),
        "",
        "## Results",
    ]
    results = result.get("results", [])
    if not results:
        lines.append("_No results._")
    for item in results:
        loc = _location(item)
        lines.append(
            f"{item.get('rank')}. `{item.get('doc_id')}` score={item.get('score')} {loc} "
            f"terms={_terms(item.get('matched_terms', []))}"
        )
        title = item.get("title")
        if title:
            lines.append(f"   title: {_clean(title)}")
        if item.get("snippet"):
            lines.append(f"   > {_clean(item['snippet'])}")
        read_actions = [action for action in item.get("actions", []) if action.get("action") in {"read_window", "read_pages", "doc_card"}]
        if read_actions:
            lines.append("   actions: " + "; ".join(_action_inline(action) for action in read_actions))
    if result.get("suggested_refinements"):
        lines.extend(["", "## Refinements"])
        for refinement in result["suggested_refinements"][:6]:
            count_label = f"~{refinement.get('estimated_result_count')}"
            if refinement.get("observed_result_count") is not None:
                count_label = f"matches={refinement.get('estimated_result_count')}"
                if refinement.get("scope_document_count") != refinement.get("estimated_result_count"):
                    count_label += f" scope={refinement.get('scope_document_count')}"
            lines.append(
                f"- {refinement.get('type')} `{_clean(refinement.get('value'))}` "
                f"{count_label} -> {_action_inline(refinement.get('action', {}))}"
            )
    return lines


def _render_doc_card(result: dict[str, Any]) -> list[str]:
    doc = result.get("doc", {})
    quality = result.get("quality", {})
    lines = [
        "## Document",
        _kv_line(
            doc=doc.get("doc_id"),
            type=doc.get("doc_type"),
            pages=doc.get("page_count"),
            chars=doc.get("char_count"),
            extraction=quality.get("extraction_method"),
            text=quality.get("text_available"),
        ),
        f"file: `{doc.get('source_uri')}`",
    ]
    if doc.get("title"):
        lines.append(f"title: {_clean(doc['title'])}")
    if result.get("topic_memberships"):
        lines.append("topics: " + ", ".join(f"{m['topic_id']}:{m['score']}" for m in result["topic_memberships"][:6]))
    if quality.get("warnings"):
        lines.extend(["", "## Extraction Warnings", *[f"- {_clean(warning)}" for warning in quality["warnings"]]])
    if result.get("page_map"):
        lines.extend(["", "## Page Map", "| page | tokens | terms | signals |", "|---:|---:|---|---|"])
        for page in result["page_map"][:12]:
            lines.append(
                f"| {page.get('page_number')} | {page.get('token_estimate')} | "
                f"{_terms(page.get('top_terms', [])[:5])} | {_clean('; '.join(page.get('signals', [])[:3]))} |"
            )
    if result.get("best_passages"):
        lines.extend(["", "## Best Passages", *_snippet_lines(result["best_passages"][:5])])
    if result.get("related_docs"):
        lines.extend(["", "## Related Docs"])
        for item in result["related_docs"][:8]:
            related = item.get("doc", {})
            lines.append(f"- `{related.get('doc_id')}` {item.get('relation')} conf={item.get('confidence')} `{related.get('source_uri')}`")
    return lines


def _render_read_window(result: dict[str, Any]) -> list[str]:
    return [
        "## Evidence Window",
        _evidence_meta(result),
        "",
        "```text",
        result.get("text", ""),
        "```",
        "",
        "## Provenance",
        _provenance_line(result.get("provenance", {})),
    ]


def _render_read_pages(result: dict[str, Any]) -> list[str]:
    lines = ["## Page Evidence", _provenance_line(result.get("provenance", {}))]
    for page in result.get("pages", []):
        lines.extend(
            [
                "",
                f"### Page {page.get('page_number')} chars={page.get('char_start')}..{page.get('char_end')}",
                "```text",
                page.get("text", ""),
                "```",
            ]
        )
        if page.get("extraction_warnings"):
            lines.append("warnings: " + "; ".join(_clean(warning) for warning in page["extraction_warnings"]))
    return lines


def _render_generic(result: Any) -> list[str]:
    if isinstance(result, dict):
        return ["## Result", *[f"- {key}: `{_clean(value)}`" for key, value in result.items() if key != "canonical_text"]]
    return ["## Result", str(result)]


def _topic_block(topic: dict[str, Any]) -> list[str]:
    quality = topic.get("quality") or {}
    scope = topic.get("scope") or {}
    lines = [
        _kv_line(
            topic=topic.get("topic_id"),
            parent=topic.get("parent_topic_id"),
            docs=topic.get("doc_count"),
            pages=topic.get("page_count"),
            chunks=topic.get("chunk_count"),
            scope=scope.get("scope_id"),
        ),
        f"terms: {_terms(topic.get('label_terms', []))}",
    ]
    if topic.get("top_entities"):
        lines.append("entities: " + _entity_counts_inline(topic.get("top_entities", [])[:8]))
    if quality:
        lines.append(
            "quality: "
            + _kv_line(
                coherence=quality.get("coherence"),
                source_concentration=quality.get("source_concentration"),
                ocr_risk=quality.get("ocr_risk"),
            )
        )
        if quality.get("warnings"):
            lines.append("warnings: " + "; ".join(_clean(warning) for warning in quality["warnings"]))
    return lines


def _topic_table(topics: list[dict[str, Any]]) -> str:
    if not topics:
        return "_No topics._"
    lines = ["| topic | docs | pages | terms | scope | warnings |", "|---|---:|---:|---|---|---|"]
    for topic in topics:
        quality = topic.get("quality") or {}
        lines.append(
            f"| `{topic.get('topic_id')}` | {topic.get('doc_count')} | {topic.get('page_count')} | "
            f"{_terms(topic.get('label_terms', [])[:6])} | `{(topic.get('scope') or {}).get('scope_id', '')}` | "
            f"{_clean('; '.join(quality.get('warnings', [])))} |"
        )
    return "\n".join(lines)


def _doc_ref_lines(docs: list[dict[str, Any]]) -> list[str]:
    lines = []
    for doc in docs:
        lines.append(
            f"- `{doc.get('doc_id')}` score={doc.get('topic_membership_score')} "
            f"type={doc.get('doc_type')} reason={','.join(doc.get('selection_reason', []))} title={_clean(doc.get('title'))}"
        )
    return lines


def _snippet_lines(snippets: list[dict[str, Any]]) -> list[str]:
    lines = []
    for snippet in snippets:
        lines.append(
            f"- `{snippet.get('snippet_id')}` `{snippet.get('doc_id')}` p{snippet.get('page_number')} "
            f"chars={snippet.get('char_start')}..{snippet.get('char_end')} terms={_terms(snippet.get('matched_terms', []))}"
        )
        lines.append(f"  > {_clean(snippet.get('text', ''))}")
        read_action = snippet.get("read_action")
        if read_action:
            lines.append(f"  read: {_action_inline(read_action)}")
    return lines


def _neighbor_lines(neighbors: list[dict[str, Any]]) -> list[str]:
    return [f"- `{item.get('topic_id')}` score={item.get('score')} terms={_terms(item.get('label_terms', []))}" for item in neighbors]


def _render_actions(actions: list[dict[str, Any]]) -> list[str]:
    return [f"- {_action_inline(action)}" for action in actions]


def _action_inline(action: dict[str, Any]) -> str:
    args = action.get("args", {})
    compact_args = ", ".join(f"{key}={_clean(value)}" for key, value in args.items())
    return f"{action.get('action')}({compact_args})"


def _evidence_meta(result: dict[str, Any]) -> str:
    return _kv_line(
        doc=result.get("doc_id"),
        page=result.get("page_number"),
        chars=f"{result.get('char_start')}..{result.get('char_end')}",
        type=result.get("doc_type"),
        source=result.get("source_batch"),
    )


def _provenance_line(provenance: dict[str, Any]) -> str:
    return _kv_line(
        doc=provenance.get("doc_id"),
        file=provenance.get("source_uri") or provenance.get("original_filename"),
        pages=",".join(str(page) for page in provenance.get("page_numbers") or []),
        chars=f"{provenance.get('char_start')}..{provenance.get('char_end')}",
        method=provenance.get("extraction_method"),
        text_version=provenance.get("canonical_text_version"),
    )


def _kv_line(**items: Any) -> str:
    return " ".join(f"{key}={_format_value(value)}" for key, value in items.items() if value is not None)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"`{_clean(value)}`"


def _counts_inline(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_none_"
    return ", ".join(f"{_clean(item.get('value'))}:{item.get('count')}" for item in items)


def _entity_counts_inline(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_none_"
    return ", ".join(f"{_clean(item.get('text') or item.get('value'))}:{item.get('count')}" for item in items)


def _terms(items: list[Any]) -> str:
    return " / ".join(_clean(item) for item in items) if items else "_none_"


def _location(item: dict[str, Any]) -> str:
    parts = []
    if item.get("page_number") is not None:
        parts.append(f"p{item['page_number']}")
    if item.get("char_start") is not None:
        parts.append(f"chars={item.get('char_start')}..{item.get('char_end')}")
    return " ".join(parts)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|")
    text = " ".join(text.split())
    return text[:900]
