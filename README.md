# Fieldguide

Fieldguide is an agent-facing corpus orientation tool. It indexes a local directory into
bounded, provenance-rich JSON records and exposes the minimal v0 API from the specification:

- `orient_corpus`
- `expand_topic`
- `topic_card`
- `search_within`
- `doc_card`
- `read_window`
- `read_pages`

The tool stores canonical text internally, but the public API only returns scoped navigation
objects or bounded evidence windows.

## Quick Start

```bash
uv run fieldguide build ./corpus --index .fieldguide_index
uv run fieldguide orient --index .fieldguide_index
uv run fieldguide search --index .fieldguide_index --scope-id S-root --query "competitive bidding"
```

Commands print compact Markdown by default. Add `--json` to any command to print the raw
structured response:

```bash
uv run fieldguide orient --index .fieldguide_index --json
```

Entity extraction combines exact regex extraction for handles such as emails, phone numbers,
money amounts, and case-like identifiers with spaCy's `en_core_web_md` statistical NER
model for people, organizations, and locations. The small model is kept as a fallback if the
medium model is unavailable.
