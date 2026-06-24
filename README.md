# FieldGuide

FieldGuide is an agent-facing corpus orientation tool. It indexes a local directory into
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

## MCP Server

Fieldguide can run as a stdio MCP server that exposes only bounded query tools. Configure the
index path server-side with `FIELDGUIDE_INDEX`; do not put the raw index in the agent workspace.

```json
{
  "mcpServers": {
    "fieldguide-cpd": {
      "command": "fieldguide-mcp",
      "env": {
        "FIELDGUIDE_INDEX": "/Users/nrh146/.fieldguide-indexes/cpd"
      }
    }
  }
}
```

Available tools:

- `fieldguide_orient`
- `fieldguide_expand_topic`
- `fieldguide_topic_card`
- `fieldguide_search`
- `fieldguide_doc_card`
- `fieldguide_read_window`
- `fieldguide_read_pages`

## LanceDB Hybrid Search

Fieldguide also ships a separate LanceDB-backed hybrid search index and MCP server. This is
an alternative search path, not a replacement for the orientation API above. It reuses the
same document extraction and chunking pipeline, defaults PDF preprocessing to docling, embeds
chunks locally with `BAAI/bge-small-en-v1.5`, and stores a LanceDB table under the requested
index directory.

```bash
uv run fieldguide-hybrid build-source ./corpus --index .fieldguide_lancedb --pdf-backend docling
uv run fieldguide-hybrid build-from-json .fieldguide_index --index .fieldguide_lancedb
uv run fieldguide-hybrid search --index .fieldguide_lancedb --query "competitive bidding" --mode hybrid
```

Run it as a separate stdio MCP server with `FIELDGUIDE_HYBRID_INDEX`:

```json
{
  "mcpServers": {
    "fieldguide-hybrid-cpd": {
      "command": "fieldguide-hybrid-mcp",
      "env": {
        "FIELDGUIDE_HYBRID_INDEX": "/Users/nrh146/.fieldguide-indexes/cpd-lancedb"
      }
    }
  }
}
```

Available tool:

- `fieldguide_hybrid_search`
