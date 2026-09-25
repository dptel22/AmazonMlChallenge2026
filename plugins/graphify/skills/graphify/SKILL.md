---
name: graphify
description: Builds and queries a knowledge graph of a codebase with the graphify CLI — use when the user wants to map a repo's structure, find god nodes/most-connected concepts, trace call/import/inheritance paths across files, explain how a feature is wired, or asks to "graphify" a codebase.
---

# Graphify

Turn a codebase into a deterministic knowledge graph (tree-sitter AST, ~40 languages, no LLM, fully local) and query it instead of grepping.

## Prerequisites check

Run `graphify --version` first. If it is missing, install it and stop for the user:

```
uv tool install graphifyy
```

(no `uv` → `pipx install graphifyy` or `pip install graphifyy`). Python 3.10+ required. Then re-run the check.

## Workflow

1. **Extract the graph** for the target directory (default: current workspace):

   ```
   graphify extract <path>
   ```

   This produces `graph.json`, `GRAPH_REPORT.md`, and `graph.html` in the project. Do not re-extract if `graph.json` exists and is newer than the last code change (check `graphify --help` for a rebuild/status command first); say the graph is being reused.

2. **Answer the user's question by querying, not grepping:**

   - `graphify query <concept>` — look up a concept/symbol
   - `graphify path <A> <B>` — trace how two things connect (calls, imports, inheritance)
   - `graphify explain <feature>` — explain how a feature is wired across files
   - `GRAPH_REPORT.md` — read for god nodes (most-connected concepts), community clusters, and overview stats

3. **Report** — summarize findings in prose with `file:line`-style references where the CLI gives locations. Mention that every edge is tagged `EXTRACTED` (from AST) vs `INFERRED` so the user knows confidence.

## Notes

- Respect ignore configuration; never force-include secrets or `.env` files into the graph.
- Large repos: extraction is local and deterministic but can take a while — run it once and reuse `graph.json`.
- If a query command name differs in the installed version (check `graphify --help`), use the actual subcommands rather than assuming.
