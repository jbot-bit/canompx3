---
description: CRG D3 graph visualization. Generates an HTML file and opens it in the browser. On-demand human-auditor surface only — no agent loop should consume the HTML.
allowed-tools: Bash
---

# /crg-visualize — interactive graph (browser)

**Phase 3 / A8.** Renders the canonical graph as a D3.js HTML page and opens it.

## Usage

`/crg-visualize [community-name]` — e.g. `/crg-visualize` (whole graph) or
`/crg-visualize test-trading-app-no` (one community).

## Returns

A path to the generated HTML and a browser-opened tab. Output is for human
audit only. Agents do NOT consume HTML — if you want graph data programmatically
use the MCP query tools instead.

## Implementation

```bash
# Default output dir lives at .code-review-graph/visualize/ on canonical worktree
out_dir="C:/Users/joshd/canompx3/.code-review-graph/visualize"
mkdir -p "$out_dir"
code-review-graph visualize \
  --repo C:/Users/joshd/canompx3 \
  --output "$out_dir/graph_$(date +%Y%m%d_%H%M%S).html" \
  ${ARGUMENTS:+--community "$ARGUMENTS"} 2>&1 | tail -5

# Try to open in default browser (Windows / Git Bash)
latest="$(ls -t "$out_dir"/graph_*.html 2>/dev/null | head -1)"
if [ -n "$latest" ]; then
  echo "Opening $latest"
  start "" "$latest" 2>/dev/null || cmd /c start "" "$latest" 2>/dev/null || echo "Open manually: $latest"
fi
```

## When NOT to use

- Programmatic graph traversal → use `mcp__code-review-graph__traverse_graph_tool`
  or `query_graph_tool`.
- Token-budget-aware exploration → `/crg-context` then `/crg-search`.
- Headless agent runs → never call this; the browser open will fail and the
  HTML is wasted.

## Hard rules

- Visualization is a **human surface**. Do not parse the HTML in an agent loop.
- The output HTML is gitignored (lives under `.code-review-graph/`).
- Do not run this from worktrees other than canonical without redirecting
  `--output` — keeps state in the canonical graph dir.

## Refs

- `docs/plans/2026-04-29-crg-integration-spec.md` § Phase 3 / A8
