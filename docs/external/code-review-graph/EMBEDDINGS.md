# CRG Embeddings — Configuration & Status

**Date verified:** 2026-04-29
**Graph version:** v2.3.2
**Graph stats:** 13,821 nodes / 150,940 edges / 1,047 files

## Current embeddings mode

**Mode: hybrid (FTS5 + keyword)**

Verified via `semantic_search_nodes_tool` call — `search_mode` field returns `"hybrid"`.

Local vector embeddings (MiniLM-L6-v2) are **NOT installed**. The graph uses
FTS5 full-text search + keyword scoring (RRF fusion) without dense vectors.

## Search quality — local benchmark (2026-04-29)

Queries run against this repo's graph DB. Mode: hybrid. No vectors.

| Query | Top hit | Latency |
|-------|---------|---------|
| `orb_utc_window session catalog` | `_session_open_utc` | 7,312ms* |
| `E2 break bar lookahead contamination` | `check_e2_lookahead_research_contamination` ✓ | 3,579ms |
| `parse_strategy_id eligibility builder` | `parse_strategy_id` ✓ | 3,578ms |
| `cost model reprice entry` | `_cost` | 3,558ms |
| `holdout sacred date policy` | `enforce_holdout_date` ✓ | 3,622ms |

*First query includes Python cold-start (~3.5s overhead). Subsequent queries ~3.6s.
Via MCP (already-running server) latency is significantly lower — no process spawn.

3 of 5 top hits are canonical for the query. The `orb_utc_window` miss returned
`_session_open_utc` (a helper it calls internally) — semantically adjacent, not wrong.

## To enable vector embeddings (optional — do NOT enable without re-measuring)

```bash
.venv/Scripts/pip install "code-review-graph[embeddings]"
.venv/Scripts/code-review-graph build  # full rebuild required (~30 min)
```

Model: `all-MiniLM-L6-v2`, 384-dim, ~90MB one-time download.
Per official LLM-OPTIMIZED-REFERENCE §embeddings: auto-used by `semantic_search_nodes`
after rebuild. Hybrid mode already returns canonical hits — measure before upgrading.

## Decision record

- 2026-04-29: Do NOT install embeddings yet. Hybrid hits 3/5 canonical. Re-measure
  quarterly. Enabling vectors requires a full graph rebuild (~30 min, CI-blocking).
  Document in this file after any change.

## Re-verification procedure

```bash
.venv/Scripts/python.exe -c "
from code_review_graph.tools.query import semantic_search_nodes
out = semantic_search_nodes(query='E2 break bar lookahead contamination', limit=3, repo_root='.')
print('mode:', out.get('search_mode'))
for r in (out.get('results') or [])[:3]:
    print(' ', r.get('name'), r.get('score'))
"
```

Expected: `check_e2_lookahead_research_contamination` in top 3.
