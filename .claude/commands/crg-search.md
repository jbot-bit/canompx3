---
description: CRG semantic/FTS search for nodes. Standout tool in 2026-04-28 benchmark — top 3 hits all canonical for "E2 break_bar look-ahead". Use for "where does X happen?" questions.
allowed-tools: Bash
---

# /crg-search — semantic node search

**Standout tool from 2026-04-28 benchmark.** Hybrid (FTS5 + keyword) search across 13,804 graph nodes. Returns ranked symbols with file:line.

## Usage

`/crg-search <free-text query>` — e.g. `/crg-search E2 break bar lookahead contamination predictor`

Optional: filter by node kind (File/Class/Function/Test). Pass via MCP arg `kind=...`.

## Verified output (2026-04-28)

Query `"E2 break bar lookahead contamination predictor"` → top 3:
1. `check_e2_lookahead_research_contamination` (the canonical drift check)
2. `BreakBarContinuesFilter` (the canonical filter class)
3. `is_e2_lookahead_filter` (the canonical filter detector)

All three were the right files. This is CRG's highest-signal tool for codebases with lots of similarly-named symbols.

## Implementation

**Preferred — MCP tool** (after merge + Claude Code restart):
- Call `mcp__code-review-graph__semantic_search_nodes_tool` with `query=$ARGUMENTS, limit=10`.

**Fallback — Python one-liner:**

```bash
.venv/Scripts/python.exe -c "
from code_review_graph.tools.query import semantic_search_nodes
import os, sys
out = semantic_search_nodes(query=sys.argv[1], limit=10, repo_root='.')
print('Summary:', out.get('summary',''))
print('Mode   :', out.get('search_mode','?'))
print()
repo = os.path.abspath('.').replace(chr(92), '/').rstrip('/') + '/'
for r in (out.get('results') or [])[:10]:
    fp = (r.get('file_path') or '').replace(chr(92), '/')
    if fp.lower().startswith(repo.lower()):
        fp = fp[len(repo):]
    score = r.get('score','')
    score_s = ' %.3f' % score if isinstance(score, (int, float)) else ''
    print('  [%-8s] %-40s %s:%s%s' % (r.get('kind','?'), r.get('name','?'), fp, r.get('line_start','?'), score_s))
" "$ARGUMENTS"
```

## Optional — upgrade to vector embeddings

Default is hybrid FTS+keyword (no embeddings installed). To enable semantic vectors:

```bash
.venv/Scripts/pip install "code-review-graph[embeddings]"
.venv/Scripts/code-review-graph build  # rebuild required
```

Then `semantic_search_nodes_tool` auto-uses vectors. Per official LLM-OPTIMIZED-REFERENCE §embeddings: local model is `all-MiniLM-L6-v2`, 384-dim, ~90MB one-time download.

**Don't enable without measuring.** Hybrid mode already returned canonical hits in the benchmark; vectors may or may not improve it for this codebase.
