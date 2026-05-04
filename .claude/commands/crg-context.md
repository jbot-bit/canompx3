---
description: CRG minimal-context entry point. Always call this first per official LLM-OPTIMIZED-REFERENCE before any other CRG tool. Returns ~80 tokens (risk, communities, flows, suggested next tools).
allowed-tools: Bash
---

# /crg-context — minimal-context entry

**Official tool ladder rung 1 of 4.** Per `docs/external/code-review-graph/LLM-OPTIMIZED-REFERENCE.md` §usage: ALWAYS start here.

## Usage

`/crg-context <task description>` — e.g. `/crg-context audit cost_model canonical chain`

If no task given, use `"review current diff"`.

## Returns (~80 tokens, smoke-tested 2026-04-28)

```
Risk    : low
Summary : 13804 nodes, 150717 edges across 1044 files. Risk: low (0.00).
Comms   : test-trading-app-no, research-load, test-pipeline-passes
Flows   : post_session, _on_bar, __init__
Next    : detect_changes, semantic_search_nodes, get_architecture_overview
```

Read **`Next:`** — that tells you which tool to call after this one.

## Implementation

**Preferred — MCP tool** (use when MCP is available / approved):
- Call `mcp__code-review-graph__get_minimal_context_tool` with `task=$ARGUMENTS`.

**Fallback — Python one-liner** (use when MCP is not approved or unavailable):

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
import sys
_p = Path('.').resolve()
_sibling = _p.parent / 'canompx3'
if _p.name.startswith('canompx3-') and (_sibling / '.code-review-graph').exists():
    _root = str(_sibling)
else:
    try:
        from code_review_graph.incremental import find_project_root
        _root = str(find_project_root(_p))
    except Exception:
        _root = '.'
from code_review_graph.tools.context import get_minimal_context
out = get_minimal_context(task=sys.argv[1], repo_root=_root)
print('Risk    :', out.get('risk'))
print('Summary :', out.get('summary'))
print('Comms   :', ', '.join((out.get('communities') or [])[:3]))
print('Flows   :', ', '.join((out.get('flows_affected') or [])[:3]))
print('Next    :', ', '.join(out.get('next_tool_suggestions') or []))
" "$ARGUMENTS"
```

## When NOT to use CRG at all

- Trading data, fitness, strategy stats → `gold-db` MCP
- Doctrine / thresholds → `docs/institutional/`
- Trivial 1-file edits → just Read + Grep

## Tool ladder

1. **`/crg-context <task>`** ← you are here
2. `/crg-search <query>` — semantic/FTS lookup (best-rated tool in 2026-04-28 benchmark)
3. `/crg-blast <file>` — impact radius (use only if context flagged risk medium/high)
4. `/crg-tests <qualified-symbol>` — affected tests
