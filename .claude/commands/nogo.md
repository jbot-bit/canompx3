---
description: One-screen NO-GO / KILL / PARK verdict summary for a topic. Replaces ad-hoc graveyard greps with a single research-catalog MCP call.
allowed-tools: Bash
---

# /nogo — kill-verdict lookup

Search the research catalog (literature + pre-regs + audit-results + `STRATEGY_BLUEPRINT.md` §5 NO-GO Registry) for prior kill verdicts on a topic. Returns up to 10 hits ranked by relevance.

## Usage

`/nogo <free-text topic>` — e.g. `/nogo NR7 IBS external retest` or `/nogo GARCH overnight`

## What it covers

- `docs/audit/results/*.md` — verdict detected from front-matter `verdict:`, body `**Verdict:** X`, or filename suffix (`-nogo.md`, `-kill.md`, `-park.md`)
- `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO Registry table rows
- Pre-reg hypotheses tagged with kill verdicts
- Literature sources matched on topic (rarely tagged kill, but surface)

Verdict tokens filtered: `NO-GO`, `KILL`, `PARK`, `UNSUPPORTED`, `DECAY`, `STALE`. Non-kill verdicts (`VALIDATED`, `CONDITIONAL`, etc.) are excluded.

## Implementation

Calls `mcp__research-catalog__search_research_catalog` with `verdict_tags=["NO-GO","KILL","PARK","UNSUPPORTED","DECAY","STALE"]` and the user's topic as `query`. Limit 10.

**Preferred — MCP tool:**
- Call `mcp__research-catalog__search_research_catalog` with:
  - `query=$ARGUMENTS`
  - `verdict_tags=["NO-GO","KILL","PARK","UNSUPPORTED","DECAY","STALE"]`
  - `limit=10`

**Fallback — Python one-liner** (when MCP unavailable):

```bash
.venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, '.')
from scripts.tools.research_catalog_mcp_server import _search_research_catalog
out = _search_research_catalog(
    query=sys.argv[1],
    verdict_tags=['NO-GO','KILL','PARK','UNSUPPORTED','DECAY','STALE'],
    limit=10,
)
print(f\"Query: {out['query']}\")
print(f\"Verdict-filter: {out['verdict_tags']}\")
print(f\"Hits: {len(out['items'])}\")
print()
for item in out['items']:
    verdict = item.get('verdict','?')
    title = item.get('title','?')
    path = item.get('path','?')
    date = item.get('date') or '-'
    print(f\"  [{verdict:8}] {title}\")
    print(f\"             {path}  ({date})\")
    snippet = item.get('snippet','').strip()
    if snippet:
        print(f\"             > {snippet[:180]}\")
    print()
" "$ARGUMENTS"
```

## Reading the output

- **NO-GO** = doctrinally banned (don't propose; reopen requires the criteria in the NO-GO Registry row)
- **KILL** = audit closed the hypothesis (specific test refuted it)
- **PARK** = held pending more data / power floor
- **UNSUPPORTED** = no literature grounding found; not the same as KILL
- **DECAY / STALE** = once worked, currently below threshold

If `/nogo` returns zero hits, the topic has no prior kill verdict in the local catalog — it does NOT mean it's never been tested. Cross-check with `/crg-search` and `/pinecone-assistant` for orthogonal coverage.

## Related

- `/research` — runs a structured research hypothesis (use AFTER `/nogo` to verify not in graveyard)
- `/pinecone-assistant` — broader project-history Q&A (chat surface, slower)
- `mcp__research-catalog__search_research_catalog` — same tool, full schema (verdict-tag union, etc.)
