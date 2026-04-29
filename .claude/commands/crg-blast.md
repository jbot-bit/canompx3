---
description: CRG blast-radius / impact-analysis for a changed file. Use AFTER /crg-context when minimal context flags risk medium/high. MCP tool MUST pass detail_level=minimal — Python API output exceeds 500 KB on real files.
allowed-tools: Bash
---

# /crg-blast — impact radius

**Official tool ladder rung 3 of 4.**

## Usage

`/crg-blast <file_path>` — e.g. `/crg-blast trading_app/strategy_discovery.py`

## CRITICAL: token discipline

Benchmarked 2026-04-28 on `trading_app/strategy_discovery.py`: 522 impacted nodes, raw Python-API JSON = **560 KB**. The MCP tool's `detail_level="minimal"` parameter is mandatory; the Python API has no equivalent so the fallback below projects to top-10 affected files only.

Per official LLM-OPTIMIZED-REFERENCE §usage: "Use detail_level=minimal on all subsequent calls unless you need more detail." 90% token savings.

## Implementation

**Preferred — MCP tool** (use when MCP is available / approved):
- Call `mcp__code-review-graph__get_impact_radius_tool` with `changed_files=[$ARGUMENTS], max_depth=2, detail_level="minimal"`.

**Fallback — Python one-liner** (use when MCP is not approved or unavailable; top-10 file projection):

```bash
.venv/Scripts/python.exe -c "
from code_review_graph.tools.query import get_impact_radius
import os, sys
out = get_impact_radius(changed_files=[sys.argv[1]], max_depth=2, repo_root='.')
imp = out.get('impacted_nodes') or []
chg = out.get('changed_nodes') or []
files = sorted({n['file_path'] for n in imp})
summary = (out.get('summary') or '').splitlines()[0] if out.get('summary') else '(no summary)'
print('Summary:', summary)
print('Changed nodes  :', len(chg))
print('Impacted nodes :', len(imp))
print('Files affected :', len(files))
print()
repo = os.path.abspath('.').replace(chr(92), '/').rstrip('/') + '/'
print('Top %d affected files:' % min(10, len(files)))
for f in files[:10]:
    fp = f.replace(chr(92), '/')
    if fp.lower().startswith(repo.lower()):
        fp = fp[len(repo):]
    print(' ', fp)
if len(files) > 10:
    print('  ... (%d more — use MCP tool with detail_level=minimal for full)' % (len(files) - 10))
" "$ARGUMENTS"
```

## Truth caveat

Blast radius is **structural** (call/import/inherits edges). Does NOT capture:
- DataFrame column-access edges (e.g. `df['rel_vol']` reads)
- SQL column dependencies
- Runtime-dispatched calls

For column-level data-flow bugs (E2 look-ahead pattern), `pipeline/check_drift.py` AST scans remain authoritative. CRG is structural, drift is semantic.
