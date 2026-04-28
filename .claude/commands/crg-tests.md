---
description: Find tests covering a qualified symbol via CRG TESTED_BY edges. Best when symbol is fully-qualified (file::function or class). File paths return 0 — use file_summary first to enumerate symbols.
allowed-tools: Bash
---

# /crg-tests — affected tests

**Official tool ladder rung 4 of 4.**

## Usage

Fully-qualified (best):
`/crg-tests pipeline/dst.py::orb_utc_window`

Short name (returns ambiguous-candidates list to qualify):
`/crg-tests orb_utc_window`

## CRG limitation (verified 2026-04-28 benchmark)

`tests_for` returns 0 results when the target is a **file path** (e.g. `pipeline/cost_model.py`). TESTED_BY edges link to specific functions/classes, not files. To find tests for a file's functions:

1. `query_graph(file_summary, "<file>.py")` → list of functions/classes
2. For each symbol, `query_graph(tests_for, "<file>::<symbol>")`

Verified working: `pipeline/dst.py::orb_utc_window` → 8 hits, all in `tests/test_pipeline/test_orb_utc_window.py`. Correct.

## Implementation

**Preferred — MCP tool** (after merge + Claude Code restart):
- Call `mcp__code-review-graph__query_graph_tool` with `pattern="tests_for", target=$ARGUMENTS`.

**Fallback — Python one-liner:**

```bash
.venv/Scripts/python.exe -c "
from code_review_graph.tools.query import query_graph
import os, sys
out = query_graph(pattern='tests_for', target=sys.argv[1], repo_root='.')
repo = os.path.abspath('.').replace(chr(92), '/').rstrip('/') + '/'
def short(p):
    s = (p or '').replace(chr(92), '/')
    return s[len(repo):] if s.lower().startswith(repo.lower()) else s
if out.get('status') == 'ambiguous':
    print('AMBIGUOUS:', out.get('summary'))
    print()
    print('Candidates (qualify and re-run):')
    for c in (out.get('candidates') or [])[:10]:
        print('  [%-8s] %s (%s:%s)' % (c.get('kind','?'), c.get('qualified_name','?'), short(c.get('file_path','')), c.get('line_start','?')))
else:
    results = out.get('results') or []
    print('Summary:', out.get('summary'))
    print()
    if not results:
        sym = sys.argv[1].split('::')[-1]
        print('No tests found via TESTED_BY edges.')
        print('Caveat: integration tests / fixtures may exist but are invisible to the static graph.')
        print('Cross-check: pytest --collect-only -q | grep ' + sym)
    else:
        for r in results[:20]:
            print('  [%-8s] %-40s %s:%s' % (r.get('kind','?'), r.get('name','?'), short(r.get('file_path','')), r.get('line_start','?')))
        if len(results) > 20:
            print('  ... (%d more)' % (len(results) - 20))
" "$ARGUMENTS"
```

## Truth caveat

A symbol with **0 TESTED_BY edges** does NOT mean "no tests exist" — it means "no test imports/calls it directly." Indirect coverage (integration tests, fixtures, parametrize) is invisible to the static graph. Cross-check with `pytest --collect-only -q | grep <symbol>` if zero hits look wrong.
