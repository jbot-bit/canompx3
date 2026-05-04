---
description: CRG dead-code preview. Wraps refactor_tool(mode=dead_code) — returns a list of unreferenced functions/classes. Preview ONLY; never auto-applies.
allowed-tools: Bash
---

# /crg-deadcode — dead-code preview (preview-only)

**Phase 3 / A7.** Identifies functions/classes the graph reports as unreferenced
(no inbound CALLS edges, no inbound REFERENCES edges).

## Usage

`/crg-deadcode [path-prefix]` — e.g. `/crg-deadcode pipeline/` or `/crg-deadcode`
(repo-wide).

## Returns

A list of candidate dead-code symbols. **All output is preview-only.** The
`refactor_tool(mode=dead_code, apply=False)` mode never modifies code.

Example shape:
```
Candidate dead code:
  pipeline/foo.py::_legacy_helper           (no callers in graph)
  trading_app/bar.py::OldHandler            (no inbound REFERENCES)
  scripts/baz.py::deprecated_main           (no flows reach this entry)
```

## Implementation

**Preferred — MCP tool** (when approved):
- Call `mcp__code-review-graph__refactor_tool` with `mode="dead_code"`,
  `apply=False`, `repo_root="C:/Users/joshd/canompx3"`, optional
  `path_prefix=$ARGUMENTS`.

**Fallback — CLI:**

```bash
code-review-graph refactor --mode dead-code --preview --repo C:/Users/joshd/canompx3 ${ARGUMENTS:+--path "$ARGUMENTS"} 2>&1 | head -60
```

## Hard rules

- **Never auto-apply.** This command is preview-only by design (spec §
  `apply_refactor_tool` SKIP). Refactors touching production go through the
  stage-gate (`/stage-gate`).
- **CRG is a frozen snapshot** (Volatile Data Rule). Verify each candidate
  with `Grep` before deleting — partial coverage on `tests_for` /
  `REFERENCES` edges in v2.1.0 means false positives are possible. A
  candidate flagged as dead may still be reflectively imported, dynamically
  loaded, or used in a test fixture the graph misses.
- Do **not** mass-delete from this output without per-symbol confirmation.

## When NOT to use

- Looking for unused tests → use coverage tools, not graph-based dead-code.
- Looking for unused config keys → grep `config.py` and the consumers.
- Removing canonical symbols → STOP, this is a stage-gate task.

## Refs

- `docs/plans/2026-04-29-crg-integration-spec.md` § Phase 3 / A7
- `feedback_crg_v2_1_0_bugs.md` (REFERENCES coverage caveat)
