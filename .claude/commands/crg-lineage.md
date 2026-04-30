---
description: Predicate-lineage auditor. Enumerates every research script that consumes a given daily_features column or canonical predicate. Use to catch contamination classes before they bite (cf. E2 break-bar look-ahead registry).
allowed-tools: Bash
---

# /crg-lineage — predicate-lineage auditor

**Phase 3 / A9 (D6 in spec).** Catches the next predicate-contamination class
before it bites. Built from the lesson of the E2 break-bar look-ahead
contamination registry (29 entries, 24 tainted) — manual grep was the only
detection mechanism, and it took weeks. This command is the automated
equivalent.

## Usage

`/crg-lineage <predicate>` — e.g.
- `/crg-lineage daily_features.rel_vol_atr20`
- `/crg-lineage break_dir`
- `/crg-lineage trading_app.entry_rules.detect_break_touch`

## Returns

Two lists:

1. **Affected flows** — entry-point flows in CRG that pass through the predicate.
2. **Direct consumers** — every research script / pipeline file that calls or
   imports the predicate's source function/column.

Example shape:
```
Predicate: daily_features.rel_vol_atr20

Affected flows (top 5):
  - pipeline/build_daily_features.py::main
  - research/2026-04-28-rel-vol-decomposition.py::run
  - trading_app/strategy_validator.py::validate_strategy
  ...

Direct consumers:
  - pipeline/build_daily_features.py:142   (assignment site)
  - research/2026-04-28-rel-vol-decomposition.py:31  (read site, post-entry?)
  - research/.../scratch_*.py:55  (read site)
  ...

⚠ POST-ENTRY SUSPECT (heuristic): scripts where the read site appears AFTER an
entry-determination block. Manual review required — graph cannot prove temporality.
```

## Implementation

**Preferred — MCP composition:**
1. `mcp__code-review-graph__get_affected_flows_tool` with the predicate as `target`.
2. `mcp__code-review-graph__query_graph_tool` with `pattern="callers_of"` and
   `target=<predicate>`.
3. Heuristic post-entry classification done client-side by reading each consumer
   and checking line ordering (read this command's caller for the heuristic).

**Fallback — CLI:**

```bash
code-review-graph affected-flows --target "$ARGUMENTS" --repo C:/Users/joshd/canompx3 2>&1 | head -30
echo "---"
code-review-graph query --pattern callers_of --target "$ARGUMENTS" --repo C:/Users/joshd/canompx3 2>&1 | head -40
```

## Hard rules

- **CRG cannot prove temporality.** A "post-entry suspect" flag is a heuristic
  to focus manual review, not a verdict. Confirm with `Read` of the actual
  source code, looking for the entry-determination block ordering.
- **Volatile Data Rule:** the graph is a frozen snapshot. After major
  refactors, re-run `code-review-graph update` before trusting lineage output.
- **Use this BEFORE writing a new research script that consumes a predicate** —
  catches re-implementations of canonical functions (D2-class) at design time
  rather than after results are in.

## When NOT to use

- Trading-data lineage (which trades / strategies use a column) → `gold-db` MCP.
- Cross-language lineage (Python → SQL → Python) → not supported by graph; use
  manual SQL grep + Python grep.

## Refs

- `docs/plans/2026-04-29-crg-integration-spec.md` § Phase 3 / D6 / A9
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` (the
  motivating incident — what manual grep cost us)
- `feedback_e2_lookahead_drift_check_landed.md`
