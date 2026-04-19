---
mode: IMPLEMENTATION
slug: fix-audit-behavioral-prose-false-positive
stage: 1/1
started: 2026-04-19
task: "audit_behavioral.py check 5 (Triple-join guard) false-positives on docstrings that mention 'JOIN daily_features' in prose"
---

# Stage 1 — Tighten SQL block detection in check 5

## Context
`scripts/tools/audit_behavioral.py` check 5 ("Triple-join guard") flags
`research/vwap_comprehensive_family_scan.py:215` as a violation. The
flagged content is the function DOCSTRING:

  """Load orb_outcomes JOIN daily_features for one cell. Triple-join correct.

The docstring contains the literal English words "JOIN daily_features"
but is not SQL. The actual SQL block immediately below (lines 223-254)
correctly contains `o.orb_minutes = d.orb_minutes` and three other
references to `orb_minutes`.

The bug: `SQL_KEYWORD_PATTERN = re.compile(r"\b(?:SELECT|INSERT|FROM|JOIN)\b")`
treats any block with the word "JOIN" as a SQL block. Prose docstrings
that mention "JOIN" trip the false positive.

This blocks every PostToolUse hook on every Edit/Write to ANY file in
this repo (the hook runs the full audit on every save), so it has
significant friction cost beyond just one wrong flag.

## Approach
Require the candidate block to look like SQL, not prose, before applying
the JOIN-without-orb_minutes check. The cheapest robust improvement:
require BOTH `SELECT` AND `FROM` keywords present. Prose almost never
contains both as standalone tokens; real SQL queries always do.

Alternative considered (rejected): rephrase the docstring. That's a
bandaid — same false positive will recur on any future docstring that
mentions "JOIN daily_features" in prose.

## Scope Lock
- scripts/tools/audit_behavioral.py
- tests/test_tools/test_audit_behavioral.py

## Acceptance criteria
1. RED test: a docstring containing "JOIN daily_features" but no SELECT
   FROM should NOT be flagged.
2. GREEN: SQL_KEYWORD_PATTERN tightened to require SELECT and FROM both.
3. Existing tests still pass (real SQL blocks with `JOIN daily_features`
   without `orb_minutes` MUST still be flagged).
4. Run `python scripts/tools/audit_behavioral.py` on the repo →
   `vwap_comprehensive_family_scan.py:215` no longer in violation list.
5. Full pytest passes.

## Blast Radius
- Heuristic tightening — strictly REDUCES false positive surface.
- Could miss violations where SQL block has `JOIN` but no `SELECT FROM`
  combination (e.g., a UPDATE/DELETE with JOIN). Mitigated by the
  heuristic's existing scope (it only checks daily_features JOINs which
  are essentially always SELECT/CTE patterns in this codebase).
- No production trading code touched.
