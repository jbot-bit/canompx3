---
task: "Lever 1 — add --skip-crg-advisory flag to check_drift.py and pass it from pre-commit step 3, removing the ~73s D5 CRG advisory cost from the commit drift gate while preserving full blocking coverage (CRG advisories still run in CI)."
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - .githooks/pre-commit
  - tests/test_pipeline/test_check_drift_skip_crg.py
---

## Blast Radius

- `pipeline/check_drift.py` — adds a new `CRG_ADVISORY_LABELS` frozenset (explicit
  D1–D5), a `_assert_crg_advisory_labels_valid()` guard (asserts every label is in
  CHECKS AND is_advisory=True — the load-bearing honesty property), a new
  `--skip-crg-advisory` argparse flag, a skip branch in the runner loop, and a
  `crg_skipped` counter in the summary. NO change to any check's logic, ordering,
  or exit-code semantics. Flag defaults OFF — full behavior unchanged unless opted in.
- `.githooks/pre-commit` — step 3 (line 367) gains `--skip-crg-advisory` on the
  drift invocation. Single-line change. Rollback = remove the flag token.
- `tests/test_pipeline/test_check_drift_skip_crg.py` — new test file: flag skips
  ONLY D1–D5; blocking checks still run; label-validity guard fails closed if a
  CRG label becomes blocking; serial/no-flag path unchanged.
- Reads: CRG graph DB (read-only, via existing checks); Writes: none.
- Honesty proof: skipped checks are all `is_advisory=True` (cannot change a commit
  verdict). CI path (no flag) runs them. In-session post-edit hook already skips
  them via `--fast`/SLOW_CHECK_LABELS, so no in-editing coverage changes.

## Decision basis (measured this session, not memory)

- D5 (`check_crg_bridge_node_test_coverage`) = **73.3s against an already-built
  fresh graph** (graph built 12:13 today, current commit). Cost is the per-node
  query loop, NOT graph staleness → reorder-3b (Lever 1-alt) measured ~0s, rejected.
- D1=4.0s, D2=4.4s, D3=3.5s, D4=0.1s, D5=73.3s. Total ~85s; D5 dominates.
- Post-edit hook already skips D1/D2/D3/D5 (in SLOW_CHECK_LABELS) → no in-session
  last-look exists to lose. Operator chose skip-at-commit with full knowledge.

## Gate

Self-review per institutional-rigor §1. No adversarial-audit gate required: no
blocking/truth-layer verdict path is touched (only advisory checks are skipped,
and a fail-closed guard enforces that invariant in code).
