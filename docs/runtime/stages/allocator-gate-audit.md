---
task: Allocator-gate attrition audit (validated_setups → lane_allocation.json)
mode: IMPLEMENTATION
slug: allocator-gate-audit
scope_lock:
  - docs/runtime/stages/allocator-gate-audit.md
  - scripts/tools/allocator_gate_audit.py
---

## Task
One-shot read-only diagnostic that maps strategy attrition from `validated_setups` (active deployable scope = 844) through every allocator gate down to the final deployed N in `lane_allocation.json`. Reports per-gate counts, top example strategy_ids, and verbatim `status_reason` exemplars. No production-code edits.

## Mode
IMPLEMENTATION (Stage 1 of 1 — v2 deferred Stages 2-4 per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`).

## Scope Lock
- `docs/runtime/stages/allocator-gate-audit.md` — this file
- `scripts/tools/allocator_gate_audit.py` — new read-only script

## Blast Radius
- `scripts/tools/allocator_gate_audit.py` — NEW file, zero callers, zero importers.
- DB access: `duckdb.connect(GOLD_DB_PATH, read_only=True)`. No `INSERT` / `UPDATE` / `DELETE` / `save_allocation` calls.
- Imports only canonical helpers: `compute_lane_scores`, `apply_chordia_gate`, `enrich_scores_with_liveness`, `compute_pairwise_correlation`, `compute_orb_size_stats`, `build_allocation` from `trading_app.lane_allocator`; `ACCOUNT_PROFILES`, `ACCOUNT_TIERS` from `trading_app.prop_profiles`; `deployable_validated_relation` via `pipeline.db_contracts`.
- Zero edits to `pipeline/` or `trading_app/`. Cannot alter live allocator output, paper trades, lane allocation JSON, or any persisted state.
- Drift surface: none — adds a `scripts/tools/` script not referenced by drift checks.

## Acceptance
1. `python scripts/tools/allocator_gate_audit.py --profile topstep_50k_mnq_auto` exits 0; Gate 0 reports 844; conservation invariant holds (sum across gates 1-6 + selection_loop + selected == 844); SELECTED count matches `docs/runtime/lane_allocation.json` for that profile.
2. `python scripts/tools/allocator_gate_audit.py --all-profiles` runs every active profile in `ACCOUNT_PROFILES` and prints per-profile tables.
3. `python pipeline/check_drift.py` exits 0 (no production-code touched).
4. Script self-review confirms: no inlined thresholds (`MIN_TRAILING_N`, `RHO_REJECT_THRESHOLD`, `HYSTERESIS_PCT` come from `trading_app.lane_allocator` if referenced at all); no re-encoded predicates (gate attribution is by parsing `status_reason` strings the canonical classifier produced).
