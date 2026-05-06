---
task: R3 — refresh stale validated_setups trade-window provenance to clear Check 50
mode: IMPLEMENTATION
scope_lock:
  - scripts/migrations/backfill_validated_trade_windows.py
  - HANDOFF.md
  - docs/plans/active/2026-05/2026-05-05-live-trading-rollout.md
---

## Blast Radius

- `scripts/migrations/backfill_validated_trade_windows.py` — read-only invocation, no source edits planned. Canonical writer for Check 50 per its own docstring ("Canonical refresh for Check 45" — drift check renumbered to 50, same function `check_active_native_trade_windows_match_provenance`).
- `gold.db` (canonical truth-layer) — `UPDATE validated_setups` on the 10 stale `status='active' AND promotion_provenance='VALIDATOR_NATIVE'` rows. Touches only `first_trade_day`, `last_trade_day`, `trade_day_count`. Performance columns + `status` + `promotion_provenance` + `promotion_git_sha` left authoritative.
- HANDOFF.md + `docs/plans/active/2026-05/2026-05-05-live-trading-rollout.md` — already-staged Phase 0 evidence packet edits will commit alongside R3 closure note.
- Live deployment: NONE of the 10 stale rows are deployed lanes (per HANDOFF + lane_allocation.json — current deployment is 3 MNQ lanes, none overlap stale set).
- Reads: `validated_setups` (active VALIDATOR_NATIVE rows only) + whatever `StrategyTradeWindowResolver` reads to recompute (`orb_outcomes` joined to `validated_setups` filter spec).
- Writes: `validated_setups` row updates only; no schema, no canonical-source code, no test edits.

## Plan

1. Dry-run the canonical refresh: `python scripts/migrations/backfill_validated_trade_windows.py --dry-run`. Confirm `drifted == 10` and capture the per-strategy drift detail.
2. Live run: `python scripts/migrations/backfill_validated_trade_windows.py`. Confirm `updated == 10` and exit 0.
3. Re-run `python pipeline/check_drift.py` end-to-end. Confirm Check 50 (`check_active_native_trade_windows_match_provenance`) is now empty + overall drift exits 0.
4. Stage R3 closure note in HANDOFF.md + plan doc, then `git commit` the already-staged Phase 0 evidence packet + closure note + this stage file.
5. `git push -u origin docs/phase-0-evidence-2026-05-06` and open PR.

## Done criteria

- Check 50 empty (`drift_count == 0` overall, including the previously failing `check_active_native_trade_windows_match_provenance` block).
- 10 `validated_setups` rows have refreshed trade-window provenance via the canonical writer (no hand-rolled UPDATE).
- HANDOFF.md + plan doc reflect R3 closed; R4 status unchanged (still queued, separate full-instrument rebuild).
- Branch pushed to `origin/docs/phase-0-evidence-2026-05-06` and PR opened.

## Notes

- This is `[mechanical]` per stage-gate / institutional-rigor: re-runs an existing canonical writer with no behavior changes. No live-trading path or truth-layer code edited. Adversarial-audit gate does NOT trigger (commit is not `[judgment]` and does not edit `trading_app/live/`, `risk_manager.py`, or `pipeline/`).
- Canonical writer per `.claude/rules/institutional-rigor.md` § 4: `scripts/migrations/backfill_validated_trade_windows.py` is the documented canonical refresh for the same drift class (Check 45 → 50 renumbering only; same function).
- No deployment risk: HANDOFF confirms none of the 10 stale rows are deployed lanes. Live trading book unaffected.
