# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13.
- **⚠ Truth note (Claude, 2026-05-30, verified):** the `--strict-live-clean` flag and the lane_allocator `feature_cache` opt described above are **NOT committed on any branch** (`git log --all -S` = 0 hits). They live only in `stash@{0}` + `docs/runtime/rescued/2026-05-30-*` (see RESCUE-MANIFEST). Capital-path, Codex-owned — must be committed by its owner, not dropped. The regenerated `lane_allocation*.json` is the OUTPUT of that un-landed code, so canonical `docs/runtime/lane_allocation.json` on HEAD does NOT reflect it.
- **⚠ Known drift (Claude, 2026-05-30, verified, UNOWNED-by-me):** `check_active_native_trade_windows_match_provenance` FAILS (1 violation): lane `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` is `status='active'` in gold.db `validated_setups` but has NULL/non-strict-IS `native_trade_window_start`. NOT in canonical `lane_allocation.json` on HEAD (0 hits) — it's a gold.db data-state row, downstream of Codex's un-landed rebalance (see Truth note above). `backfill_validated_trade_windows.py --dry-run` = drifted=0 (scopes 848 other rows, not this active lane). Capital-path; fix belongs with the rebalance owner (re-run backfill after the strict-live-clean work lands) — do NOT patch the downstream row to silence the check (Source-of-Truth Chain Rule). My commit `59fa70a6` is orthogonal (touched 0 validated_setups/alloc/drift files).

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-30
- **Commit:** 59fa70a6 — fix(live): scope _ensure_repo_python to main() so import is preflight-safe (pushed to origin/main)
- **Files changed:** 3 files
  - `HANDOFF.md`
  - `scripts/tools/live_readiness_report.py`
  - `tests/test_tools/test_live_readiness_report.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
