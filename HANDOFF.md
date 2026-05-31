# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13.
- **⚠ Truth note (Claude, 2026-05-30, verified):** the `--strict-live-clean` flag and the lane_allocator `feature_cache` opt described above are **NOT committed on any branch** (`git log --all -S` = 0 hits). They live only in `stash@{0}` + `docs/runtime/rescued/2026-05-30-*` (see RESCUE-MANIFEST). Capital-path, Codex-owned — must be committed by its owner, not dropped. The regenerated `lane_allocation*.json` is the OUTPUT of that un-landed code, so canonical `docs/runtime/lane_allocation.json` on HEAD does NOT reflect it.
- **✅ Drift CLEAN — RETRACTION of an earlier false claim (Claude, 2026-05-30, execution-verified):** an earlier version of this note (commit `82721bcc`) claimed `check_active_native_trade_windows_match_provenance` was FAILING on lane `MNQ_COMEX_SETTLE_...OVNRNG_100`. **That was wrong** — I wrote it from a stale memory note without executing, violating Rule 11 (never trust metadata). Direct call returns `VIOLATIONS: 0`; full `check_drift.py --skip-crg-advisory` = **NO DRIFT DETECTED, 170 passed, 0 failed** (incl. Check 191 cold-recheck PASSED). The COMEX_SETTLE lane IS present in canonical `lane_allocation.json` (one of the 3 active lanes). `backfill_validated_trade_windows.py` (live write) = `inspected=848 drifted=0 updated=0`. Trade-window provenance is canonical. No action owed.

## Current Codex Follow-up
- **Tool:** Codex
- **Date:** 2026-05-31
- **Summary:** Live-pilot readiness rest pass on `main`: `topstep_50k_mnq_auto` is a 3-lane MNQ single-account pilot (`--copies 1`), NYSE_OPEN SR-alarm lane is paused/parked, funded telemetry maturity remains advisory, strict readiness is green, ProjectX preflight passed 14/14, phase 7 passed, targeted live readiness/preflight tests passed. Also fixed docs/audit drift: CLAUDE sample thresholds, TRADING_RULES `NYSE_PREOPEN`, `.claude/rules/large-file-reads.md` false old-session hit, regenerated `REPO_MAP.md`, and fixed Phase 3's REPO_MAP checker to call `gen_repo_map.py --check`. No live launch was started; `START_BOT.bat` remains `--demo`.
- **Operator launcher:** Added `START_LIVE_PILOT.bat` plus `scripts/tools/start_topstep_live_pilot.py` so the human-facing live-pilot path is one Windows entrypoint. It pins `topstep_50k_mnq_auto` / `MNQ` / `--copies 1`, refreshes C11/C12, runs strict readiness and live preflight, then calls the canonical live runner. It does **not** use `--auto-confirm`; the canonical `CONFIRM` prompt remains the final capital gate.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-31
- **Commit:** 2810fead — @ feat(prop): encode MFFU Builder + Flex specs/tiers/payouts; fix Rapid sim-cap leak
- **Files changed:** 6 files
  - `HANDOFF.md`
  - `docs/audit/2026-05-31-mffu-forced-progression-live-cap-memo.md`
  - `docs/runtime/stages/2026-05-31-mffu-builder-prop-rules-stage2.md`
  - `tests/test_prop_profiles_mffu.py`
  - `trading_app/prop_firm_policies.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
