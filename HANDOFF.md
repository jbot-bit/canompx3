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
- **Summary:** PR #327 on `codex/db-mcp-safe-access` implements and formalizes local-first DB MCP safe access: `gold-db` exposes read-only health, freshness, snapshot-manifest, and access-policy tools; `scripts/tools/export_gold_db_snapshot.py` exports approved read snapshots with stamped manifests; durable write-broker boundaries live in `docs/plans/active/2026-05/2026-05-30-db-mcp-safe-access.md`; implementation-grade plan lives in `docs/superpowers/plans/2026-05-31-db-mcp-safe-access.md`; a remote-consumer test now proves exported Parquet can be read without opening source `gold.db`. Verification passes targeted tests/ruff/CLI, targeted drift checks, and full `pipeline/check_drift.py --quiet` (`SUMMARY: clean passed=170 advisory=21`). No live DB writes, allocation edits, order routing, or `paper_trades` mutation.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-31
- **Commit:** 78d48eb1 — research(adaptive-stops): H4 time-based no-progress exit pre-reg DRAFT (K=1)
- **This session:** human-reviewed + PROMOTED the H4 draft to LOCKED. Moved
  `drafts/2026-05-31-...draft.yaml` → `docs/audit/hypotheses/2026-05-31-adaptive-stops-h4-time-based-exit-stack-v1.yaml`,
  status DRAFT→LOCKED. 3 reviewer_todo cleared: (1) Howard extract line-188
  parameter (0.5R/120s) verified verbatim — flagged as the extract's "e.g."
  illustrative pre-reg value, not a measured constant (honesty correction in
  pre_committed_parameters); (2) K-budget evaluated PASS (N=1, MinBTL 0.00yr,
  6.65yr headroom, fits cap) after adding `instruments: [MNQ, MES]`; (3) canonical-
  delegation / READ-not-resimulate enforced via methodology_gates.
  **NEXT:** write read-only `research/adaptive_stops_h4_time_exit_paired.py`
  (paired-ΔR, reads orb_outcomes excursion, no DB write) — only after this
  promotion commit lands (do_not_run_until_committed).
- **Files changed:** HANDOFF.md + rename draft→locked H4 prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
