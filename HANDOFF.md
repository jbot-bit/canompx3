# HANDOFF.md - Cross-Tool Session Baton

---

## 🔴 RESUME HERE (2026-05-31, Claude Code) — MFFU Builder backtest

**Next action:** Model the deployed MNQ book against MFFU **Builder** plan constraints — read-only `gold.db` sim, NO config/schema/capital edits.

Replay deployed MNQ lane daily P&L vs MFFU Builder (live-verified from help.myfundedfutures.com this session — repo snapshots are STALE, firms update constantly; re-fetch live before any firm-rule claim):
- **$2,000 EOD trailing DD, locks permanently at +$100** above start
- **$1,000 daily-loss soft-pause**; **40 micro** max (50k); **50% payout-consistency** (eval has none); 80/20, $500 min, **$2k/cycle cap**

Per micro-contract size compute: max-DD-to-breach, P(breach), worst-day-vs-cycle. Output "profitable at N micros, X% breach prob." Use `COST_SPECS` (MNQ), `GOLD_DB_PATH`, `validated_setups WHERE status='active'`. Backtest books realized-eod (does NOT model intraday equity vs trailing DD — OK for Builder's EOD-lock; would NOT be OK for Rapid's intraday trailing).

**Decisions locked (grounded, read-only):**
1. **NQ vs MNQ = sizing/cost only, ZERO revalidation.** Same Nasdaq/R-multiples. NQ friction −34%/R but 10× coarser. Mechanism already designed: `prop_profiles.py:123-169` (Stage 1 landed, Stage 2 order-build unbuilt). `orb_active=False` does NOT block live use. **MNQ-micro <~$139k; hybrid $150–500k; NQ-primary $500k+.** `prop_profiles.py:420` "1 NQ/$25k" = STALE (Tier B flag, don't edit unprompted). All firms 10:1 mini:micro.
2. **Firm fit for automated bot:** automation policy = dominant filter. **Apex DISQUALIFIED** (bans automation). **Bulenox + MFFU best.** TopStep auto-policy for autonomous bots UNVERIFIED → open risk on current live `topstep_50k_mnq_auto`.
3. **MFFU plan = BUILDER** (EOD-lock-+$100 safe for E2 reversals; 40 micro granularity vs Pro's 5; no eval consistency). Rapid REJECTED (funded=intraday trailing, hostile to E2). NQ on 50k = ~3 losers to breach → micros only.

**Task B PARKED:** adaptive stops/liquidity-sweep — heavy NO-GO overlap (breakeven trail DEAD, vol-regime DEAD, Chan stop-cascade twice-exhausted, Howard 2026 price-level stops EV-negative). Only seam = continuous Carver-Ch11-12 R4 modifier, blocked on lit extraction not a scan. Behind NQ + NYSE_PREOPEN O30 + powered-OOS.

**Constraints:** no live config/profiles/strategy/schema/deploy edits without explicit approval + adversarial-audit gate. `--holdout-date 2026-01-01`.

---

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
- **Commit:** 4c3228b9 — @ docs(handoff): close-out — live-preflight landed, capital-path work rescued
- **Files changed:** 7 files
  - `HANDOFF.md`
  - `docs/runtime/rescued/2026-05-30-RESCUE-MANIFEST.md`
  - `docs/runtime/rescued/2026-05-30-lane-allocator-feature-cache-WIP.patch`
  - `docs/runtime/rescued/2026-05-30-lane_allocation.STASHED.json`
  - `docs/runtime/rescued/2026-05-30-topstep_50k_mnq_auto.STASHED.json`
  - `docs/runtime/stages/2026-05-30-opus-48-effort-wiring.md`
  - `docs/runtime/stages/2026-05-30-worktree-lease-real-mutex.md`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
