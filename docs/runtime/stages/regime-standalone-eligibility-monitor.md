---
slug: regime-standalone-eligibility-monitor
stage: 1
mode: IMPLEMENTATION
status: SCOPED_AWAITING_PARAM_SIGNOFF
created: 2026-06-03
design_doc: docs/plans/2026-06-03-regime-standalone-eligibility-monitor-design.md
capital_path: true
worktree: REQUIRED (isolated, off clean origin/main; START_WORKTREE.bat)
---

# Stage 1 — REGIME standalone-eligibility monitor: predicate + shadow ledger ONLY

## Goal
Prove the four-gate monitor WOULD have gated REGIME lanes correctly, with ZERO
live allocation change. Paper-shadow only.

## scope_lock (files this stage MAY touch)
- CREATE `trading_app/regime_eligibility.py` — pure predicate
  (inputs: session-regime, fitness status, rolling-N, last-trade date,
  account-DD headroom [injected], cooldown state -> eligible/paused + reason).
  No I/O. No import of the live allocation decision path.
- CREATE `trading_app/regime_eligibility_shadow.py` — shadow ledger runner:
  reads live `LaneScore`s read-only, applies the predicate to REGIME-tier
  lanes only, writes verdicts to a shadow ledger file. Changes NOTHING live.
- CREATE `tests/test_trading_app/test_regime_eligibility.py` — predicate unit
  tests (the six failure-mode tests in the design).
- CREATE `tests/test_trading_app/test_regime_eligibility_shadow.py` — proves
  shadow run changes zero live allocation rows; CORE lanes never gated.
- CREATE `docs/runtime/regime_eligibility_shadow_ledger.yaml` — the shadow
  output (generated artifact; multi-terminal hygiene applies).

## FORBIDDEN (hard scope wall)
- NO edit to `trading_app/lane_allocator.py` (no live wiring this stage).
- NO edit to `trading_app/account_survival.py` (peer-dirty; DD is INJECTED).
- NO edit to `trading_app/strategy_fitness.py` (read its public classify only).
- NO profile / live_config / prop_profiles / deployment-sizing edit.
- NO SR-UNKNOWN gate change.
- NO deployment-threshold change of any kind.

## Capital-risk parameters — LITERATURE-GROUNDED (no gut guesses)
- `REGIME_STANDALONE_MIN_ROLLING_N` = 100. Grounded: Pepelyshev-Polunchenko
  2015 — SR detector estimates pre-anomaly mu/sigma from first 100 live trades;
  below 100 = NO_DATA = un-monitorable. (Supersedes earlier guess of 30.)
- `REGIME_STANDALONE_RECENCY_TRADING_DAYS` = 60. Grounded: P-P SR ARL ~= 60
  trading days (~one quarter). Recency aligns to detector clock.
  (Supersedes earlier 45 calendar days.)
- Auto-pause uses EXISTING `sr_status == "ALARM"` (Pepelyshev-Polunchenko,
  implemented `trading_app/sr_monitor.py`). NO new cooldown param — consume the
  canonical detector. (Removes the earlier hand-rolled cooldown knobs.)
All knobs are vetoes that tighten, never loosen.

## Acceptance (Stage 1 done = all true, with shown evidence)
1. Predicate unit tests pass (six failure modes) — show pytest output.
2. Shadow-leak test proves zero live allocation rows change — show output.
3. CORE lanes (N>=100) are never gated by the predicate — show output.
4. Shadow ledger generated against live scores; REGIME verdicts present with
   per-gate reasons — show a sample of the ledger.
5. `py_compile` + scoped `ruff` pass on new files.
6. `git diff --check` clean.
7. Drift check: record result; if it times out under contention, record the
   exact timeout and what remains unverified (do not block on it).
8. Self-review: confirm no FORBIDDEN file touched (`git diff --name-only`).

## Verification order (narrowest first)
predicate unit tests -> shadow-leak test -> py_compile -> ruff -> git diff --check
-> drift (resource-permitting).
