---
task: |
  Build NYSE_PREOPEN as a tradeable MNQ session so the parked draft prereg
  (docs/audit/hypotheses/drafts/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.draft.yaml,
  commit 9c5b2b26, K=27, /nogo clean, K-budget PASS at ceiling) can run and,
  if it clears strict Chordia (t>=3.79, NO_THEORY_GRANT), become a new
  UNCORRELATED additive lane. This is the disciplined "grow new inventory"
  path chosen over forcing an MGC/MES prereg (see decision rationale below).
  STAGE 1 ONLY in this file: session definition in pipeline/dst.py.
mode: DESIGN
updated: 2026-05-25T16:00Z
agent: claude (opus 4.7)

scope_lock:
  - pipeline/dst.py
  - tests/test_pipeline/test_dst.py

blast_radius:
  - pipeline/dst.py — CANONICAL session catalog. Adds nyse_preopen_brisbane resolver + SESSION_CATALOG["NYSE_PREOPEN"] + DOW_MISALIGNED_SESSIONS entry (offset -1). Every outcome/feature/execution path reads session timing from here — HIGH blast. Verified in isolation before any downstream rebuild.
  - tests/test_pipeline/test_dst.py — resolver DST-season + DOW-misalignment regression.
  - Reads: none new. Writes: none (pure timing definition). orb_outcomes/daily_features only change when Stage 3 rebuild runs (separate stage).
  - CRIT-path canonical edit -> adversarial-audit gate fires after this stage.
---

# NYSE_PREOPEN session build — Stage 1: session definition

## Decision rationale (why this, not an MGC/MES prereg)

User asked to "get more trades live-promoted". Ground-truth findings 2026-05-25
(HEAD e4ae20a6 at investigation time):

- Every Chordia-cleared edge is ALREADY accounted for: 4 LIVE, 10 DISPLACED by
  correlation gate, 3 PAUSED on C8 OOS. ZERO unblocked slots. No quick promotion exists.
- Promotable-candidate lists are selection-biased sample frames (per
  feedback_chordia_candidate_selection_population_vs_sample_frame). Verified against
  canonical orb_outcomes with strict IS/OOS split at HOLDOUT_SACRED_FROM=2026-01-01:
  - MGC_LONDON_METALS E2 RR2.0 CB1 O30 + ORB_VOL_8K: IS +0.7327R (N=45, WR 60%, first day 2025-04-09)
    -> OOS +0.0673R (N=33, WR 36%). Magnitude collapsed; RR2.0 WR below breakeven OOS.
  - MGC_LONDON_METALS RR1.5 variant: IS +0.6039R (N=48) -> OOS +0.1495R (N=34, WR 47%). ~75% decay.
  - MGC_CME_REOPEN E2 RR1.0 CB1 O5 base: IS -0.1402R (N=464) — base session is a loser;
    validated +0.28R N=86 is the ORB_G4 size-filtered subset only.
  All three would likely PARK on FAILED_RATIO -> burn MinBTL budget. NOT clean prereg targets.
- NYSE_PREOPEN is a NEW SESSION (not a saturated-session filter-variant), so it is genuine
  additive territory. LONDON_METALS vs TOKYO_OPEN corr = -0.39 per STRATEGY_BLUEPRINT;
  a new US-preopen session is plausibly low-corr with the 4 live MNQ lanes (allocator
  correlation gate adjudicates downstream — out of scope for the prereg).

## Blockers the draft itself enumerates (the plan spine)

1. NYSE_PREOPEN resolver NOT in pipeline.dst.DYNAMIC_ORB_RESOLVERS — runner cannot
   compute ORB windows. (THIS STAGE resolves.)
2. NYSE holiday-exclusion source NOT canonical — no is_nyse_holiday in
   pipeline/calendar_filters.py (only is_nfp_day confirmed at line 29). Holiday-bar
   contamination is an explicit kill criterion. (Stage 2.)

## Staged chain (upstream -> downstream)

- **Stage 1 (this file):** pipeline/dst.py — add nyse_preopen_brisbane resolver
  (anchor 09:00 ET NYSE Order Imbalance publication; EDT->23:00 Brisbane same-day,
  EST->00:00 next-cal-day; mirror us_equity_open_brisbane tz math EXACTLY, never
  hardcode offsets), register SESSION_CATALOG["NYSE_PREOPEN"] (break_group "us"),
  add DOW_MISALIGNED_SESSIONS entry offset -1 (EST Brisbane-Fri 00:00 = US-Thu 09:00 ET).
  Verify orb_utc_window(d, "NYSE_PREOPEN", 5/15/30) resolves both DST seasons.
- **Stage 2:** pipeline/calendar_filters.py — add is_nyse_holiday(d) backed by a
  COMMITTED canonical holiday list (grounded like is_nfp_day, not a library guess).
- **Stage 3 (no code):** bash scripts/tools/run_rebuild_with_sync.sh MNQ — populate
  orb_outcomes + daily_features for NYSE_PREOPEN. Capture IS/OOS day counts; confirm
  DST-imbalance kill floor (N_EST>=30 AND N_EDT>=30) and the draft's N=27 budget hold.
- **Stage 4 (research, gated on 1-3):** promote draft out of drafts/ quarantine, run
  K=27 Chordia strict-unlock (t>=3.79, theory_grant=false, PASS_PROTOCOL_A unavailable),
  produce verdict MD. PASS_CHORDIA -> eligible next allocator rebalance.

## Canonical contracts (verified this session)

- SESSION_CATALOG entry shape: dict {type:"dynamic", resolver:<fn>, break_group:str, event:str}.
  Resolvers return (hour, minute) Brisbane. DYNAMIC_ORB_RESOLVERS = {label: entry["resolver"]
  for dynamic entries} (pipeline/dst.py ~line 539). orb_utc_window at ~line 543.
- Sibling resolver to mirror: us_equity_open_brisbane (pipeline/dst.py:315) — 09:30 ET.
  NYSE_PREOPEN is 09:00 ET (30 min earlier), same tz arithmetic.
- DOW_MISALIGNED_SESSIONS lives in pipeline/dst.py (also referenced in TRADING_RULES.md,
  strategy_discovery.py). Existing precedent: NYSE_OPEN midnight-crossing guard.

## Honest risk

Mechanism is NO_THEORY_GRANT (plausible synthesis, no /resources PDF anchor) -> strict
t>=3.79 bar. May not clear. But it is a clean a-priori session test at honest power, not
a decayed filter-selected artifact. The session build is reusable regardless of verdict.

## NOT done by this stage

- No is_nyse_holiday (Stage 2). No orb_outcomes rebuild (Stage 3). No prereg run (Stage 4).
- No allocator/sizing/deployment change. No write to validated_setups.
- Draft stays in drafts/ quarantine until Stage 4.

## Resume pointer (next session)

This stage is DESIGN-approved-by-user, not yet implemented. Next session: flip mode to
IMPLEMENTATION, write the resolver + catalog entry + DOW entry + test in pipeline/dst.py,
verify orb_utc_window both DST seasons, run drift + dst tests, adversarial-audit gate,
then return for Stage 2 go-ahead. Peer live-bar-ring work was in-flight on main at
commit ba53cdf0 (bar_ring.py / session_orchestrator.py / recover_ring.py) — NOT this
session's; do not entangle.

COMMIT BLOCKER (carry-over): this stage file is staged-but-uncommitted. Pre-commit step
2b/8 (trade-window backfill, scripts/migrations/backfill_validated_trade_windows.py) BLOCKS
with rc=1 — pre-existing validated_setups staleness (likely the partial-refresh class bug,
feedback_validated_setups_partial_refresh_n1_2026_05_21), NOT caused by this doc-only change.
User did not authorize the DB migration. Resolve the backfill (or diagnose the staleness)
before this commit + push can land. Do NOT --no-verify.

SEPARATE REQUEST (user, 2026-05-25, deferred to next session due to pre-clear): port the
"MPX3 - No HUD" TradingView Pine v6 ORB indicator to draw ALL canonical sessions (not just
the 6 in the user's script, not just auto-traded ones). Key correctness requirements:
(1) the user's script hardcodes fixed GMT+10 clock hours — WRONG for US/UK sessions which
are DST-aware (NYSE_OPEN 09:30 ET = 23:30 Brisbane EDT but 00:30 EST). Port must use
TradingView timezone-aware anchors (e.g. "America/New_York", "Europe/London", "America/Chicago")
mirroring pipeline.dst SESSION_CATALOG resolvers, letting TV do DST math. (2) Draw all 12
catalog sessions: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, EUROPE_FLOW,
US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, NYSE_CLOSE, BRISBANE_1025 (+ NYSE_PREOPEN
once built). (3) ORB aperture should reflect real 5/15/30-min apertures (precomputed outcomes),
not the script's fixed 300000ms freeze. Source event anchors from SESSION_CATALOG event strings.
This is a standalone .pine deliverable (not in pipeline/ or trading_app/) — own stage, not
entangled with the dst.py session build.
