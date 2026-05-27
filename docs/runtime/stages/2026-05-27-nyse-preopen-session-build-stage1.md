---
task: |
  Lane B — Build NYSE_PREOPEN as a tradeable session so the parked prereg draft
  (docs/audit/hypotheses/drafts/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.draft.yaml)
  can later run and, if it clears strict Chordia (t>=3.79, NO_THEORY_GRANT),
  become a NEW UNCORRELATED additive MNQ lane. This is the disciplined
  "grow new inventory" path (every Chordia-cleared edge already allocated;
  MGC/MES promotable candidates are decayed sample frames).

  STAGE 1 = the atomic five-list session DEFINITION + one hardening drift check.
  This stage does NOT run the prereg, does NOT rebuild outcomes, does NOT add
  the NYSE-holiday source, does NOT promote anything to the allocator.

  Scope is wider than the superseded 2026-05-25 stage file's "dst.py only"
  framing: that framing CANNOT COMMIT (Drift Check #32 + the
  build_daily_features import guard both fail-closed unless ORB_LABELS and
  DST_CLEAN_SESSIONS move in the same change). The five canonical session
  lists are one atomic unit. User approved the widened scope 2026-05-27.
mode: IMPLEMENTATION
updated: 2026-05-27T00:00Z
agent: claude (opus 4.7)
supersedes: docs/runtime/stages/2026-05-25-nyse-preopen-session-build.md

scope_lock:
  - pipeline/dst.py
  - pipeline/init_db.py
  - pipeline/check_drift.py
  - pipeline/build_daily_features.py
  - tests/test_pipeline/test_dst.py
  - tests/test_pipeline/test_check_drift.py
  - tests/test_app_sync.py

# Scope expansion (2026-05-27, anti-creep disclosure): added
# pipeline/build_daily_features.py and tests/test_app_sync.py. Reason: the
# roster grew 12->13, which made a hardcoded "12 dynamic sessions" doc-comment
# (build_daily_features.py:7) and the EXPECTED_ORB_LABELS roster
# (test_app_sync.py:64) factually wrong. These are doc/test consequences of the
# canonical roster change, not new logic. build_daily_features.py edit is a
# comment-only reword to a count-free phrasing so it cannot go stale again.

## Blast Radius
- pipeline/dst.py — CANONICAL session catalog. Adds the 09:00-ET pre-open resolver (mirrors us_equity_open_brisbane:315 tz arithmetic EXACTLY, no hardcoded offsets), SESSION_CATALOG["NYSE_PREOPEN"] (break_group "us"), DST_CLEAN_SESSIONS membership, and DOW_MISALIGNED_SESSIONS["NYSE_PREOPEN"] = -1 (EST midnight-crossing parity with NYSE_OPEN). Every outcome/feature/execution path reads session timing from here — HIGH blast. Pure timing definition: NO data write, NO schema DDL.
- pipeline/init_db.py — adds "NYSE_PREOPEN" to ORB_LABELS_DYNAMIC. This auto-generates daily_features schema columns via the existing `for label in ORB_LABELS` loops (no hardcoded column names exist — verified by empty grep for orb_NYSE_OPEN_5). Satisfies Drift Check #32 mirror.
- pipeline/check_drift.py — adds a NEW class-invariant check: every dynamic SESSION_CATALOG session must be in exactly one of DOW_ALIGNED_SESSIONS / DOW_MISALIGNED_SESSIONS. Closes the currently-uncovered DOW coupling (drift #32 covers ORB_LABELS↔catalog; build_daily_features:119 import guard covers DST_CLEAN; DOW was enforced by convention only). Future-proofs every session added after NYSE_PREOPEN.
- tests/test_pipeline/test_dst.py — resolver DST-season values, EST midnight date-bump, fail-closed in-window guard, adjacency (no overlap with NYSE_OPEN, end-exclusive abutment with US_DATA_830 O30), DOW guard raises, validate_catalog still passes.
- tests/test_pipeline/test_check_drift.py — known-violation injection test for the new DOW-classification check (Rule 11: verify the check actually catches what it claims).
- Reads: none new. Writes: none (no DB write, no migration). daily_features columns only materialize when a LATER rebuild stage runs run_rebuild_with_sync.sh MNQ.
- trading_app/portfolio.py derives valid_sessions = set(SESSION_CATALOG.keys()) at load — auto-propagates, NO edit required.
- CRIT-path canonical edit (pipeline/dst.py) -> adversarial-audit gate fires after this stage (evidence-auditor, independent context) BEFORE any Stage 2 dispatch.
---

# Lane B — NYSE_PREOPEN session build, Stage 1: atomic session definition + DOW hardening

## Why this, not a quick promotion (grounded decision)

Ground-truth 2026-05-25 (carried into 2026-05-27): all 14 PASS_CHORDIA + 3
PASS_PROTOCOL_A edges are already allocated (4 LIVE, 10 DISPLACED on the
correlation gate, 3 PAUSED on C8 OOS) — ZERO unblocked slots. MGC/MES
list_promotable_candidates are selection-biased sample frames that collapse
under strict IS/OOS split (MGC_LONDON_METALS +0.733R IS -> +0.067R OOS) and
would likely PARK on FAILED_RATIO, burning MinBTL budget. NYSE_PREOPEN is a
genuinely NEW session (never tested), so it is honest additive territory.
The prereg makes NO theory grant — strict t>=3.79 applies — and this stage
builds the session REGARDLESS of the eventual verdict (no bias toward "passing").

## The atomic five-list lock-step (the real Stage 1 unit)

A new dynamic session is defined by five canonical lists that MUST move
together. The original 2026-05-25 stage file scoped only the first; that
cannot commit. Verified fail-closed paths:
- Drift Check #32 (check_drift.py:2359) BLOCKS if SESSION_CATALOG has a dynamic
  session not in ORB_LABELS, or vice versa.
- build_daily_features.py:119-123 RAISES at import if an ORB label is in
  neither DST_AFFECTED_SESSIONS nor DST_CLEAN_SESSIONS.
- DOW classification (DOW_ALIGNED_SESSIONS / DOW_MISALIGNED_SESSIONS) had NO
  mechanical check — convention only. THIS STAGE adds one.

| List | File | NYSE_PREOPEN value |
|---|---|---|
| resolver fn | dst.py | new fn mirroring us_equity_open_brisbane (09:00 ET) |
| SESSION_CATALOG | dst.py | {type:dynamic, resolver, break_group:"us", event:"NYSE Order Imbalance publication 9:00 AM ET"} |
| DST_CLEAN_SESSIONS | dst.py | member (all dynamic sessions are clean by construction) |
| DOW_MISALIGNED_SESSIONS | dst.py | -1 (EST: Brisbane-Fri 00:00 = US-Thu 09:00 ET) |
| ORB_LABELS_DYNAMIC | init_db.py | "NYSE_PREOPEN" |

## DST behavior (grounded in orb_utc_window:602 date-bump)

The resolver returns only (hour, minute) Brisbane; orb_utc_window handles the
calendar-date bump. Mirrors NYSE_OPEN exactly — a PROVEN path, not invented:
- Summer EDT: 09:00 ET = 13:00 UTC = 23:00 Brisbane, SAME calendar day (hour>=9, no bump).
- Winter EST: 09:00 ET = 14:00 UTC = 00:00 Brisbane, NEXT calendar day (hour<9 -> cal_date+1).
The EST midnight-crossing is also WHY the DOW offset is -1 (identical reasoning
to NYSE_OPEN, dst.py:198-207).

## Ordered implementation steps

1. pipeline/dst.py — add the 09:00-ET resolver near us_equity_open_brisbane
   (line 315). Build a 09:00 US/Eastern aware datetime, convert to Brisbane,
   return (hour, minute). Docstring states BOTH regimes. NO hardcoded offsets.
2. pipeline/dst.py — register SESSION_CATALOG["NYSE_PREOPEN"] (break_group "us").
3. pipeline/dst.py — add "NYSE_PREOPEN" to DST_CLEAN_SESSIONS.
4. pipeline/dst.py — add DOW_MISALIGNED_SESSIONS["NYSE_PREOPEN"] = -1 and extend
   the DOW comment block (mirror the NYSE_OPEN explanation). NOT in DOW_ALIGNED.
5. pipeline/init_db.py — add "NYSE_PREOPEN" to ORB_LABELS_DYNAMIC.
6. pipeline/check_drift.py — add check_dow_classification_complete: every dynamic
   SESSION_CATALOG session in exactly one of DOW_ALIGNED_SESSIONS /
   DOW_MISALIGNED_SESSIONS (not zero, not both). Register in the checks tuple
   list. Pattern: mirror check_orb_labels_session_catalog_sync (check #32).
7. tests/test_pipeline/test_dst.py — resolver values both regimes; orb_utc_window
   5/15/30 both regimes; EST next-cal-day; in-window fail-closed guard satisfied;
   adjacency (NYSE_PREOPEN end <= NYSE_OPEN start; US_DATA_830 O30 end == NYSE_PREOPEN
   O-start, end-exclusive); validate_dow_filter_alignment raises for NYSE_PREOPEN;
   validate_catalog() passes.
8. tests/test_pipeline/test_check_drift.py — inject a SESSION_CATALOG session
   absent from both DOW sets, assert the new check returns a violation; assert
   clean catalog returns none.

## Acceptance (all required before deleting this stage file)

- tests/test_pipeline/test_dst.py + test_check_drift.py PASS — show output.
- python pipeline/check_drift.py PASSES (Check #32 + the new DOW check + the
  build_daily_features import guard all green together = atomic proof).
- dead-code sweep: grep confirms no orphaned half-edit (no list with NYSE_PREOPEN
  missing from another).
- self-review (line citations, not narrative).
- THEN STOP: dispatch the adversarial-audit gate (evidence-auditor, independent
  context) per adversarial-audit-gate.md — pipeline/dst.py is CRIT-path. Report
  verdict. Do NOT start Stage 2 until the audit returns.

## NOT done by this stage (downstream, gated)

- Stage 2: pipeline/calendar_filters.py is_nyse_holiday backed by a COMMITTED
  canonical holiday source (currently only is_nfp_day:29 exists). Holiday-bar
  contamination is an explicit prereg kill criterion — runner must fail-closed
  on missing source.
- Stage 3 (no code): run_rebuild_with_sync.sh MNQ — populate orb_outcomes +
  daily_features for NYSE_PREOPEN. Capture IS/OOS day counts; confirm DST-imbalance
  kill floor (N_EST>=30 AND N_EDT>=30) and the draft's N=27 budget hold.
  STAGE 3 BLOCKER (found during Stage 1 sweep): pipeline/session_guard.py
  _SESSION_ORDER (line 38) is a DUPLICATED chronological roster that FAILS-CLOSED
  (raise ValueError "Unknown session") for any session not listed. It does NOT
  break now (the guard is only invoked once daily_features columns exist), but it
  WILL raise the moment NYSE_PREOPEN features are built. Stage 3 must insert
  "NYSE_PREOPEN" between US_DATA_830 (~23:30) and NYSE_OPEN (~00:30) in
  _SESSION_ORDER AND add its _WINDOW_FEATURES safe-after entries (look-ahead
  validity — backtesting-methodology RULE 1). NOTE: the file's stated
  source-of-truth pointer "trading_app/ml/config.py SESSION_CHRONOLOGICAL_ORDER"
  is STALE — grep finds no such constant; the list is now a free-standing
  duplicate. A parity check (or single-source derivation from SESSION_CATALOG
  ordered by resolved Brisbane time) is a future-hardening candidate, NOT this
  stage's scope (look-ahead-correctness path, feature-builder territory).
- NOT a blocker / correctly omitted: trading_app/deployability.py
  MNQ_ROUTINE_TBBO_SLIPPAGE_SESSIONS is a frozenset of the 9 sessions where MNQ
  slippage was empirically MEASURED. NYSE_PREOPEN has ZERO measured slippage —
  adding it would be a lie about evidence (institutional honesty violation).
  It stays out until a real slippage pilot measures it (post-Stage-4).
- Stage 4 (research, gated on 1-3 + live-engine audit): promote draft out of
  drafts/ quarantine, run K=27 strict Chordia. PASS_CHORDIA -> eligible next
  allocator rebalance. NO promotion before runner produces a verdict MD.
- No allocator/sizing/deployment change. No write to validated_setups. Draft
  stays in drafts/ quarantine until Stage 4.

## Honesty / grounding ledger (user mandate 2026-05-27)

- Resolver grounded in sibling us_equity_open_brisbane, not training memory.
- Adjacency CLEAN_GAP claim from the draft is RE-VERIFIED mechanically in a test,
  not trusted as prose (Rule 11: never trust metadata).
- New drift check verified by known-violation injection (Rule 11).
- No tuning toward a hoped-for prereg result; strict t>=3.79 NO_THEORY_GRANT bar
  untouched; session built regardless of verdict.
- Hardening is a real class invariant (DOW coupling was uncovered), not
  speculative gold-plating — passes the n>=1-latent-gap test, not a hypothetical.
