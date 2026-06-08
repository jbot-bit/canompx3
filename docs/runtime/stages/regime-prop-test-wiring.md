---
task: Wire 3 REGIME-tier lanes (3 independent sessions) onto a NEW dedicated express profile (default-OFF) as a zero-evidence forward live-test
slug: regime-prop-test-wiring
mode: IMPLEMENTATION
status: BUILD_COMPLETE_AWAITING_OPERATOR_DIFF_REVIEW
created: 2026-06-08
capital_path: true
worktree: C:/Users/joshd/canompx3-regime-prop-test (branch session/joshd-regime-trim, off origin/main a6eaaef5)
design_gate: PRESENTED + OPERATOR-APPROVED 2026-06-08 ("Build it, show me the diff, I approve before live")
build_note: FINAL SLATE = 3 lanes / 3 INDEPENDENT sessions (one best lane per session, after measuring intra-session correlation). MNQ COMEX_SETTLE E2 RR2.0 (ExpR 0.617, cap 50.1) + MNQ CME_PRECLOSE X_MES_ATR70 E2 RR1.0 O15 (ExpR 0.371, cap 99.2) + MGC LONDON_METALS E2 RR2.0 (ExpR 0.451, cap 25.8). DROPPED same-session redundant variants (MNQ COMEX RR1.5/RR2.5 rho~0.83; MGC LONDON E2RR1.5/E1RR1.5 rho~0.90) + MES N=16. Caps = canonical P90 per-aperture (O15 cap 99.2 != O5 50.1). Worst-case 1ct ~$557 < $2000 MLL. Default-OFF = active=False (hard fail-closed arm gate). NOTHING ARMED.
---

# Stage 1 — REGIME prop-test wiring

## Operator decisions (LOCKED 2026-06-08, this session)
- Deploy REGIME standalone DESPITE the "never REGIME standalone" doctrine —
  operator explicit, express-funded only (firm absorbs DD), accepts ZERO forward
  evidence (shadow pipeline recorded 0 trades; predicate never built).
- ALL 7 lanes on ONE new dedicated profile `topstep_50k_regime_test`
  (multi-instrument). NOT spread across 3 accounts (earlier plan superseded).
- Lanes ship **default-OFF** (via `lane_ctl.pause_strategy_id`) — nothing trades
  until operator clicks each ON from the dashboard `/api/lane-control/toggle`.
- NO new account purchase. Bind to an existing $50k/$100k express account when
  arming. (100k-vs-150k decision DEFERRED until forward evidence exists — do not
  buy to test a no-evidence hypothesis.)
- self_funded_tradovate (real capital, is_express_funded=False) — NEVER touched.

## The slate (7 lanes — all FIT, validated sample 30-99, from regime_shadow_universe.yaml)
MNQ (COMEX_SETTLE, orb 5m, ORB_VOL_16K):
  - MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K   (val_N=87, roll_ExpR=0.617)
  - MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K   (val_N=90, roll_ExpR=0.546)
  - MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K   (val_N=81, roll_ExpR=0.504)
MGC (LONDON_METALS, orb 30m, ORB_VOL_8K):
  - MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30 (val_N=45, roll_ExpR=0.451)
  - MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30 (val_N=48, roll_ExpR=0.416)
  - MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30 (val_N=46, roll_ExpR=0.319)
MES (CME_PRECLOSE, orb 30m, ATR_P30):
  - MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30   (val_N=37, roll_ExpR=0.412)

## CRITICAL build input — risk caps MUST be computed, never guessed
Each DailyLaneSpec needs `max_orb_size_pts` (load-bearing: account_survival.py:439
and session_orchestrator.py:395 read it as the per-lane size guard — a None cap =
NO size guard on real capital, test MUST assert non-None for all 7).
DERIVE from canonical `orb_size_stats` (lane_allocator.py:1264), keyed on
(instrument, orb_label, orb_minutes) -> (avg_orb_pts, p90_orb_pts). Deployed lanes
show the cap sits between avg and p90 (e.g. avg=29.1, p90=49.8, cap=37.35). Use the
SAME method the allocator uses — do not invent a formula. ("measure, don't assume"
— [[feedback_measure_dd_scaling_dont_assume_linear_2026_06_07]].)

## scope_lock (files this stage MAY touch)
- trading_app/prop_profiles.py        # NEW profile topstep_50k_regime_test + 7 DailyLaneSpec (Tier B canonical)
- scripts/tools/<compute_regime_caps>.py  # one-shot: compute max_orb_size_pts from orb_size_stats
- tests/test_trading_app/test_regime_prop_test_profile.py  # 7 lanes parse, caps non-None, required_fitness=FIT, correct instrument
- docs/runtime/stages/regime-prop-test-wiring.md

## FORBIDDEN (hard scope wall)
- NO edit to lane_allocator.py / account_survival.py / session_orchestrator.py /
  strategy_fitness.py / dst.py / cost_model.py / asset_configs.py.
- NO edit to any OTHER profile (topstep_50k MGC 1-contract cap, _mnq_auto $450
  daily-loss all stay untouched).
- NO touch to self_funded_tradovate.
- NO live arm. NO broker account-id hardcoded (bind at arm time).
- Lanes default-OFF — do NOT ship them resumed/active.

## Blast radius (verified this session)
- prop_profiles.py read by allocator, session_orchestrator, account_survival,
  derived_state (C11/C12 code-fingerprint), bot_dashboard. Adding a profile is
  additive but CHANGES the code fingerprint -> the 3 fingerprint registries
  (_criterion11/_lane/_sr/_overlay per [[project_code_fingerprint_staleness_class_of_four_2026_06_07]])
  may need the new profile counted; C11 survival report for the new profile must
  be generated before arm (or arm fails closed = SAFE direction).
- Dashboard toggle works FOR FREE: /api/lane-control/toggle reads
  effective_daily_lanes(profile); refuses while a session runs (409). REGIME lanes
  become toggleable the moment they're in the profile.
- required_fitness=('FIT',) -> allocator auto-pauses a decaying lane. No manual gate.

## Acceptance (Stage 1 done = all true, shown evidence)
1. New profile topstep_50k_regime_test exists, is_express_funded=True, holds all 7
   lanes; multi-instrument (MNQ+MGC+MES) resolves in effective_daily_lanes. Show repr.
2. All 7 max_orb_size_pts non-None and within [avg_orb_pts, p90_orb_pts] band. Show the computed table.
3. Lanes default-OFF (lane_ctl shows all 7 paused/disabled on a fresh profile). Show.
4. Tests pass (parse, caps, fitness, instrument-match). Show pytest output.
5. NO forbidden file touched (git diff --name-only). Show.
6. check_drift.py passes (record count; if fingerprint check flags the new profile,
   resolve per the class-of-four pattern, do not band-aid). Show.
7. py_compile + scoped ruff clean. git diff --check clean.
8. Self-review: dead-code sweep, simulate happy/decay/None-cap/fingerprint-stale.

## Then: SHOW THE FULL DIFF to operator. NOTHING arms. Operator runs live arm themselves.

## Verification order (narrowest first)
compute caps -> wire profile -> unit tests -> py_compile -> ruff -> git diff --check
-> drift (resolve fingerprint if flagged) -> self-review -> present diff.
