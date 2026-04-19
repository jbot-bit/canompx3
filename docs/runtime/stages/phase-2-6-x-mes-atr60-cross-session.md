---
mode: IMPLEMENTATION
task: Phase 2.6 — X_MES_ATR60 cross-session extension (K=6)
slug: phase-2-6-x-mes-atr60-cross-session
branch: research/campaign-2026-04-19-phase-2
updated: 2026-04-19
origin: Phase 2.5 Tier-1 unlock (commit 051c2851) — X_MES_ATR60 is 3 of 9 PASS lanes
pre_reg: docs/audit/hypotheses/2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml (to be locked from stub)
---

## Why

X_MES_ATR60 is a Tier-1 filter class in Phase 2.5 (passes Chordia on
COMEX_SETTLE both RRs + CME_PRECLOSE RR1.0). Untested on US_DATA_830 and
CME_PRECLOSE RR1.5/2.0. Theory (Chan Ch 7 + Carver Ch 9-10) supports
extension to US-session cells where MES is actively trading.

## Scope

K=6 cells, all MNQ E2 O5 CB1 long with X_MES_ATR60 filter:
1. CME_PRECLOSE RR1.5
2. CME_PRECLOSE RR2.0
3. US_DATA_830 RR1.0
4. US_DATA_830 RR1.5
5. US_DATA_830 RR2.0
6. US_DATA_1000 RR2.0

## Scope lock

- `docs/audit/hypotheses/2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml` (lock from stub)
- `research/phase_2_6_x_mes_atr60_cross_session_audit.py` (NEW)
- `tests/test_research/test_phase_2_6_x_mes_atr60_cross_session_audit.py` (NEW)
- `docs/audit/results/2026-04-19-x-mes-atr60-cross-session-audit.md` (NEW)
- delete: `docs/audit/hypotheses/2026-04-19-x-mes-atr60-cross-session-extension-stub.md` (after lock)

## Blast radius

Pure research layer, read-only. No production / schema / validator changes.

## Canonical delegations

- `compute_mode_a`, `C4_T_WITH_THEORY`, `C7_MIN_N`, `C9_*` from `research.mode_a_revalidation_active_setups`
- `filter_signal` via compute_mode_a (delegates to ALL_FILTERS["X_MES_ATR60"])
- `HOLDOUT_SACRED_FROM` from `trading_app.holdout_policy`
- `GOLD_DB_PATH` from `pipeline.paths`
- `SESSION_CATALOG` whitelist from `pipeline.dst`

## Kill criteria (per cell)

- C3 p < 0.05 at K=1 per-cell (OR BH-FDR at K=6 family if stricter)
- C4 Chordia t ≥ 3.00 (with-theory)
- C6 WFE ≥ 0.50
- C7 N ≥ 100
- C9 no year N≥50 with ExpR < −0.05
- T8 direction consistency (long vs short on same cell)

## Acceptance

1. pre-reg yaml locked and committed BEFORE script runs (C1)
2. Script runs 6 cells end-to-end, reports per-cell pass/fail per criterion
3. Result doc shows per-cell verdict table + family BH-FDR at K=6
4. Tests green (canonical delegation + loop correctness + smoke test)
5. Commit + push
