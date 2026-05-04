# 2026-04-15 Session Handover — Volume Finding Stress-Tested

**Status at handoff:** 13 commits pushed to `origin/main`. Context ~60% used. Fresh terminal ready for next task.

---

## TL;DR

- **rel_vol_HIGH_Q3 finding is EDGE_WITH_CAVEAT** — real edge, modest effect size, deploy-or-wait decision depends on N_eff assumption
- **Look-ahead free + repeatable** (verified `pipeline/build_daily_features.py:1326-1380`)
- **Not deployed anywhere yet** — research-only this session
- **Multiple horizon filters untested** at T0-T8 (overnight range, vol regime, calendar, ovn_took_pdh) — clear next-session candidates
- **v1 stress test was over-punishing** — v2 corrected (empirical var_sr, DSR labeled informational not hard gate)

---

## The 5 lanes in question

| Lane | CORE gates | DSR realistic K | Per-trade SR | Verdict |
|------|-----------|-----------------|--------------|---------|
| MES COMEX_SETTLE O5 RR1.0 short | 4/4 | 0.02 @ K=36 | +0.094 | EDGE_WITH_CAVEAT |
| MGC LONDON_METALS O5 RR1.0 short | 4/4 | 0.19 @ K=36 | +0.160 | EDGE_WITH_CAVEAT (strongest) |
| MES TOKYO_OPEN O5 RR1.5 long | 4/4 | 0.005 @ K=36 | +0.082 | EDGE_WITH_CAVEAT |
| MNQ SINGAPORE_OPEN O5 RR1.0 short | 4/4 | 0.13 @ K=36 | +0.166 | EDGE_WITH_CAVEAT |
| MES COMEX_SETTLE O5 RR1.5 short | 3/4 | 0.002 @ K=36 | +0.053 | EDGE_WITH_CAVEAT |

CORE gates all passed: block bootstrap (autocorrelation-robust) p=0.0005, temporal split both halves sign-match + |t|≥2, |t| exceeds E[max_t from K=14261 noise], per-day aggregated t significant.

DSR informational (dsr.py line 35 explicit): fails at K≥36, passes partially at K=5 (lane-level).

---

## Next concrete actions (priority-ordered for fresh terminal)

### Tier 1 — Horizon audit (untested non-volume filters)
T0-T8 these NEW candidates from comprehensive scan (|t|≥3, dir_match, not yet audited):
1. **MES LONDON_METALS O30 RR1.5 long `ovn_range_pct_GT80`** (t=+3.54, Δ_OOS=+0.690) — overnight-vol signal, orthogonal to rel_vol
2. **MNQ COMEX_SETTLE O5 RR1.0 long `garch_vol_pct_GT70`** (t=+3.18, Δ_OOS=+0.236) — forward vol forecast
3. **MNQ BRISBANE_1025 O30 RR2.0 long `is_monday`** (t=+3.27, Δ_OOS=+0.854) — Monday effect
4. **MNQ COMEX_SETTLE O15 RR1.0 short `dow_thu`** (t=+3.27, Δ_OOS=+0.552) — Thursday effect
5. **MES COMEX_SETTLE O30 RR1.0 long `ovn_took_pdh_LONG_INTERACT`** (t=-3.82) — SKIP signal

Script template: copy `research/t0_t8_audit_mgc_level_cells.py`, adapt feature_sql.

### Tier 2 — Composite with orthogonal signals
Build `rel_vol_HIGH × ovn_range_pct_GT80` composite — orthogonal mechanisms may multiply edge. Per-trade Sharpe from 0.17 to 0.25+ would clear DSR at K=36.

### Tier 3 — Cross-RR family audit of deployed lanes
Stage R-1 from `docs/institutional/regime-and-rr-handling-framework.md` — retroactive audit of 6 deployed MNQ lanes to check all 3 RRs pass core T0-T8 criteria. Any SINGLE_RR_ISOLATED cells get demoted to research-provisional.

### Tier 4 — Phase D pilot start
Spec exists at `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`. D-0 stage is 1-week backtest on MNQ COMEX_SETTLE with discrete size-scaling. Can start anytime after Tier 1/2/3.

---

## Key files tonight

### Research scripts
- `research/comprehensive_deployed_lane_scan.py` — 324-combo scan, canonical helpers
- `research/volume_confluence_scan.py` — 2-factor AND confluence
- `research/t0_t8_audit_volume_cells.py` — V1-V4 + M1-M4 volume cells
- `research/mgc_level_scan.py` — MGC-only level features
- `research/t0_t8_audit_mgc_level_cells.py` — MGC level T0-T8 including cross-RR
- `research/rel_vol_mechanism_decomposition.py` — partial-dependence + multivar regression
- `research/stress_test_rel_vol_finding_v2.py` — corrected DSR methodology

### Results
- `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` — 14,261 cells, 13 BH-global
- `docs/audit/results/2026-04-15-rel-vol-stress-test-v2.md` — corrected stress test
- `docs/audit/results/2026-04-15-rel-vol-mechanism-decomposition.md` — independence check
- `docs/audit/results/2026-04-15-mgc-level-scan.md` — MGC level universe

### Frameworks + rules
- `.claude/rules/backtesting-methodology.md` — 13 mandatory rules
- `docs/institutional/edge-finding-playbook.md` — 12 commandments, niche-hunting ladder
- `docs/institutional/regime-and-rr-handling-framework.md` — Stages R-1 through R-5
- `docs/institutional/literature/chan_2008_ch7_regime_switching.md` — canonical regime extract

### Pre-regs
- `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md`
- `docs/audit/hypotheses/2026-04-15-volume-exploitation-research-plan.md`
- `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`
- `docs/audit/hypotheses/phase-c-e-retest-entry-model.md`
- `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md`

---

## Critical cautions for next session

1. **Don't deploy rel_vol yet** — DSR caveat stands. Signal-only shadow or composite first.
2. **Use backtesting-methodology.md canonical helpers** — don't re-derive `_valid_session_features` or `bh_fdr_multi_framing`. Import.
3. **Aronson Ch 6 is Data-Mining Bias** not volume — don't cite it for volume again.
4. **day_type is LOOK-AHEAD** per `pipeline/build_daily_features.py:510`. Never use as feature for intraday signals.
5. **T8 cross-instrument twin is wrong for MGC** — code uses MNQ, should be same-asset-class (gold). Flag any MGC T8 result as methodologically suspect.
6. **DSR is INFORMATIONAL not a hard gate** per dsr.py line 35. Don't over-weight.
7. **N_eff for DSR is UNKNOWN** — report at multiple framings (5, 36, 72, 300, 14261).

---

## Commits tonight (12, all pushed)

```
456b4476 research: stress test v2 (self-audited) + horizon audit of non-volume filters
8d162b18 research: honest stress test of rel_vol_HIGH finding — MARGINAL at every lane
f6944d71 research: rel_vol_HIGH mechanism decomposition + Chan 2008 Ch 7 regime extract
efed6e24 research: comprehensive MGC level scan + T0-T8 on 4 new MGC level cells
cf1079ee research: T0-T8 on MGC volume cells — cross-instrument concordance for rel_vol
7181ec7a research: T0-T8 on 4 volume cells + Phase D pilot spec + edge-finding playbook
62fc7ff7 research: volume confluence scan — 1 strict survivor, mixed OOS
4271fb49 research: volume exploitation plan — institutional-grade alignment
152e74a1 research: full-universe scan (324 combos, 14K cells) + backtesting methodology rules
b1249062 research: institutional-grade comprehensive deployed-lane scan + phase stubs
35fe611d research: scoped mega on 6 deployed-lane (session,aperture) pairs × MNQ
e9667b77 research: T0-T8 batch audit of 27 mega-exploration cells + adversarial fade audit
```

All pushed to `origin/main`. No production code changed. No validated_setups writes. No live-trading disruption.

---

## rel_vol look-ahead / repeatability verification (confirmed)

`pipeline/build_daily_features.py:1326-1380`:
- Numerator: `break_bar_volume` — 1-minute bar volume at break timestamp (known at break time)
- Denominator: median of prior 20 days at same UTC minute-of-day (line 1337: "No look-ahead: history uses only bars before today")
- Matches VolumeFilter contract, verified 257/257 rows zero diff with independent enrichment path
- Deterministic, repeatable, ✓ production-grade

---

## One-line summary for next-session intro

> "Tonight validated volume signal as EDGE_WITH_CAVEAT (real, modest Sharpe). Horizon has untested ovn_range / garch_vol / calendar filters. Start with Tier 1 T0-T8 on those, then composite or Phase D."
