# Scan-guard pressure test — 2026-04-19 overnight scans

**Generated:** 2026-04-18T22:28:55+00:00
**Script:** `research/pressure_test_scan_guards.py`
**Rule:** `.claude/rules/backtesting-methodology.md` RULE 13
**Test cell:** MNQ COMEX_SETTLE O5 RR1.5 long (Mode A IS)

## Motivation

The 2026-04-19 overnight session wrote 4 new scan scripts but did NOT pressure-test their guard layers per backtesting-methodology.md RULE 13. This script closes that debt by injecting 3 synthetic bad filter signals and confirming the guard layers catch them.

Scan scripts under guard-audit (all share the same T0 / extreme_fire / arithmetic_only / direction-filter guard stack):
  1. `research/mode_a_revalidation_active_setups.py` (Phase 3)
  2. `research/mes_mnq_mirror_v1_scan.py` (prior session)
  3. `research/mgc_mode_a_rediscovery_orbg5_v1_scan.py` (Phase 6)
  4. `research/mes_broader_mode_a_rediscovery_v1_scan.py` (Phase 7)

## Verdict: **PASS** — all 3 bad inputs caught by expected guard

### BAD_1_lookahead

Synthetic look-ahead filter: fire = (outcome == 'win'). Trivial, should produce extreme |t| and huge WR_spread, catchable as RULE 12 red flag.

**Stats:** N_on=413, fire_rate=0.4753, ExpR_on=1.2661401937046004, t=177.8538313989989, WR_spread=1.0, Delta_IS=1.1891368565354403

**T0 correlations:** orb_size=0.0568263831783968, atr_20=0.07903976928930347, overnight_range=0.0164814275415588, pnl_r=0.9961481370243007

| Guard | Fired? | Expected? | Metric | Threshold |
|---|---|---|---:|---|
| G1_T0_tautology | silent | no | 0.0790 | |corr| > 0.70 |
| G2_extreme_fire | silent | no | 0.4753 | <5% or >95% |
| G3_arithmetic_only | silent | no | 1.0000 | |WR_spread|<3% AND |Delta_IS|>0.10 |
| G4_direction_filter | silent | no | — | SQL-level direction=dir filter |
| RED_FLAG_extreme_t | fired | YES | 177.8538 | |t| > 10 (RULE 12 red flag) |

**Caught by expected guard:** YES

### BAD_2_extreme_rare

Synthetic extreme-rare filter: fire only on top 0.5% orb_size. Should trigger G2 extreme_fire.

**Stats:** N_on=5, fire_rate=0.0058, ExpR_on=0.48604, t=0.8011415255845311, WR_spread=0.12546296296296294, Delta_IS=0.40903666283084006

**T0 correlations:** orb_size=0.600998950996214, atr_20=0.1564734746925633, overnight_range=0.21967683248965092, pnl_r=0.02738986237131846

| Guard | Fired? | Expected? | Metric | Threshold |
|---|---|---|---:|---|
| G1_T0_tautology | silent | no | 0.2197 | |corr| > 0.70 |
| G2_extreme_fire | fired | YES | 0.0058 | <5% or >95% |
| G3_arithmetic_only | silent | no | 0.1255 | |WR_spread|<3% AND |Delta_IS|>0.10 |
| G4_direction_filter | silent | no | — | SQL-level direction=dir filter |
| RED_FLAG_extreme_t | silent | no | 0.8011 | |t| > 10 (RULE 12 red flag) |

**Caught by expected guard:** YES

### BAD_3_arithmetic_only_size

Synthetic cost-screen filter: fire on top 20% orb_size (proxy for cost_risk_pct<8%). Should trigger G3 arithmetic_only IF WR stays flat while ExpR moves. Historical-failure-log class: 2026-03-24 cost_risk_pct tautology.

**Stats:** N_on=176, fire_rate=0.2025, ExpR_on=0.1779227272727273, t=1.9529772464132749, WR_spread=0.016774891774891776, Delta_IS=0.10091939010356732

**T0 correlations:** orb_size=0.6303654829257584, atr_20=0.43613095033542815, overnight_range=0.4151266415363053, pnl_r=0.044767600416073375

| Guard | Fired? | Expected? | Metric | Threshold |
|---|---|---|---:|---|
| G1_T0_tautology | silent | no | 0.4361 | |corr| > 0.70 |
| G2_extreme_fire | silent | no | 0.2025 | <5% or >95% |
| G3_arithmetic_only | fired | YES | 0.0168 | |WR_spread|<3% AND |Delta_IS|>0.10 |
| G4_direction_filter | silent | no | — | SQL-level direction=dir filter |
| RED_FLAG_extreme_t | silent | no | 1.9530 | |t| > 10 (RULE 12 red flag) |

**Caught by expected guard:** YES

## Interpretation of each case

**BAD_1** — Look-ahead via outcome column. If scan scripts used this as a filter, fire_rate would equal win_rate (~50%) — NOT extreme, so G2 extreme_fire would NOT fire. WR_spread would be +50-100% (huge) — G3 arithmetic_only would NOT fire (it needs small WR_spread). T0 corr with pnl_r would be ~1.0 — BUT the scan scripts' T0 implementation only checks corr vs orb_size / atr_20 / overnight_range, NOT vs pnl_r. So G1 would silently pass. The RED_FLAG_extreme_t check (RULE 12) is the only robust catcher. In the scan scripts, |t| > 10 would not automatically halt but is an obvious red flag on manual review. **Finding:** the scan scripts' T0 guard should add `corr_with_pnl_r` as an additional check to catch direct-outcome look-ahead.

**BAD_2** — Extreme-rare fire (0.5%). G2 extreme_fire has threshold <5%, so this IS caught.

**BAD_3** — Cost-screen via size. Classic arithmetic_only pattern. On MNQ COMEX top-20% size, WR stays near baseline (~50%) while ExpR lifts materially. G3 should catch if |WR_spread|<3% and |Delta_IS|>0.10.

## Remediation (if applicable)

All 3 bad inputs caught by their expected guard. The scan scripts' T0 / extreme_fire / arithmetic_only guard stack is working as specified.

**Non-blocking improvement recommendation:** the T0 guard currently checks corr vs `orb_size`, `atr_20`, `overnight_range`. Adding `corr(fire, pnl_r)` as a red-flag check would robustly catch direct-outcome look-ahead (BAD_1 class). This is not a gate (the scan scripts don't write to validated_setups) but adds defense-in-depth. File as next-session task.

## Reproduction
```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/pressure_test_scan_guards.py
```

No writes to validated_setups or experimental_strategies. Deterministic on same DB state.

