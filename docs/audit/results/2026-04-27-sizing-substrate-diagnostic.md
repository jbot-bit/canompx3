# Sizing-Substrate Diagnostic — Result

- Design doc: `docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md`
- Pre-reg: `docs\audit\hypotheses\2026-04-27-sizing-substrate-prereg.yaml`
- Git HEAD: `9d63ea4c35d9b68d4fd8c68d5568d8cc1e7f09f6`
- DB schema fingerprint: `27885f79609cd9b7d4ffc45b50d2e5ab`
- Bootstrap seed: 42; B=10000
- K = 48
- **VERDICT: SUBSTRATE_WEAK**
- Lanes with substrate: ['MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5', 'MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12']

## Scope / Question

Stage-1 falsifier of the thesis that deployed binary filters in the live
`topstep_50k_mnq_auto` 6-lane portfolio have continuous substrate justifying
a Carver-style forecast→sizing layer. K=48 cells = 6 deployed lanes ×
(3 Tier-A substrate forms + 5 Tier-B orthogonal continuous features).
Pass criterion (per cell, ALL gates): |ρ|≥0.10, monotonic Q1→Q5, |Q5−Q1|≥0.20R,
sized-vs-flat 95% CI > 0, BH-FDR survives at q=0.05, split-half sign match,
ex-ante prediction sign match. Lane has substrate iff ≥1 cell passes.
Substrate confirmed globally iff ≥3 lanes pass.

## Verdict / Decision

**SUBSTRATE_WEAK.** 2 of 6 lanes have substrate.
Decision: park sizing thesis. Pre-reg requires ≥3 lanes for global confirmation.
Possible single-lane Stage-2 study only under a fresh pre-reg with
explicit mechanism citation per lane.

## Reproduction / Outputs

- Script: `research/audit_sizing_substrate_diagnostic.py` at git HEAD above
- Pre-reg: `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`
- Bootstrap: numpy default_rng seed=42, B=10000
- DB: `pipeline.paths.GOLD_DB_PATH`, schema fingerprint `27885f79609cd9b7d4ffc45b50d2e5ab`
- Re-run: `python research/audit_sizing_substrate_diagnostic.py` (read-only;
  raises RuntimeError on any trading_day ≥ 2026-01-01)
- Tests: `pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py`
- Output JSON twin: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json`

## Caveats / Limitations / Disconfirming Considerations

- **Selection-bias compounding (per spec §6):** the 6 lanes are themselves
  survivors of an earlier, larger trial space. K=48 BH-FDR controls only the
  new diagnostic family; it does NOT deflate prior lane-discovery multiplicity.
  Stage-2 deployment must apply DSR with cumulative trial count.
- **Lookahead-validity gate enforcement:** cells using `overnight_*` features
  on TOKYO_OPEN/SINGAPORE_OPEN (sessions starting <17:00 Brisbane) are marked
  INVALID per `.claude/rules/backtesting-methodology.md` RULE 1.2. The
  unguarded run had additional apparent passes that disappeared under the gate.
- **Linear-rank weights {0.6, 0.8, 1.0, 1.2, 1.4} are diagnostic-only,**
  NOT Carver's actual recipe (Ch. 7 forecast scalar to abs-mean=10, cap=±20).
  Stage-2 must implement the canonical Carver scaling, not the rank proxy.
- **Per-cell bootstrap underestimates serial dependence.** Acceptable for Stage 1;
  Stage 2 must consider block bootstrap if substrate were confirmed.
- **AFML Ch. 19 sigmoid bet-sizer is NOT in `resources/`.** Sigmoid functional
  form deferred. Stage 2 (if any) defaults to Carver-only sizing.
- **Single-pass discipline.** Re-running with different feature lists, weight
  schemas, thresholds, or expanded K is a NEW pre-reg, not a re-run.

## Per-cell results

| lane | tier | feature/form | n | rho | p | bh-fdr | Q5-Q1 R | mono | delta CI | split | stable | pred | real | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | raw | 1718 | +0.248 | 0.0000 | Y | +0.181 | N | [+0.007, +0.036] | Y | U | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | vol_norm | 1657 | +0.144 | 0.0000 | Y | +0.307 | N | [+0.010, +0.039] | Y | U | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | rank_252d | 1656 | +0.200 | 0.0000 | Y | +0.089 | N | [-0.003, +0.027] | Y | S | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | rel_vol_session | 1712 | +0.129 | 0.0000 | Y | +0.276 | Y | [+0.009, +0.038] | Y | U | + | + | **PASS** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | overnight_range_pct | 1698 | +0.135 | 0.0000 | Y | +0.207 | N | [+0.009, +0.039] | Y | S | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | atr_vel_ratio | 1713 | +0.056 | 0.0206 | Y | +0.138 | N | [-0.011, +0.019] | Y | U | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | garch_forecast_vol_pct | 1446 | +0.099 | 0.0002 | Y | +0.020 | N | [-0.013, +0.021] | N | U | + | + | **FAIL** |
| MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | pit_range_atr | 1695 | +0.098 | 0.0001 | Y | +0.169 | Y | [+0.002, +0.032] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_a | raw | 1717 | +0.143 | 0.0000 | Y | +0.089 | N | [+0.003, +0.033] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_a | vol_norm | 1717 | +0.143 | 0.0000 | Y | +0.089 | N | [+0.003, +0.033] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_a | rank_252d | 1655 | +0.065 | 0.0081 | Y | +0.167 | N | [-0.007, +0.024] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_b | rel_vol_session | 1713 | +0.065 | 0.0072 | Y | +0.127 | N | [-0.010, +0.020] | N | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_b | overnight_range_pct | 0 | +0.000 | 1.0000 | N | +0.000 | N | [+0.000, +0.000] | N | n | + | ? | **INVALID** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_b | atr_vel_ratio | 1714 | +0.062 | 0.0098 | Y | +0.153 | N | [-0.007, +0.023] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_b | garch_forecast_vol_pct | 1445 | +0.140 | 0.0000 | Y | +0.183 | N | [+0.004, +0.036] | Y | U | + | + | **FAIL** |
| MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50 | tier_b | pit_range_atr | 1696 | +0.073 | 0.0025 | Y | +0.089 | N | [-0.010, +0.020] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | raw | 1636 | +0.278 | 0.0000 | Y | +0.376 | N | [+0.011, +0.042] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | vol_norm | 1581 | +0.113 | 0.0000 | Y | +0.093 | N | [-0.006, +0.025] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_a | rank_252d | 1574 | +0.215 | 0.0000 | Y | +0.177 | N | [-0.002, +0.030] | Y | S | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | rel_vol_session | 1631 | +0.148 | 0.0000 | Y | +0.270 | N | [+0.014, +0.046] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | overnight_range_pct | 1617 | +0.080 | 0.0013 | Y | +0.112 | N | [-0.007, +0.024] | Y | S | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | atr_vel_ratio | 1631 | +0.029 | 0.2439 | N | -0.020 | N | [-0.021, +0.010] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | garch_forecast_vol_pct | 1376 | +0.140 | 0.0000 | Y | +0.224 | N | [-0.000, +0.034] | Y | U | + | + | **FAIL** |
| MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5 | tier_b | pit_range_atr | 1618 | +0.041 | 0.0986 | N | -0.003 | N | [-0.009, +0.023] | Y | U | + | + | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_a | raw | 1693 | -0.324 | 0.0000 | Y | -0.061 | N | [-0.005, +0.021] | Y | U | - | - | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_a | vol_norm | 1633 | -0.224 | 0.0000 | Y | -0.015 | N | [-0.006, +0.021] | N | U | - | - | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_a | rank_252d | 1631 | -0.236 | 0.0000 | Y | -0.005 | N | [-0.012, +0.014] | N | S | - | - | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_b | rel_vol_session | 1688 | +0.045 | 0.0642 | N | -0.038 | N | [-0.017, +0.009] | Y | U | + | + | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_b | overnight_range_pct | 1673 | +0.075 | 0.0022 | Y | -0.105 | N | [-0.020, +0.006] | Y | S | + | + | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_b | atr_vel_ratio | 1688 | +0.062 | 0.0111 | Y | -0.059 | N | [-0.016, +0.010] | N | U | + | + | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_b | garch_forecast_vol_pct | 1425 | +0.116 | 0.0000 | Y | -0.020 | N | [-0.018, +0.010] | N | U | + | + | **FAIL** |
| MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12 | tier_b | pit_range_atr | 1672 | +0.015 | 0.5297 | N | -0.088 | N | [-0.019, +0.007] | N | U | + | + | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_a | raw | 1722 | -0.262 | 0.0000 | Y | -0.247 | N | [+0.007, +0.036] | Y | U | - | - | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_a | vol_norm | 1661 | -0.207 | 0.0000 | Y | -0.344 | N | [+0.009, +0.040] | Y | U | - | - | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_a | rank_252d | 1660 | -0.239 | 0.0000 | Y | -0.195 | N | [+0.004, +0.034] | Y | S | - | - | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_b | rel_vol_session | 1716 | +0.174 | 0.0000 | Y | +0.419 | Y | [+0.024, +0.053] | Y | U | + | + | **PASS** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_b | overnight_range_pct | 0 | +0.000 | 1.0000 | N | +0.000 | N | [+0.000, +0.000] | N | n | + | ? | **INVALID** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_b | atr_vel_ratio | 1717 | +0.092 | 0.0001 | Y | +0.247 | N | [+0.005, +0.034] | Y | U | + | + | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_b | garch_forecast_vol_pct | 1447 | +0.132 | 0.0000 | Y | +0.187 | N | [+0.001, +0.034] | Y | U | + | + | **FAIL** |
| MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12 | tier_b | pit_range_atr | 1699 | +0.113 | 0.0000 | Y | +0.174 | N | [+0.000, +0.030] | Y | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_a | raw | 1495 | +0.172 | 0.0000 | Y | -0.135 | N | [-0.029, +0.005] | Y | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_a | vol_norm | 1443 | +0.050 | 0.0553 | N | +0.048 | N | [-0.025, +0.010] | N | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_a | rank_252d | 1433 | +0.145 | 0.0000 | Y | -0.116 | N | [-0.025, +0.011] | Y | S | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_b | rel_vol_session | 1490 | +0.066 | 0.0105 | Y | +0.061 | N | [-0.005, +0.029] | Y | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_b | overnight_range_pct | 1476 | +0.069 | 0.0081 | Y | +0.064 | N | [-0.015, +0.021] | N | S | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_b | atr_vel_ratio | 1490 | +0.023 | 0.3682 | N | -0.065 | N | [-0.025, +0.010] | Y | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_b | garch_forecast_vol_pct | 1253 | +0.083 | 0.0035 | Y | +0.070 | N | [-0.022, +0.016] | N | U | + | + | **FAIL** |
| MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5 | tier_b | pit_range_atr | 1476 | +0.009 | 0.7203 | N | -0.037 | N | [-0.021, +0.014] | N | U | + | + | **FAIL** |
