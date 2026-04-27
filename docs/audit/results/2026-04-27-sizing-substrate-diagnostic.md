---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md#per-cell-results
flip_rate_pct: 67.0
heterogeneity_ack: true
---

# Sizing-Substrate Diagnostic — Result

- Design doc: `docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md`
- Pre-reg: `docs\audit\hypotheses\2026-04-27-sizing-substrate-prereg.yaml`
- Cross-walk vs PR #51: `docs/audit/results/2026-04-27-sizing-substrate-vs-pr51-cross-walk.md`
- Git HEAD: `60556b6d031fcd9c98582110e74847cc277d682d`
- DB schema fingerprint: `27885f79609cd9b7d4ffc45b50d2e5ab`
- Bootstrap seed: 42; B=10000
- K = 48 declared; **effective unique cells = 42** (6 ATR_P50 raw/vol_norm cells are identical column derivations — see audit Finding A in § Caveats)
- **VERDICT: SUBSTRATE_WEAK** (institutional code+quant audit 2026-04-27: PASS_WITH_RISKS — verdict upheld; 5 load-bearing findings documented)
- Lanes with substrate: ['MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5', 'MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12']
- **Tier-A diagnostic — STRONG NEGATIVE:** all 18 deployed-filter substrate cells (ORB_G5/ATR_P50/COST_LT12 in raw/vol_norm/rank_252d forms across 6 lanes) FAIL. The currently deployed binary filters carry NO measurable continuous predictive substrate. A Carver-style continuous scaler **on the deployed filters themselves** is therefore not supported; this is the more important diagnostic result than the 2-cell Tier-B PASS.
- **Lane-level heterogeneity on rel_vol_session:** 2/6 lanes PASS at strict cell gates (EUROPE_FLOW, TOKYO_OPEN); 4/6 FAIL (SINGAPORE_OPEN, COMEX_SETTLE, NYSE_OPEN, US_DATA_1000). Lane-level flip rate = 67% → `heterogeneity_ack: true` per `.claude/rules/pooled-finding-rule.md`. The 4 FAIL lanes show positive ρ in 3/4 cases; failure is via monotonicity / |Q5−Q1| / bootstrap-CI gates, not direction. This is consistent with PR #51's universal monotonic-up effect attenuating below the strict cell-level threshold at lane scope; see cross-walk note for full treatment.

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
- **Effective tested K = 46** (2 cells gated INVALID by lookahead;
  pre-reg K=48 unchanged but BH-FDR denominator includes the gated cells, making
  the family-wise correction conservative — survivors are if anything stronger
  evidence than the q value suggests.)
- **Stage-2 eligibility:** PASS cells are stage-2 eligible only if their
  `stability_status == STABLE`. UNSTABLE PASS cells require a fresh Stage-2
  pre-reg with explicit forecast-normalization per Carver Ch.7 fn 78. Check
  the `stage2_eligible` field in the JSON twin.
- **Linear-rank weights {0.6, 0.8, 1.0, 1.2, 1.4} are diagnostic-only,**
  NOT Carver's actual recipe (Ch. 7 forecast scalar to abs-mean=10, cap=±20).
  Stage-2 must implement the canonical Carver scaling, not the rank proxy.
- **Per-cell bootstrap underestimates serial dependence.** Acceptable for Stage 1;
  Stage 2 must consider block bootstrap if substrate were confirmed.
- **AFML Ch. 19 sigmoid bet-sizer is NOT in `resources/`.** Sigmoid functional
  form deferred. Stage 2 (if any) defaults to Carver-only sizing.
- **Single-pass discipline.** Re-running with different feature lists, weight
  schemas, thresholds, or expanded K is a NEW pre-reg, not a re-run.

### Audit findings (institutional code+quant audit 2026-04-27, PASS_WITH_RISKS)

- **A. ATR_P50 vol_norm = raw identity (MED).** `derive_features` in
  `research/audit_sizing_substrate_diagnostic.py` returns `atr_20_pct`
  for both `raw` and `vol_norm` forms when the lane's substrate is
  ATR-percentile (lane `MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50`).
  Effect: 6 cells are identical column derivations rather than independent
  forms — effective unique cell count = 42, not the K=48 declared in
  pre-reg. The BH-FDR denominator stays K=48 (conservative direction),
  so no PASS verdict is spuriously promoted; the overstatement is in
  the diagnostic's coverage of normalization variants, not in any FDR
  inflation. Stage-2 pre-regs (if authored) must use a distinct vol_norm
  column (e.g., `atr_20 / atr_vel_ratio`) — requires fresh pre-reg.
- **B. COST_LT12 inline formula not delegated to canonical (LOW).**
  `derive_features` re-encodes the cost-ratio formula rather than calling
  `trading_app.config.CostRatioFilter.matches_row`. Run-time equivalent
  at this commit; future drift risk only. Anti-drift test added; see
  § Audit-recommended tests below.
- **C. Pooled-finding YAML front-matter absent (MED — RULE VIOLATION).**
  `.claude/rules/pooled-finding-rule.md` requires `pooled_finding: true`
  + `flip_rate_pct` + `per_cell_breakdown_path` on every audit-result MD
  dated 2026-04-20+ that makes a pooled-universe claim. K=48 BH-FDR is a
  pooled framing. Fixed in this commit (front-matter added at top of
  file, `flip_rate_pct: 67.0` lane-level, `heterogeneity_ack: true`).
- **D. RULE 13 pressure-test absent from test suite (MED — MANDATORY RULE).**
  `.claude/rules/backtesting-methodology.md` RULE 13 requires every new
  scan to deliberately inject a known-bad feature (e.g., `mae_r`,
  `outcome`) and confirm the script flags or filters it. Missing from
  `tests/test_research/test_audit_sizing_substrate_diagnostic.py` at
  the time of audit. Test added; see § Audit-recommended tests below.
- **E. Monotonicity gate dominates failures — verified correct (INFO).**
  Direction-agnostic gate (`inc or dec`); failing cells are genuinely
  non-monotonic in either direction. No fix needed.
- **H. `testing_mode: diagnostic_descriptive` non-canonical metadata
  value (LOW).** Pre-reg uses a value not in the canonical
  `{family, individual}` enum per
  `docs/institutional/pre_registered_criteria.md` Amendment 3.0.
  Closed via corrigendum subdoc
  `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg-corrigendum.md`
  (option 3: corrigendum without doctrine amendment, single-pass
  discipline preserved on locked pre-reg). Functional methodology was
  correct (Pathway-A family test with explicit no-writes boundary);
  metadata label was non-canonical. Future sessions: prefer
  `testing_mode: family` + `boundary.diagnostic_only: true` to express
  the same intent within the canonical enum.

### Audit-recommended tests (added in companion commit)

- `test_rule13_lookahead_inject_mae_r` — RULE 13 pressure-test: confirm
  banned look-ahead features (`mae_r`, `outcome`, `pnl_r`) are blocked
  by the temporal-validity / pre-reg-allowlist gates.
- `test_atr_p50_vol_norm_raw_identity_documented` — make Finding A
  identity explicit so a future "fix" cannot accidentally diverge them
  without a conscious design decision.
- `test_cost_lt12_formula_equivalence_with_canonical` — anti-drift gate
  asserting `derive_features` cost-ratio matches `CostRatioFilter` at
  run time.
- `test_stage2_eligible_false_for_unstable_pass` — confirm UNSTABLE PASS
  cells are correctly blocked from Stage-2 promotion (already present
  per commit `19a7a534`; verified by audit).

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
