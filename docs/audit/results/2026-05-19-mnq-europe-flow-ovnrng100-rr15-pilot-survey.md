---
pooled_finding: false
audit_target: "Read-only pilot-candidate survey: MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100"
auditor_context: opus-4-7-state-first-survey
canonical_layers: [orb_outcomes, daily_features, validated_setups]
db_freshness: "orb_outcomes MNQ max=2026-05-17 (read 2026-05-19)"
verdict: "DEAD"
parent_claims:
  - docs/audit/results/2026-05-05-capital-review-ovnrng100.md
  - docs/audit/results/2026-05-05-capital-review-ovnrng100-addendum.md
  - docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md
---

# Pilot-candidate survey — MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100

**Mode:** READ-ONLY state-first survey. No pre-reg, no Chordia unlock, no allocator mutation.
**Date:** 2026-05-19
**Origin:** Resume pointer `project_chordia_audit_unblock_real_edge_location_2026_05_19.md` flagged this lane as a tier-2 "next-tier unblock". Survey commissioned by user with explicit goal: determine whether the strategy can become a valid controlled-live pilot candidate. VWAP_MID_ALIGNED_O15 (the tier-1 candidate) is owned by parallel terminal and out of scope.

## Verdict: DEAD

Cannot route under any active pathway. Six independent blockers, any one of which is sufficient.

## Blockers

| # | Blocker | Evidence |
|---|---|---|
| B1 | **No Chordia audit-log entry exists** | `docs/runtime/chordia_audit_log.yaml` grep → 0 hits for strategy_id. `lane_allocation.json` paused row: `chordia_verdict: MISSING, chordia_audit_age_days: null` |
| B2 | **IS t = 3.518 < 3.79 strict Chordia threshold** (canonical recompute), **no theory citation exists** for Pathway-B 3.00 unlock. Amendment 3.3 (2026-05-17) requires explicit `theory_grant: true` with on-disk literature extract; none exists for "EUROPE_FLOW + overnight_range>=100 stop-cascade" | Canonical SQL: `IS m=0.1757 sd=1.1593 N=539 t=3.518`. Snapshot t=3.40 differs from canonical by ~0.12 (likely scratch-policy delta). `pre_registered_criteria.md` § Criterion 4 + Amendment 3.3 |
| B3 | **F1 VESTIGIAL — filter is mis-tuned to scale.** Fire-rate 4.1% (2019) → 92.5% (2026), 22.6× drift on a 4.7× avg_ovnrng shift. Structurally identical to parent COMEX_SETTLE F1 finding (4.3% → 91.7%). This is filter-class drift, not lane-specific noise. | Per-year SQL output below; parent audit `docs/audit/results/2026-05-05-capital-review-ovnrng100.md` § F1; `feedback_absolute_threshold_scale_audit.md` |
| B4 | **F3 DEPLOYMENT_MATH_BROKEN — MinBTL violation factor ≈121×** | `validated_setups.n_trials_at_discovery=36,372`. Clean-MNQ bound N≤300 per Criterion 2 + Amendment 2.8 + Bailey 2013 |
| B5 | **C8 OOS power tier = STATISTICALLY_USELESS** (power=0.251 « 0.50 RULE 3.3 floor). `validated_setups.c8_oos_status='PASSED'` is structurally meaningless under RULE 3.3 — verdict overrides to UNVERIFIED per `feedback_chordia_oos_park_vs_unverified_power_floor.md`. The +0.356R OOS ExpR is DIRECTIONAL_ONLY descriptive evidence, NOT confirmatory. | Canonical SQL: OOS N=74, cohen_d=0.1515, ncp=1.3036, df=73, power=0.251 (`scipy.stats.nct`). Amendment 3.1 (2026-04-22) power-floor gate |
| B6 | **Filter-class portfolio concentration risk** — admitting EUROPE_FLOW_OVNRNG_100 would put OVNRNG_100 in 2/3 deployable MNQ lane families, with both same-filter sibling already on a DOWNSIZE verdict | `2026-05-05-capital-review-ovnrng100-addendum.md` § Framing C |

## Canonical evidence (verbatim from `pipeline.paths.GOLD_DB_PATH`, max trading_day 2026-05-17)

### validated_setups row

```
strategy_id: MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100
sample_size: 532       win_rate: 0.5056       expectancy_r: 0.1714
sharpe_ratio: 0.1476   oos_exp_r: 0.2218      c8_oos_status: PASSED
wfe: 1.593             wfe_verdict: None      era_dependent: False
max_year_pct: 0.3128   p_value: 6.62e-04      fdr_adjusted_p: 6.43e-03
fdr_significant: True  dsr_score: 5.72e-15    sr0_at_discovery: 0.4831
n_trials_at_discovery: 36372  sharpe_haircut: -0.7638
max_drawdown_r: 9.9651        trades_per_year: 88.7
promoted_at: 2026-05-10 13:38:11+10:00   status: active
```

### Per-year fire rate (overnight_range>=100 on MNQ EUROPE_FLOW E2 RR1.5 CB1 O5)

```
yr=2019 N=171 fired=  7 fire%=  4.1 avg_ovnrng= 42.6 exprF=+0.500 exprNF=-0.187 exprUni=-0.159
yr=2020 N=256 fired=104 fire%= 40.6 avg_ovnrng=107.5 exprF=+0.035 exprNF=-0.059 exprUni=-0.021
yr=2021 N=259 fired= 67 fire%= 25.9 avg_ovnrng= 80.1 exprF=+0.230 exprNF=-0.052 exprUni=+0.021
yr=2022 N=258 fired=121 fire%= 46.9 avg_ovnrng=111.7 exprF=+0.166 exprNF=+0.039 exprUni=+0.098
yr=2023 N=258 fired= 20 fire%=  7.8 avg_ovnrng= 61.0 exprF=+0.348 exprNF=+0.093 exprUni=+0.113
yr=2024 N=259 fired= 74 fire%= 28.6 avg_ovnrng= 91.1 exprF=+0.224 exprNF=-0.035 exprUni=+0.039
yr=2025 N=257 fired=146 fire%= 56.8 avg_ovnrng=138.9 exprF=+0.195 exprNF=+0.075 exprUni=+0.143
yr=2026 N= 80 fired= 74 fire%= 92.5 avg_ovnrng=199.1 exprF=+0.356 exprNF=-0.612 exprUni=+0.283
```

### IS/OOS split (Mode A holdout 2026-01-01)

```
IS N=539  ExpR=+0.1757  sd=1.1593  t=3.518
OOS N=74  ExpR=+0.3559  cohen_d=0.1515  ncp=1.3036  df=73  power=0.2509  tier=STATISTICALLY_USELESS
```

### Era stability (post-WF_START_OVERRIDE 2020-01-01)

```
2020-2022   N=292   ExpR=+0.134   PASS (≥-0.05)
2023        N= 20   ExpR=+0.348   exempt (N<50)
2024-2025   N=220   ExpR=+0.205   PASS
2026        N= 74   ExpR=+0.356   PASS
[PRE-2020 excluded per Amendment 3.1 (2026-04-09)]
```

Era stability is the **one criterion this passes**. Every other gate fails or is structurally blocked.

## Why not CONDITIONAL or UNVERIFIED

- **UNVERIFIED** (Amendment 3.1, 2026-04-22) is reserved for low-power OOS where IS gates would otherwise pass. Here IS t fails 3.79 strict and no literature extract exists to grant Pathway-B 3.00. The IS veto is structural, not power-conditional.
- **CONDITIONAL_DEPLOY** (Amendment 3.2) requires `theory_grant: true` with cited mechanism. None available for this exact session × filter pair.

## What this survey does NOT establish

- Does NOT establish that the OVNRNG_100 mechanism is dead in general — only that the absolute-100-point threshold is vestigial.
- Does NOT propose a re-parametrisation pre-reg (e.g., `overnight_range >= prev_atr_20 * X`). That would be new discovery, out of scope.
- Does NOT affect the COMEX_SETTLE sibling lane's DOWNSIZE verdict (separate doc).

## Files / queries read

**Doctrine:** `docs/institutional/pre_registered_criteria.md` Criteria 1–13, Amendments 2.1 / 2.7 / 2.8 / 3.0 / 3.1×2 / 3.2 / 3.3; `.claude/rules/backtesting-methodology.md` RULE 3.3 / 1.2 / 4 / 8.1; `.claude/rules/pooled-finding-rule.md`.

**Prior audits:** `docs/audit/results/2026-05-05-capital-review-ovnrng100.md`, `docs/audit/results/2026-05-05-capital-review-ovnrng100-addendum.md`.

**Runtime / canonical:** `docs/runtime/chordia_audit_log.yaml`, `docs/runtime/fast_lane_status.yaml`, `docs/runtime/lane_allocation.json` (via strategy-lab MCP), `validated_setups` + `orb_outcomes` + `daily_features` against `pipeline.paths.GOLD_DB_PATH`.

**NO-GO catalog:** research-catalog MCP disconnected mid-survey; sibling COMEX_SETTLE OVNRNG_100 DOWNSIZE + 2026-05-01 vestigialness audit are the class-equivalent kill evidence.

## Smallest safe next command

Do not author a new pre-reg on this lane. Pivot to the next tier-2 candidate from the resume pointer (e.g. `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` or `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075`) and run the same state-first survey. The TOKYO_OPEN pair has a literature-grounded mechanism (Chan 2013 Ch7 intraday momentum, extract on disk) that this lane lacks.

## Lessons / linked memory

- `feedback_absolute_threshold_scale_audit.md` — absolute-points thresholds drift with price; filter-class drift ports across sessions.
- `feedback_chordia_oos_park_vs_unverified_power_floor.md` — `c8_oos_status='PASSED'` does not survive RULE 3.3 power floor.
- `feedback_max_profit_grow_chordia_inventory_not_force_slots.md` — do not admit same-filter siblings when one is on DOWNSIZE.
- Resume pointer `project_chordia_audit_unblock_real_edge_location_2026_05_19.md` should be annotated (next session) that EUROPE_FLOW OVNRNG_100 surveyed → DEAD.
