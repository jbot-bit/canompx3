---
pooled_finding: false
---

# MNQ Chordia batch K=20 — strict no-theory unlock audit (2026-05-15)

**Driver doc:** `docs/plans/2026-05-12-deployment-throughput-leverage.md` Lever 1
**Runner:** `research/chordia_strict_unlock_v1.py` (bounded exact-lane K=1 per pre-reg)
**Pre-regs:** `docs/audit/hypotheses/2026-05-15-NN-mnq-*-chordia-unlock.yaml` (20 files)
**Per-cell result MDs:** `docs/audit/results/2026-05-15-NN-mnq-*-chordia-unlock.md` (20 files + 20 CSVs)
**Audit log delta:** `docs/runtime/chordia_audit_log.yaml` (12 → 32 entries, +20)

## Scope

Bounded K=20 strict-replay Chordia audit on currently-MISSING MNQ lanes from `validated_setups`, ranked by `expectancy_r * trades_per_year` (annual_r proxy). Each cell is a Pathway-A K=1 single-cell pre-reg routed through `research/chordia_strict_unlock_v1.py`. The audit does NOT retest validated_setups discovery; it asks whether each candidate's IS sample clears Chordia 2018's strict no-theory hurdle (t ≥ 3.79) under canonical Mode-A replay.

## The question

For each of the 20 candidates: does the IS replay still clear t ≥ 3.79 (Criterion 4 strict no-theory threshold per `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`), and does the 2026 OOS sign confirm the IS direction?

## Why this batch

Per the 2026-05-12 throughput plan: 771 of 783 active MNQ validated_setups (~98.5%) had `chordia_verdict=MISSING` and were paused at allocator Gate 4 (CHORDIA). Each new audit unlocks one strategy from the gate. The plan's open question — "is there a batch-runner script?" — was answered NO; this batch instead loops the existing canonical bounded runner over 20 individually pre-registered K=1 audits per `feedback_meta_tooling_n1_tunnel_2026_05_01.md` (no n=1 meta-tooling).

## Candidate selection methodology

20 candidates ranked by `expectancy_r * trades_per_year` (annual_r proxy) from `validated_setups` (status=active, MNQ only), with these diversification constraints:

- Excluded: COLD sessions (LONDON_METALS, US_DATA_830 — confirmed COLD via canonical `_compute_session_regime`)
- Excluded: DIR_LONG / DIR_SHORT direction filters (not orthogonal alpha)
- Max 2 per (session, RR, orb_minutes) triplet
- Max 4 per session

Final session distribution: SINGAPORE_OPEN=4, COMEX_SETTLE=4, US_DATA_1000=4, NYSE_OPEN=4, CME_PRECLOSE=2, EUROPE_FLOW=2.

## Methodology

Each pre-reg: Pathway-A `testing_mode: family` with K=1 (per existing 13 unlock pre-reg pattern + hypothesis_loader Amendment 3.0 constraint that no-theory pre-regs cannot use Pathway B). No `theory_citation` field on any hypothesis (per `feedback_chordia_theory_citation_field_presence_trap.md` — field-presence flips threshold). Strict no-theory Chordia threshold (t ≥ 3.79) per Criterion 4.

Mode A holdout sacred from 2026-01-01 (`HOLDOUT_SACRED_FROM`). IS = trading_day < 2026-01-01; OOS = trading_day >= 2026-01-01 (descriptive only). Scratch policy: `realized-eod` (`COALESCE(pnl_r, 0.0)`).

Canonical filter delegation via `research.filter_utils.filter_signal()`. Cohort matched to validated_setups via `WF_START_OVERRIDE` (MNQ = 2020-01-01 micro-launch exclusion).

## Verdicts (canonical 5-state taxonomy)

| Rank | Strategy ID | Runner verdict | Canonical | t_IS | N_IS | ExpR_IS |
|---:|---|---|---|---:|---:|---:|
| 01 | MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_COST_LT12_O30 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.072 | 1387 | 0.1661 |
| 02 | MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_COST_LT12_O15 | FAIL_STRICT_CHORDIA | FAIL_BOTH | 2.976 | 1114 | 0.1799 |
| 03 | MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT12_O30 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.074 | 1387 | 0.1421 |
| 04 | MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_G8_O15 | FAIL_STRICT_CHORDIA | FAIL_BOTH | 2.992 | 1337 | 0.1390 |
| 05 | MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER | PARK | PARK | 4.691 | 1479 | 0.1051 |
| 06 | MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4 | PARK | PARK | 4.677 | 1478 | 0.1049 |
| 07 | **MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K** | **PASS_CHORDIA** | **PASS_CHORDIA** | **3.956** | **1437** | **0.1197** |
| 08 | **MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K** | **PASS_CHORDIA** | **PASS_CHORDIA** | **4.678** | **1406** | **0.1179** |
| 09 | **MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15** | **PASS_CHORDIA** | **PASS_CHORDIA** | **3.988** | **1372** | **0.1241** |
| 10 | **MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25** | **PASS_CHORDIA** | **PASS_CHORDIA** | **3.802** | **1524** | **0.1156** |
| 11 | MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.709 | 1547 | 0.1118 |
| 12 | MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_8K | PARK | PARK | 3.838 | 1406 | 0.1213 |
| 13 | MNQ_NYSE_OPEN_E2_RR2.0_CB1_X_MGC_ATR70 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.373 | 427 | 0.2276 |
| 14 | MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.604 | 1487 | 0.1106 |
| 15 | **MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25** | **PASS_CHORDIA** | **PASS_CHORDIA** | **4.103** | **1523** | **0.0991** |
| 16 | MNQ_NYSE_OPEN_E2_RR2.5_CB1_X_MGC_ATR70 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.424 | 427 | 0.2579 |
| 17 | MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G8 | FAIL_STRICT_CHORDIA | FAIL_BOTH | 2.988 | 1291 | 0.1121 |
| 18 | MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | FAIL_STRICT_CHORDIA | FAIL_BOTH | 2.783 | 1485 | 0.0961 |
| 19 | MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT12 | FAIL_STRICT_CHORDIA | FAIL_CHORDIA | 3.252 | 1252 | 0.1245 |
| 20 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_O15 | PARK | PARK | 3.865 | 412 | 0.2151 |

## Tally

| Canonical verdict | Count | Allocator deploy? |
|---|---:|---|
| PASS_CHORDIA | 5 | yes (gate 4 clears) |
| PARK | 4 | NO (IS-clean, OOS sign opposes) |
| FAIL_CHORDIA | 7 | NO (3.00 ≤ t < 3.79, no theory) |
| FAIL_BOTH | 4 | NO (t < 3.00) |

**Pass rate: 5/20 = 25%.** Lower than the throughput plan's prior empirical 8/12 = 67% on MNQ (n=12). The drop comes from systematic over-promotion in `validated_setups` for high-RR variants — every RR ≥ 2.0 candidate failed strict Chordia, despite headline annual_r > 23.

## Verdict

**Bottom line: 5 of 20 PASS_CHORDIA, allocator selects 2 NEW from this batch + 2 RR-tripwire swaps = 4 deployed lanes total (was 2). Annual_r 63 → 89.5 (+42% throughput).**

Per-cell verdicts in the table above. Allocator decision (post-batch + post-rebalance) lands in `docs/runtime/lane_allocation.json` and is the binding deployment record.

## Newly deployable lanes

The 5 PASS_CHORDIA strategies clear allocator Gate 4 (CHORDIA) for the first time. Whether they actually deploy is downstream of this audit:

1. **MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K** — t=3.956, N=1437, ExpR=0.120
2. **MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K** — t=4.678, N=1406, ExpR=0.118
3. **MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15** — t=3.988, N=1372, ExpR=0.124
4. **MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25** — t=3.802, N=1524, ExpR=0.116
5. **MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25** — t=4.103, N=1523, ExpR=0.099

Allocator audit (`scripts/tools/allocator_gate_audit.py --all-profiles`) shows the post-batch selection set (4 lanes for `topstep_50k_mnq_auto`):
- MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K (NEW from this batch)
- MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25 (NEW from this batch)
- MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 (RR1.5→RR1.0 SR-tripwire swap)
- MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 (RR1.5→RR1.0 SR-tripwire swap)

The remaining 3 PASS_CHORDIA strategies (#9, #10, #8 of the batch) lose to the correlation gate (RHO_REJECT_THRESHOLD=0.70) against same-session siblings.

## Findings of note

1. **Validated_setups inflates t-stats for high-RR variants.** Every RR ≥ 2.0 candidate in this batch (5 strategies, ranks 13/16/17/18/19) failed strict Chordia despite high annual_r proxies. The proxy is `expectancy_r × trades_per_year`; high-RR strategies trade slightly less but win much bigger when they win — inflating ExpR while suppressing per-trade Sharpe → lower t. Chordia strict t-hurdle is a Sharpe-based test, not an ExpR-based test.

2. **OOS direction-flip is a real signal-killer at this batch's sample size.** 4 of 9 IS-clean candidates flipped sign in 2026 OOS at N_OOS ≥ 30 (PARK rate = 4/9 = 44%). Per `.claude/rules/backtesting-methodology.md` § 3.3, OOS power floor would normally rescue some of these, but at N_OOS ~ 70-80 with effect sizes ~ 0.10R, the OOS power is moderate (CAN_REFUTE tier). Treat PARK verdicts as honest kills.

3. **CME_PRECLOSE NO_FILTER PARK (rank 5) is high-info.** t=4.69 is very strong on IS but OOS sign opposes at N=75. This is the canonical "regime-shifted" signature; matches the throughput plan's note that CME_PRECLOSE was COLD in May audit (now barely HOT at 0.014). Do not deploy this strategy even if regime turns warmer — the OOS-sign opposition is the binding evidence.

## Drift check

`python pipeline/check_drift.py` after audit log + 20 pre-regs + 20 result MDs: **131/131 checks passed**, 0 failures, 20 advisory (none related to this batch). Check 146 (lane_allocation.json lanes pass Chordia gate) PASSED.

## Reproduction

```bash
# 1. Re-run any single audit
python research/chordia_strict_unlock_v1.py \
  --hypothesis docs/audit/hypotheses/2026-05-15-NN-mnq-*-chordia-unlock.yaml

# 2. Re-run all 20 in a loop (PowerShell-friendly)
Get-ChildItem docs/audit/hypotheses/2026-05-15-*-chordia-unlock.yaml | ForEach-Object {
  python research/chordia_strict_unlock_v1.py --hypothesis $_.FullName
}

# 3. Inspect canonical state
python -c "import yaml; print(len(yaml.safe_load(open('docs/runtime/chordia_audit_log.yaml')).get('audits',[])))"

# 4. Re-run allocator (regenerates lane_allocation.json)
python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto

# 5. Drift check (must pass)
python pipeline/check_drift.py
```

## Outputs

- `docs/audit/hypotheses/2026-05-15-NN-mnq-*-chordia-unlock.yaml` (20 pre-reg files)
- `docs/audit/results/2026-05-15-NN-mnq-*-chordia-unlock.{md,csv}` (20 result MDs + 20 row-level CSVs)
- `docs/audit/results/2026-05-15-mnq-chordia-batch-k20.md` (this file — batch summary)
- `docs/runtime/chordia_audit_log.yaml` (12 → 32 audit entries; +20 with `audit_date: 2026-05-15`)
- `docs/runtime/lane_allocation.json` (rebalance_date 2026-05-14, 2 → 4 deployed lanes)

## Limitations

- **Per-cell K=1 pilots, not a family-wide K=20 BH-FDR claim.** Each pre-reg is its own confirmatory K=1 audit per `pre_registered_criteria.md` Amendment 3.0. Multi-framing K applies to the upstream `validated_setups` promotion (which already passed BH-FDR), not to this batch.
- **No theory grants claimed.** All 20 use the strict no-theory threshold (t ≥ 3.79). A future Pathway-B promotion to t ≥ 3.00 would require new literature extracts under `docs/institutional/literature/` and is out of scope.
- **OOS power floor not formally computed per Pathway-B doctrine.** N_OOS in the 2026 window is ~70-80 for full-history candidates and ~40-50 for cross-asset variants. At per-trade effect sizes ~ 0.10R, OOS power is moderate (CAN_REFUTE tier per `.claude/rules/backtesting-methodology.md` § 3.3) but not formally calculated. PARK verdicts could in principle be UNVERIFIED rather than honest kills if power is below 0.50.
- **3-month decay not assessed in this audit.** The allocator's `recent_3mo_expr` field shows 248 strategies in decay state at this rebalance — this batch did not stratify candidates by decay state.

## What this audit does NOT do

- Does not mutate `lane_allocation.json` directly (allocator-output truth, regenerated separately by `scripts/tools/rebalance_lanes.py`).
- Does not assign theory grants (would require new literature extracts under `docs/institutional/literature/`).
- Does not audit MGC or MES (Lever 2 of throughput plan, gated on profile activation).
- Does not address the 2026-05-20 SR-tripwire RR1.5→RR1.0 decision as a standalone gate (the rebalance triggered it 6 days early as a side effect of running the rebalance with fresh inputs).
- Does not validate live execution path (Stage A of operator plan — DEMO mode flip + smoke checklist — is a separate session).
- Does not retroactively reclassify the 12 prior audit entries; they remain on their original audit_date and original verdicts.

## Acceptance gates from the plan

| Gate | Status |
|---|---|
| ≥1 new live_trades row (Stage A) | DEFERRED — operator action |
| ≥3 new deployed lanes in lane_allocation.json (Stage B) | conditional on rebalance run; allocator audit shows 2 NEW from batch + 2 RR-tripwire swaps = 4 total |
| `pipeline/check_drift.py` passes | PASSED 131/131 |
| `git status --short` clean except for plan files | true |

---

*Batch executed by Claude Code (Opus 4.7) on 2026-05-14, against gold.db at canonical project root. All 20 audits ran the canonical bounded runner without modification.*
