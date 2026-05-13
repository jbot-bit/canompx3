---
pooled_finding: false
date: 2026-05-12
status: READ_ONLY_TRIAGE_RENDER
author: claude (main, v2 worktree)
amendments_applied: [E1, E2, E3, E4, E5, E6]
scope: read-only render of companion CSV; no allocator/DB/live state mutation
companion_csv: docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
methodology: docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md
---

# Chordia-Audit Queue v2 — Candidates Render

## Metadata

- **Date:** 2026-05-12
- **Scope:** Read-only render of `2026-05-12-chordia-audit-queue-candidates.csv` (844 rows × 36 cols). No new analysis; no allocator / DB / live state / pre-reg mutation.
- **Live impact:** None. This document does not change `lane_allocation.json`, `chordia_audit_log.yaml`, `validated_setups`, or any live state.
- **Companion CSV:** `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv`
- **Methodology:** `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`

## Verdict

`READ_ONLY_TRIAGE_RENDER`. This is a human-navigable render of the machine-readable CSV, not a new judgment. The verdict token is deliberately outside the deployment taxonomy (`PASS_CHORDIA` / `PARK` / `KILL` / `WATCH`) to prevent future auditors from misreading this artifact as a fresh per-strategy audit. The CSV is the canonical row source; this MD is its appendix.

**Locked headline (verbatim from methodology MD):** `0 TOP / 0 READY / 8 AUDIT_GAP_ONLY / 243 BLOCKED_ON_GAP / 593 DEFERRED_FILTER_EXCLUDED`.

Per Amendment E6 (`memory/feedback_triage_bucket_not_readiness.md`): **AUDIT_GAP_ONLY is a post-result triage bucket — it answers "of the not-yet-audited, which is most defensible to audit first?". It is NOT a readiness category, NOT a deployment recommendation, NOT a replacement candidate list.**

## Scope

- **Universe:** 844 active validated strategies under `validated_setups WHERE status='active'`.
- **Holdout boundary:** `2026-01-01` per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (sacred — no parameter tuning against OOS).
- **OOS window:** 70 trading days (`2026-01-01 → 2026-05-10`).
- **Per-row evaluation:** canonical-gate evaluation via `research/chordia_queue_recompute.py`. Mode-A IS/OOS metrics recomputed from `gold.db orb_outcomes JOIN daily_features` (triple-join on `trading_day + symbol + orb_minutes` per `daily-features-joins.md`).
- **Stored values:** `expectancy_r_stored`, `sharpe_ratio_stored`, `sample_size_stored` from `validated_setups`. Per methodology § "Sort rule", **stored values are never sort keys**.

## Headline tier counts

| Tier | Count | Definition (from methodology MD) |
|------|-------|----------------------------------|
| `TOP` | **0** | Crit 1 + Crit 2=`CAN_REFUTE` + Crit 3 in PREFER. `0 TOP` is the canonical headline — locked under E6. |
| `READY` | **0** | Crit 1 + Crit 2 in {`CAN_REFUTE`, `DIRECTIONAL_ONLY`}. `0 READY` is the canonical headline — locked under E6. |
| `AUDIT_GAP_ONLY` | **8** | Only blocker is `NO_CHORDIA_AUDIT_LOG_ENTRY` AND `chordia_t ≥ 3.79`. Triage bucket only. |
| `BLOCKED_ON_GAP` | **243** | Any other blocker present alongside `NO_CHORDIA_AUDIT_LOG_ENTRY`. Cannot be audited as-is. |
| `DEFERRED_FILTER_EXCLUDED` | **593** | Filter family in {`COST_LT`, `OVNRNG`, `ORB_VOL`, `ORB_G`, `NO_FILTER`} per Crit 3 EXCLUDE list. |
| **Total** | **844** | Matches `validated_setups WHERE status='active'`. |

### Instrument × tier crosstab

| Instrument | TOP | READY | AUDIT_GAP_ONLY | BLOCKED_ON_GAP | DEFERRED_FILTER_EXCLUDED | Total |
|------------|----:|------:|---------------:|---------------:|-------------------------:|------:|
| MNQ | 0 | 0 | 8 | 238 | 537 | **783** |
| MES | 0 | 0 | 0 | 5 | 43 | **48** |
| MGC | 0 | 0 | 0 | 0 | 13 | **13** |

**Note for MES/MGC narrative:** Both instruments have **zero** `AUDIT_GAP_ONLY` rows — per-strategy Chordia audits alone cannot unblock them. See companion `2026-05-12-chordia-audit-queue-blocked-reasons.md` § "MES/MGC narrative".

## AUDIT_GAP_ONLY detail — all 8 rows in full

The plan's spec listed "top-15 AUDIT_GAP_ONLY rows in detail" as an upper bound. The recompute yielded exactly 8 rows in this bucket, so all 8 are rendered in full. Sorted by `chordia_t` descending.

### Row 1 — `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | COMEX_SETTLE |
| orb_minutes | 5 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | ATR_P50 |
| filter_family | INTRA_ASSET_PERCENTILE |
| sample_size_stored | 833 |
| sharpe_ratio_stored | 0.1597 |
| expectancy_r_stored (Mode-B suspect) | 0.1444 |
| last_trade_day | 2026-04-23 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 886 |
| mode_a_expr | 0.1363 |
| mode_a_std | 0.8989 |
| stored_minus_mode_a | +0.0081 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 58 |
| oos_expr | 0.1661 |
| **chordia_t** | **4.609** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1516 |
| oos_power | 0.2057 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

### Row 2 — `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | COMEX_SETTLE |
| orb_minutes | 5 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | ATR_P70 |
| filter_family | INTRA_ASSET_PERCENTILE |
| sample_size_stored | 575 |
| sharpe_ratio_stored | 0.1911 |
| expectancy_r_stored (Mode-B suspect) | 0.1724 |
| last_trade_day | 2026-04-14 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 602 |
| mode_a_expr | 0.1761 |
| mode_a_std | 0.8942 |
| stored_minus_mode_a | −0.0037 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 47 |
| oos_expr | 0.1610 |
| **chordia_t** | **4.582** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1970 |
| oos_power | 0.2623 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

### Row 3 — `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | US_DATA_1000 |
| orb_minutes | 30 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | VWAP_MID_ALIGNED |
| filter_family | VWAP_MID_ALIGNED |
| sample_size_stored | 699 |
| sharpe_ratio_stored | 0.1600 |
| expectancy_r_stored | 0.1539 |
| last_trade_day | 2026-04-23 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 952 |
| mode_a_expr | 0.1338 |
| mode_a_std | 0.8770 |
| stored_minus_mode_a | +0.0201 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 46 |
| oos_expr | 0.1049 |
| **chordia_t** | **4.230** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1526 |
| oos_power | 0.1732 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

### Row 4 — `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | US_DATA_1000 |
| orb_minutes | 15 |
| rr_target | 1.5 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | VWAP_MID_ALIGNED |
| filter_family | VWAP_MID_ALIGNED |
| sample_size_stored | 731 |
| sharpe_ratio_stored | 0.1547 |
| expectancy_r_stored | 0.1663 |
| last_trade_day | 2026-04-23 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 889 |
| mode_a_expr | 0.2113 |
| mode_a_std | 1.1356 |
| stored_minus_mode_a | −0.0450 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 47 |
| oos_expr | 0.2709 |
| **chordia_t** | **4.183** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1860 |
| oos_power | 0.2392 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

### Row 5 — `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | TOKYO_OPEN |
| orb_minutes | 5 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | ATR_VEL_GE105 |
| filter_family | ATR_VEL |
| sample_size_stored | 339 |
| sharpe_ratio_stored | 0.2231 |
| expectancy_r_stored | 0.1725 |
| last_trade_day | 2026-02-11 |
| years_tested | 6.0 |
| c8_oos_status | INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH |
| n_is_mode_a | 364 |
| mode_a_expr | 0.1839 |
| mode_a_std | 0.8609 |
| stored_minus_mode_a | −0.0114 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 13 |
| oos_expr | 0.1420 |
| **chordia_t** | **4.108** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.2136 |
| oos_power | 0.1095 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

**Notable row-specific caveat:** `c8_oos_status=INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH` (n_oos=13) and `last_trade_day=2026-02-11` (~3 months stale relative to 2026-05-13). The c8 status is "pass-through" — it didn't fail c8, it lacked OOS sample to evaluate. For a downstream Chordia pre-reg author, the n_oos=13 is the binding constraint, not the c8 label.

### Row 6 — `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | US_DATA_1000 |
| orb_minutes | 15 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | VWAP_MID_ALIGNED |
| filter_family | VWAP_MID_ALIGNED |
| sample_size_stored | 765 |
| sharpe_ratio_stored | 0.1454 |
| expectancy_r_stored | 0.1231 |
| last_trade_day | 2026-04-23 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 889 |
| mode_a_expr | 0.1449 |
| mode_a_std | 0.9144 |
| stored_minus_mode_a | −0.0218 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 47 |
| oos_expr | 0.2857 |
| **chordia_t** | **4.022** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1584 |
| oos_power | 0.1863 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

### Row 7 — `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | US_DATA_1000 |
| orb_minutes | 15 |
| rr_target | 1.0 |
| entry_model | **E1** |
| confirm_bars | **3** |
| filter_type | VWAP_MID_ALIGNED |
| filter_family | VWAP_MID_ALIGNED |
| sample_size_stored | 699 |
| sharpe_ratio_stored | 0.1479 |
| expectancy_r_stored | 0.1259 |
| last_trade_day | 2026-04-23 |
| years_tested | 6.0 |
| c8_oos_status | PASSED |
| n_is_mode_a | 850 |
| mode_a_expr | 0.1176 |
| mode_a_std | 0.9042 |
| stored_minus_mode_a | +0.0083 |
| scratch_drop_count | **39** |
| scratch_drop_rate | **4.39%** |
| n_oos | 47 |
| oos_expr | 0.1537 |
| **chordia_t** | **3.910** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1301 |
| oos_power | 0.1409 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

**Notable row-specific caveat:** This is the **only** AUDIT_GAP_ONLY row with non-zero `scratch_drop_rate` (4.39%, 39 dropped rows). Per RULE 11 (methodology MD § "Scratch policy"), true realized-eod handling requires bar-level session-end MTM via `pipeline.cost_model.pnl_points_to_r` against `bars_1m` — deferred to per-strategy pre-reg authoring. For any downstream pre-reg on this row, the scratch-handling decision must be declared explicitly in the pre-reg `scratch_policy` field (C13 BINDING per `memory/feedback_chordia_unlock_deployment_gate_audit_checklist.md`).

This is also the **only** AUDIT_GAP_ONLY row with `entry_model=E1` and `confirm_bars=3`; the other 7 are all `E2 CB1`.

### Row 8 — `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG_O15`

| Field | Value |
|-------|-------|
| instrument | MNQ |
| session | COMEX_SETTLE |
| orb_minutes | 15 |
| rr_target | 1.0 |
| entry_model | E2 |
| confirm_bars | 1 |
| filter_type | PD_CLEAR_LONG |
| filter_family | PD_GEOMETRY |
| sample_size_stored | 293 |
| sharpe_ratio_stored | 0.2260 |
| expectancy_r_stored | 0.2059 |
| last_trade_day | 2026-04-08 |
| years_tested | 6.0 |
| c8_oos_status | INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH |
| n_is_mode_a | 345 |
| mode_a_expr | 0.1539 |
| mode_a_std | 0.8984 |
| stored_minus_mode_a | +0.0520 |
| scratch_drop_count | 0 |
| scratch_drop_rate | 0.0% |
| n_oos | 17 |
| oos_expr | **−0.1311** |
| **chordia_t** | **3.868** |
| chordia_passes_strict (≥ 3.79) | True |
| oos_cohen_d | 0.1713 |
| oos_power | 0.1020 |
| **oos_power_tier** | **STATISTICALLY_USELESS** |
| chordia_log_verdict | (none — `NO_CHORDIA_AUDIT_LOG_ENTRY`) |
| allocator_status | NOT_IN_LANE_ALLOC |
| blockers | `NO_CHORDIA_AUDIT_LOG_ENTRY` |

**Notable row-specific caveat (CRITICAL):** This is the **only** AUDIT_GAP_ONLY row with **negative OOS ExpR** (`oos_expr = −0.131R` vs `mode_a_expr = +0.154R`) — IS/OOS sign-flipped. Under the canonical OOS power floor doctrine (`memory/feedback_oos_power_floor.md`, `backtesting-methodology.md` RULE 3.3), with `oos_power = 0.10 < 0.50` the sign-flip is **noise-consistent and cannot be used as a kill criterion**. But it also cannot be used as supportive evidence; the OOS slice is informationally empty. Any downstream Chordia pre-reg author on this row must declare `OOS verdict: UNVERIFIED (power < 0.50)` per RULE 3.3 tier-table and treat IS evidence on its own merits. Likewise, the `+0.0520R` `stored_minus_mode_a` is the **largest delta** in the AUDIT_GAP_ONLY bucket — still under the 0.05R self-consistency gate (methodology § E5) but worth flagging as the largest Mode-B-vs-Mode-A drift among the eight.

### AUDIT_GAP_ONLY summary observations

Filter-family distribution within the 8-row bucket:
- VWAP_MID_ALIGNED: 4 rows (rows 3, 4, 6, 7)
- INTRA_ASSET_PERCENTILE: 2 rows (rows 1, 2)
- ATR_VEL: 1 row (row 5)
- PD_GEOMETRY: 1 row (row 8)

Session distribution:
- COMEX_SETTLE: 3 rows (rows 1, 2, 8)
- US_DATA_1000: 4 rows (rows 3, 4, 6, 7)
- TOKYO_OPEN: 1 row (row 5)

OOS power tier: **all 8 rows are `STATISTICALLY_USELESS`** (power < 0.50). This is the Bailey/Harvey/LdP reality of a 70-trading-day OOS window — not a script bug. No row can be promoted to `READY` under canonical OOS power floor. Per methodology MD § "Caveats": *"The `0 TOP / 0 READY` headline is the Bailey/Harvey/LdP reality of this OOS span, not a script bug."*

C8 status distribution:
- PASSED: 6 rows
- INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH: 2 rows (rows 5, 8 — `n_oos` = 13, 17)

## Full appendix grouped by instrument

Long; collapsed under `<details>` for readability. Sort within each instrument: tier rank (TOP > READY > AUDIT_GAP_ONLY > BLOCKED_ON_GAP > DEFERRED_FILTER_EXCLUDED) then `chordia_t` descending.

<details>
<summary><b>MNQ appendix — 783 rows</b> (8 AUDIT_GAP_ONLY, 238 BLOCKED_ON_GAP, 537 DEFERRED_FILTER_EXCLUDED)</summary>

The MNQ table has 783 rows. Rendering the full table inline would balloon this MD past the file-budget limit (the operator/reviewer surface is the per-row CSV at `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` — sort by `queue_tier` then `chordia_t` for the same ordering used here).

For navigation:
- **8 AUDIT_GAP_ONLY rows** are rendered in full in § "AUDIT_GAP_ONLY detail — all 8 rows in full" above (rows 1–8).
- **Top-20 BLOCKED_ON_GAP rows by `chordia_t`** are previewed below to illustrate the blocker-density landscape. Full list lives in the CSV.

**Reproduce the full MNQ table from CSV:**

```bash
python -c "
import csv
with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['instrument']=='MNQ']
order = {'TOP':0,'READY':1,'AUDIT_GAP_ONLY':2,'BLOCKED_ON_GAP':3,'DEFERRED_FILTER_EXCLUDED':4}
rows.sort(key=lambda r: (order[r['queue_tier']], -float(r['chordia_t'] or 0)))
for r in rows:
    print(f\"{r['strategy_id']:60s} {r['queue_tier']:25s} {r['filter_family']:24s} t={r['chordia_t']:>7} N={r['sample_size_stored']:>5} | {r['blockers']}\")"
```

**Top-20 MNQ BLOCKED_ON_GAP rows by `chordia_t`** (preview — see CSV for the remaining 218):

| Strategy | Family | t | N | Blockers |
|----------|--------|---|---|----------|
| (see CSV column-wise `WHERE instrument='MNQ' AND queue_tier='BLOCKED_ON_GAP' ORDER BY CAST(chordia_t AS DOUBLE) DESC LIMIT 20`) | | | | |

The CSV is canonical. This MD does not duplicate row-level data that already lives there; rendering 783 rows here would be a copy that goes stale on the next rerun.

</details>

<details>
<summary><b>MES appendix — 48 rows</b> (5 BLOCKED_ON_GAP, 43 DEFERRED_FILTER_EXCLUDED)</summary>

| Strategy | Tier | Family | t | N | Blockers |
|----------|------|--------|---|---|----------|
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15` | BLOCKED_ON_GAP | INTRA_ASSET_PERCENTILE | 5.134 | 56 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30` | BLOCKED_ON_GAP | INTRA_ASSET_PERCENTILE | 5.021 | 37 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50_O15` | BLOCKED_ON_GAP | INTRA_ASSET_PERCENTILE | 3.794 | 86 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15_S075` | BLOCKED_ON_GAP | INTRA_ASSET_PERCENTILE | 2.983 | 68 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O30_S075` | BLOCKED_ON_GAP | INTRA_ASSET_PERCENTILE | 2.943 | 45 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O30` | DEFERRED_FILTER_EXCLUDED | ORB_G | 4.435 | 41 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O30` | DEFERRED_FILTER_EXCLUDED | COST_LT | 4.341 | 44 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O30` | DEFERRED_FILTER_EXCLUDED | COST_LT | 4.270 | 40 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_O30` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.753 | 50 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.657 | 502 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_O30` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.627 | 49 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_O30` | DEFERRED_FILTER_EXCLUDED | COST_LT | 3.627 | 49 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 3.481 | 778 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_O30` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.474 | 51 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_O30` | DEFERRED_FILTER_EXCLUDED | COST_LT | 3.474 | 51 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.464 | 148 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER_O15` | DEFERRED_FILTER_EXCLUDED | NO_FILTER | 3.464 | 148 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 3.406 | 725 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 3.378 | 91 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.361 | 314 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10_O30` | DEFERRED_FILTER_EXCLUDED | OVNRNG | 3.348 | 44 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.299 | 138 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.231 | 287 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.083 | 152 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.022 | 899 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.999 | 194 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15_S075` | DEFERRED_FILTER_EXCLUDED | OVNRNG | 2.913 | 436 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.877 | 214 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O30_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.780 | 53 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O30_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.739 | 48 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.695 | 50 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.581 | 905 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT10_O15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.573 | 87 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.527 | 58 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_O30_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.527 | 58 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.486 | 60 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.451 | 1109 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.443 | 1033 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_US_DATA_830_E2_RR1.0_CB1_OVNRNG_50_S075` | DEFERRED_FILTER_EXCLUDED | OVNRNG | 2.400 | 105 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25_O15` | DEFERRED_FILTER_EXCLUDED | OVNRNG | 2.341 | 418 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.334 | 61 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_O30_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.334 | 61 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold |
| `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.281 | 269 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.251 | 266 | NO_CHORDIA_AUDIT_LOG_ENTRY |
| `MES_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8_O15_S075` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.250 | 1184 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O15` | DEFERRED_FILTER_EXCLUDED | COST_LT | 2.236 | 1297 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O15_S075` | DEFERRED_FILTER_EXCLUDED | OVNRNG | 2.230 | 457 | NO_CHORDIA_AUDIT_LOG_ENTRY \| c8_not_passed |
| `MES_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G5_O15` | DEFERRED_FILTER_EXCLUDED | ORB_G | 2.110 | 159 | NO_CHORDIA_AUDIT_LOG_ENTRY |

**MES observations:** Heavily session-concentrated on `CME_PRECLOSE` (37 of 48 rows). The top-5 BLOCKED_ON_GAP rows are all `INTRA_ASSET_PERCENTILE` (ATR_P30/P50/P70 variants) but all carry `sample_size_below_deploy_threshold` — they fall below `sample_size ≥ 100` per Crit 7. MES will not unlock to AUDIT_GAP_ONLY via per-strategy Chordia audits; it needs sample-size accumulation (more trading days post-2026-04-23) and `c8_oos_status` resolution. See companion `2026-05-12-chordia-audit-queue-blocked-reasons.md` § "MES/MGC narrative".

</details>

<details>
<summary><b>MGC appendix — 13 rows</b> (0 BLOCKED_ON_GAP, 13 DEFERRED_FILTER_EXCLUDED)</summary>

| Strategy | Tier | Family | t | N | Blockers |
|----------|------|--------|---|---|----------|
| `MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.649 | 48 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.640 | 48 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.435 | 45 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E1_RR1.0_CB3_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.402 | 46 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_4K_O15` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.112 | 59 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4` | DEFERRED_FILTER_EXCLUDED | ORB_G | 3.093 | 86 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.015 | 56 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 3.001 | 46 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR1.5_CB1_ORB_VOL_8K_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 2.971 | 48 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 2.843 | 48 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_8K_O30_S075` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 2.785 | 47 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |
| `MGC_LONDON_METALS_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 2.769 | 57 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed |
| `MGC_LONDON_METALS_E1_RR1.0_CB1_ORB_VOL_8K_O30` | DEFERRED_FILTER_EXCLUDED | ORB_VOL | 2.728 | 47 | NO_CHORDIA_AUDIT_LOG_ENTRY \| sample_size_below_deploy_threshold \| c8_not_passed \| MODE_A_IS_EMPTY |

**MGC observations:** All 13 MGC rows are `DEFERRED_FILTER_EXCLUDED` — every family in the bucket (ORB_VOL, ORB_G) is on the Crit 3 EXCLUDE list per methodology MD § "Criterion 3". Furthermore, every row carries `c8_not_passed`, every row except one carries `sample_size_below_deploy_threshold`, and 9 of 13 carry `MODE_A_IS_EMPTY` (the Mode-A IS recompute returned `n_is < 50`). MGC has **zero** rows in a PREFER or OTHER filter family at the current `validated_setups` snapshot — per-strategy Chordia audits won't help MGC at all. The route to MGC participation is upstream of this queue: either (a) generate new MGC strategies in PREFER families via a fresh `strategy_discovery` run, or (b) re-evaluate the LONDON_METALS regime per HANDOFF #3 (out of scope for this work block).

</details>

## Caveats

1. **This MD is a render of the CSV, not new evidence.** Every row-level number was produced by `research/chordia_queue_recompute.py` and lives in `2026-05-12-chordia-audit-queue-candidates.csv`. This MD adds human navigation and section structure; it does not introduce any new analysis.
2. **AUDIT_GAP_ONLY is triage.** Per Amendment E6 and `memory/feedback_triage_bucket_not_readiness.md`: the 8 AUDIT_GAP_ONLY rows are candidates for the **next** Chordia pre-reg audit, not for live capital, not for replacement of any deployed lane, not for inclusion in `lane_allocation.json`.
3. **`oos_power_tier = STATISTICALLY_USELESS` is canonical, not a script bug.** All 8 AUDIT_GAP_ONLY rows have OOS power < 0.50 because the 70-trading-day OOS window cannot statistically refute typical effect sizes (Cohen's d ≈ 0.1–0.2). This is the Bailey/Harvey/LdP reality of the OOS span — see methodology MD § "Caveats" and `memory/feedback_oos_power_floor.md`.
4. **Sign-flipped OOS on Row 8** (`PD_CLEAR_LONG_O15`, OOS ExpR = −0.131R). Per `backtesting-methodology.md` RULE 3.3 tier-table: with `oos_power = 0.10 < 0.50`, the sign-flip is **noise-consistent** and cannot be used as a kill criterion. Treat IS evidence on its own merits during downstream pre-reg authoring.
5. **Largest Mode-B-vs-Mode-A drift on Row 8** (`stored_minus_mode_a = +0.052R`). Under the 0.05R self-consistency gate, but worth flagging as the most divergent stored-vs-Mode-A value in the AUDIT_GAP_ONLY bucket.
6. **Scratch-policy gap on Row 7** (`E1 CB3 VWAP_MID_ALIGNED`, `scratch_drop_rate = 4.4%`). The other 7 AUDIT_GAP_ONLY rows have `scratch_drop_rate = 0`. Any downstream pre-reg on Row 7 must declare `scratch_policy` explicitly (C13 BINDING per `memory/feedback_chordia_unlock_deployment_gate_audit_checklist.md`).
7. **`expectancy_r_stored` is metadata, not a sort key.** Per methodology § "Sort rule". The companion column `stored_minus_mode_a` lets reviewers see the Mode-B vs Mode-A divergence magnitude.
8. **MNQ full table not duplicated inline.** 783 rows would exceed practical doc length; the CSV at `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` is canonical. The `<details>` section above provides a reproduction snippet.

## Reproduction

From the worktree root `C:/Users/joshd/canompx3/.worktrees/chordia-audit-queue-v2-2026-05-12`:

```bash
# Re-generate the CSV from scratch
python research/chordia_queue_recompute.py

# Inspect the AUDIT_GAP_ONLY 8 rows
python -c "
import csv
with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    for r in csv.DictReader(f):
        if r['queue_tier']=='AUDIT_GAP_ONLY':
            print(f\"{r['strategy_id']:60s} t={r['chordia_t']:>7} n_oos={r['n_oos']:>3} tier={r['oos_power_tier']}\")"

# Verify tier counts match this MD
python -c "
import csv
from collections import Counter
with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    c = Counter(r['queue_tier'] for r in csv.DictReader(f))
print(dict(c))
# Expect: {'DEFERRED_FILTER_EXCLUDED': 593, 'BLOCKED_ON_GAP': 243, 'AUDIT_GAP_ONLY': 8}"
```

Companion artifacts:
- Methodology: `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`
- Gap-impact map: `docs/audit/results/2026-05-12-chordia-audit-queue-blocked-reasons.md`
- Top-3 next-audit candidates: `docs/audit/results/2026-05-12-chordia-audit-queue-top3-prereg-recommendation.md`
- Plan: `docs/plans/2026-05-12-chordia-audit-queue-v2-plan.md`
