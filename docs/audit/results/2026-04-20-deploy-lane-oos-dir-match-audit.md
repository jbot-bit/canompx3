# 2026 OOS per-lane dir_match audit — 6 DEPLOY lanes

**Generated:** 2026-04-20
**Script:** `research/deploy_lane_oos_dir_match_audit.py`
**Mode A sacred cutoff:** `HOLDOUT_SACRED_FROM` = 2026-01-01 (from `trading_app.holdout_policy`)
**Filter source:** canonical `trading_app.config.ALL_FILTERS` via `research.filter_utils.filter_signal` (no re-encoded logic)
**Power thresholds:** canonical `research.oos_power.POWER_TIERS` (80% / 50%)

## Audited claim

Mandate #4 from `memory/next_session_mandates_2026_04_20.md`: per-lane 2026 OOS dir_match audit on the 6 DEPLOY lanes in `docs/runtime/lane_allocation.json` (`topstep_50k_mnq_auto` rebalance 2026-04-18). Must use canonical `oos_power` helper per RULE 3.3; must provide per-lane breakdown per RULE 14 — no pooled p-value claim.

## Pre-committed decision rule

Locked in `docs/runtime/stages/deploy_lane_oos_dir_match_audit.md` before implementation.

| dir_match | OOS power tier | Lane verdict |
|---|---|---|
| TRUE  | CAN_REFUTE            | ALIVE_CONFIRMED |
| TRUE  | DIRECTIONAL_ONLY      | ALIVE_PROVISIONAL |
| TRUE  | STATISTICALLY_USELESS | UNVERIFIED_ALIVE |
| FALSE | CAN_REFUTE            | **DEAD** |
| FALSE | DIRECTIONAL_ONLY      | WATCH |
| FALSE | STATISTICALLY_USELESS | UNVERIFIED |

## Lanes under test

| # | strategy_id | session | orb_min | RR | filter |
|---:|---|---|---:|---:|---|
| L1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | EUROPE_FLOW | 5 | 1.5 | ORB_G5 |
| L2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | SINGAPORE_OPEN | 15 | 1.5 | ATR_P50 |
| L3 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | COMEX_SETTLE | 5 | 1.5 | ORB_G5 |
| L4 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | NYSE_OPEN | 5 | 1.0 | COST_LT12 |
| L5 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | TOKYO_OPEN | 5 | 1.5 | COST_LT12 |
| L6 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | US_DATA_1000 | 15 | 1.5 | ORB_G5 |

## IS baseline + 2026 OOS per-lane statistics

| # | strategy_id | IS N | IS mean | IS std | IS WR | OOS N | OOS mean | OOS std | OOS WR | OOS t (1-sample) | p (1-sided) | OOS date span |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 1583 | +0.0643 | 1.123 | 0.474 | 72 | +0.2928 | 1.166 | 0.556 | +2.13 | 0.018 | 2026-01-02 … 2026-04-16 |
| L2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 913 | +0.1130 | 1.152 | 0.484 | 54 | +0.0617 | 1.199 | 0.444 | +0.38 | 0.353 | 2026-01-09 … 2026-04-06 |
| L3 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 1555 | +0.0915 | 1.147 | 0.477 | 66 | +0.0058 | 1.181 | 0.424 | +0.04 | 0.484 | 2026-01-02 … 2026-04-16 |
| L4 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 1669 | +0.0820 | 0.958 | 0.561 | 71 | +0.1360 | 0.979 | 0.577 | +1.17 | 0.123 | 2026-01-02 … 2026-04-16 |
| L5 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 950 | +0.1222 | 1.152 | 0.487 | 70 | +0.1530 | 1.195 | 0.486 | +1.07 | 0.144 | 2026-01-02 … 2026-04-16 |
| L6 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1491 | +0.1055 | 1.202 | 0.459 | 55 | +0.1609 | 1.237 | 0.473 | +0.96 | 0.170 | 2026-01-02 … 2026-04-02 |

## RULE 3.3 power-floor evaluation per lane

| # | strategy_id | Cohen's d (IS) | OOS N | OOS power | Tier (canonical) | N needed for 80% | dir_match | **Verdict** |
|---:|---|---:|---:|---:|---|---:|:---:|---|
| L1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 0.057 | 72 | 7.7% | STATISTICALLY_USELESS | 2,401 | TRUE | **UNVERIFIED_ALIVE** |
| L2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 0.098 | 54 | 10.9% | STATISTICALLY_USELESS | 818 | TRUE | **UNVERIFIED_ALIVE** |
| L3 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 0.080 | 66 | 9.8% | STATISTICALLY_USELESS | 1,235 | TRUE | **UNVERIFIED_ALIVE** |
| L4 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 0.086 | 71 | 11.0% | STATISTICALLY_USELESS | 1,073 | TRUE | **UNVERIFIED_ALIVE** |
| L5 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 0.106 | 70 | 14.1% | STATISTICALLY_USELESS | 701 | TRUE | **UNVERIFIED_ALIVE** |
| L6 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 0.088 | 55 | 9.8% | STATISTICALLY_USELESS | 1,021 | TRUE | **UNVERIFIED_ALIVE** |

## Portfolio summary

- **UNVERIFIED_ALIVE:** 6/6

**UNVERIFIED lanes (power < 50%, do NOT kill):** 6
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` — OOS power 7.7% (STATISTICALLY_USELESS); needs N=2,401 for 80% power
  - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` — OOS power 10.9% (STATISTICALLY_USELESS); needs N=818 for 80% power
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` — OOS power 9.8% (STATISTICALLY_USELESS); needs N=1,235 for 80% power
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — OOS power 11.0% (STATISTICALLY_USELESS); needs N=1,073 for 80% power
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` — OOS power 14.1% (STATISTICALLY_USELESS); needs N=701 for 80% power
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` — OOS power 9.8% (STATISTICALLY_USELESS); needs N=1,021 for 80% power

## RULE 3.3 compliance note

Every `dir_match=FALSE` finding above is paired with its RULE 3.3 power tier. No lane is labeled DEAD unless its OOS power to detect the IS effect reaches CAN_REFUTE (≥80%). Where OOS power is STATISTICALLY_USELESS (<50%), the dir_match=FALSE outcome is noise-consistent and the lane is labeled UNVERIFIED — not DEAD. This replaces the pre-correction 2026-04-20 `bull_short_avoidance` error where a 7.9%-power OOS was used as a binary kill criterion. See `feedback_oos_power_floor.md` + PR #32.

## RULE 14 compliance note

This audit is per-lane by construction — no pooled p-value is computed across the 6 lanes. RULE 14 explicitly requires a per-lane breakdown before any capital action rests on pooled evidence. See `feedback_per_lane_breakdown_required.md`.

## Classification

- **VALID** if canonical filter delegation verified + `oos_power.power_verdict` invoked + per-lane table + no pooled claim.
- **CONDITIONAL** if any caveat applies (e.g., lanes with IS std near zero causing degenerate Cohen's d).

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/deploy_lane_oos_dir_match_audit.py
```

No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.