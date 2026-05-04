# MNQ retirement queue — formal committee-action doc

**Generated:** 2026-04-19
**Purpose:** Replaces the SUPERSEDED "Category D 4 CRITICAL lanes" framing in `2026-04-19-mnq-mode-a-committee-review-pack.md` with the environment-controlled retirement candidates identified by `2026-04-19-regime-drift-control-critical-lanes.md`. This doc is an ACTION ITEM — a structured list for a committee vote, with rationale per lane.

## Methodology — what qualifies for this queue

A lane qualifies for RETIREMENT CONSIDERATION if its early→late Mode A Sharpe drop EXCEEDS the portfolio-wide MNQ drop by more than 0.30 (in the losing direction). The portfolio baseline is computed by aggregating per-trade pnl_r across all 36 active MNQ lanes (Mode A IS rr_target, entry_model, confirm_bars, filter applied canonically).

**Portfolio-wide MNQ Sharpe drop 2022-23 → 2024-25: −0.41.**

Retirement tiers (pre-declared, see `research/regime_drift_control_critical_lanes.py:13-22`):
- **Tier 1 — DECAY (vote THIS WEEK):** excess-drop > 0.60 vs portfolio (i.e., lane-specific decay > 1.00 absolute Sharpe).
- **Tier 2 — REVIEW (vote NEXT 2 WEEKS):** excess-drop 0.10 – 0.60 vs portfolio (i.e., lane drop 0.51 – 1.00 absolute).
- **HOLD — REGIME-STRESSED:** excess-drop within ±0.30 of portfolio. Do NOT retire; continue monitoring.
- **BETTER-THAN-PEERS:** excess-drop > +0.30 vs portfolio (lane holds up BETTER under stress). Do NOT retire.

All thresholds are from the regime-drift control's pre-declared rubric, not rescued post-hoc.

---

## Tier 1 — DECAY (vote recommended THIS WEEK)

4 lanes with Sharpe drop > 1.00 absolute, excess-drop > 0.60 vs portfolio. These are the honest retirement candidates the original 2026-04-19 committee pack missed.

| # | Strategy ID | Early Sh | Late Sh | Drop | Excess vs portfolio | Rationale for retirement |
|---|---|---:|---:|---:|---:|---|
| T1.1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | 1.59 | 0.18 | −1.41 | **−1.00** | Sharpe collapsed from 1.59 to 0.18 — 89% of edge disappeared. Environment alone explains only −0.41. Lane-specific decay is the dominant factor. Mechanism (cross-session momentum from SGP to EUR) may have decayed as more participants trade the pattern. |
| T1.2 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 1.48 | 0.21 | −1.28 | **−0.87** | Same mechanism as T1.1, higher RR target. Sibling decay — correlated retirement. Voting retire T1.1 without T1.2 would leave a decayed twin active. |
| T1.3 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 1.42 | 0.16 | −1.26 | **−0.85** | COST_LT12 friction screen edge decayed on EUROPE_FLOW. Note: COST_LT12 on other sessions (COMEX_SETTLE, NYSE_OPEN) holds up — this is session-specific decay, not filter-wide. |
| T1.4 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 0.74 | **−0.43** | −1.17 | **−0.76** | **LATE SHARPE NEGATIVE.** This lane is currently LOSING MONEY in 2024-25 IS. Not regime stress — dead edge. Highest-urgency retirement. |

### Special call-out: T1.4

`MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` is the ONLY Tier 1 lane with a NEGATIVE late-period Sharpe. Every other retirement candidate is still positive — just decayed. T1.4 is actively bleeding. If the committee only has time to vote on ONE lane this week, it should be T1.4.

---

## Tier 2 — REVIEW (vote recommended next 2 weeks)

9 lanes with excess-drop 0.10 – 0.60 vs portfolio (i.e., Sharpe drop 0.51 – 1.00 absolute). Less urgent but non-trivial.

| # | Strategy ID | Early Sh | Late Sh | Drop | Excess vs portfolio |
|---|---|---:|---:|---:|---:|
| T2.1 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | 2.20 | 1.29 | −0.91 | −0.50 |
| T2.2 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 1.18 | 0.37 | −0.81 | −0.40 |
| T2.3 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | 0.92 | 0.12 | −0.80 | −0.39 |
| T2.4 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 0.98 | 0.37 | −0.61 | −0.20 |
| T2.5 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 0.79 | 0.20 | −0.59 | −0.18 |
| T2.6 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 0.81 | 0.29 | −0.53 | −0.12 |
| T2.7 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 0.64 | 0.12 | −0.51 | −0.10 |
| T2.8 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 1.20 | 0.19 | −1.01 | −0.60 |
| T2.9 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | (see 2026-04-19 mode-a-revalidation for exact) | — | varies | borderline |

**Pattern observation:** 5 of the 9 Tier 2 lanes are on `EUROPE_FLOW` — combined with the 3 EUROPE_FLOW entries in Tier 1, 8 of the 13 decay candidates are EUROPE_FLOW lanes. EUROPE_FLOW may itself be a regime-stressed session requiring broader review (see follow-up below).

---

## HOLD — REGIME-STRESSED (do NOT retire on current evidence)

The 4 lanes originally flagged as Category D CRITICAL in the committee review pack:

| # | Strategy ID | Drop | Excess | Original verdict | Reframed verdict |
|---|---|---:|---:|---|---|
| H1 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | −0.36 | +0.05 | RETIRE | HOLD (tracks portfolio) |
| H2 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | −0.02 | +0.39 | RETIRE | **BETTER-THAN-PEERS (keep)** |
| H3 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | +0.49 | +0.90 | RETIRE | **IMPROVING (keep)** |
| H4 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | −0.20 | +0.21 | RETIRE | HOLD |

All 4 false-positives of the original raw-Sharpe-drop framing. Do NOT retire.

---

## BETTER-THAN-PEERS (genuinely outperforming under stress — do NOT retire)

5 lanes whose Sharpe actually WENT UP early → late, despite the −0.41 portfolio-wide drop:

| Strategy ID | Early Sh | Late Sh | Drop | Excess |
|---|---:|---:|---:|---:|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 0.53 | 1.94 | +1.42 | +1.83 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | 0.75 | 2.04 | +1.28 | +1.69 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 1.32 | 2.71 | +1.39 | +1.80 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 1.25 | 2.29 | +1.04 | +1.45 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 1.46 | 2.43 | +0.96 | +1.37 |

**Observation:** COMEX_SETTLE and SINGAPORE_OPEN are the two sessions carrying the portfolio through 2024-25. Both are pre-US-liquidity sessions. Worth confirming in a separate regime diagnostic whether overnight / Asian-session ORB lanes have a structural lead vs US-liquidity sessions in the current regime.

---

## Summary counts

| Tier | Count | Action |
|---|---:|---|
| Tier 1 DECAY | 4 | Vote this week (T1.4 is urgent — negative late Sharpe) |
| Tier 2 REVIEW | 9 | Vote within 2 weeks |
| HOLD / regime-stressed | 4 | Do NOT retire |
| BETTER-than-peers | 5 | Do NOT retire |
| Remaining on-portfolio | 14 | No action required |
| **Total active MNQ** | **36** | — |

---

## Follow-up research suggested by this queue

1. **EUROPE_FLOW session regime audit:** 8 of 13 decay candidates are on EUROPE_FLOW. Separate analysis of whether EUROPE_FLOW (Brisbane 18:00–23:00 = London pre-/early session) has deteriorated structurally as opposed to hosting lanes that decayed independently.
2. **COMEX_SETTLE + SINGAPORE_OPEN regime dominance:** 4 of 5 BETTER-than-peers lanes are on these two sessions. May indicate a regime shift favouring Asian-to-European overnight liquidity breaks over US-session breaks.
3. **CROSS_SGP_MOMENTUM mechanism retrospective:** T1.1/T1.2/T2.8 all use this mechanism and all decayed. Mechanism-level post-mortem before attempting to rediscover a related signal.

## Audit trail

- Primary source: `docs/audit/results/2026-04-19-regime-drift-control-critical-lanes.md`
- Parent pack reframed: `docs/audit/results/2026-04-19-mnq-mode-a-committee-review-pack.md`
- Supersedes Category D 4-lane RETIRE framing
- Read-only audit; `validated_setups` NOT mutated. Committee decision required to flip status.
