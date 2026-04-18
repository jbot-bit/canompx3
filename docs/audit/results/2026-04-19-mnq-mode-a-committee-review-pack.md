# Mode A committee review pack — live-book action recommendations

**Generated:** 2026-04-19 (overnight session)
**Reframed:** 2026-04-19 post-regime-drift control (see Addendum at top)
**Primary source:** `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`
**Purpose:** Synthesize the 38/38 Mode A drift finding into per-lane action categories that a committee can vote on. No auto-promote; no auto-retire. This document recommends; decisions require human sign-off.

---

## ADDENDUM — Regime-drift reframe (2026-04-19, supersedes Category D RETIRE framing)

**Source:** `docs/audit/results/2026-04-19-regime-drift-control-critical-lanes.md` (commit 9937ebf6).

The original Category D classification (4 lanes flagged for RETIRE) attributed Mode A Sharpe drop to **lane-specific decay**. A regime-drift control was run to partition lane-drop from environment-drop.

**Environment baseline:** all 36 active MNQ lanes pooled early (2022-2023) vs late (2024-2025) Mode A IS subsets → Sharpe drop **−0.41 annualized portfolio-wide**.

**Per-lane excess drop vs environment:**

| Lane | Early Sh | Late Sh | Lane drop | Excess drop vs portfolio (−0.41) | Original verdict | **Reframed verdict** |
|---|---:|---:|---:|---:|---|---|
| CR1 MNQ EUROPE_FLOW OVNRNG_100 long RR1.0 | 0.78 | 0.43 | −0.36 | +0.05 | CRITICAL (RETIRE) | **REGIME (hold)** |
| CR2 MNQ EUROPE_FLOW OVNRNG_100 long RR1.5 | 0.75 | 0.73 | −0.02 | +0.39 | CRITICAL (RETIRE) | **BETTER-THAN-PEERS (keep)** |
| CR3 MNQ NYSE_OPEN X_MES_ATR60 long RR1.0 | 0.47 | 0.95 | +0.49 | +0.90 | CRITICAL (RETIRE) | **BETTER-THAN-PEERS (improving)** |
| CR4 MNQ NYSE_OPEN X_MES_ATR60 long RR1.5 | 0.59 | 0.40 | −0.20 | +0.21 | CRITICAL (RETIRE) | **REGIME (hold)** |

**None of the 4 CRITICAL lanes drop more than 0.30 worse than the broader MNQ environment.** Two are actually better-than-peers; the deepest-drop of the four (CR1) tracks the environment almost exactly.

### Revised committee recommendation

- **RETIRE vote withdrawn for all 4 CRITICAL lanes.** The RETIRE framing was over-attributed to lane decay without controlling for environment stress.
- **Reclassify Category D → REGIME-STRESSED WATCH.** Continue live; re-evaluate Q3 2026 against fresh OOS. Downgrade from CORE to REGIME tier is optional but not mandatory.
- **Broader finding:** 13/36 MNQ lanes show Sharpe drops > 0.50 (twice the environment). THOSE are the honest retirement candidates — see "All 36 MNQ lanes early/late" table in the control doc. Notable deep-drop non-flagged lanes:
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` (drop −1.41)
  - `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` (drop −1.28)
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` (drop −1.26)
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` (drop −1.17, Sharpe NEGATIVE late)
- **Committee should re-prioritize:** instead of voting on the 4 (falsely-flagged) lanes, vote on the 4-to-13 lanes with genuine excess decay (beyond portfolio).

### Methodology note (Mode A → environment-controlled)

Raw Mode A Sharpe drop conflates (a) honest lane-specific decay with (b) environment-wide regime compression. The 2024-2025 MNQ intraday-breakout environment produced a broad −0.41 Sharpe compression across all 36 lanes. A retirement framing anchored only on raw Mode A drift is systematically biased toward false-positive retirement.

Going forward: **lane retirement decisions require environment-controlled excess drop**, not raw Sharpe drop alone. Recommended threshold: `lane_drop - portfolio_drop < -0.50` → candidate for RETIRE vote. Between -0.30 and -0.50 → downgrade to REGIME tier. Within -0.30 of portfolio → HOLD (regime stress, not lane decay).

The body of this document (below) is preserved for audit trail. The Category D 4-lane RETIRE framing in § "Category D — CRITICAL" is SUPERSEDED by this addendum.

---



## Framework

Each of the 38 active lanes is classified into one of four categories:

| Category | Criteria | Recommended action |
|---|---|---|
| **A — CONFIRM** | Mode A ExpR is WITHIN 0.03 of stored; Mode A Sharpe within 0.20; no Mode-B contamination | KEEP active. No review needed. |
| **B — KEEP WITH NOTE** | Mode A ExpR drift modest (Δ 0.03-0.05 absolute) OR Sharpe drift 0.20-0.5 | KEEP but downgrade confidence; cite Mode A baseline going forward. |
| **C — REVIEW** | Mode A ExpR drop > 0.05 OR Sharpe drop > 0.5 OR Mode-B contaminated with Mode A < 0.05 ExpR | Committee vote: retire / downgrade to REGIME tier / accept lower expected return? |
| **D — CRITICAL** | Mode A ExpR drop > 0.10 OR Sharpe drop > 0.8 OR Mode A ExpR < 0.03 | Strong case for RETIRE; committee vote urgent. |

## Per-lane classification (all 38 active lanes)

### Category A — CONFIRM (0 lanes)

Every active lane fails at least one drift threshold under Mode A. **None are in Category A.**

### Category B — KEEP WITH NOTE (9 lanes)

Lanes where Mode A ExpR is within 0.03 of stored but Sharpe or sample drifted. These lanes produce similar per-trade expectancy under Mode A but different Sharpe due to the removed 2026 Q1 trades.

| Instr | Lane | Stored ExpR / Mode A | Stored Sh / Mode A | Note |
|---|---|---|---|---|
| MES | CME_PRECLOSE ORB_G8 long RR1.0 | 0.173 / 0.280 | 1.34 / 1.49 | ExpR UP 0.11 — the 2026 Q1 trades were below average for this lane |
| MNQ | CME_PRECLOSE X_MES_ATR60 long RR1.0 | 0.170 / 0.214 | 1.88 / 1.58 | ExpR UP 0.04; Sh drift -0.30 (in-band) |
| MNQ | COMEX_SETTLE COST_LT12 long RR1.0 | 0.110 / 0.104 | 1.74 / 1.11 | ExpR flat; Sh drift -0.63 Category C by Sharpe |
| MNQ | COMEX_SETTLE OVNRNG_100 long RR1.0 | 0.173 / 0.184 | 1.76 / 1.29 | ExpR UP 0.01 |
| MNQ | COMEX_SETTLE X_MES_ATR60 long RR1.0 | 0.151 / 0.195 | 1.78 / 1.62 | ExpR UP 0.04 |
| MNQ | COMEX_SETTLE X_MES_ATR60 long RR1.5 | 0.161 / 0.198 | 1.46 / 1.26 | ExpR UP 0.04 |
| MNQ | COMEX_SETTLE ORB_G5 long RR2.0 | 0.089 / 0.106 | 1.02 / 0.85 | ExpR UP 0.02 |
| MNQ | EUROPE_FLOW CROSS_SGP_MOMENTUM long RR2.0 | 0.123 / 0.112 | 1.12 / 0.75 | ExpR down 0.01; Mode-B contaminated |
| MNQ | TOKYO_OPEN ORB_G5 long RR1.5 | 0.095 / 0.099 | 1.33 / 0.95 | ExpR flat |

### Category C — REVIEW (15 lanes)

Lanes where the committee should discuss: is this lane still deployable at its reduced Mode A ExpR? Does the Sharpe drop reflect honest decay or sample-smaller variance?

| Instr | Lane | Stored ExpR / Mode A | Stored Sh / Mode A | ΔExpR | Mode-B |
|---|---|---|---|---|---|
| MES | CME_PRECLOSE COST_LT08 long RR1.0 | 0.196 / 0.328 | 1.25 / 1.45 | +0.13 | N |
| MNQ | COMEX_SETTLE ORB_G5 long RR1.0 | 0.089 / 0.078 | 1.54 / 0.95 | -0.01 Sh-heavy | N |
| MNQ | COMEX_SETTLE ORB_G5 long RR1.5 | 0.112 / 0.092 | 1.52 / 0.88 | -0.02 Sh-heavy | N |
| MNQ | COMEX_SETTLE OVNRNG_100 long RR1.5 | 0.215 / 0.187 | 1.70 / 1.00 | -0.03 Sh-heavy | N |
| MNQ | EUROPE_FLOW COST_LT12 long RR1.0 | 0.092 / 0.054 | 1.32 / 0.51 | -0.04 | N |
| MNQ | EUROPE_FLOW CROSS_SGP_MOMENTUM long RR1.0 | 0.085 / 0.050 | 1.18 / 0.50 | -0.03 | Y |
| MNQ | EUROPE_FLOW ORB_G5 long RR1.0 | 0.066 / 0.035 | 1.16 / 0.42 | -0.03 | N |
| MNQ | EUROPE_FLOW CROSS_SGP_MOMENTUM long RR1.5 | 0.094 / 0.081 | 1.01 / 0.64 | -0.01 | Y |
| MNQ | EUROPE_FLOW ORB_G5 long RR1.5 | 0.074 / 0.077 | 1.03 / 0.72 | +0.00 | N |
| MNQ | EUROPE_FLOW COST_LT12 long RR1.5 | 0.107 / 0.105 | 1.21 / 0.78 | -0.00 Sh-heavy | N |
| MNQ | EUROPE_FLOW ORB_G5 long RR2.0 | 0.096 / 0.074 | 1.14 / 0.58 | -0.02 Sh-heavy | N |
| MNQ | NYSE_OPEN COST_LT12 long RR1.0 | 0.087 / 0.069 | 1.43 / 0.79 | -0.02 Sh-heavy | N |
| MNQ | NYSE_OPEN ORB_G5 long RR1.0 | 0.089 / 0.066 | 1.47 / 0.77 | -0.02 | N |
| MNQ | NYSE_OPEN COST_LT12 long RR1.5 | 0.105 / 0.089 | 1.36 / 0.80 | -0.02 | N |
| MNQ | NYSE_OPEN ORB_G5 long RR1.5 | 0.107 / 0.086 | 1.39 / 0.78 | -0.02 | N |
| MNQ | SINGAPORE_OPEN ATR_P50 long RR1.5 O15 | 0.109 / 0.205 | 1.14 / 1.50 | +0.10 (UP) | Y |
| MNQ | SINGAPORE_OPEN ATR_P50 long RR1.5 O30 | 0.125 / 0.221 | 1.28 / 1.56 | +0.10 (UP) | Y |
| MNQ | TOKYO_OPEN COST_LT12 long RR1.0 | 0.096 / 0.079 | 1.31 / 0.71 | -0.02 Sh-heavy | N |
| MNQ | TOKYO_OPEN COST_LT12 long RR1.5 | 0.129 / 0.104 | 1.39 / 0.74 | -0.03 | N |
| MNQ | TOKYO_OPEN ORB_G5 long RR2.0 | 0.087 / 0.123 | 1.04 / 0.99 | +0.04 (UP) | N |
| MNQ | US_DATA_1000 X_MES_ATR60 long RR1.0 | 0.100 / 0.077 | 1.14 / 0.59 | -0.02 Sh-heavy | N |
| MNQ | US_DATA_1000 VWAP_MID_ALIGNED long RR1.0 | 0.149 / 0.132 | 1.74 / 1.13 | -0.02 | Y |
| MNQ | US_DATA_1000 VWAP_MID_ALIGNED long RR1.5 | 0.210 / 0.184 | 1.87 / 1.21 | -0.03 | Y |
| MNQ | US_DATA_1000 VWAP_MID_ALIGNED long RR2.0 | 0.176 / 0.148 | 1.27 / 0.79 | -0.03 | Y |

### Category D — CRITICAL (4 lanes) — SUPERSEDED, see Addendum at top

**[SUPERSEDED 2026-04-19]** Regime-drift control shows all 4 lanes track the portfolio-wide −0.41 Sharpe drop. RETIRE framing withdrawn. Preserved below for audit trail only.

Strong case for RETIRE under Mode A. ExpR drop > 0.05 absolute OR Mode A ExpR below 0.05.

| Instr | Lane | Stored ExpR / Mode A | Stored Sh / Mode A | ΔExpR | ΔSh |
|---|---|---|---|---|---|
| MNQ | EUROPE_FLOW OVNRNG_100 long RR1.0 | 0.118 / 0.056 | 1.22 / 0.37 | **-0.062** | **-0.85** |
| MNQ | EUROPE_FLOW OVNRNG_100 long RR1.5 | 0.171 / 0.118 | 1.39 / 0.62 | **-0.053** | **-0.77** |
| MNQ | NYSE_OPEN X_MES_ATR60 long RR1.0 | 0.137 / 0.078 | 1.54 / 0.57 | **-0.058** | **-0.97** |
| MNQ | NYSE_OPEN X_MES_ATR60 long RR1.5 | 0.132 / 0.066 | 1.17 / 0.38 | **-0.067** | **-0.79** |
| MNQ | US_DATA_1000 ORB_G5 long RR1.5 O15 | 0.093 / 0.057 | 1.16 / 0.51 | **-0.037** | **-0.65** (borderline C/D) |

Committee recommendation: **RETIRE** the 4 lanes above, OR downgrade them from CORE to REGIME tier with reduced position sizing per `trading_app.lane_allocator` config.

Rationale:
- The 4 CRITICAL lanes' Mode A Sharpe is below 0.60 annualized. That's below institutional deployment bar (typical prop-firm rules implicitly require Sharpe > 1.0 for stable performance).
- The 4 CRITICAL lanes' Mode A ExpR is 0.056 to 0.078. At MNQ's $2.74 RT cost (canonical `pipeline.cost_model.COST_SPECS`) and typical ORB risk $30-50, that's barely above breakeven net of friction.
- The stored values (ExpR 0.12-0.17, Sharpe 1.2-1.5) are Mode-B grandfathered and reflect a different sample — they misrepresent what the lane has done IS-strict.

## Summary counts

| Category | Count | Note |
|---|---|---|
| A — CONFIRM | 0 | All 38 lanes have at least one drift flag |
| B — KEEP WITH NOTE | 9 | Sharpe drifted but ExpR flat; low committee-urgency |
| C — REVIEW | 25 | Many Sh-heavy (stored Sharpe 1.3-1.9 under Mode B, Mode A 0.5-0.9) — Mode-B sample-size inflated the Sharpe; not necessarily decay |
| D — CRITICAL | 4 | Strong case for RETIRE |

## Recommended committee agenda

1. **IMMEDIATE (this week):** Vote on Category D 4 lanes. Retire or downgrade.
2. **NEAR-TERM (next 2 weeks):** Review the 25 Category C lanes. For each, decide:
   - Accept lower Mode A ExpR as realistic expectation?
   - Trigger a lane-specific re-validation pre-reg?
   - Retire proactively (if Mode A Sharpe < 0.80)?
3. **DEFERRED:** The 9 Mode-B-contaminated lanes (last_trade_day ≥ 2026-01-01) should receive a fresh Mode A re-validation once OOS accumulates further. Committee pre-decides: stay active, or pause pending re-validation?

## Not in this review

- This is NOT a retirement decree. `validated_setups` rows are NOT mutated by this audit. The committee retains final call.
- This review does NOT include per-lane walk-forward efficiency or DSR. If needed, commission a per-lane WFE+DSR deep-dive as a follow-up Phase 8b.
- This review does NOT account for portfolio-level correlation; a lane's Sharpe drop may be offset by its diversification value. Lane-correlation gate evaluation is separate (Phase 12 design-doc queue).

## Audit trail

- Primary source: `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md` (commit ef0494d9)
- Canonical filter delegation: `research.filter_utils.filter_signal` (per Phase 2 fix, commit 09d5cc0f)
- Mode-B grandfather warning: `.claude/rules/research-truth-protocol.md` § Mode B grandfathered validated_setups baselines
- Holdout boundary: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` = 2026-01-01 (Mode A)
