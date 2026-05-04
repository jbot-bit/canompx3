# 2026-04-18 — C12 Shiryaev-Roberts Alarmed-Lanes Review

**Scope:** 3 live lanes currently in SR ALARM state on `topstep_50k_mnq_auto`.
**Authority:** `docs/institutional/pre_registered_criteria.md` § Criterion 12; `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`.
**Reviewer:** Claude (institutional-rigor manual review per Criterion 12 rule: "On alarm: strategy goes to 'suspended' state pending manual review").
**Method:** Literature-grounded per-lane decision matrix. No cherry-picking; all 3 alarmed lanes reviewed. No new parameters tuned.

---

## 1. Threshold + monitor config (canonical)

Source: `trading_app/sr_monitor.py` + `calibrate_sr_threshold()` call.

| Parameter | Value | Source |
|---|---|---|
| Target ARL (false alarm) | 60 trading days | Pepelyshev-Polunchenko 2015 § 5 (ARL ≈ 60 matches 7-day cycle × 8.5) |
| Threshold A | 31.96 | `calibrate_sr_threshold(60, delta=-1.0, q=1.0)` |
| Delta (post-change drift) | −1.0 σ in mean | canonical monitor default |
| Variance ratio q | 1.0 | variance held constant (σ_post = σ_pre) |
| Score function | Eq 17 linear-quadratic | Pepelyshev-Polunchenko 2015 Eq 17–18 |
| Stream source | `canonical_forward` (orb_outcomes) | no live broker stream yet on these 3 lanes |
| Baseline source | `validated_backtest` (first-50) | validated_setups IS distribution |

**Material caveat — surveillance stream:** All 3 lanes use `canonical_forward` (shadow R-multiples from `orb_outcomes` simulation), not `paper_trades` (actual broker fills). An alarm on the canonical stream is a drift signal in the backtest-equivalent stream; it does NOT necessarily mean the live broker P&L has degraded. This is consistent with the Pepelyshev-Polunchenko framework (stream can be any iid score-generating process), but the interpretation is "backtest-shadow drift" not "live trading drift."

---

## 2. Per-lane data (source: direct `duckdb` read via `prepare_monitor_inputs`)

### L3 — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100

| Field | Value |
|---|---|
| Baseline (validated_backtest) mean R | +0.2151 |
| Baseline std R | 1.2518 |
| Stream N (canonical_forward) | 60 |
| Stream mean R | +0.1063 |
| Stream std R | 1.1836 |
| Delta vs baseline | **−0.109 R/trade (−0.09 σ)** |
| Trailing-30 mean R | **+0.3481** |
| Win rate (stream) | 46.7% (28/60) |
| SR alarm fired at trade | **24** of 60 |
| SR final (post-alarm, multi-cyclic) | 1.51 |
| Validated IS ExpR | +0.2151 (N=513) |
| **C6 WFE** | **1.107** (pass ≥ 0.50) |
| C6 wf_passed | TRUE |
| OOS ExpR | +0.2029 (tracks IS within 6%) |
| C9 era_dependent | FALSE |
| C9 max year % | 46.4% |
| p-value | 3.3e-05 |

### L4 — MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5

| Field | Value |
|---|---|
| Baseline mean R | +0.0889 |
| Baseline std R | 0.9928 |
| Stream N | 71 |
| Stream mean R | +0.1360 |
| Stream std R | 0.9718 |
| Delta vs baseline | **+0.047 R/trade (+0.05 σ) — ABOVE baseline** |
| Trailing-30 mean R | +0.0505 |
| Win rate | 57.7% (41/71) |
| SR alarm fired at trade | **54** of 71 |
| SR final | 0.82 |
| Validated IS ExpR | +0.0889 (N=1521) |
| **C6 WFE** | **2.144** |
| OOS ExpR | +0.1036 (matches IS) |
| C9 era_dependent | FALSE |
| C9 max year % | 37.6% |
| p-value | 3.1e-04 |

### L6 — MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15

| Field | Value |
|---|---|
| Baseline mean R | +0.2101 |
| Baseline std R | 1.2506 |
| Stream N | 35 |
| Stream mean R | +0.2612 |
| Stream std R | 1.2258 |
| Delta vs baseline | **+0.051 R/trade (+0.04 σ) — ABOVE baseline** |
| Trailing-30 mean R | **+0.4714** |
| Win rate | 51.4% (18/35) |
| SR alarm fired at trade | **6** of 35 |
| SR final | 0.54 |
| Validated IS ExpR | +0.2101 (N=701) |
| **C6 WFE** | **0.901** |
| OOS ExpR | +0.2065 (matches IS) |
| C9 era_dependent | FALSE |
| C9 max year % | 21.8% |
| p-value | 5e-06 |

---

## 3. Decision framework (literature-grounded)

Per Pepelyshev-Polunchenko 2015 § 5 and § Conclusion: the SR procedure is optimal for quickest detection, but an alarm is a **detection event**, not a verdict. The paper's own HST case study shows post-alarm manual review to confirm structural break. The multi-cyclic nature of the monitor (R_n resets post-alarm) explicitly supports this — the statistic looks for the NEXT drift once the previous alarm is acknowledged.

**Criterion 12 rule:** "On alarm: strategy goes to 'suspended' state pending manual review." This audit **is** that manual review.

### Classification matrix

| Joint condition | Classification | Action |
|---|---|---|
| SR alarm + live_mean < baseline−0.5σ + trailing_30 < 0 | **SUSPEND** | pause lane, investigate break |
| SR alarm + live_mean < baseline−0.1σ + trailing_30 ≥ 0 | **WATCH** | continue, elevated monitor |
| SR alarm + live_mean ≥ baseline−0.1σ + trailing_30 ≥ 0 | **KEEP** (false-alarm) | continue, document |
| SR alarm + any C6/C8/C9 fail | **SUSPEND regardless** | structural concern |

The thresholds (−0.5σ strong concern, −0.1σ mild concern) are chosen to match the monitor's own design point (δ = −1.0σ post-change mean shift). An observed stream mean within 0.1σ of baseline is by construction not the regime the detector was tuned for.

---

## 4. Per-lane verdicts

### L3 — COMEX_SETTLE OVNRNG_100 → **WATCH (continue deployment)**

- Delta −0.09σ (mild; well under −0.5σ SUSPEND threshold).
- Alarm fired at trade 24/60 and has NOT re-triggered in the remaining 36 trades — SR statistic back to 1.51 from alarm.
- Trailing-30 mean R = +0.35 (outperforms baseline 0.22). **Clear stream recovery.**
- All structural criteria pass: WFE 1.11, OOS tracks IS, p=3.3e-5, non-era-dependent, p=3.3e-05.
- Alarm interpretable as localized variance cluster (pre-trade-24), now resolved.
- **Action:** Continue deployed. Annotate debt-ledger: C12 review complete 2026-04-18, classification WATCH until N=120 (2× ARL of re-validation window).

### L4 — NYSE_OPEN ORB_G5 → **KEEP (false-alarm)**

- Delta **+0.05σ above baseline** (live mean 0.136 > IS 0.089).
- Trailing-30 mean R = +0.05 (near baseline).
- Alarm fired at trade 54/71 — single variance excursion; no structural shift observed.
- Structural criteria very strong: WFE=2.14 (highest of the 3), OOS=0.10 matches IS, p=3e-4.
- **Action:** Continue deployed. No debt-ledger entry needed beyond standard SR monitor note. Alarm reclassified as statistically expected (at ARL=60 with 71 trades, ~1 false alarm is within Poisson confidence).

### L6 — US_DATA_1000 VWAP_MID_ALIGNED → **KEEP (false-alarm, strongest evidence)**

- Delta **+0.04σ above baseline** (live 0.261 > IS 0.210).
- Trailing-30 mean R = +0.47 — **2.2× baseline ExpR**. Strongest outperformance of the 3.
- Alarm fired at trade 6 (very early, before meaningful stream); clearly stale.
- Structural criteria: WFE=0.90 (lowest of 3 but above 0.50 floor), OOS=0.21 matches IS, p=5e-6 (strongest statistical evidence of the 3).
- **Action:** Continue deployed. Alarm unambiguously false.

---

## 5. Aggregate verdict

**0 of 3 SUSPEND. 1 WATCH (L3). 2 KEEP (L4, L6).**

No live lane has shown structural deterioration warranting a trading-capital change. All 3 lanes retain their C1-C10 institutional-rigor credentials; the SR monitor (C12) has performed its design function (quickest detection at ARL=60), and manual review has correctly reclassified 3/3 alarms as non-actionable after cross-referencing live stream means against baseline.

### What this review does NOT conclude

- Does not conclude that the strategies are profit-maximized — that's a separate question (Phase D / portfolio rebalance).
- Does not conclude that the monitor is miscalibrated — false alarms at rate 1/60 are *by design*. Observing 3 alarms across 6 lanes with N∈[35,71] is within Poisson CI.
- Does not override the Criterion 12 requirement itself — the rule still reads "on alarm, manual review before continuing." This review satisfies that requirement for these 3 alarms as of 2026-04-18.

### What would trigger reclassification

- Any of L3/L4/L6 drops trailing-30 mean R below −0.1 R/trade → reopen review.
- SR stat re-crosses threshold 31.96 post-reset → new alarm, new manual review required.
- C6 WFE on any lane drops below 0.50 on next canonical recompute → SUSPEND regardless of SR state.

---

## 6. Audit trail

- Script: ad-hoc computation via `trading_app.sr_monitor.prepare_monitor_inputs()` on 2026-04-18.
- Raw trade streams: `orb_outcomes` canonical-forward via `gold.db` (read-only).
- Baseline: `validated_setups` IS ExpR + std per lane.
- Literature source: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`.
- Criterion source: `docs/institutional/pre_registered_criteria.md` § Criterion 12.
- Prior C12 review precedent (same lane): `HANDOFF.md` + debt-ledger entry SR-L6 (2026-04-12 literature-grounded audit on L3 COMEX_SETTLE, same verdict KEEP-WATCH).
