# L2 MNQ SINGAPORE_OPEN ATR_P50 — regime-gate vs arithmetic-gate confirmatory audit

**Date:** 2026-04-21
**Branch:** `research/l2-atr-p50-arithmetic-vs-regime-audit`
**Script:** `research/audit_l2_atr_p50_regime_vs_arithmetic.py`
**Parent claim:** `docs/runtime/lane_allocation.json` 2026-04-18 rebalance — lane deployed with annual_r=44.0, ExpR=+0.2407R, N=137, WR=53.3%, status HOT/DEPLOY.
**Rule:** `backtesting-methodology.md § RULE 8.2` ARITHMETIC_ONLY; § RULE 7 TAUTOLOGY; `research-truth-protocol.md § 10` confirmatory audit.
**Classification:** confirmatory audit (no new pre-reg required).

---

## TL;DR

**ATR_P50 passes neither RULE 8.2 flag nor RULE 8.2 behavioral flag cleanly.** Δ_WR = +3.51pp (z-p=0.146), Δ_ExpR = +0.1340R. Not arithmetic; not conclusively behavioral either at p < 0.05. Needs narrower follow-up.

**No RULE 7 TAUTOLOGY with ORB_G5**: fire-event correlation = 0.148 (< 0.7 threshold). ATR_P50 is a distinct gate from ORB_G5 on this lane.

---

## Scope

- Lane: MNQ × SINGAPORE_OPEN × E2 × CB=1 × O15 × RR=1.5 × ATR_P50
- IS window: trading_day < 2026-01-01 (Mode A holdout)
- Total N (eligible, atr_20_pct + orb_size known): **1,717**
- Source: canonical `orb_outcomes` ⨝ `daily_features` on (trading_day, symbol, orb_minutes)
- Filter spec (canonical `trading_app/config.py:2832-2836`): `atr_20_pct >= 50` (pre-session, STARTUP-resolved)

---

## RULE 8.2 decomposition

| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB | mean_ATR_pct |
|---|---|---|---|---|---|---|---|
| FIRE (atr_pct ≥ 50) | 913 | 48.41% | +0.1130R | 1.152 | 23.19 | 18.00 | 78.14 |
| NON-FIRE (atr_pct < 50) | 804 | 44.90% | -0.0210R | 1.090 | 13.17 | 10.75 | 21.14 |

- **Δ_WR** = +3.51pp (≥ 3pp threshold)
- **Two-proportion z on WR** = +1.455, p = 0.1456 → WR spread statistically zero
- **Δ_ExpR** = +0.1340R (> 0.10 threshold)
- **Welch t** on pnl_r = +2.477, p = 0.0134
- **mean ORB size fire - non-fire** = +10.02 pts (ratio = 1.76x)
- **RULE 8.2**: |wr_spread|=3.51pp, |Δ_ExpR|=0.1340R
- **Verdict**: inconclusive at p=0.05

---

## Does ATR correlate with ORB size? (arithmetic-confound check)

If ATR_P50 fire days have systematically larger ORBs, the ExpR lift could be mechanical cost amplification (same pattern as ORB_G5 on L1 per PR #71), not volatility-regime directional content.

- Mean ORB size on ATR_P50 fire days: **23.19 pts**
- Mean ORB size on ATR_P50 non-fire days: **13.17 pts**
- Size ratio (fire / non-fire): **1.76x** (compare to L1 ORB_G5 ratio = 4.73x per PR #71)
- Ratio ≥ 1.5 — sizeable arithmetic confound present. ATR_P50 is partially a size-proxy. Interpret Δ_ExpR accordingly.

---

## RULE 7 TAUTOLOGY check vs ORB_G5

For each lane trade, classify ATR_P50 fire and ORB_G5 fire. Compute overlap and Pearson correlation of the two binary fire vectors.

| | ORB_G5 fire | ORB_G5 non-fire | row total |
|---|---|---|---|
| ATR_P50 fire | 899 | 14 | 913 |
| ATR_P50 non-fire | 743 | 61 | 804 |
| column total | 1642 | 75 | 1717 |

- Pearson correlation (binary fire vectors): **+0.1478**
- RULE 7 threshold: |corr| > 0.7 → flag TAUTOLOGY
- **No tautology** — ATR_P50 is operationally distinct from ORB_G5 on this lane.

---

## Literature grounding

- **Chan 2008 Ch 7** (`docs/institutional/literature/chan_2008_ch7_regime_switching.md`) — volatility-regime-conditional strategy activation. The canonical prior for "ATR percentile gates a momentum strategy" is that HIGH vol = trending regime = higher WR on a momentum-continuation lane. This audit tests whether ATR_P50 fits that pattern on L2.
- **Chordia et al 2018** (`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`) — K_family and t-threshold framing. K is small here (1 lane, 1 filter, 1 aperture); the relevant bar is Chordia strict t ≥ 3.79 (no prior literature directly validates ATR_P50 as a specific pre-session filter on E2 CB1 RR1.5, so we have no theory-backed pathway claim to relax the threshold).

---

## Follow-up (not in this PR)

1. If RULE 8.2 arithmetic-only fired, propose metadata reclassification on `OwnATRPercentileFilter.CONFIDENCE_TIER` to reflect cost-gate class rather than regime-gate class (per institutional-rigor.md § 5).
2. If behavioral content confirmed, the lane's H4 hypothesis status is reinforced — but the RR=1.0 / RR=2.0 variants should be audited before generalizing.
3. If TAUTOLOGY flagged, ATR_P50 should be replaced in L2's filter spec with whichever of ORB_G5 / ATR_P50 has tighter confidence evidence (separate audit required).
4. Cross-session ATR_P50 audit (other sessions × MNQ × O15 × RR=1.5) to test whether the L2 finding generalizes or is session-specific. Separate PR.

---

## Reproduction

```bash
python research/audit_l2_atr_p50_regime_vs_arithmetic.py
```

Writes this document to `docs/audit/results/2026-04-21-l2-atr-p50-regime-vs-arithmetic-audit.md`.
