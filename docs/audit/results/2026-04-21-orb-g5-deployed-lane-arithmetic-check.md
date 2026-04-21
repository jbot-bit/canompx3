# ORB_G5 deployed-lane ARITHMETIC_ONLY confirmatory audit

**Date:** 2026-04-21
**Branch:** `research/l1-orb-g5-arithmetic-only-check`
**Script:** `research/audit_l1_orb_g5_arithmetic_only_check.py`
**Parent claim:** PR #57 (merge `5a39ea20`) classified L1 as
  FILTER_CORRELATES_WITH_EDGE. This audit asks whether L1 — and the other
  two deployed ORB_G5 lanes — are behavioral edge or RULE 8.2 cost-screen.
**Rule:** `.claude/rules/backtesting-methodology.md` § RULE 8.2 ARITHMETIC_ONLY.
**Classification:** confirmatory audit (RULE 10: no new prereg required).

---

## TL;DR

**Cross-lane verdict is heterogeneous** — the 3 deployed ORB_G5 lanes do NOT share
a common class under RULE 8.2:

- **Cost-gate (ARITHMETIC_ONLY)**: `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`. WR spread statistically zero (p > 0.05 on proportion z-test); all ExpR lift is cost-amplification.
- **Behavioral content possible**: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`. WR spread significant at p < 0.05 on proportion z-test; fire group has real directional-accuracy advantage beyond cost-math.
- **Underpowered (RULE 3.2)**: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` (N_nofire=4). Non-fire bucket below 30-sample power floor — directional-only, not statistically conclusive.

**Consequence for action queue item #2**: the original premise ("replace ORB_G5 selectivity on L1 with pre-break context") is supported ONLY for L1 EUROPE_FLOW, where ORB_G5 is a cost-gate and not a behavioral filter. On other ORB_G5 lanes the filter is carrying behavioral information or the test is underpowered. Lane-specific reframing is required — a single portfolio-level rewrite is wrong.

---

## Per-lane summary table

| Lane | N | N_fire | N_nofire | Δ_WR (pp) | z_WR | p_WR | Δ_ExpR | Welch t | Welch p | Size ratio | RULE 8.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 1718 | 1583 | 135 | +2.26 | +0.50 | 0.6142 | +0.2685 | +3.29 | 0.0012 | 4.73x | **ARITH** |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 1636 | 1555 | 81 | +14.32 | +2.52 | 0.0118 | +0.5022 | +5.11 | 0.0000 | 6.17x | edge? |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 1495 | 1491 | 4* | -29.12 | -1.17 | 0.2431 | -0.3061 | -0.65 | 0.5628 | 14.66x | edge? |

`*` = N_nofire < 30 (RULE 3.2 power floor; directional-only on that cell).

---

## Per-lane detail

### MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5

- Spec: MNQ × EUROPE_FLOW × E2 × CB=1 × O5 × RR=1.5 × ORB_G5
- Total N (IS, eligible): **1,718**

| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB |
|---|---|---|---|---|---|---|
| FIRE (orb_size ≥ 5) | 1,583 | 47.44% | +0.0643R | 1.123 | 16.39 | 13.50 |
| NON-FIRE (orb_size < 5) | 135 | 45.19% | -0.2043R | 0.890 | 3.46 | 3.75 |

- **Δ_WR** = +2.26pp
- **Two-proportion z on WR** = +0.504, p = 0.6142 → WR spread statistically zero (no directional content)
- **Δ_ExpR** = +0.2685R
- **Welch t** on pnl_r = +3.288, p = 0.0012
- **Size ratio** (mean_ORB fire / non-fire) = 4.73x
- **RULE 8.2**: |wr_spread|=2.26pp (< 3.0?  True) AND |Delta_ExpR|=0.2685 (> 0.1?  True)
- **Verdict**: ARITHMETIC_ONLY (cost-screen, not behavioral edge)

### MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5

- Spec: MNQ × COMEX_SETTLE × E2 × CB=1 × O5 × RR=1.5 × ORB_G5
- Total N (IS, eligible): **1,636**

| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB |
|---|---|---|---|---|---|---|
| FIRE (orb_size ≥ 5) | 1,555 | 47.65% | +0.0915R | 1.147 | 21.11 | 17.00 |
| NON-FIRE (orb_size < 5) | 81 | 33.33% | -0.4107R | 0.844 | 3.42 | 3.50 |

- **Δ_WR** = +14.32pp
- **Two-proportion z on WR** = +2.518, p = 0.0118 → WR spread significant
- **Δ_ExpR** = +0.5022R
- **Welch t** on pnl_r = +5.115, p = 0.0000
- **Size ratio** (mean_ORB fire / non-fire) = 6.17x
- **RULE 8.2**: |wr_spread|=14.32pp (< 3.0?  False) AND |Delta_ExpR|=0.5022 (> 0.1?  True)
- **Verdict**: NOT arithmetic_only (behavioral content possible)

### MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15

- Spec: MNQ × US_DATA_1000 × E2 × CB=1 × O15 × RR=1.5 × ORB_G5
- Total N (IS, eligible): **1,495**

| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB |
|---|---|---|---|---|---|---|
| FIRE (orb_size ≥ 5) | 1,491 | 45.88% | +0.1055R | 1.202 | 58.66 | 51.25 |
| NON-FIRE (orb_size < 5) | 4 | 75.00% | +0.4117R | 0.943 | 4.00 | 4.00 |

- **Δ_WR** = -29.12pp
- **Two-proportion z on WR** = -1.167, p = 0.2431 → WR spread statistically zero (no directional content)
- **Δ_ExpR** = -0.3061R
- **Welch t** on pnl_r = -0.648, p = 0.5628
- **Size ratio** (mean_ORB fire / non-fire) = 14.66x
- **RULE 8.2**: |wr_spread|=29.12pp (< 3.0?  False) AND |Delta_ExpR|=0.3061 (> 0.1?  True)
- **Verdict**: NOT arithmetic_only (behavioral content possible)
- **Caveat**: N_nofire = 4 is below RULE 3.2 power floor (30). Treat as directional-only evidence on this cell.

---

## Mechanism

MNQ: 1 point = $2.00 per micro contract. A 5-pt ORB stop = $10 risk; a 16-pt ORB stop = $32 risk. Fixed slippage + commission per trade (canonical `cost_model.COST_SPECS`) is a much larger fraction of small-ORB R-risk than large-ORB R-risk. `pnl_r` in `orb_outcomes` is net of canonical costs, so the fire-vs-non-fire Δ_ExpR reflects this cost amplification directly.

Reference: RULE 7 (`backtesting-methodology.md`) — `cost_risk_pct ∝ 1 / orb_size_pts` has near-perfect inverse correlation with ORB_G5. ORB_G5 is mechanically equivalent to an upper-bound cost-risk-pct filter.

---

## Deployed lanes NOT in this audit (filter class differs)

- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` — ATR_P50 filter (different class; separate audit warranted to ask whether it is behavioral vs volatility-regime-gate).
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — COST_LT12 is **cost-gate by construction** (canonical filter name is explicitly cost-based).
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` — same as above.

Combined portfolio composition (6 lanes):
- 3 ORB_G5 lanes — **audited here; result is heterogeneous (see TL;DR and per-lane)**
- 2 COST_LT12 lanes — cost-gate by construction
- 1 ATR_P50 lane — different class (not audited here)

The heterogeneous ORB_G5 result means the portfolio's filter composition is more nuanced than either "all cost-gate" or "all behavioral edge" framings. Before any portfolio-level reclassification, each ORB_G5 lane should be revisited with the lane-specific evidence from this audit.

---

## Follow-up (not in this PR)

1. **Scope action queue item #2 to L1 only.** The "replace ORB_G5 selectivity with pre-break context" premise is supported only for L1 EUROPE_FLOW, where ORB_G5 is a cost-gate. The COMEX_SETTLE ORB_G5 lane has real behavioral content (p_WR=0.012) and should not be part of a behavioral-overlay prereg. The US_DATA_1000_O15 lane is underpowered (N_nofire=4) and cannot be classified from this audit.
2. **L1-specific rewrite**: the L1 behavioral-overlay prereg should compare candidate signals against the **unfiltered** L1 baseline (all trades, no ORB_G5 pre-gate), not against the filtered baseline, since ORB_G5 on L1 does not carry directional information.
3. **COMEX_SETTLE ORB_G5 standalone follow-up**: the +14pp WR spread at p=0.012 is real behavioral evidence. Separate diagnostic warranted — what specifically about COMEX_SETTLE's small-ORB days causes -0.41R ExpR on N=81? Possible mechanisms: thin liquidity around settlement, NY-close overhang, overnight chop that persists.
4. **US_DATA_1000_O15 power question**: only 4 non-fire days across 7 years of IS data. Either the ORB_G5 threshold rarely bites at O15 (typical mechanism: the 15m ORB is almost always ≥5 pts), or there is a sampling issue. Cannot conclude from 4 trades.
5. **ATR_P50 audit on L2.** Separate question; ATR_P50 is a different filter class (volatility-regime, not size). Needs its own fire-vs-non-fire decomposition. Probably not arithmetic_only but needs explicit real-data verification.
6. **Cross-session pre-break descriptive diagnostic.** Highest-EV next-step research per the earlier brutal audit: measure `pre_velocity` / `vwap` directional content on the UNFILTERED universe across 12 × 3 × 3 lane combos. Separate PR.
7. **No portfolio metadata reclassification yet.** Earlier draft of this audit assumed all ORB_G5 lanes were cost-gate. Real data rejects that. Portfolio `CONFIDENCE_TIER` on `OrbSizeFilter` should remain PROVEN; a finer per-lane classification is premature until (3) and (4) resolve.

---

## Reproduction

```bash
python research/audit_l1_orb_g5_arithmetic_only_check.py
```

Writes this document to `docs/audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md`.
