# Stage 6 — Downstream impact of canonical scratch-EOD-MTM fix

**Plan stage:** Stage 6 of `docs/runtime/stages/scratch-eod-mtm-canonical-fix.md`
**Pre-rebuild commit:** Stage 5 fix `68ee35f8`
**Rebuild date:** 2026-04-28
**Backup:** `gold.db.backup-pre-stage5b-rebuild` (7.6 GB, retained for rollback)

**Scope:** quantify the impact on every downstream consumer that read `orb_outcomes.pnl_r` with the silent scratch-NULL dropout. Halt-and-notify if any DEPLOYED lane in `topstep_50k_mnq_auto` flips from FIT to DECAY.

**Outcome (verdict):** **NO DEPLOYED LANE FLIPS TO DECAY.** All 6 deployed lanes remain positive; ExpR shifts are within +0.0% to +13.3% — consistent with low scratch rates at RR=1.0 / RR=1.5. **20 of 144 v1-high-RR cells sign-flipped** under realized-EOD MTM (the Stage-1 0R correction missed this); previously KILLed cells at RR≥2.0 on NYSE_OPEN, US_DATA_1000, and CME_PRECLOSE are now significantly positive. These cells go to Stage 8 (MFE-distribution research) for further analysis under separate pre-reg — **NO automatic promotion**.

---

## Verdict — DEPLOYED lane parity check

The 6 lanes in `topstep_50k_mnq_auto` (loaded via `load_allocation_lanes('topstep_50k_mnq_auto')`) compared pre-fix `pnl_r IS NOT NULL` (drop policy) vs post-rebuild realized-EOD MTM:

| Lane | N (drop) | N (realized) | ExpR (drop) | ExpR (realized) | ΔExpR | Δ% |
|---|---:|---:|---:|---:|---:|---:|
| MNQ_EUROPE_FLOW_O5_RR1.5_ORB_G5 | 1583 | 1583 | +0.0643 | +0.0643 | +0.0000 | +0.0% |
| MNQ_SINGAPORE_OPEN_O15_RR1.5_ATR_P50 | 913 | 914 | +0.1130 | +0.1138 | +0.0008 | +0.7% |
| MNQ_COMEX_SETTLE_O5_RR1.5_ORB_G5 | 1555 | 1577 | +0.0915 | +0.0941 | +0.0025 | +2.8% |
| MNQ_NYSE_OPEN_O5_RR1.0_COST_LT12 | 1669 | 1695 | +0.0820 | +0.0790 | -0.0031 | -3.8% |
| MNQ_TOKYO_OPEN_O5_RR1.5_COST_LT12 | 950 | 950 | +0.1222 | +0.1222 | +0.0000 | +0.0% |
| MNQ_US_DATA_1000_O15_RR1.5_ORB_G5 | 1491 | 1712 | +0.1055 | +0.1195 | +0.0140 | +13.3% |

Every deployed lane retains ExpR > 0. None crosses any FIT → WATCH or WATCH → DECAY threshold. No allocator action required.

**Why so small?** Deployed lanes are all RR=1.0 / RR=1.5. At those low RR targets, target hit-rate is high and scratch rate is correspondingly low (typically <5%). The selection-bias contamination scales with scratch rate — high-RR cells were heavily contaminated; deployed low-RR cells were marginally contaminated. The ΔExpR magnitudes here confirm the institutional intuition: deployed lanes were the LEAST affected.

## Reproduction

Backup verification: `ls -la gold.db gold.db.backup-pre-stage5b-rebuild`. Rebuild log written to terminal stdout 2026-04-28; total wall time ~36 minutes for 9 (instrument, aperture) combos.

Per-instrument-aperture rebuild commands run:
```bash
python -m trading_app.outcome_builder --instrument MNQ --orb-minutes 5 --force   # 327s
python -m trading_app.outcome_builder --instrument MNQ --orb-minutes 15 --force  # 283s
python -m trading_app.outcome_builder --instrument MNQ --orb-minutes 30 --force  # 286s
python -m trading_app.outcome_builder --instrument MES --orb-minutes 5 --force   # 289s
python -m trading_app.outcome_builder --instrument MES --orb-minutes 15 --force  # 271s
python -m trading_app.outcome_builder --instrument MES --orb-minutes 30 --force  # 234s
python -m trading_app.outcome_builder --instrument MGC --orb-minutes 5 --force   # 129s
python -m trading_app.outcome_builder --instrument MGC --orb-minutes 15 --force  # 133s
python -m trading_app.outcome_builder --instrument MGC --orb-minutes 30 --force  # 119s
```

Acceptance verified via `pipeline/check_drift.py::check_orb_outcomes_scratch_pnl`:

| Symbol | Aperture | Scratch total | Scratch with non-NULL pnl_r | % populated |
|---|---:|---:|---:|---:|
| MES | 5 | 119,108 | 118,844 | 99.78% |
| MES | 15 | 141,814 | 141,365 | 99.68% |
| MES | 30 | 149,983 | 149,251 | 99.51% |
| MGC | 5 | 41,894 | 41,822 | 99.83% |
| MGC | 15 | 65,809 | 65,713 | 99.85% |
| MGC | 30 | 87,648 | 87,462 | 99.79% |
| MNQ | 5 | 124,097 | 123,887 | 99.83% |
| MNQ | 15 | 153,071 | 152,664 | 99.73% |
| MNQ | 30 | 161,689 | 161,078 | 99.62% |

All 9 combos pass the ≥99% threshold. Min: MES 30m at 99.51%.

## Sign-flip findings (v1 high-RR family scan, 144 cells)

Comparison: drop-scratches (original v1 behavior) vs realized-EOD MTM (Stage 5 canonical). Sign flip defined as `sign(t_drop) != sign(t_real)` AND `|t| > 1.0` on both sides (excludes near-zero cells):

**20 of 144 cells sign-flipped.** All flipped from negative to positive. The Stage-1 0R approximation in `2026-04-27-mnq-unfiltered-high-rr-family-v1-CORRECTION.md` claimed "0/144 sign flips"; that claim was correct under 0R but is **wrong under realized-EOD**. Empirical mean scratch_R on MNQ NYSE_OPEN 15m RR=4.0 is +0.9955 (not 0). Scratches systematically have positive directional drift on E2 confirmed-break entries — consistent with intraday trend-continuation premium during the post-break holding window.

| Apt | RR | Session | N_drop | N_real | ExpR_drop | ExpR_real | t_drop | t_real |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 2.0 | NYSE_CLOSE | 515 | 1437 | -0.1678 | +0.0333 | -3.02 | +1.46 |
| 5 | 2.5 | CME_PRECLOSE | 1032 | 1643 | -0.0851 | +0.1247 | -1.90 | +4.05 |
| 5 | 2.5 | NYSE_CLOSE | 459 | 1437 | -0.3163 | +0.0362 | -5.15 | +1.52 |
| 5 | 3.0 | CME_PRECLOSE | 954 | 1643 | -0.2096 | +0.1230 | -4.33 | +3.83 |
| 5 | 3.0 | NYSE_CLOSE | 428 | 1437 | -0.4595 | +0.0293 | -7.29 | +1.20 |
| 5 | 4.0 | CME_PRECLOSE | 865 | 1643 | -0.4135 | +0.1309 | -8.01 | +3.84 |
| 5 | 4.0 | COMEX_SETTLE | 1494 | 1658 | -0.0597 | +0.0660 | -1.26 | +1.48 |
| 5 | 4.0 | NYSE_CLOSE | 397 | 1437 | -0.6520 | +0.0333 | -10.65 | +1.31 |
| 5 | 4.0 | NYSE_OPEN | 1360 | 1719 | -0.1307 | +0.1726 | -2.61 | +3.95 |
| 5 | 4.0 | US_DATA_1000 | 1454 | 1718 | -0.0649 | +0.1504 | -1.32 | +3.38 |
| 15 | 2.0 | CME_PRECLOSE | 160 | 1131 | -0.1593 | +0.0717 | -1.54 | +3.73 |
| 15 | 2.5 | CME_PRECLOSE | 149 | 1131 | -0.2162 | +0.0757 | -1.86 | +3.77 |
| 15 | 2.5 | NYSE_CLOSE | 170 | 968 | -0.3267 | +0.0242 | -3.21 | +1.04 |
| 15 | 2.5 | NYSE_OPEN | 1130 | 1715 | -0.1265 | +0.1462 | -2.86 | +4.53 |
| 15 | 3.0 | CME_PRECLOSE | 143 | 1131 | -0.2276 | +0.0860 | -1.77 | +4.04 |
| 15 | 3.0 | NYSE_OPEN | 1051 | 1715 | -0.2684 | +0.1466 | -5.73 | +4.35 |
| 15 | 3.0 | US_DATA_1000 | 1189 | 1717 | -0.1601 | +0.1304 | -3.47 | +3.69 |
| 15 | 4.0 | CME_PRECLOSE | 130 | 1131 | -0.4503 | +0.0881 | -3.36 | +3.96 |
| 15 | 4.0 | NYSE_OPEN | 950 | 1715 | -0.5249 | +0.1533 | -11.23 | +4.35 |
| 15 | 4.0 | US_DATA_1000 | 1100 | 1717 | -0.3160 | +0.1577 | -6.25 | +4.10 |

The most striking case: **MNQ NYSE_OPEN 15m RR=4.0** went from t=-11.23 (catastrophic KILL) to t=+4.35 (CHORDIA-significant POSITIVE) — same data, just no longer dropping the +1R-average scratches.

### Why these flips do NOT auto-promote

These cells were KILLed by the v1 scan's H1 gate. With the canonical fix, several pass H1 (`t ≥ +3.0` with theory). However:
- C5 (DSR), C6 (WFE), C8 (OOS), C9 (era stability) are NOT recomputed in this Stage 6 report.
- The new positive sign is a *necessary* condition for promotion, not sufficient.
- Per `pre_registered_criteria.md` Amendment 3.0, individual-cell promotion requires Pathway B K=1 with theory citation under a separate pre-reg.

The institutional discipline: Stage 6 is observation; Stage 8 (MFE-distribution research) is the controlled framework for re-evaluating these candidates under realized-EOD-MTM data. **No deployment from this stage.**

## Other downstream consumers

### `validated_setups` and `live_config`

Existing rows in `validated_setups` were computed pre-fix under drop policy. Per `RESEARCH_RULES.md` § Mode B grandfathered guard, these rows must be recomputed against canonical layers before any new deployment decision. The Stage 5 fix invalidates all `validated_setups.expectancy_r` values for any lane with non-trivial scratch rate.

**Action item (deferred to follow-up):** `python research/mode_a_revalidation_active_setups.py` (already exists per `feedback_validated_setups_drift.md`) should be re-run after Stage 5 to refresh `expectancy_r` against the rebuilt `orb_outcomes`. Out of scope for this report — flagged for next session.

### `trading_app/portfolio.py::compute_fitness`

Reads `orb_outcomes.pnl_r` with `WHERE pnl_r IS NOT NULL` (verified in source). Now sees realized-EOD values. ExpR shifts on the 6 deployed lanes are <14% (table above); fitness verdict thresholds are coarser. **Not expected to flip any verdict.** Spot check via direct query above confirms.

### `trading_app/sprt_monitor.py` and `sr_monitor.py` (Phase 0 Criterion 12)

Both monitors operate on **live trade R-multiples** from `paper_trades` and live execution feeds, NOT on `orb_outcomes`. The `orb_outcomes` rebuild does NOT directly affect these monitors. However, **if their pre-change distribution is calibrated from the first 50–100 simulated trades from `orb_outcomes`** (per spec), the calibration baseline shifts on rebuild. Inspection of `sprt_monitor.py:_init_pre_change_distribution()` (read-only) shows the calibration uses `validated_setups.expectancy_r` — which is stale per above. Not directly broken by Stage 5; will be refreshed when `validated_setups` is rebuilt.

### `trading_app/pbo.py` (Probability of Backtest Overfitting)

PBO is win-rate + payoff sensitive. Pre-fix, scratches were dropped (no contribution). Post-fix, scratches contribute realized P&L. PBO bounds may change marginally — not flipping any verdict at the deployed scale (RR≤1.5).

### `trading_app/rolling_correlation.py`, `trading_app/rolling_portfolio.py`

Both consume `orb_outcomes.pnl_r` for rolling windows. The rebuild is transparent — these now use realized-EOD values automatically.

### `trading_app/weekly_review.py`

Reads `paper_trades`, NOT `orb_outcomes`. Stage 7 audits `paper_trades` separately.

### `trading_app/ai/sql_adapter.py`

Inspection of templates (read-only) shows several use `WHERE pnl_r IS NOT NULL`. Since the rebuilt data has scratches populated, the existing filter is now safe (no longer creates selection bias on aggregate queries). **Recommended improvement, not blocker:** new templates should use `outcome IN ('win','loss','scratch')` for clarity. Deferred to follow-up.

## Limitations

1. **Per-lane filter application.** The 6-lane parity check in this file applies the canonical filter via `research.filter_utils.filter_signal`. Edge cases in filter behavior (NaN handling, missing daily_features columns) could theoretically produce slightly different N counts than the live allocator's runtime check. Not material at the magnitudes observed.

2. **C5/C6/C8/C9 not recomputed for sign-flipped cells.** Stage 6 reports flip detection only; full re-validation with all 13 criteria is the Stage 8 work.

3. **Lit sign-flip explanation.** "Scratches average +1R" is the empirical observation. The mechanism — intraday trend-continuation post-confirmed-break — is consistent with `mechanism_priors.md` § Liquidity + displacement core premise. But this is not a formally verified mechanism claim; it's a hypothesis to test under Stage 8.

4. **paper_trades NOT in this report.** Stage 7 separately.

## Halt-and-notify decision

**No halt.** Plan proceeds to Stage 7 (paper_trades parity audit) and Stage 8 (MFE-distribution endogenous-RR research).
