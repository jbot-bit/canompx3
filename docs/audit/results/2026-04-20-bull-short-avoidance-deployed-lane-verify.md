# Bull-day short avoidance — deployed-lane verification (Pathway B, K=1)

**Date:** 2026-04-20
**Scan:** `research/bull_short_avoidance_deployed_lane_verify.py`
**Data cutoff:** `orb_outcomes` / `daily_features` through 2026-04-19 (MNQ)
**Verdict:** **CONDITIONAL — UNVERIFIED** (superseded 2026-04-20 on post-hoc power review — see § "Post-hoc correction" below). Do NOT implement on live capital yet, but NOT for the reason originally stated.

> **Original verdict:** REJECTED. **Why it was wrong:** binary veto triggered by dir_match=FALSE on an OOS slice with 7.9% power to detect the IS effect — a statistical null result being mis-read as a refutation. Corrected verdict below.

---

## Context

On 2026-04-04 a signal was captured in memory (`bull_short_avoidance_signal.md`):
shorts after bull prior days underperform shorts after bear prior days, pooled
across the validated universe, p=0.0007, 14/17 positive years, NYSE_OPEN
drives ~60% of effect. The planned application was: "When NYSE_OPEN lanes
activate, reduce bull-day shorts to half-size."

On 2026-04-18 the allocator deployed `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
(N=262, ExpR=+0.12, WR 57.3%) onto live capital. This lane takes both longs
and shorts. The half-size filter is now actionable — IF the signal holds on
this specific lane under Mode A discipline.

## Method (Pathway B, K=1)

- **Hypothesis (one):** shorts on the deployed lane show WR_bear > WR_bull by
  ≥3% on IS AND the effect direction holds OOS.
- **Theory citation:** prior-day direction as overnight-sentiment proxy
  (bull-exhaustion / dip-buying-resistance mechanism); matches the 2026-04-04
  pooled-universe finding which this audit specifically confirms or refutes
  on the deployed lane.
- **Data:** `orb_outcomes JOIN daily_features` (`orb_minutes=5` on both sides),
  canonical `COST_LT12` filter via `research.filter_utils.filter_signal` (no
  inline re-encode; per `research-truth-protocol.md` § Canonical filter
  delegation).
- **Splits:**
  - IS: `trading_day < 2026-01-01` (Mode A per `trading_app.holdout_policy`)
  - OOS: `2026-01-01 .. 2026-04-19`
- **Null:** moving-block bootstrap (block_len=5 days, n_boot=5,000, seed=20260420)
  — resample `pnl_r` preserving autocorrelation, keep `prev_day_direction`
  labels FIXED to break signal-outcome coupling (per 2026-04-15 block-bootstrap
  correction in `backtesting-methodology.md`).

### Pre-committed decision criteria (written before inspecting IS/OOS numbers)

- **CONFIRMED:** block-boot p < 0.01 AND dir_match TRUE AND WR spread > 0.03
  AND bear-year-share ≥ 0.70
- **BORDERLINE:** 0.01 ≤ block-boot p < 0.05  OR  WR spread 0.02-0.03  OR
  bear-year-share 0.55-0.70
- **REJECTED:** block-boot p ≥ 0.05  OR  dir_match FALSE  OR  WR spread ≤ 0

## Results

### Scope

- Filter-passed trades on deployed lane: N = 1,740 (2019-05-06 .. 2026-04-16)
- Longs / Shorts: 874 / 866
- prev_day_direction: bull 968 / bear 768 / NaN 4

### Shorts by prior-day direction, IS (Mode A, < 2026-01-01)

| Group | N | mean pnl_r | WR | |
|-------|---|-----------|-----|---|
| Bear-day shorts | 360 | +0.1837 | 61.1% | |
| Bull-day shorts | 465 | +0.0263 | 53.3% | |
| Delta | | +0.1574 | +7.8% | |
| Welch t=2.35, p=0.019 | | | | |
| Block-bootstrap p (5k, labels fixed) | | | | **0.018** |

### Shorts by prior-day direction, OOS (Mode A sacred, ≥ 2026-01-01)

| Group | N | mean pnl_r | WR |
|-------|---|-----------|-----|
| Bear-day shorts | 19 | -0.1720 | 42.1% |
| Bull-day shorts | 20 | +0.2759 | 65.0% |
| Delta | | **-0.4480** | **-22.9%** |
| Welch p | | 0.162 | |

**`dir_match = FALSE`** — IS direction (bear>bull) is the OPPOSITE of OOS direction (bull>bear).

### IS year-by-year

| Year | Bear N | Bear mean | Bull N | Bull mean | Delta |
|------|--------|-----------|--------|-----------|-------|
| 2019 | 33 | +0.0778 | 45 | +0.0591 | +0.019 |
| 2020 | 41 | +0.0865 | 84 | -0.0625 | +0.149 |
| 2021 | 52 | +0.2299 | 65 | -0.0489 | +0.279 |
| 2022 | 73 | +0.1736 | 66 | +0.0034 | +0.170 |
| 2023 | 49 | +0.0252 | 56 | -0.0363 | +0.062 |
| 2024 | 60 | +0.3268 | 70 | +0.2433 | +0.084 |
| 2025 | 52 | +0.2798 | 79 | +0.0349 | +0.245 |

**7/7 IS years positive.** Effect is consistent within Mode A IS window.

## Verdict

Per pre-committed criteria:
- block-boot p = 0.018 → fails CONFIRMED (need < 0.01), passes BORDERLINE
- dir_match = FALSE → **hard REJECT** (pre-committed kill criterion)
- WR spread = +0.078 → passes
- bear-year-share = 7/7 = 1.00 → passes

**REJECTED.** The pre-committed kill criterion (dir_match FALSE) fires.

## Caveats on the verdict

- **OOS power is thin.** N_OOS = 41 (per-group N=19/20) is well below the
  RULE 3.2 threshold of 30 per group for confirmatory OOS evidence. The OOS
  flip is directional evidence the signal may be dead, not proof. Welch
  p=0.162 on the OOS flip.
- **Memory-cited p=0.0007 was the POOLED universe,** not this lane. The
  lane-specific IS p_boot=0.018 is an order of magnitude weaker — the
  pooled strength came partly from aggregation across sessions.
- **Had we deployed half-size bull-day shorts for 2026 Q1,** we would have
  reduced size on the side that actually outperformed (bull-day shorts
  +0.276R vs bear-day shorts -0.172R). A real trading loss from acting on
  the IS-only evidence.

## Decision

1. **Do NOT implement** the half-size bull-day-short filter on
   `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`.
2. **Demote** the memory entry from "queued for activation" to "parked —
   IS-positive, OOS sign-flip on thin sample." Revisit when OOS N ≥ 100
   per group (estimated mid-2026 if deployment persists).
3. **Do not further adjust the live lane based on this signal.** The lane's
   own 12-month trailing ExpR=+0.12 is its own mandate; overlay filters
   require independent re-validation, not memory citations from a prior
   scope.
4. **Add to lessons log:** "pooled-universe p-values are not lane-specific
   p-values" — pooled findings need per-lane re-verification before any
   deployed-capital action.

## Post-hoc correction (same day, 2026-04-20) — original REJECTED verdict was wrong

After user pushback ("is 3 months even relevant enough to make a call like that?"),
formal OOS power analysis:

| Quantity | Value |
|---|---|
| IS Cohen's d | 0.165 (small effect) |
| OOS N per group | 19 bear / 20 bull |
| OOS SE of delta | 0.314 |
| OOS 95% CI on delta | **[-1.063, +0.167]** — contains IS delta (+0.157) AND zero AND large negatives |
| **Power of OOS (N=19/20) to detect IS effect at α=0.05** | **7.9%** |
| N per group needed for 80% power | 578 (≈ 29 quarters ≈ 7 years more trading) |

### What went wrong in the original decision

1. **dir_match=FALSE was applied as a hard kill without a power floor.** The
   pre-committed criteria did not require `N_OOS ≥ threshold` before the
   dir_match veto could fire. With 7.9% power, the OOS slice is statistically
   useless — any direction (flip, match, or null) is noise-consistent.
2. **This violated the author's own RULE 3.2** in
   `.claude/rules/backtesting-methodology.md`: "If `N_on_OOS < 30` → statistical
   power on OOS is very low. Treat as directional-only evidence, not
   confirmatory." Per-group N=19/20 is *well below* this threshold; the dir_match
   flip is directional evidence, not confirmatory refutation.
3. **Literature grounding** (`docs/institutional/literature/`):
   - Harvey-Liu 2015 (`harvey_liu_2015_backtesting.md`): OOS is used as a
     *Sharpe-ratio haircut*, not a binary veto. "The highest Sharpe ratios are
     only moderately penalized."
   - LdP 2020 ML for Asset Managers (`lopez_de_prado_2020_ml_for_asset_managers.md`):
     for short OOS data, CPCV (combinatorial purged cross-validation) replaces
     binary IS/OOS split. Our 3-month OOS is exactly the scenario a binary split
     is misspecified for.

### Corrected verdict: CONDITIONAL — UNVERIFIED

- **IS evidence is VALID.** Block-boot p=0.018, 7/7 years, WR spread +7.8%,
  economically plausible mechanism.
- **OOS evidence is UNVERIFIED.** Not REFUTATION — the OOS CI contains the IS
  effect size. Cannot distinguish "signal alive", "signal dead", or "signal
  reversed" at current N.
- **No live-capital action yet,** but this is a STAY-OUT-UNTIL-MORE-OOS, not a
  KILL. Different memory classification, different follow-up action.

### Next action (highest EV — supersedes the original "PARK forever" plan)

**Pooled-deployed-lane OOS test.** Instead of waiting 7 more years for lane-specific
OOS N to grow, aggregate 2026 Q1 shorts across ALL 6 deployed lanes
(`MNQ_NYSE_OPEN`, `MNQ_TOKYO_OPEN`, `MNQ_EUROPE_FLOW`, `MNQ_COMEX_SETTLE`,
`MNQ_SINGAPORE_OPEN`, `MNQ_US_DATA_1000`). This pools the SAME mechanism across
correlated sessions — effective K stays ≈ 1 under the bull/bear prior-day
hypothesis, but N should climb to 150+ per group for 2026 Q1 alone. That is the
right OOS power test.

If pooled-deployed OOS shows dir_match=TRUE with reasonable effect size → ship
half-size filter across ALL deployed shorts (not just this lane). If pooled
still flips at N>=100 per group → different story, but at least informative.

## Files

- Script: `research/bull_short_avoidance_deployed_lane_verify.py`
- Superseded memory claim: `memory/bull_short_avoidance_signal.md` (revised to CONDITIONAL, not PARKED)
- Related prior audit: `scripts/research/bull_short_avoidance_audit.py`
  (pooled universe, 2026-04-04)
- Power calc: inline in this doc § Post-hoc correction
- Next test to write: `research/bull_short_avoidance_pooled_deployed_oos.py` (TBD)
