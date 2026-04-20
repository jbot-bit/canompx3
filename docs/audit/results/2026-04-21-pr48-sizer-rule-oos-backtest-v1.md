# PR #48 sizer-rule OOS backtest v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-21-pr48-sizer-rule-oos-backtest-v1.yaml`

**Script:** `research/pr48_sizer_rule_oos_backtest_v1.py`

**Rule (pre-committed):** IS per-lane rel_vol 5-quintile thresholds, multipliers {Q1: 0.5, Q2: 0.75, Q3: 1.0, Q4: 1.25, Q5: 1.5}; mean multiplier = 1.0 (capital-neutral).

**Pass criterion:** delta > 0 AND paired t >= +2.0.

## Headline

| Instrument | N_OOS | Uniform ExpR | Sizer ExpR | Delta | Paired t | p (one-tail) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| MNQ | 771 | +0.05895 | +0.06522 | +0.00627 | +0.440 | 0.3302 | **SIZER_WEAK** |
| MES | 702 | -0.09022 | -0.05997 | +0.03025 | +2.084 | 0.0188 | **SIZER_ALIVE** |
| MGC | 601 | +0.06955 | +0.10130 | +0.03175 | +2.000 | 0.0230 | **SIZER_ALIVE** |

## Per-quintile OOS diagnostics (rank→pnl monotonicity check)

| Instrument | Q1 N / ExpR | Q2 N / ExpR | Q3 N / ExpR | Q4 N / ExpR | Q5 N / ExpR |
|---|---|---|---|---|---|
| MNQ | 135 / +0.0157 | 141 / +0.0353 | 175 / +0.0635 | 183 / +0.1417 | 137 / +0.0096 |
| MES | 118 / -0.1987 | 136 / -0.1760 | 152 / -0.0832 | 141 / -0.1469 | 155 / +0.1123 |
| MGC | 96 / -0.0261 | 145 / -0.0225 | 138 / +0.0325 | 127 / +0.1428 | 95 / +0.2626 |

## Summary

- SIZER_ALIVE: 2 — MES, MGC
- SIZER_WEAK: 1 — MNQ
- SIZER_ADVERSE: 0 — none
- SIZER_NULL: 0 — none

## Interpretation

**Sizer rule deploy-eligible on:** MES, MGC. For each ALIVE instrument: IS-trained quintile thresholds applied to fresh OOS rel_vol produce a paired-positive, statistically-significant uplift vs uniform sizing at capital-neutral budget. Next bounded step per pre_registered_criteria.md is **shadow-deployment design** — document the exact rule, the per-lane thresholds (frozen snapshot), and the shadow monitor (Shiryaev-Roberts or fixed-horizon fire-rate).

## Methodology notes

- IS thresholds computed per-lane (session × direction) on pre-2026-01-01 data only. OOS `rel_vol` is searchsort-bucketed into those fixed IS thresholds — no OOS information leaks into the rule.
- Lanes with fewer than 100 IS trades fall back to uniform size=1.0 (no rule applied). Prevents sparse-lane distortion.
- Paired one-tailed t-test on per-trade weighted vs uniform P&L is the correct discriminator here (same trades, different sizes). One-sample t on the mean difference.
- Multipliers {0.5, 0.75, 1.0, 1.25, 1.5} have mean 1.0 → total capital deployed is identical to uniform baseline. This is the honest capital-neutral comparison.

## Not done by this result

- No writes to validated_setups / edge_families / lane_allocation / live_config.
- No deployment or capital action.
- Does NOT tune the multiplier curve (pre-committed).
- Does NOT test alternative rank-basis features (atr_vel_ratio, break_delay_min) — future pre-regs.
- Does NOT test per-direction or per-session (pooled by instrument is the pre-reg scope).