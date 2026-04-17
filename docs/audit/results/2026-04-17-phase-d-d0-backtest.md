# Phase D Volume Pilot — Stage D-0 Backtest

**Date:** 2026-04-17
**Pre-reg:** `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`
**Stage file:** `docs/runtime/stages/phase-d-volume-pilot-d0.md`
**Verdict:** `PASS`

## Universe
- Instrument: MNQ
- Session: COMEX_SETTLE
- Aperture: O5
- RR: 1.5
- Entry model: E2 / CB1
- Window: IS-only, trading_day < 2026-01-01
- Date range: 2019-05-13 00:00:00 to 2025-12-31 00:00:00
- N = 1631

## Sizing rule (discrete bucketing)
- P33 of rel_vol_COMEX_SETTLE = 1.0529
- P67 of rel_vol_COMEX_SETTLE = 1.7880
- rel_vol < P33 -> 0.5x
- P33 <= rel_vol <= P67 -> 1.0x
- rel_vol > P67 -> 1.5x

## Headline

| Metric | Baseline 1x | Scaled | Ratio |
|---|---|---|---|
| N trades | 1631 | 1631 | 1.000 |
| WR | 0.4697 | 0.4697 | 1.000 |
| ExpR | 0.0672 | 0.1135 | 1.690 |
| Sharpe_ann | 0.9250 | 1.4257 | 1.541 |
| MaxDD (R) | 49.5979 | 50.4279 | 1.017 |

## Gate evaluation
- **primary (Sharpe uplift >= 15%):** PASS (uplift = 54.12%, ratio = 1.541)
- **secondary MaxDD (<= 1.5x baseline):** PASS (scaled 50.43 / baseline 49.60 = 1.017x)
- **secondary per-year (scaled >= 0.8x baseline Sharpe each year):** PASS
- **secondary corr(size, pnl) > 0.05:** PASS (corr = 0.1002)
- **T0 tautology |corr(size, overnight_range_pct)| < 0.70:** PASS (corr = 0.0672)

- Correlation(size_multiplier, pnl_r) = 0.1002 (floor 0.05)
- T0 tautology: |corr(size_multiplier, overnight_range_pct)| = 0.0672 (max 0.7)

## Per-year breakdown — baseline

| Year | N | WR | ExpR | Sharpe | SumR |
|---|---|---|---|---|---|
| 2019 | 156 | 0.3718 | -0.2666 | -3.4435 | -41.59 |
| 2020 | 247 | 0.4615 | 0.0488 | 0.6742 | 12.05 |
| 2021 | 246 | 0.4715 | 0.0522 | 0.7315 | 12.84 |
| 2022 | 249 | 0.4257 | -0.0013 | -0.0170 | -0.31 |
| 2023 | 246 | 0.5000 | 0.1407 | 1.9282 | 34.62 |
| 2024 | 241 | 0.4896 | 0.1282 | 1.7219 | 30.89 |
| 2025 | 246 | 0.5325 | 0.2483 | 3.3187 | 61.08 |

## Per-year breakdown — scaled

| Year | N | WR | ExpR | Sharpe | SumR |
|---|---|---|---|---|---|
| 2019 | 156 | 0.3718 | -0.2729 | -3.1212 | -42.57 |
| 2020 | 247 | 0.4615 | 0.1330 | 1.6536 | 32.84 |
| 2021 | 246 | 0.4715 | 0.0987 | 1.2590 | 24.28 |
| 2022 | 249 | 0.4257 | 0.0240 | 0.3109 | 5.97 |
| 2023 | 246 | 0.5000 | 0.2189 | 2.6688 | 53.84 |
| 2024 | 241 | 0.4896 | 0.1926 | 2.3360 | 46.43 |
| 2025 | 246 | 0.5325 | 0.2618 | 3.2463 | 64.39 |

## Interpretation
- D-0 gate cleared. Proceed to D-1 (signal-only shadow) per the Phase D pre-reg, pending user approval of the 4-week shadow timeline and the `pre_registered_criteria.md` secondary review (DSR at multi-K framings, Carver Ch 9-10 grounding).
