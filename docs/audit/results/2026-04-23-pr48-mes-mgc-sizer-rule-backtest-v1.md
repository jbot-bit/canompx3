# PR48 MES/MGC frozen rel-vol sizer-rule backtest v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
**Script:** `research/pr48_mes_mgc_sizer_rule_backtest_v1.py`
**Breakpoints:** `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
**Metrics artifact:** `research/output/pr48_mes_mgc_sizer_rule_metrics_v1.csv`

## Verdict summary

| Instrument | Verdict | Reason |
|---|---|---|
| MES | **SIZER_ALIVE_NOT_READY** | OOS take-home improves (-0.030 vs -0.090) and bucket ordering stays constructive (Q5 +0.112 > Q1 -0.199), but normalized ExpR or absolute OOS sign is not yet strong enough. |
| MGC | **SIZER_DEPLOY_CANDIDATE** | OOS full ExpR +0.133 > baseline +0.070, normalized ExpR +0.135 > baseline, and Q5 +0.263 > Q1 -0.026. |

## Replay metrics

| Instrument | Split | N | Mean Size | Base ExpR | Sized ExpR | Normalized ExpR | Full Delta | Norm Delta | Base Sharpe | Sized Sharpe | Active Share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MES | IS | 16058 | 1.00 | -0.115 | -0.051 | -0.051 | +0.065 | +0.065 | -0.111 | -0.039 | 79.8% |
| MES | OOS | 702 | 1.06 | -0.090 | -0.030 | -0.028 | +0.061 | +0.062 | -0.082 | -0.021 | 83.2% |
| MGC | IS | 7485 | 1.00 | -0.134 | -0.074 | -0.074 | +0.060 | +0.060 | -0.135 | -0.059 | 79.6% |
| MGC | OOS | 601 | 0.98 | +0.070 | +0.133 | +0.135 | +0.063 | +0.066 | +0.059 | +0.094 | 84.0% |

## Frozen bucket surfaces

### MES IS buckets

| Bucket | N | ExpR | Win Rate | Mean Multiplier |
|---|---:|---:|---:|---:|
| Q1 | 3207 | -0.256 | 37.8% | 0.00 |
| Q2 | 3199 | -0.153 | 41.4% | 0.50 |
| Q3 | 3198 | -0.143 | 41.2% | 1.00 |
| Q4 | 3200 | -0.040 | 45.4% | 1.50 |
| Q5 | 3210 | +0.013 | 47.3% | 2.00 |

### MES OOS buckets

| Bucket | N | ExpR | Win Rate | Mean Multiplier |
|---|---:|---:|---:|---:|
| Q1 | 118 | -0.199 | 37.3% | 0.00 |
| Q2 | 136 | -0.176 | 37.5% | 0.50 |
| Q3 | 152 | -0.083 | 41.4% | 1.00 |
| Q4 | 141 | -0.147 | 38.3% | 1.50 |
| Q5 | 155 | +0.112 | 49.0% | 2.00 |

### MGC IS buckets

| Bucket | N | ExpR | Win Rate | Mean Multiplier |
|---|---:|---:|---:|---:|
| Q1 | 1489 | -0.229 | 41.5% | 0.00 |
| Q2 | 1486 | -0.204 | 41.1% | 0.50 |
| Q3 | 1484 | -0.161 | 42.4% | 1.00 |
| Q4 | 1490 | -0.095 | 45.0% | 1.50 |
| Q5 | 1495 | +0.016 | 49.2% | 2.00 |

### MGC OOS buckets

| Bucket | N | ExpR | Win Rate | Mean Multiplier |
|---|---:|---:|---:|---:|
| Q1 | 96 | -0.026 | 41.7% | 0.00 |
| Q2 | 145 | -0.023 | 41.4% | 0.50 |
| Q3 | 138 | +0.032 | 43.5% | 1.00 |
| Q4 | 127 | +0.143 | 48.0% | 1.50 |
| Q5 | 95 | +0.263 | 52.6% | 2.00 |

## Notes

- `Sized ExpR` is the actual take-home replay `mean(size * pnl_r)`.
- `Normalized ExpR` is `sum(size * pnl_r) / sum(size)` so leverage drift does not masquerade as edge improvement.
- `Mean Size` is reported for both IS and OOS because the frozen quintile map averages 1.0x only under stationary bucket frequencies.
- This result is research-only. No live profile, lane allocation, or runtime sizing was modified.
