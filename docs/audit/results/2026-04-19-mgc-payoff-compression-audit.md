# MGC 5-minute Payoff-Compression Audit

Date: 2026-04-19

## Scope

Narrow diagnostic audit of the warm `GC -> MGC` translated 5-minute families.
This is not a new discovery run and not a deployment memo.

Locked surface:

- symbol: `MGC`
- overlap era only: `2022-06-13 <= trading_day < 2026-01-01`
- `orb_minutes = 5`, `entry_model = E2`, `confirm_bars = 1`, `rr_target = 1.0`
- sessions: `EUROPE_FLOW`, `NYSE_OPEN`, `US_DATA_1000`
- diagnostics only:
  - raw canonical `pnl_r`
  - canonical `ts_pnl_r` time-stop
  - conservative lower-target rewrites at `0.5R` and `0.75R`

## Executive Verdict

The question is no longer whether `GC` triggers translate; they do. The
question is whether narrow exit handling can recover the warm `MGC` bridge.

Under the locked diagnostic:

- strongest warm lower-0.5R improvement: `US_DATA_1000_ATR_P70_RR1`
  (`avg_raw_r=+0.0008` -> `avg_lr05_r=+0.0710`)
- strongest broad lower-0.5R improvement: `EUROPE_FLOW_BROAD_RR1` (`avg_raw_r=-0.1297` -> `avg_lr05_r=-0.0095`)

- canonical time-stop is a null lens here:
  `ts_pnl_r` is identical to raw `pnl_r` across all tested families, and these
  sessions currently have no configured early-exit threshold in runtime policy
  (`EARLY_EXIT_MINUTES` is `None` for all three)

- lower-0.5R improvement is **not** confined to the warm translated rows:
  2/3 broad session comparators also flip positive
  under the same conservative rewrite

That means the remaining question has widened slightly:

- this is still a gold-specific 5-minute payoff-compression problem
- but it now looks more like a broader `MGC` target-shape issue in these sessions,
  not just a narrow proxy-rescue path

Interpretation should stay disciplined:

- this is not evidence that the retired `GC` shelf should be revived
- this is not evidence that `MGC` is solved by one lower target
- it is evidence that `RR1.0` may still be too ambitious for 5-minute `MGC` in these
  sessions, including beyond the narrow translated warm rows

## Verdict Labels

- `PAYOFF_COMPRESSION_REAL`: YES
- `LOW_RR_RESCUE_PLAUSIBLE`: YES
- `NO_RESCUE_SIGNAL`: NO

## Recommended Next Move

- Treat this item as actioned and closed at the diagnostic stage.
- If revisited, do it as a narrow MGC 5-minute exit-shape / lower-target prereg.
- Do not reopen broad GC proxy discovery or treat this as a generic gold deployment claim.

## Family Summary

| Family | Kind | N | Raw avg R | T80 avg R | 0.5R avg | 0.75R avg | reach 0.5 | reach 0.75 | time-stop | ambiguous |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EUROPE_FLOW_BROAD_RR1 | broad | 917 | -0.1297 | -0.1297 | -0.0095 | -0.0132 | 88.7% | 76.1% | 0.0% | 0.0% |
| NYSE_OPEN_BROAD_RR1 | broad | 918 | -0.0384 | -0.0384 | +0.0380 | +0.0390 | 81.3% | 69.6% | 0.0% | 0.2% |
| US_DATA_1000_BROAD_RR1 | broad | 918 | -0.0301 | -0.0301 | +0.0488 | +0.0342 | 80.6% | 66.6% | 0.0% | 0.0% |
| EUROPE_FLOW_OVNRNG_50_RR1 | warm | 52 | +0.0356 | +0.0356 | +0.0896 | +0.0604 | 80.8% | 67.3% | 0.0% | 0.0% |
| NYSE_OPEN_OVNRNG_50_RR1 | warm | 52 | +0.1468 | +0.1468 | +0.1293 | +0.2226 | 80.8% | 75.0% | 0.0% | 0.0% |
| US_DATA_1000_ATR_P70_RR1 | warm | 428 | +0.0008 | +0.0008 | +0.0710 | +0.0484 | 81.3% | 66.8% | 0.0% | 0.0% |
| US_DATA_1000_ORB_G5_RR1 | warm | 331 | +0.0807 | +0.0807 | +0.0579 | +0.0669 | 74.9% | 60.7% | 0.0% | 0.0% |
| US_DATA_1000_OVNRNG_10_RR1 | warm | 589 | +0.0194 | +0.0194 | +0.0685 | +0.0558 | 80.6% | 67.1% | 0.0% | 0.0% |

## Guardrails

- Lower-target rewrites are diagnostic only. They are not live-ready exits.
- Ambiguous loss bars are left as losses rather than rescued.
- This audit does **not** say whether MGC is dead overall. It only addresses the
  warm translated 5-minute families from the prior `GC -> MGC` audit.

## Outputs

- `research/output/mgc_payoff_compression_audit_family_summary.csv`
- `research/output/mgc_payoff_compression_audit_trade_matrix.csv`
