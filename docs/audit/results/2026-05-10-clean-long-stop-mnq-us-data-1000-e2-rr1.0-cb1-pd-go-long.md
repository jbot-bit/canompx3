# Clean E2 long-stop replay — MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG

## Question

Does the prior-day geometry context work when expressed as a live-placeable long stop, without using close-confirmed `break_dir` as a selector?

## Method

- Canonical layers: `bars_1m` + `daily_features`.
- Entry: long E2 stop at ORB high after ORB formation, with canonical E2 slippage.
- Selector: prior-day geometry state from ORB midpoint and prior-day levels only.
- Forbidden: `orb_<session>_break_dir` selection, sibling rescue, threshold tuning.

## Summary

| Split | N context | N fired | ExpR fired | Policy EV/context | t | Wins | Losses | Scratches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 643 | 563 | 0.0747 | 0.0654 | 1.885 | 316 | 247 | 0 |
| OOS | 36 | 33 | 0.0049 | 0.0045 | 0.030 | 17 | 16 | 0 |

## Verdict / Decision

**WRONG as tested for production.** The clean live-placeable long-stop replay
does not clear the statistical or OOS evidence needed to promote this
prior-day branch.

## Classification Use

This replay is a clean falsification surface for the prior-day long idea. It is not a live promotion by itself; deployment still requires the normal Criterion 4/8/9, additivity, runtime, SR, survival, and preflight gates.

## Reproduction / Outputs

Runner: `research/mnq_e2_long_stop_replay_v1.py`

CSV: `docs/audit/results/2026-05-10-clean-long-stop-mnq-us-data-1000-e2-rr1.0-cb1-pd-go-long.csv`

## Caveats / Limitations

- This replay only tests the clean long-stop framing for this exact branch.
- It does not test short-side variants, allocator replacement, or portfolio
  additivity.
- It must not be used as evidence for the original close-confirmed
  `break_dir` selector.
