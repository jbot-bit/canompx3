# Clean E2 long-stop replay — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG

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
| IS | 567 | 519 | -0.0192 | -0.0176 | -0.480 | 278 | 241 | 0 |
| OOS | 38 | 34 | 0.0628 | 0.0562 | 0.382 | 19 | 15 | 0 |

## Verdict / Decision

**WRONG as tested for production.** The clean live-placeable long-stop replay
does not support promoting this prior-day branch.

## Classification Use

This replay is a clean falsification surface for the prior-day long idea. It is not a live promotion by itself; deployment still requires the normal Criterion 4/8/9, additivity, runtime, SR, survival, and preflight gates.

## Reproduction / Outputs

Runner: `research/mnq_e2_long_stop_replay_v1.py`

CSV: `docs/audit/results/2026-05-10-clean-long-stop-mnq-comex-settle-e2-rr1.0-cb1-pd-clear-long.csv`

## Caveats / Limitations

- This replay only tests the clean long-stop framing for this exact branch.
- It does not test short-side variants, allocator replacement, or portfolio
  additivity.
- It must not be used as evidence for the original close-confirmed
  `break_dir` selector.
