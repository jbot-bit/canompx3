# MNQ 0900 Dalton Filter OOS Check

No-lookahead enforced: entry_ts >= A/B gate_ts.

- Years in sample: [2024, 2025, 2026]
- Train year: 2024
- Train uplift (ON-OFF avgR): +0.9662
- Test uplift (ON-OFF avgR): +0.4559

## Aggregate
- ON: N=90, WR=62.2%, avgR=+0.8429, totalR=+75.86, maxDD=14.87
- OFF: N=1499, WR=32.3%, avgR=+0.1071, totalR=+160.60, maxDD=151.25

## Verdict guide
- KEEP if test uplift positive with adequate N and stable monthly behavior.
- WATCH if positive but sparse/volatile.
- KILL if test uplift flips negative.