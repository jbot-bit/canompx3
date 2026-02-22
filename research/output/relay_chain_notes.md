# Relay Chain Hypothesis Results

Follower slice: M2K US_POST_EQUITY E1/CB5/RR1.5
Single filter: M6E_US_EQUITY_OPEN direction match
Relay filter: Single + MES_US_DATA_OPEN direction match
No-lookahead: leader break_ts <= follower entry_ts

## Summary
- baseline: N=964, WR=40.8%, avgR=-0.0624, totalR=-60.15
- single_m6e: N=491, WR=46.8%, avgR=+0.0765, totalR=+37.56
- relay_m6e_mes: N=225, WR=46.7%, avgR=+0.0725, totalR=+16.32

## Quick OOS (test year 2025)
- single_m6e: trainΔ=+0.2453, testΔ=+0.4401, n_test_on=104
- relay_m6e_mes: trainΔ=+0.1149, testΔ=+0.5056, n_test_on=44

## Verdict: KILL