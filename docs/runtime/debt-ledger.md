# Debt Ledger

Use this file for unresolved integrity, runtime, and context debt that should remain visible until closed.

## Open Debt

- `phase4-hypothesis-sha` — Phase 4 hypothesis SHA integrity debt remains an explicit unresolved item until repaired or quarantined.
- `operator-pulse-latency` — `project_pulse --fast` should stay bounded enough for frequent shell use and must avoid broad startup scans.
- `capsule-hygiene` — Work capsules need disciplined upkeep so scope, verification, and ledger refs remain accurate.
- `cost-realism-slippage-pilot` — Book-wide cost-realism debt. **MGC, MNQ, and MES routine-central-tendency portions CLOSED:**
   - MGC TBBO pilot (N=40, `research/output/mgc_e2_slippage_analysis.json`): MEDIAN=0 ticks, p95=2.05, max=263 (dominated by 2018-01-18 gap event). Trimmed mean ≈0.18 ticks. Raw mean=6.75 is outlier-driven.
   - **MNQ TBBO pilot v2 (N=142, 2026-04-20 gap fill):** MEDIAN=0 ticks, p95=0.00, max=+2, **100% of days ≤ 2 ticks across all 9 deployed sessions**. Gap fill added EUROPE_FLOW / COMEX_SETTLE / US_DATA_1000 (10 days each, clean medians). Full result: `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md`. Modeled friction UNCHANGED.
   - **MES TBBO pilot v1 (N=40, 2026-04-24):** MEDIAN=0 ticks, p95=0.00, max=0, **100% of days ≤ 1 modeled tick across all current deployable MES sessions** (`CME_PRECLOSE`, `COMEX_SETTLE`, `SINGAPORE_OPEN`, `US_DATA_830`). Full result: `docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md`. Modeled friction UNCHANGED.
   - **Phase D 2026-05-15 MNQ COMEX_SETTLE gate:** no slippage-based blocker. 10-sample COMEX_SETTLE median=0, mean=0.0.
   - MGC + MNQ + MES routine central tendency: median = 0 ticks. Honest central tendency is modeled-conservative.
   - STILL OPEN: event-day tail NOT measured for MNQ/MES routine samples (known-unknown, not refuted concern).
- ~~`mnq-tbbo-pilot-script-broken`~~ **CLOSED 2026-04-20.** Script rewritten with canonical `reprice_e2_entry` delegation + real `orb_high/orb_low` from daily_features + new `--reprice-cache` mode. Canonical regression + caller tests at `tests/test_research/test_reprice_e2_entry_regression.py` + `tests/test_research/test_mnq_pilot_caller.py`. Ran clean against existing 119-file cache: 114/119 valid, 5 legitimate `no_trigger_trade_found` errors.
