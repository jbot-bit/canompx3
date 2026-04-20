# Debt Ledger

Use this file for unresolved integrity, runtime, and context debt that should remain visible until closed.

## Open Debt

- `phase4-hypothesis-sha` — Phase 4 hypothesis SHA integrity debt remains an explicit unresolved item until repaired or quarantined.
- `operator-pulse-latency` — `project_pulse --fast` should stay bounded enough for frequent shell use and must avoid broad startup scans.
- `capsule-hygiene` — Work capsules need disciplined upkeep so scope, verification, and ledger refs remain accurate.
- `cost-realism-slippage-pilot` — Book-wide cost-realism debt. **Partial measurement landed 2026-04-20:**
   - MGC TBBO pilot (N=40, `research/output/mgc_e2_slippage_analysis.json`): MEDIAN=0 ticks, p95=2.05, max=263 (dominated by 2018-01-18 gap event). Trimmed mean ≈0.18 ticks. Raw mean=6.75 is outlier-driven.
   - **MNQ TBBO pilot (N=114, 2026-04-20):** MEDIAN=0 ticks, p95=0.35, max=+2, 100% of days ≤ 2 ticks. Modeled slippage is CONSERVATIVE on routine days. Full result: `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md`. Modeled friction UNCHANGED (no numeric COST_SPECS edit needed).
   - Both instruments: median = 0 ticks routine. Honest central tendency is modeled-conservative.
   - STILL OPEN: (a) MES TBBO pilot NOT run; (b) MNQ sample missing EUROPE_FLOW / COMEX_SETTLE / US_DATA_1000 (3 of 5 deployed sessions absent from cache); (c) event-day tail NOT measured for MNQ (2021-2026 sample had no MGC-2018-type gap); (d) Phase D MNQ COMEX_SETTLE baseline specifically would benefit from a targeted pull before 2026-05-15 gate evaluation.
- ~~`mnq-tbbo-pilot-script-broken`~~ **CLOSED 2026-04-20.** Script rewritten with canonical `reprice_e2_entry` delegation + real `orb_high/orb_low` from daily_features + new `--reprice-cache` mode. Canonical regression + caller tests at `tests/test_research/test_reprice_e2_entry_regression.py` + `tests/test_research/test_mnq_pilot_caller.py`. Ran clean against existing 119-file cache: 114/119 valid, 5 legitimate `no_trigger_trade_found` errors.
