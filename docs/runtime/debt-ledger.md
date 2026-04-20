# Debt Ledger

Use this file for unresolved integrity, runtime, and context debt that should remain visible until closed.

## Open Debt

- `phase4-hypothesis-sha` — Phase 4 hypothesis SHA integrity debt remains an explicit unresolved item until repaired or quarantined.
- `operator-pulse-latency` — `project_pulse --fast` should stay bounded enough for frequent shell use and must avoid broad startup scans.
- `capsule-hygiene` — Work capsules need disciplined upkeep so scope, verification, and ledger refs remain accurate.
- `cost-realism-slippage-pilot` — MGC TBBO pilot (`research/research_mgc_e2_microstructure_pilot.py`) measured mean slippage = 6.75 ticks vs modeled 2 ticks = **~3.4× optimism in dollar friction** (`pipeline/cost_model.py:145`). MNQ TBBO pilot (`research/research_mnq_e2_slippage_pilot.py`) exists but has NOT been run. Every backtest `pnl_r` in `orb_outcomes` is systematically optimistic by an unknown amount per instrument. Book-wide. Repaired by: (1) running MNQ pilot, (2) running MES pilot (not yet scheduled), (3) rebuilding `orb_outcomes` with pilot-calibrated per-session slippage, (4) re-auditing 38 deployed lanes. See `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md` §4 for full analysis. Break-even sensitivity per `scripts/tools/slippage_scenario.py`: COMEX_SETTLE and SINGAPORE_OPEN lanes are within 1σ of the MGC pilot's mean → may be zero-expectancy in live. Pilot itself has NOT been adversarially audited — flagged as prerequisite for H0 sensitivity test.
