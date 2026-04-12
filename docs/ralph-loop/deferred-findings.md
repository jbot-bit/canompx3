# Ralph Loop — Deferred Findings Ledger

> **Purpose:** Structured tracking of deferred findings. Nothing gets lost.
> **Rule:** Every finding deferred in ralph-loop-history.md MUST have a row here.
> **Enforcement:** Drift check scans this file. Zero-item file = clean. Items here = known debt.
> **Lifecycle:** Add when deferred. Remove when fixed (cite commit hash). Never delete silently.

## Open Findings

| ID | Iter | Severity | Target | Description | Deferred Reason |
|----|------|----------|--------|-------------|-----------------|
| SR-L6L7 | n/a | MEDIUM | trading_app/prop_profiles.py topstep_50k_mnq_auto | Both 2026-04-12 expansion lanes (L6 MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100, L7 MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60) tripped Shiryaev-Roberts ALARM on first SR monitor pass at N=58 and N=42 respectively. Per-lane 2026 forward ExpR positive but materially below IS validation: L6 +0.104 vs IS +0.215 (-52%); L7 +0.037 vs IS +0.170 (-78%). Same regime is benefiting core ORB_G5 lanes (+112%/+286% over IS), so the directional asymmetry is consistent with a vol-conditional-filter mismatch in the 2026 high-vol regime (MNQ ORB median +95%). Aggregate C11 still 88.4% (improved from 86.2% baseline) because the diversification benefit from 2 uncorrelated lanes more than offsets the per-lane weakness. | Triaged as WATCH following the same standard as L3 (NYSE_OPEN ORB_G5 RR1.5, in WATCH since 2026-04-10 with the same SR-alarm + still-positive + vol-regime pattern). Re-check trigger: after N>=100 monitored trades per lane, rerun `python -m trading_app.sr_monitor`. Auto-revert per lane if SR remains ALARM AND per-lane ExpR < +0.05. Procedural improvement deferred: add SR-clean check as a pre-flight gate for future profit-expansion stages so this is caught before the prop_profiles commit, not after. |

## Won't Fix (ACCEPTABLE)

| ID | Iter | Target | Description | Reasoning |
|----|------|--------|-------------|-----------|
| WF-01 | 39 | scoring.py:SC1 | Hardcoded SINGAPORE_OPEN/TOKYO_OPEN in heuristic bonus | Intentional per-session adjustments, not a canonical list. Worst case on rename: bonus silently stops. Not safety/correctness. |
| WF-02 | 19 | execution_engine.py:EE3 | IB hardcoded 23:00 UTC for TOKYO_OPEN | Correctly documents Brisbane UTC+10 fixed offset. No DST. IB_DURATION_MINUTES from config. |
| WF-03 | 44 | strategy_fitness.py | Full scan clean — no findings | Audited iter 44, no actionable findings |
| WF-04 | 58 | projectx/positions.py:35 | `avg_price: p.get("averagePrice", 0)` uses int 0 vs float 0.0 | Style difference, no correctness impact. avg_price is only used for logging in session_orchestrator (never for P&L computation). |
| WF-05 | 96 | scripts/tools/audit_15m30m.py:29,44,62,88 | Hardcoded `IN ('MGC','MNQ','MES','M2K')` in SQL queries | Read-only investigation script. Matches current active instruments exactly. If instrument removed, SQL returns 0 rows — not dangerous. Pattern: one-off diagnostic, not canonical source. |
| WF-06 | 161 | trading_app/live/rithmic/contracts.py:22-26 | `INSTRUMENT_ROOTS` hardcodes `{"MES","MNQ","MGC"}` | Translation dict; fallback `INSTRUMENT_ROOTS.get(instrument, instrument)` is functionally correct for all CME micros (root == symbol name). No safety impact. |
| WF-07 | 161 | trading_app/live/rithmic/positions.py:93-95 | `query_equity()` returns `None` on exception | Intentional contract: `float | None`. HWM tracker `update_equity(None)` designed for this — tracks consecutive failures and halts after N. All callers guard for None. |

## Resolved Findings

| ID | Iter Found | Resolved | Commit | Description |
|----|-----------|----------|--------|-------------|
| DF-01 | 9/11 | 23 | f7bd0c4 | Conditional EXITED trade prune — made unconditional; silent-exit paths now pruned correctly |
| DF-03 | 9/11 | 40 | ACCEPTABLE | IB hardcoded 23:00 UTC — reassessed: correctly documents Brisbane UTC+10 fixed offset, no DST, IB_DURATION_MINUTES from config. Not a defect. |
| DF-07 | 13 | slate-clear | 7cf57cb | HOT tier thresholds unannotated — @research-source annotation added |
| DF-09 | 21 | 25 | 8261a0e | OR2: No fill_price parsing tests — unit tests added for both Tradovate and ProjectX routers |
| DF-10 | 26 | post-26 | (this session) | E1/E3 zero-risk paths silent continue — REJECT events added; E3 block documented as defensive dead code (unreachable by construction); E1/E2 tests added |
| DF-05 | 13 | 28 | already-present | build_edge_families.py thresholds — annotations already present at iter 28 audit; stale ledger entry |
| DF-06 | 13 | 28 | already-present | strategy_validator.py WF thresholds — annotations already present at iter 28 audit; stale ledger entry |
| DF-08 | 13 | 28 | 43a86ba | live_config.py LIVE_MIN_EXPECTANCY_R + LIVE_MIN_EXPECTANCY_DOLLARS_MULT — @research-source annotations added |
| DF-11 | 27 | 31 | 9158b77 | Hardcoded ("E1","E2","E3") in rolling_portfolio + paper_trader → canonical ENTRY_MODELS import |
| DF-02 | 9/11 | 45 | 4c6bc4d | ARMED/CONFIRMING silent exit at session_end — logger.debug() added; no behavior change |
| DF-04 | 12 | 2026-04-11 | (pending commit) | `compute_day_of_week_stats` now threads `orb_minutes: int = 5` through both the `daily_features` eligibility query and the `orb_outcomes` query (PIPELINE_AUDIT_2026-02-27 F1 sibling bug also fixed). 4 regression tests added. |
| MGC-FP | n/a | 2026-04-12 | (pending commit) | Pulse MGC false positives structurally eliminated: added `deployable_expected` flag to `pipeline/asset_configs.py`, derived `DEPLOYABLE_ORB_INSTRUMENTS` constant + `get_deployable_instruments()` helper. `collect_fitness_fast` + `collect_staleness` now scope alerts to deployable subset. New drift check 98 enforces `DEPLOYABLE ⊆ ACTIVE` invariant. 10 regression tests (asset_configs + pulse + drift). Root cause: `orb_active=True` conflated "pipeline runs on it" with "expected to have validated strategies" — MGC's 3.8yr real-micro horizon is insufficient for T7 era-discipline survival, so the empty shelf was by-design, not decaying. |
| LaneReg | n/a | 2026-04-12 | (pending commit) | Latent production bug: `trading_app.prop_profiles.get_lane_registry` raised `ValueError: Profile has multiple lanes for session(s)` on the active `topstep_50k_mnq_auto` profile after the 2026-04-10 multi-RR discovery added duplicate EUROPE_FLOW and TOKYO_OPEN lanes. `trading_app/live/session_orchestrator.py:130` calls this function in `__init__` and re-raises for profile-backed portfolios (fail-closed) — meaning the live bot could not initialize against the active profile in live mode. Masked because the bot was running in signal/demo mode. Fix: `get_lane_registry` now allows duplicate sessions as long as every lane on the same session shares the same `max_orb_size_pts` (ORB cap is a session-level attribute by convention); raises only when caps disagree. New regression test asserts consistent-duplicate sessions are accepted on the live profile and inconsistent caps on `topstep_50k_type_a` still fail closed. Fixed in-scope during profit expansion because it was blocking that stage's verification path and is the same code region. |
| ProfitX | n/a | 2026-04-12 | (pending commit) | Profit expansion of `topstep_50k_mnq_auto` via C11 Monte Carlo self-audit loop. Screened 24 validated-not-deployed candidates down to 2 highest-honest-EV additions: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` and `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`. Both are pure new-session lanes (COMEX_SETTLE and CME_PRECLOSE were not in `allowed_sessions` before) with distinct families from the deployed core. Two additional same-session candidates (NYSE_OPEN X_MES_ATR60, EUROPE_FLOW OVNRNG_100) were evaluated but REJECTED — 9-lane MC run showed C11 dropped from 86.2% to 75.8% due to same-session drawdown compounding. 7-lane config actually IMPROVES C11 to 88.4% and reduces trailing_dd_breach from 13.8% to 11.6% while raising p50 90d PnL from $1,133 to $1,762 per copy. Deterministic annualized incremental EV ≈ +$5,100/yr at 2 copies. `max_slots` 5 → 7; `allowed_sessions` expanded to include COMEX_SETTLE and CME_PRECLOSE; `allowed_instruments` unchanged (MNQ only). Drift check 95 validates every new lane is in `validated_setups` with `status='active'`. |
