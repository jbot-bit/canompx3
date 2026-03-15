# Ralph Loop — Deferred Findings Ledger

> **Purpose:** Structured tracking of deferred findings. Nothing gets lost.
> **Rule:** Every finding deferred in ralph-loop-history.md MUST have a row here.
> **Enforcement:** Drift check scans this file. Zero-item file = clean. Items here = known debt.
> **Lifecycle:** Add when deferred. Remove when fixed (cite commit hash). Never delete silently.

## Open Findings

| ID | Iter | Severity | Target | Description | Deferred Reason |
|----|------|----------|--------|-------------|-----------------|
| DF-04 | 12 | LOW | rolling_portfolio.py:304 | Dormant `orb_minutes=5` in rolling DOW stats — multi-aperture TODO | Annotated TODO, not blocking |

## Won't Fix (ACCEPTABLE)

| ID | Iter | Target | Description | Reasoning |
|----|------|--------|-------------|-----------|
| WF-01 | 39 | scoring.py:SC1 | Hardcoded SINGAPORE_OPEN/TOKYO_OPEN in heuristic bonus | Intentional per-session adjustments, not a canonical list. Worst case on rename: bonus silently stops. Not safety/correctness. |
| WF-02 | 19 | execution_engine.py:EE3 | IB hardcoded 23:00 UTC for TOKYO_OPEN | Correctly documents Brisbane UTC+10 fixed offset. No DST. IB_DURATION_MINUTES from config. |
| WF-03 | 44 | strategy_fitness.py | Full scan clean — no findings | Audited iter 44, no actionable findings |
| WF-04 | 58 | projectx/positions.py:35 | `avg_price: p.get("averagePrice", 0)` uses int 0 vs float 0.0 | Style difference, no correctness impact. avg_price is only used for logging in session_orchestrator (never for P&L computation). |

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
