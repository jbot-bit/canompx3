# Ralph Loop — Deferred Findings Ledger

> **Purpose:** Structured tracking of deferred findings. Nothing gets lost.
> **Rule:** Every finding deferred in ralph-loop-history.md MUST have a row here.
> **Enforcement:** Drift check scans this file. Zero-item file = clean. Items here = known debt.
> **Lifecycle:** Add when deferred. Remove when fixed (cite commit hash). Never delete silently.

## Open Findings

| ID | Iter | Severity | Target | Description | Deferred Reason |
|----|------|----------|--------|-------------|-----------------|
| DF-02 | 9/11 | LOW | execution_engine.py:~1020 | E3 silent exit — no log when E3 limit order expires unfilled | Dormant: E3 soft-retired |
| DF-03 | 9/11 | LOW | execution_engine.py:~879 | IB hardcoded 23:00 UTC close time — only affects TOKYO_OPEN on IB | Dormant: IB not in active use |
| DF-04 | 12 | LOW | rolling_portfolio.py:304 | Dormant `orb_minutes=5` in rolling DOW stats — multi-aperture TODO | Annotated TODO, not blocking |
| DF-11 | 27 | LOW | rolling_portfolio.py:228 | Hardcoded ("E1", "E2", "E3") set in aggregate_rolling_performance — StopIteration if new entry model added | Dormant: no E4 yet; should reference canonical config.ENTRY_MODELS when E4 arrives |

## Resolved Findings

| ID | Iter Found | Resolved | Commit | Description |
|----|-----------|----------|--------|-------------|
| DF-01 | 9/11 | 23 | f7bd0c4 | Conditional EXITED trade prune — made unconditional; silent-exit paths now pruned correctly |
| DF-07 | 13 | slate-clear | 7cf57cb | HOT tier thresholds unannotated — @research-source annotation added |
| DF-09 | 21 | 25 | 8261a0e | OR2: No fill_price parsing tests — unit tests added for both Tradovate and ProjectX routers |
| DF-10 | 26 | post-26 | (this session) | E1/E3 zero-risk paths silent continue — REJECT events added; E3 block documented as defensive dead code (unreachable by construction); E1/E2 tests added |
| DF-05 | 13 | 28 | already-present | build_edge_families.py thresholds — annotations already present at iter 28 audit; stale ledger entry |
| DF-06 | 13 | 28 | already-present | strategy_validator.py WF thresholds — annotations already present at iter 28 audit; stale ledger entry |
| DF-08 | 13 | 28 | 43a86ba | live_config.py LIVE_MIN_EXPECTANCY_R + LIVE_MIN_EXPECTANCY_DOLLARS_MULT — @research-source annotations added |
