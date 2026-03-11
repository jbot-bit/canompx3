# Ralph Loop — Deferred Findings Ledger

> **Purpose:** Structured tracking of deferred findings. Nothing gets lost.
> **Rule:** Every finding deferred in ralph-loop-history.md MUST have a row here.
> **Enforcement:** Drift check scans this file. Zero-item file = clean. Items here = known debt.
> **Lifecycle:** Add when deferred. Remove when fixed (cite commit hash). Never delete silently.

## Open Findings

| ID | Iter | Severity | Target | Description | Deferred Reason |
|----|------|----------|--------|-------------|-----------------|
| DF-01 | 9/11 | LOW | execution_engine.py:~688 | Conditional EXITED trade prune — may skip trades that should count | Dormant: E3 soft-retired, prune harmless in current config |
| DF-02 | 9/11 | LOW | execution_engine.py:~1020 | E3 silent exit — no log when E3 limit order expires unfilled | Dormant: E3 soft-retired |
| DF-03 | 9/11 | LOW | execution_engine.py:~879 | IB hardcoded 23:00 UTC close time — only affects TOKYO_OPEN on IB | Dormant: IB not in active use |
| DF-04 | 12 | LOW | rolling_portfolio.py:304 | Dormant `orb_minutes=5` in rolling DOW stats — multi-aperture TODO | Annotated TODO, not blocking |
| DF-05 | 13 | LOW | build_edge_families.py:31-38 | Edge family thresholds missing @research-source | Annotation debt |
| DF-06 | 13 | LOW | strategy_validator.py:654-656 | WF gate thresholds missing @research-source | Annotation debt |
| DF-07 | 13 | LOW | live_config.py:54-57 | HOT tier thresholds unannotated | Annotation debt |
| DF-08 | 13 | LOW | live_config.py:354-355,583-584 | Live portfolio constructor magic numbers inline | Annotation debt |
| DF-09 | 21 | LOW | order routers (both) | OR2: No fill_price parsing tests | Test coverage gap |

## Resolved Findings

| ID | Iter Found | Resolved | Commit | Description |
|----|-----------|----------|--------|-------------|
| — | — | — | — | *(none yet)* |
