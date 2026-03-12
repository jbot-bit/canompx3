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
| DF-05 | 13 | LOW | build_edge_families.py:31-38 | Edge family thresholds missing @research-source | Annotation debt |
| DF-06 | 13 | LOW | strategy_validator.py:654-656 | WF gate thresholds missing @research-source | Annotation debt |
| DF-08 | 13 | LOW | live_config.py:354-355,583-584 | Live portfolio constructor magic numbers inline | Annotation debt |

## Resolved Findings

| ID | Iter Found | Resolved | Commit | Description |
|----|-----------|----------|--------|-------------|
| DF-01 | 9/11 | 23 | f7bd0c4 | Conditional EXITED trade prune — made unconditional; silent-exit paths now pruned correctly |
| DF-07 | 13 | slate-clear | 7cf57cb | HOT tier thresholds unannotated — @research-source annotation added |
| DF-09 | 21 | 25 | 8261a0e | OR2: No fill_price parsing tests — unit tests added for both Tradovate and ProjectX routers |
| DF-10 | 26 | post-26 | (this session) | E1/E3 zero-risk paths silent continue — REJECT events added; E3 block documented as defensive dead code (unreachable by construction); E1/E2 tests added |
