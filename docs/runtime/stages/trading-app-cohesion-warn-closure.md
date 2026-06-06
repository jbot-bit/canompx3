---
task: "trading_app cohesion — close verified WARNs to A+ (canonical-delegation HIGHs, import cycles, orphans)"
mode: CLOSED
---

## Scope Lock

- trading_app/live/session_orchestrator.py
- trading_app/account_hwm_tracker.py
- trading_app/portfolio.py
- trading_app/prop_portfolio.py
- trading_app/prop_profiles.py
- trading_app/strategy_validator.py
- trading_app/strategy_discovery.py
- trading_app/dsr.py
- trading_app/deployability.py
- tests/test_trading_app/test_strategy_validator.py
- trading_app/setup_detector.py
- trading_app/strategy_lineage_ast.py
- research/archive/setup_detector.py
- research/archive/strategy_lineage_ast.py
- tests/test_research/test_setup_detector.py
- tests/test_research/test_strategy_lineage_ast.py
- tests/test_trading_app/test_account_hwm_tracker.py
- tests/test_trading_app/test_session_orchestrator.py
- tests/test_trading_app/test_prop_portfolio.py

## Blast Radius

- session_orchestrator.py:1762-1767 — LIVE capital path (EOD rollover). Swaps inline `bris.hour<9` boundary for `pipeline.dst.compute_trading_day_from_timestamp`. Behavior must be byte-identical (returns a `date`); canonical raises on naive ts (caller already passes aware `bar_ts_utc`). TDD-gated + adversarial audit.
- account_hwm_tracker.py:427-428,444-445 — LIVE capital path (daily/weekly loss-limit resets). Two inline boundaries. MUST preserve string return (`strftime`) and naive→UTC coercion contract; wrap canonical call. TDD-gated + adversarial audit.
- portfolio.py / prop_portfolio.py — bidirectional import cycle broken only by a lazy import. Relocate `_compute_dd_per_contract` into prop_profiles (leaf both already import). Capital path; verify all importers.
- strategy_validator.py / strategy_discovery.py — research-layer cycle. Relocate `benjamini_hochberg` into an existing stats leaf; remove lazy import at strategy_discovery.py:1664.
- setup_detector.py, strategy_lineage_ast.py — VERIFIED zero production callers (broad grep across repo incl. check_drift/fingerprint/Path/bat/yaml). git mv to research/archive/. Reversible.
- Reads: gold.db (none — pure code). Writes: none. No schema, no entry-model logic.

## WARN Ledger (the persisted grade artifact — prior session's grade was never written down)

All findings INDEPENDENTLY VERIFIED by the lead (read + grep + execution), not trusted from the audit agents. Of ~30 candidate findings across 3 dimension agents, 6 survived falsification (matching the repo's documented 6/10-false audit-finding rate, feedback_audit_findings_are_claims_falsify_each_before_acting_2026_06_07).

Baseline PASS (not WARNs): pipeline/→trading_app/ layering clean; no hardcoded cost/commission/slippage; ORB windows use orb_utc_window; instrument lists import ACTIVE_ORB_INSTRUMENTS; config/db_manager/db_access cohesive.

| # | WARN | Dim | Sev | Status |
|---|------|-----|-----|--------|
| W1 | session_orchestrator.py:1762-1767 re-encodes trading-day boundary | Duplicated canonical | HIGH | ✅ CLOSED |
| W2 | account_hwm_tracker.py:427-428,444-445 re-encode boundary ×2 | Duplicated canonical | HIGH | ✅ CLOSED |
| W3 | portfolio↔prop_portfolio import cycle (lazy-import workaround) | Coupling/cycle | MED | ✅ CLOSED |
| W4 | strategy_validator↔strategy_discovery cycle (lazy benjamini_hochberg) | Coupling/cycle | MED | ✅ CLOSED |
| W5 | setup_detector.py orphan (0 production callers) | Dead code | LOW | ✅ CLOSED |
| W6 | strategy_lineage_ast.py orphan (unwired CLI MVP) | Dead code | LOW | ✅ CLOSED |

### Closure evidence
- W1: 2 equivalence tests (pre-09:00 + at-09:00 vs canonical) in test_session_orchestrator.py; 30 rollover tests pass. Boundary delegates to compute_trading_day_from_timestamp.
- W2: 6 characterization tests (TestBrisbaneBoundaryCanonicalEquivalence) pin string-return + naive-UTC + week-Monday contract; 97 hwm tests pass.
- W3: portfolio no longer imports prop_portfolio; `_compute_dd_per_contract` + 3 constants → prop_profiles leaf; 119 tests pass.
- W4: strategy_discovery no longer imports strategy_validator; `benjamini_hochberg` → dsr leaf (re-exported from validator for back-compat); 80 tests pass.
- W5/W6: git mv to research/archive/ + tests to tests/test_research/ (imports rewired); 12 tests pass. Zero production callers re-confirmed by broad grep.
- Global: ruff clean; check_drift 180/0; 714-test affected-module sweep green.
- **Adversarial audit (evidence-auditor, independent context): VERDICT PASS.** Highest-risk path (naive-ts reaching the canonical raise in W1) disproven — projectx/data_feed.py always passes datetime.now(UTC); BarAggregator.replace() preserves tzinfo.

### Noted follow-up (OUT OF SCOPE — not a cohesion WARN)
Auditor advisory: `Bar.ts_utc` (bar_aggregator.py:21) is annotated `datetime` with no runtime tz-awareness guard. W1's safety rests on the data-feed caller contract. A structural `replace(tzinfo=UTC)` coercion in `BarAggregator._open_bar` would make it unconditionally safe against a future naive feed. Defense-in-depth on the data-feed boundary — file as its own ticket; raising on bad input is already safer than the old silent mis-assignment.

## Verification plan

- W1/W2: equivalence tests FIRST (old inline result == canonical result) across a DST-boundary day + a pre-09:00 case + a naive-ts case; then implement; then evidence-auditor adversarial gate (institutional-rigor §2, capital path).
- W3/W4: ruff + import-resolution smoke + confirm cycle gone (`python -c` import both top-level) + check_drift.
- W5/W6: confirm zero callers post-move; check_drift basename allowlists preserved.
- Global: ruff clean; full targeted test run (show output); check_drift.py 180+/0; dead-code grep.
