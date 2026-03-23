# Comprehensive Audit of the Canompx3 Trading Platform

## Summary
- Audit the production path first: pipeline, trading runtime, UI/API, CI, and operational scripts. Treat research, archive, and `tmp_*` files as secondary unless they are imported or operationally coupled.
- Use three evidence sources together: source inspection, read-only interrogation of the live DuckDB catalog, and the existing audit scaffold under `scripts/audits`. The scaffold is baseline evidence, not the final authority.
- Start from the hotspots already exposed by exploration: live E2 order semantics, live use of prior-day feature proxies, inline schema evolution on a single canonical DuckDB file, an empty pipeline audit log despite append-only audit code, permissive CI gates, and large control-tower modules.

## Audit Workstreams
- Architecture and operability: map boundaries between data pipeline, research, live execution, webhook, and UI; assess single-DB blast radius, concurrency discipline, module concentration, and deployment exposure.
- Trading correctness: verify backtest/live parity for entry handling, feature availability, risk controls, kill-switch behavior, and broker adapters. Prioritize `execution_engine`, `live/session_orchestrator`, `live/*/order_router`, and the webhook path.
- Database and data governance: audit `gold.db` as the canonical store, the 249-column `daily_features` design, large fact tables, constraint coverage, `SELECT *` usage, inline `ALTER TABLE` migrations, and whether lock/audit controls are actually active in write paths.
- Engineering quality and controls: review CI/test effectiveness, type/lint gates, existing audit phases, and whether tests prove trading semantics or only mocked structure.

## Validation
- Run automated checks only in the repo-managed Python environment (`uv`/synced env), not the global interpreter, so dependency issues do not contaminate repo findings.
- Explicitly prove or falsify these scenarios:
  - E2 live orders reproduce engine semantics rather than placing a second stop after the signal.
  - Live filters use same-day data or a defensible real-time substitute.
  - Pipeline writes emit immutable audit rows into `pipeline_audit_log`.
  - Public/exposed order endpoints are replay-safe, duplicate-safe, and bounded.
- Separate environment/tooling problems from codebase findings so the final report only contains actionable system risks.

## Deliverable
- Produce one narrative audit organized as `Critical Issues`, `High-Impact Improvements`, `Quality Enhancements`, and `Future Considerations`.
- For each recommendation include: observed evidence, why it matters in a trading/financial context, and the next concrete step.
- Add a short executive summary and a compact evidence appendix with key table stats, freshness, and the highest-signal code anchors.

## Assumptions
- Deliver the audit in chat, not as a committed file.
- Production scope means `pipeline`, `trading_app`, `ui`, `ui_v2`, CI/workflows, and operational `scripts`; `research`, `archive`, and `tmp_*` content is only reviewed for bleed-through or hygiene risk.
- Baseline facts (STALE snapshot — Mar 6 2026, pre-rebuild): data across 8 symbols; validated/outcome counts change after every rebuild — query DB for current state. Several 1k-3k line control modules.
