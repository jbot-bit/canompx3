# Codex Current State

Snapshot for fast project awareness. Confirm changing numbers from code or DB before citing them externally.

## Core Reality

- The repo is beyond basic research scaffolding. Pipeline, discovery, validation, portfolio construction, execution engine, risk manager, paper trading, and broker adapters all exist.
- The live trading stack is partially built, not fully finished. Core runtime pieces exist, but `ROADMAP.md` still leaves monitoring and alerting as TODO, and `trading_app/live_config.py` marks HOT real-time wiring as dormant/not yet production-ready.
- The system uses one canonical `gold.db` at the repo root unless `DUCKDB_PATH` overrides it.
- Event-based sessions and DST cleanup are complete. Session logic lives in `pipeline/dst.py`.
- Live portfolio selection is declarative in `trading_app/live_config.py`, but that should not be read as "live setup complete."

## What Is Established

- 5m ORB remains the default production aperture.
- Wider-aperture support exists in the standard pipeline, but aperture expansion should be treated as targeted research, not assumed default behavior.
- E2 is the main live entry model. E0 is purged. E3 is soft-retired.
- Calendar effects are instrument x session specific. Blanket skip rules are wrong.
- Cross-instrument stacking at TOKYO_OPEN is not a free diversification win. MNQ and MES are too correlated there.

## What Still Matters Operationally

- Finishing the live trading setup/coding remains active work; do not describe the repo as fully live-ready yet.
- Monitoring and alerting remains an explicit TODO in `ROADMAP.md`.
- Re-running rolling evaluation over the longer validation window remains a follow-up task after the 2016-2020 outcomes backfill work.
- Some research findings are still trigger-based, not action-ready. Example: overlap-day revalidation when MES and MNQ overlap days reach 800.

## Live / Runtime Surfaces

- Live portfolio: `trading_app/live_config.py`
- Live orchestration: `trading_app/live/session_orchestrator.py`
- Execution engine: `trading_app/execution_engine.py`
- Broker implementations: `trading_app/live/projectx/`, `trading_app/live/tradovate/`
- Webhook path: `trading_app/live/webhook_server.py`

## Quality / Guardrails

- Drift detection is a first-class safety surface in `pipeline/check_drift.py`.
- Validation and sync discipline are mandatory, especially around config, schema, and live parity.
- Research claims must respect the statistical and mechanism rules in `RESEARCH_RULES.md`.

## Current Bias For Action

- Push toward robust, deployable edges and finishing live safety, not decorative research.
- Prefer family-level and portfolio-level robustness over isolated single-strategy backtest wins.
- Treat monitoring, drift detection, live parity, and honest runtime-status reporting as high leverage.
