# Codex Next Steps

This file captures the highest-signal open work implied by the canonical roadmap and current live setup.

## Highest-Leverage Open Work

- Finish the remaining live-trading setup/coding rather than talking about the stack as if it is already complete.
- Implement minimal monitoring and alerting for live trading:
  - drawdown vs historical
  - win-rate divergence
  - ORB size regime shift
  - live strategy status surface
- Keep HOT/rolling real-time logic out of any "done" narrative until the dormant wiring in `trading_app/live_config.py` is actually reactivated and verified.
- Re-run rolling evaluation using the longer post-backfill range and update stability scoring inputs if the data window is now sufficient.

## Research Follow-Ups Worth Respecting

- Test the MES NYSE_OPEN summer signal-stacking result at E1 before treating it as actionable.
- Only revisit overlap-dependent cross-instrument findings when the overlap-day trigger is actually met.
- Treat aperture expansion as targeted research with explicit comparison against the 5m baseline, not as a default migration path.

## Good Default Priorities

- Live safety and monitoring
- Backtest/live parity checks
- Robustness of currently traded families
- Audits of live execution paths and webhook safety

## Things To Avoid

- Building a second rule system in `.codex/`
- Repeating or restating `TRADING_RULES.md` and `RESEARCH_RULES.md` in detail
- Optimizing for sample size instead of robustness
- Treating in-sample observations as validated edges
