# Codex Next Steps

This file captures stable priority categories only. For changing session-level priorities, use `HANDOFF.md`.

## Default Priority Order

- Current shared baton first: `HANDOFF.md`
- Canonical roadmap and unfinished platform work: `ROADMAP.md`
- Trading truth and what is actually deployable: `TRADING_RULES.md`
- Research discipline for any new claim or analysis: `RESEARCH_RULES.md`

## Good Default Priorities

- Live safety and monitoring
- Backtest/live parity checks
- Robustness of currently traded families
- Audits of live execution paths and webhook safety
- Codex adapter maintenance only when it removes real friction

## Things To Avoid

- Building a second rule system in `.codex/`
- Repeating or restating `TRADING_RULES.md`, `RESEARCH_RULES.md`, or `HANDOFF.md` in detail
- Freezing volatile project state into Codex summaries
- Optimizing for sample size instead of robustness
- Treating in-sample observations as validated edges
