---
slug: triage-e2-lookahead-9-candidates
mode: IMPLEMENTATION
created: 2026-04-28
classification: judgment
---

# Triage 9 E2 Look-Ahead Drift-Check Candidates

Closes 2026-04-21 postmortem § 5.1 deferred audit; sweeps 9 scripts surfaced by drift check `check_e2_lookahead_research_contamination` beyond the 18-file registry.

## Phases

- Phase 0 — doctrine fix (§ 6.1 / 6.3 + postmortem closure)
- Phase A — Phase D D0 backtest verification + runbook flag
- Phase B — cleared/late-fill/not-predictor annotations (cheap)
- Phase C — predictor-tainted annotations + registry sweep section
- Phase D — re-derivation dispatch note + HANDOFF
- Phase E — drift check + tests + commit

## Scope lock

- `.claude/rules/backtesting-methodology.md` (§ 6.1, § 6.3 only)
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` (closure append)
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` (sweep + dispatch sections)
- `research/phase_d_d0_backtest.py`
- `research/audit_sizing_substrate_diagnostic.py`
- `research/break_delay_filtered.py`
- `research/break_delay_nuggets.py`
- `research/mnq_comex_unfiltered_overlay_v1.py`
- `research/mnq_l1_europe_flow_prebreak_context_v1.py`
- `research/l1_europe_flow_pre_break_context_scan.py`
- `research/output/confluence_program/phase0_run.py`
- `research/shadow_htf_mes_europe_flow_long_skip.py`
- `memory/phase_d_daily_runbook.md` (one-bullet contamination flag — written to user's auto-memory dir)
- `HANDOFF.md` (one-line dispatch pointer)

## Acceptance

- `python pipeline/check_drift.py` — zero new violations beyond 18 grandfathered
- `python -m pytest tests/test_pipeline/test_check_drift_e2_lookahead.py -v` — 10/10 pass
- No `pipeline/`, `trading_app/`, schema, or canonical-config files touched
- Registry doc + postmortem + § 6.1/6.3 cross-link consistently
