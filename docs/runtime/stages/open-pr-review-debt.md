---
mode: TRIVIAL
task: Review and decide on 6 deferred open PRs that touch live trading or pipeline canonical code (and 1 DRAFT)
scope_lock:
  - PR #99 (DRAFT — Codex's call when ready)
  - PR #72 (research/audit_pre_break_context_cross_session.py + 8 production files)
  - PR #74 (research/audit_l2_atr_p50_regime_vs_arithmetic.py + 6 production files)
  - PR #36 (pipeline/backfill_pit_range_atr.py + 9 files)
  - PR #59 (sizer-rule OOS, 91 files, +10469 LOC)
  - PR #8 (DSR allocator, 18 files, +3175 LOC)
  - PR #12 (Bailey-LdP Eq 9 fix, paired with #8 — base = audit/a2b-1-regime-gate-phase2)
acceptance:
  - Each PR either merged, closed-stale, or has an explicit decision recorded with reasoning.
  - No bulk-merging without dispatching evidence-auditor on PRs that touch live trading or pipeline canonical files.
updated: 2026-04-26
---

# Open PR Review Debt

Carved out from the 2026-04-26 autonomous PR burndown session. 25 PRs were merged or closed. These 7 require human or evidence-auditor review before merge — bulk-merging would violate institutional rigor.

## Why these are deferred

| PR | Title | Defer reason |
|---|---|---|
| #99 | research(mnq): harden geometry transfer + stamp COMEX family | DRAFT — wait for Codex |
| #72 | research(audit): cross-session pre-break context — universal null at K=89 | +830 LOC, touches `bar_aggregator.py`, `session_orchestrator.py`, `derived_state.py`, `eligibility/builder.py`, `CLAUDE.md`, `.claude/settings.json` |
| #74 | research(audit): L2 ATR_P50 — weak mixed signal | +597 LOC, touches same live trading files as #72 (likely paired) |
| #36 | hardening(infrastructure): close three ghost-deployment classes | +1033 LOC, new `pipeline/backfill_pit_range_atr.py` + `build_daily_features` + `check_drift` + `ingest_statistics` modifications |
| #59 | PR #48 sizer-rule OOS backtest: SIZER_ALIVE on MES + MGC at capital-neutral | +10469 LOC, 91 files, modifies `trading_app/cpcv.py` + `strategy_validator.py` + 6 hooks |
| #8 | A2b-2 Shape E: DSR exposure + L6 swap retrospective | +3175 LOC, 18 files, modifies `lane_allocator.py` and rebalancer |
| #12 | fix(A2b-2 Shape E): Bailey-LdP Eq 9 population mismatch + multi-pair test | base=audit/a2b-1-regime-gate-phase2 — stacked on #8; only landable after #8 |

## Recommended next-session workflow

1. Dispatch `evidence-auditor` in parallel on each non-draft PR (#72, #74, #36, #59, #8/#12 as a pair).
2. For each, the auditor reports: does the change preserve canonical sources? does it regress live trading? are tests included?
3. Merge the PASS verdicts; close the FAILs with a comment citing the auditor finding.
4. Save any institutional-rigor lessons surfaced.

## Context for future readers

The 2026-04-26 session merged 25 PRs covering research verdicts (KILL/PARK/null), locked preregs, plans, and infra hardening. All those were research-doc-only or low-blast. The 7 remaining hold real production-code changes — they need eyes, not bulk action.

Stack-collapse pattern (from earlier in the same session): when Codex's stacked PRs have their prereg-base merged with `--delete-branch`, the audit-run PR auto-closes and the verdict files become orphaned on the runner branch. Mitigation: revive via fresh PR from runner→main. Six PRs (#111–#117) were revived this way during the session.
