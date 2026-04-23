# Recent PR Follow-Through Queue

**Date:** 2026-04-22
**Purpose:** reconcile the recent PR/audit burst against what actually landed, what findings were already actioned, and what remains unactioned. This is a follow-through queue, not a fresh discovery memo.

## Scope

Recent merged PR window reviewed:

- PR #56 — `mnq-pr51-dsr-audit`
- PR #67 — `l1-europe-flow-raw-break-quality-prereg`
- PR #69 — `garch-r3-shadow-ledger`
- PR #71 — `l1-orb-g5-arithmetic-only-check`
- PR #73 — `comex-settle-orb-g5-diagnostic`
- PR #75 — `wt-codex-backlog-cleanup`

Canonical source docs cross-read:

- `docs/audit/results/2026-04-21-recent-claims-skeptical-reaudit-v1.md`
- `docs/audit/results/2026-04-21-pr48-participation-shape-oos-replication-v1.md`
- `docs/audit/results/2026-04-19-gc-mgc-translation-audit.md`
- `docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md`
- `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`
- `docs/audit/results/2026-04-19-sr-monitor-stream-audit.md`
- `docs/plans/2026-04-21-post-stale-lock-action-queue.md`

## Already Actioned

These findings are not open backlog anymore.

| Item | Source | Current state |
|---|---|---|
| PR #51 five `CANDIDATE_READY` cells were overstated | PR #56 + skeptical re-audit | **ACTIONED.** Reclassified by DSR audit; do not re-open as deploy candidates. |
| PR #48 lacked OOS replication | skeptical re-audit | **ACTIONED.** OOS β₁ replication landed; MES and MGC now OOS-confirmed, MNQ remains UNVERIFIED. |
| PR #48 deployable sizer-rule derivation for MES/MGC | skeptical re-audit + OOS replication | **ACTIONED.** Frozen-rule replay landed in `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`: `MGC` = `SIZER_DEPLOY_CANDIDATE`, `MES` = `SIZER_ALIVE_NOT_READY`. Do not treat PR48 as a pooled MES/MGC promotion story anymore. |
| MGC 5m payoff-compression audit on warm GC→MGC families | `docs/audit/results/2026-04-19-gc-mgc-translation-audit.md` | **ACTIONED.** Diagnostic audit is already landed and now explicitly stamped in `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md`: `PAYOFF_COMPRESSION_REAL=YES`, `LOW_RR_RESCUE_PLAUSIBLE=YES`, `NO_RESCUE_SIGNAL=NO`. Future follow-up is a narrow MGC exit-shape prereg only. |
| L1 EUROPE_FLOW break-bar columns were wrongly treated as E2-safe | PR #67 | **ACTIONED.** Doctrine corrected, postmortem landed, prereg landed. |
| L1 EUROPE_FLOW pre-break-context scan stage | PR #67 prereg + 2026-04-21 L1 diagnostics | **ACTIONED.** Restored missing prereg and executed the frozen `K=2` family in `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md`. Verdict: **KILL**. `pre_velocity_HIGH_Q3` never reached raw significance; `rel_vol_HIGH_Q3` failed BH-FDR at `q=0.0537` and flipped OOS sign. |
| MNQ NYSE_CLOSE RR1.0 failure-mode / governance follow-up | `docs/audit/results/2026-04-19-mnq-nyse-close-failure-mode-audit.md` | **ACTIONED.** Executed in `docs/audit/results/2026-04-23-mnq-nyse-close-rr10-followup.md`. Verdict: **CONTINUE with narrow prereg**. The family is alive-but-underexecuted; the direct next move is the exact `MNQ NYSE_CLOSE ORB_G8 RR1.0` prereg, not a raw portfolio unblock and not another broad sweep. |
| MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg execution | `docs/audit/results/2026-04-23-mnq-nyse-close-rr10-followup.md` | **ACTIONED.** Executed in `docs/audit/results/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.md`. Verdict: **KILL**. The exact native filter path fails era stability (`2025` on-signal negative at `N=132`) and gives no 2026 selector contrast (`42/42` OOS fire). This kills the filter role, not the broad RR1.0 NYSE_CLOSE family. |
| ORB_G5 “replace filter everywhere” framing | PR #71 + PR #73 | **ACTIONED.** Reframed as lane-specific. L1 only is the surviving path; COMEX_SETTLE is not a generic replacement story. |
| GARCH R3 path was blocked by bad builder state | PR #69 + 2026-04-22 checker hardening | **ACTIONED.** Shadow artifact is now `READY_FOR_FORWARD_MONITORING`; builder + drift guard hardened. |
| Recent checker bugs around GARCH warmup and lane-corr monitor semantics | 2026-04-22 review follow-through | **ACTIONED.** Drift threshold fixed; monitor gap/no-data semantics fixed; full test suite passed. |

## Ranked Open Queue

Scoring:

- **EV:** expected strategic or operational value if the task succeeds
- **Effort:** expected implementation / audit effort
- **ROI rank:** judgment over EV divided by effort, adjusted for prerequisite readiness and blast radius

| ROI rank | Task | Why still open | EV | Effort | Why now |
|---:|---|---|---:|---:|---|
| 1 | MNQ NYSE_CLOSE RR1.0 role audit | The exact native filter path is now closed `KILL`, but the broad RR1.0 family remains positive on canonical baselines; the unresolved question is role, not filter | 4 | 2 | Highest-EV way to avoid missing a real broad-session edge while also preventing more wrong-layer NYSE_CLOSE work |
| 2 | Prior-day Pathway-B bridge execution triage | The branch no longer lacks locks: exact MNQ prior-day bridge hypotheses are now saved, but the repo still does not say which already-locked path should be executed next | 3 | 2 | Highest-signal way to turn the saved prior-day bridge package into one honest confirm-or-kill execution instead of leaving it as dormant scaffolding |
| 3 | Cross-asset earlier-session → later-ORB quality chronology spec | Still blocked on chronology discipline; no narrow spec written yet | 3 | 3 | Plausible, but easier to get wrong than the items above |
| 4 | GC→MGC 15m/30m translation question | Audit explicitly said no honest statement exists yet for wider apertures | 3 | 4 | Real question, but not until the 5m payoff-compression path is resolved |

## Recommended Execution Order

1. `MNQ NYSE_CLOSE RR1.0 role audit`
2. `Prior-day Pathway-B bridge execution triage`
3. `Cross-asset chronology spec`
4. `GC→MGC 15m/30m translation question`

Reasoning:

- `PR48` has now been executed honestly: the frozen-rule replay rescued `MGC` only and left `MES` below zero, so it no longer belongs in the open queue.
- `GC -> MGC` 5-minute payoff-compression has now been actioned honestly. The diagnostic says the rescue, if any, is a narrow MGC exit-shape question rather than a broad proxy revival.
- `L1` has now been run exactly as frozen and closed `KILL`; it should not remain in the open queue.
- `NYSE_CLOSE` filter follow-through has now been run to closure. The exact native `ORB_G8` prereg is killed, so the remaining honest question is role: standalone/allocator versus no remaining opportunity.
- The prior-day branch is no longer missing locked next-step artifacts. Multiple exact bridge hypotheses are now saved, so the open problem has shifted from "write a prereg" to "triage and execute one already-locked path."

## PR-Ready Tasks Created Now

This session converts the open ranked items into explicit stage files:

1. `docs/runtime/stages/pr48-mes-mgc-sizer-rule-backtest.md` — now executed via `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
2. `docs/runtime/stages/mgc-5m-payoff-compression-audit.md` — now executed via `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md`
3. `docs/runtime/stages/l1-europe-flow-pre-break-context-scan.md` — now executed via `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md`
4. `docs/runtime/stages/mnq-nyse-close-rr10-followup.md`
5. `docs/runtime/stages/mnq-nyse-close-orbg8-rr10-prereg.md`
6. `docs/runtime/stages/mnq-nyse-close-rr10-role-audit.md`
7. `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md`
8. `docs/runtime/stages/cross-asset-session-chronology-spec.md`
9. `docs/runtime/stages/gc-mgc-15m-30m-translation-question.md`

## Not Put Into Action Now

- Broad `GC -> MGC` reopening
- Broad prior-day rescan
- Cross-asset chronology work before a narrow chronology spec exists
- Any resurrection of PR #50 / #51 “candidate-ready” framing
- Any retuning of the frozen GARCH R3 shadow policy

## Bottom Line

The recent PR burst did produce real follow-through, but it also left a smaller set of clean next moves. After the 2026-04-23 PR48 replay split, the explicit MGC payoff-compression closure, the L1 pre-break scan `KILL`, and the NYSE_CLOSE follow-up closure to `CONTINUE`, the highest-ROI open path is no longer “do more audits in general”; it is:

1. resolve the `MNQ NYSE_CLOSE RR1.0` branch at the correct role layer after the `ORB_G8` kill,
2. triage and execute one already-locked prior-day bridge path instead of reopening that branch broadly,
3. write the chronology spec before any cross-asset timing scan.
