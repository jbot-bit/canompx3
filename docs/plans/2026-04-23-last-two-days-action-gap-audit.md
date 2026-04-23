# 2026-04-23 Last-Two-Days Action-Gap Audit

## Purpose

Close the "good work happened but never got properly actioned" failure mode for
the last 48 hours of repo activity.

Scope:

- review recent finding-bearing commits from `2026-04-21` through `2026-04-23`
- verify whether each meaningful thread now has a durable operational home
- distinguish:
  - `CLOSED_AND_ACTIONED`
  - `SAVED_BUT_INTENTIONALLY_OPEN`
  - `OUT_OF_SCOPE_FOR_THIS_CLOSEOUT`

This is a closeout audit, not a fresh discovery memo.

## Canonical Durable Surfaces Used

- `docs/runtime/decision-ledger.md`
- `docs/plans/2026-04-22-recent-pr-followthrough-queue.md`
- `docs/plans/2026-04-21-post-stale-lock-action-queue.md`
- `HANDOFF.md`
- locked preregs / stage docs / result docs under `docs/audit/` and `docs/runtime/stages/`

## Closed And Actioned

| Thread | Evidence now in repo | Status |
|---|---|---|
| MNQ NYSE_CLOSE RR1.0 governance follow-up | `9630a38e`, `3df72841`, result docs, role-audit stage, decision ledger | `CLOSED_AND_ACTIONED` |
| Exact `MNQ NYSE_CLOSE ORB_G8 RR1.0` prereg | `3df72841`, `docs/audit/results/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.md` | `CLOSED_AND_ACTIONED` |
| L1 EUROPE_FLOW pre-break-context scan | `1af9e25a`, restored prereg, result doc, stage outcome | `CLOSED_AND_ACTIONED` |
| PR48 frozen MES/MGC sizer replay | `439ccfd5`, locked prereg, result doc, decision ledger split | `CLOSED_AND_ACTIONED` |
| MGC 5m payoff-compression diagnostic | `505f65e6`, result doc, queue + ledger language | `CLOSED_AND_ACTIONED` |
| PP-167 ORB-cap registry fix | `7c249b91`, design doc, ralph surfaces, production/tests | `CLOSED_AND_ACTIONED` |
| Handoff baton compaction | `d9366c66`, archived snapshots, tool, tests | `CLOSED_AND_ACTIONED` |
| Commit-path hardening | `34d6db39`, hook no longer hangs on advisory M2.5 | `CLOSED_AND_ACTIONED` |
| GARCH warmup constants centralization | `d8eb7ada`, producer constants now own downstream warmup contract | `CLOSED_AND_ACTIONED` |

## Saved But Intentionally Open

These are not lost. They are now explicit bounded next-step surfaces.

| Thread | Durable home | Why still open |
|---|---|---|
| MNQ NYSE_CLOSE RR1.0 role question | `docs/runtime/stages/mnq-nyse-close-rr10-role-audit.md` | `ORB_G8` was killed, but the broad family still requires role-layer resolution |
| Prior-day Pathway-B / MNQ prior-day bridge execution | `f11cfacb` locked hypotheses + `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md` | locks are written, but the confirm-or-kill executions are not all run |
| Cross-asset earlier-session chronology | `docs/runtime/stages/cross-asset-session-chronology-spec.md` | queue item now has a bounded docs-first stage, but the chronology contract itself is not written yet |
| GC->MGC 15m/30m question | `docs/runtime/stages/gc-mgc-15m-30m-translation-question.md` | still dependent on the already-closed 5m path; wider-aperture claim not yet resolved |

## Recent Commit Findings Now Properly Framed

### `phase4_candidate_precheck` / `566` claim

- The historical precheck script is real and reports `n_preholdout_outcomes`.
- For `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-near-pivot-50-avoid-v1.yaml`,
  the precheck resolves exactly one accepted raw combo with `566` pre-holdout
  outcomes.
- That count is **not** a survival verdict. It is only a scope-resolution /
  raw-availability fact.
- Repo truth later shows that exact `F3_NEAR_PIVOT_50` bridge failed, while the
  broader `PD_*` family was the surviving route.

Conclusion:

- the count is `SUPPORTED`
- the interpretation "566 proves the edge survived" is `UNSUPPORTED`

## Last-Two-Days Gaps That Were Real And Are Now Closed

- Finished research and runtime decisions were sitting in uncommitted files.
  Those slices are now committed.
- The decision ledger still implied "`ORB_G8` next" even after `ORB_G8`
  had been executed. This audit removes that contradiction.
- The post-stale queue still left the L1 path looking open. This audit marks
  the pre-break route as closed `KILL`.

## Out Of Scope For This Closeout

These files remain dirty but are not part of the trading-research closeout:

- `.codex/config.toml`
- `REPO_MAP.md`
- `GEMINI.md`
- `gemini-cli/`

They may matter operationally, but they do not represent stranded edge /
strategy-research findings from the last two days.

## Bottom Line

For the last two days, the meaningful trading-research work is no longer living
only in branch history or chat interpretation.

It is now one of:

1. closed and actioned in durable repo surfaces, or
2. saved as an explicit bounded next-step stage.

The remaining open work is therefore honest open work, not lost work.
