# O-SR Debt Frame Audit — Design

**Date:** 2026-04-25
**Mode:** Design (4-turn flow per `.claude/skills/design`)
**Worktree:** `C:\Users\joshd\canompx3-osr-audit`
**Branch:** `research/osr-shiryaev-roberts-audit` (anchored to `origin/main` `73329cd1`)
**Stage classification (when implemented):** RESEARCH — read-only audit, doc-only write.

## What this design covers

Whether the HANDOFF "Next Steps - Active" line "O-SR debt - `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous." corresponds to actual missing canonical work, or is stale framing.

The design produces a single audit verdict doc plus the staging artefacts needed to run it under institutional rigor.

## Why this matters

Three patterns from project history that this audit must avoid repeating:

1. Misframed-debt churn. A HANDOFF line that says "X is wrong" without a decision-ledger entry has caused multiple sessions to chase already-closed work. The fix pattern in this project is: audit, then either close in the ledger or escalate as a real debt with citations.
2. Source-of-truth chain violation. The literature passage's claim "Implemented 2026-04-10 as: ..." is itself a derived layer; canonical proof must trace through to the actual file, the canonical consumer, and the test that locks the intent. The audit must not accept the literature passage's claim at face value.
3. Confusing "monitor" purposes. The intraday performance-alert monitor and the binding Criterion 12 post-deployment drift gate live at different cadences, scopes, and consequences. Past sessions have conflated them. The audit must explicitly separate the two.

## Authority chain

- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` -- literature anchor (CUSUM Eq 3, original SR Eq 10 / 11, score-based SR Eq 13-14, score function Eq 17-18).
- `docs/institutional/pre_registered_criteria.md` Criterion 12 -- binding rule for post-deployment Shiryaev-Roberts drift monitoring.
- `trading_app/live/cusum_monitor.py` -- intraday CUSUM alert.
- `trading_app/live/sr_monitor.py` -- score-based Shiryaev-Roberts core.
- `trading_app/sr_monitor.py` -- Criterion 12 deployed-lane runner.
- `trading_app/live/performance_monitor.py` -- intraday CUSUM consumer.
- `tests/test_trading_app/test_cusum_monitor.py`, `tests/test_trading_app/test_sr_monitor.py` -- intent locks.
- `scripts/tools/refresh_control_state.py`, `scripts/tools/project_pulse.py`, `trading_app/lifecycle_state.py` -- downstream wiring.
- `docs/runtime/decision-ledger.md` -- silent on this question (the silence is itself evidence under the project's "decision-ledger is canonical" rule from the `runtime-shell-unification` decision).
- `HANDOFF.md` "Next Steps - Active" line 32 -- the audited claim.

## Turn 1 -- ORIENT

Already complete in design conversation. Read every file in the authority chain above; no decision-ledger entry on O-SR / SR / CUSUM exists; downstream wiring of the SR runner is confirmed via `lifecycle_state.expected_state_type` and `project_pulse` registration; intraday CUSUM is reachable only through `performance_monitor`.

## Turn 2 -- DESIGN (multi-take deliberation)

### Take 1 -- failure-mode prevention

The three failure modes above are the design constraints. The audit doc must satisfy four grounding bars before any conclusion is written:

1. Literature anchor: quote verbatim equations from the literature passage.
2. Code anchor: cite exact files, line numbers, behaviour for both monitors.
3. Consumer anchor: show who calls each monitor, at what cadence, with what consequence.
4. Criterion 12 anchor: quote the binding rule and check each parameter against the wired runner.

The verdict then sits in one of three buckets:

- CLOSED-ALREADY -- debt was real, was closed by some prior commit, never logged in the ledger. Action: write a decision-ledger entry, remove the HANDOFF line.
- MISFRAMED -- the named file (cusum) was never the binding monitor; the binding monitor lives elsewhere. Action: rewrite or remove the HANDOFF line; optionally clarify the cusum module-level docstring.
- GENUINELY-OPEN -- the binding monitor doesn't actually satisfy Criterion 12 in some respect (parameter mismatch, no live wiring, missing tests). Action: scope the actual debt and reframe the HANDOFF line precisely or open an action-queue item.

### Take 2 -- bottom-up rigor

The audit's only job is the verdict. The verdict's quality depends on whether anchors are read and grounded BEFORE the verdict section is drafted. Order discipline is the failure-mode protection.

### Take 3 -- challenge

- Too complex? No. Every grounding bar is mandatory under institutional rigor.
- Too simple? Possible: a MISFRAMED outcome may feel like a docs-only finding. Rebuttal: even a docs-only verdict counts -- it removes recurring distraction and prevents future sessions from reopening the question.
- Right ordering? Yes. Anchors first, verdict second. Reverse ordering is bias-prone.

### Take 4 -- pressure-test

- Primary sources only -- HANDOFF and the literature passage are themselves audit subjects, not authorities.
- No backtest / no canonical query / no DB-write side effect.
- Smallest blast radius: one new audit doc plus the stage doc. No canonical-surface edits at this stage.
- Rework rule: if the isolated-context evidence-auditor refutes any anchor, the audit is rewritten, not patched.

### Three approaches

- Approach 1 -- single verdict doc, single PR (audit + decision-ledger + HANDOFF edit + optional docstring). Risk: if verdict is GENUINELY-OPEN, scope must widen mid-flight.
- Approach 2 -- two-stage: audit doc first, action stage second (recommended).
- Approach 3 -- audit doc only, no canonical-surface change. Risk: next session re-runs the same audit because the canonical surfaces still mislead.

Recommendation: Approach 2. The user's "no ad-hoc, no bias, planned and iterated" instruction maps directly to this discipline. Stage 1 ships only the verdict; Stage 2 (separate design pass) takes the verdict's recommendation.

## Turn 3 -- DETAIL

### Files this design's Stage 1 will create

- `docs/audit/results/2026-04-25-osr-debt-frame-audit.md` -- the single new file in this stage.
- `docs/runtime/stages/osr-debt-frame-audit.md` -- RESEARCH-mode stage doc with scope_lock and acceptance criteria.

### Files this design's Stage 1 will read (no writes)

- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`
- `docs/institutional/pre_registered_criteria.md` (Criterion 12 region only)
- `trading_app/live/cusum_monitor.py`
- `trading_app/live/sr_monitor.py`
- `trading_app/sr_monitor.py`
- `trading_app/live/performance_monitor.py`
- `tests/test_trading_app/test_cusum_monitor.py`
- `tests/test_trading_app/test_sr_monitor.py`
- `scripts/tools/refresh_control_state.py`, `scripts/tools/project_pulse.py`, `trading_app/lifecycle_state.py`
- `docs/runtime/decision-ledger.md`
- `HANDOFF.md`

### Order of work inside the audit doc

1. Section: Scope. State exactly what is being audited. Out-of-scope: any change to monitor implementations themselves, any backtest, any canonical-layer query, any retrospective audit of whether ARL approximately 60 days is the right number for our trade cadence.
2. Section: Literature anchor. Quote the four equations verbatim. Note any prose claims the verdict relies on.
3. Section: Code anchor -- CUSUM. Cite the file and trace its update step against Eq 3. Note where the implementation differs from textbook CUSUM and whether differences are faithful to its self-described intent (intraday alert).
4. Section: Code anchor -- SR. Cite the SR file and trace its update step against Eq 13-14, score function against Eq 17-18, coefficients against Eq 18. Cite the ARL calibration routine.
5. Section: Consumer anchor. Map every production caller of each monitor with file and line. Confirm the cusum file is reachable only via `performance_monitor`. Confirm the SR file is reachable via the deployed-lane runner and via `lifecycle_state` plus `project_pulse`. Capture whether the SR runner is actually invoked by anything live.
6. Section: Criterion 12 anchor. Quote the four parameters verbatim and audit each one against the wired SR runner: pre-change-window (50-100 trades), score function (Eq 17-18), threshold calibrated to ARL approximately 60 days, alarm action ("suspended"). Each parameter receives verified / refuted / partial with code citation.
7. Section: HANDOFF claim audit. Quote the HANDOFF line verbatim. Compare against the four anchors. State the verdict (CLOSED-ALREADY / MISFRAMED / GENUINELY-OPEN) with specific supporting evidence and specific evidence that would refute it.
8. Section: Action recommendation. One paragraph per intended downstream action, cleanly separated from the verdict. The verdict drives the action shape; the action shape does not drive the verdict.
9. Section: Limitations. Honest list of what the audit does NOT cover (ARL calibration empirical adequacy, parameter-tuning quality, integration smoke-test, any data-state question).
10. Section: References. Every file-and-line citation from steps 2-7, plus the literature passage and the Criterion 12 reference.

### Validation checks before commit

- All section headings present per claim-hygiene hook regex (`Scope`, `Verdict`, `Outputs` or `Reproduction`, `Limitations`).
- Every claim has a `file:line` citation.
- No claim depends on memory or training data instead of the read sources.
- Drift check passes (idempotent; expect no changes).
- Run the targeted tests on both monitors to confirm doc claims about test-locked intent are accurate at commit time.
- Stage classification: RESEARCH (read-only audit + doc-only write).

### No tests created or modified in Stage 1

The audit consumes existing tests as anchors only.

### No drift-check additions in Stage 1

Drift checks are an action question for Stage 2 if the verdict warrants one.

## Turn 4 -- VALIDATE

### Failure modes

- Verdict-first bias. Mitigation: anchor sections drafted before the HANDOFF claim audit section.
- Stale source. Mitigation: branch anchored to `origin/main` `73329cd1`; cited line numbers reproducible against that exact tree; commit hash recorded at the top of the audit doc.
- Single-context bias. Mitigation: dispatch evidence-auditor in isolated context with the four-anchor structure and the verdict. Treat any REFUTED finding as a blocker.
- Scope creep. Mitigation: scope_lock Stage 1 to exactly the new audit doc plus the stage doc.

### What proves correctness

- Each of the four anchor sections cites a primary source for every claim.
- Isolated-context auditor confirms each anchor's grounding.
- Verdict classification is mechanically derivable from the anchor sections -- a reader who reads only anchors arrives at the same verdict without reading the verdict section first.
- Limitations section truthfully lists every question the audit does NOT answer.

### Rollback plan

Stage 1 ships one new audit doc on a branch. Rollback = abandon the branch, drop the worktree. No canonical surface touched.

### Guardian prompts needed

None at Stage 1 (no entry-model change, no pipeline-data change, no schema change).

### Adversarial-audit gate

Mandatory before Stage 1 commit. Dispatch `evidence-auditor` with explicit anchor-by-anchor verification scope. Treat output the same way the PR #105 audit was treated: any REFUTED finding blocks merge until rewritten. Per `feedback_canonical_value_unit_verification.md`, do not trust same-context self-review to catch unit / interpretation errors.

## Acceptance criteria for Stage 1

- Audit doc exists at `docs/audit/results/2026-04-25-osr-debt-frame-audit.md` with the ten sections listed in Turn 3.
- Every factual claim has a `file:line` citation.
- Isolated-context evidence-auditor returns VERIFIED on each of the four anchors and on the verdict classification.
- Drift check passes.
- Pre-commit gates pass on commit.
- Stage doc deletion happens only after the audit doc is committed and the auditor has cleared it.
