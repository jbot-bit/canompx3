---
status: design-pending-approval
author: Claude Code
date: 2026-06-05
topic: C11 path-to-live Stage 1 (close bracket-parity audit) + Stage 2 (pre-register cap_x0.80)
canonical_commit: 9429c540
scope: design-only — no code, no main-tree mutation, no prop_profiles.py, no arming
---

# C11 → Live: Stage 1 + Stage 2 Design

## Turn 1 — ORIENT

### Purpose
`topstep_50k_mnq_auto` Criterion-11 is a measured capital NO-GO. The settled fix is a
per-lane ORB-size cap at 0.80x the current per-lane cap, which brings strict 90-day
max-drawdown from the uncapped baseline (2,038.84 dollars at commit 9429c540) down to
about 1,594 dollars, clearing the 1,800-dollar Express budget at roughly 97.5 percent
operational survival. Analysis is done. Two peer-independent stages stand between
"analysis done" and "cap wiring may begin":

- **Stage 1** closes the still-open adversarial-audit gate on the bracket-risk-parity
  fix (commit 9b3fc530), which touches the live order path and therefore cannot pass to
  any capital wiring un-audited.
- **Stage 2** pre-registers the cap remediation so the cap value and its acceptance
  gates are locked before the survival re-run that judges them — preventing
  tune-to-pass.

Without Stage 1, a CRITICAL live-order-path change ships with only implementer
self-review, which the audit-gate rule explicitly forbids. Without Stage 2, the cap is
chosen and then validated against gates that could be quietly relaxed to fit it.

### Authority and canon consulted
- `docs/audit/results/2026-06-03-bracket-risk-parity-closeout.md` — the closeout whose
  Limitations section records the audit gate as OPEN ("no independent spawned reviewer
  clearance ... must run before any live arming").
- `.claude/rules/adversarial-audit-gate.md` — the gate is non-negotiable for a
  judgment-classified CRITICAL change touching `trading_app/live/`; actor is the
  independent evidence-auditor; artifact fields are fixed.
- `.claude/rules/backtesting-methodology.md` RULE 10 — pre-reg file must exist before
  the run, with numbered hypotheses, economic-theory citations, and exact dimensions.
- `.claude/rules/self-funded-sizing-doctrine.md` — a cap is a tail-risk / margin guard,
  never an earnings ceiling. The remediation is framed as risk truncation, not a cap on
  how much the strategy may earn.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` —
  the risk-first sizing the cap appeals to.

### Files actually read (blast-radius trace, read-only)
- `trading_app/live/session_orchestrator.py` at commit 9b3fc530 — the live order surface
  the audit examines. Two findings that reshape the plan:
  - The bracket stop-distance helper reads the event risk-points value first and returns
    it directly when present; only when it is absent does it fall back to median risk
    times stop-multiplier; and when neither exists it returns the empty value, which the
    caller treats as fail-closed (flatten rather than place a guessed bracket). This is
    the parity fix, and it is the thing the audit must prove is correct under every
    input shape.
  - A live ORB-cap enforcement path ALREADY EXISTS — a cap registry keyed by
    the triple (orb-label, instrument, orb-minutes) is consulted before each entry, and
    an oversized trade whose risk-points meets or exceeds the cap is skipped and
    journaled. This means the eventual cap-wiring stage (Stage 3, out of scope here) is
    not building a new mechanism; it is populating the cap VALUE into the registry that
    feeds this existing gate. Recorded here because it materially shrinks Stage 3 and
    confirms the cap is live-runnable.
- `trading_app/account_survival.py` at 9b3fc530 — the replay/sim side of the same
  parity fix (the other half the audit checks for bidirectional parity).
- `scripts/tools/prereg_front_door.py` — confirmed present; it is the validator the
  Stage 2 yaml must pass.
- A recent hypothesis yaml — confirmed the expected top-level shape (metadata,
  execution, hypotheses, controls).

### Canonical sources that would be touched LATER (not in these two stages)
- `prop_profiles.py` cap value — Stage 3, peer-owned, out of scope.
- The ORB-cap registry that feeds the live cap gate — Stage 3.
None of Stage 1 or Stage 2 touches a canonical source; Stage 1 is read-and-audit, Stage
2 is a new pre-registration document.

---

## Turn 2 — DESIGN (multi-take deliberation)

### Take 1 — what went wrong before in this domain
The audit-gate rule's own proof case: a bracket-submit fix landed with four
mutation-proof passing tests and clean drift, and self-review declared it sound. An
independent auditor then found a CRITICAL multi-event-per-bar race the implementer's
single-event tests never probed. The lesson: passing tests plus self-review is the same
mental model that produced any latent bug. The bracket-parity fix here is the same class
of change (live bracket construction), so the audit must specifically send a bar with
two entry events and trace stop-distance for both — not trust the single-event tests.

A second past failure: a stale memory note once asserted a parity fix was "necessary and
sufficient" and another that the uncapped baseline was the capped figure. The design must
keep the audit grounded in execution output, not prior summaries, and keep the prereg
baseline pinned to the canonical-loader figure (2,038.84 at 9429c540), not a remembered
number.

### Take 2 — design bottom-up from failure prevention
- Stage 1 is structured so the auditor receives only the commit diff, the six
  falsification points, and the cited tests as the baseline — and is told to treat the
  closeout's claims as assertions to disprove. The independent context is the whole
  point; reusing the implementing context would reproduce the blind spot.
- Stage 2 locks the five acceptance gates BEFORE the survival re-run, so the gates can
  never be relaxed to fit the cap. The baseline and budget numbers are written from the
  canonical loader, not memory.

### Take 3 — challenge the shape
- Too complex? No. Stage 1 is one auditor spawn plus a verdict recorded into the existing
  closeout doc. Stage 2 is one new yaml validated by the existing front-door. Neither
  invents tooling.
- Too simple / missing a step? The one real subtlety is that Stage 2's K-budget field
  wants the research-catalog estimate, and that MCP is currently disconnected. For a
  single deterministic configuration (or a configuration plus one robustness sibling) the
  MinBTL bound is not binding, but the field is still recorded honestly once the server
  reconnects; the design notes this rather than fabricating a number.
- Right ordering? Stage 1 and Stage 2 are mutually independent and both independent of
  the peer. Stage 1 is the critical path (it gates Stage 3). They can be done in either
  order or together; the audit is the higher-value one because it is the actual blocker.

### One-way dependency check
Both stages are doc/audit only and read pipeline-and-trading-app code without writing it.
No dependency is reversed.

### Approaches and recommendation
- Approach A: do both stages now in one clean isolated worktree off origin/main.
- Approach B: do Stage 1 only (the critical-path blocker), defer Stage 2 until the
  research-catalog MCP reconnects so its K-budget field is real.
- Approach C: do Stage 2 only (cheap, no auditor budget) and hold Stage 1 for a dedicated
  pass.
Recommendation: **Approach A**, with the Stage 2 K-budget field left as a marked TODO
until the MCP reconnects, OR Approach B if you want the audit verdict in hand before
spending any words on the prereg. Stage 1 carries the higher information value.

---

## Turn 3 — DETAIL (ordered, executable later)

### Stage 1 — close the bracket-parity audit (9b3fc530)
1. Re-confirm the commit is reachable and read its diff for the two touched production
   files (read-only).
2. Re-run the cited baseline tests so the auditor reasons against a known-green state:
   the bracket-orders and naked-position test classes in the session-orchestrator test
   file.
3. Dispatch exactly ONE independent evidence-auditor (separate context) with these six
   falsification points:
   - both live bracket-construction paths route through the single stop-distance helper;
     grep for any third path that bypasses it;
   - when the event risk-points value is present it is used raw, when only median risk is
     present the stop-multiplier is applied exactly once, and when neither is present the
     path fails closed (flatten, not a guessed bracket);
   - the replay side and the live side compute the same effective stop distance
     (bidirectional parity, the point of the fix);
   - no scope leak into exit-record actual-R, journal dollar fallback, or naked-position
     fail-closed behavior;
   - a bar carrying two entry events yields a correct stop distance for BOTH events (the
     canonical multi-event counterexample);
   - tight-stop handling runs before the helper reads the event risk-points value, so the
     value the helper trusts is already the effective post-tight-stop distance.
4. Receive the structured artifact with all required fields (verdict, critical issues
   with file-and-line, silent gaps, unsupported assumptions, missing tests, do-not-touch,
   single highest-priority fix).
5. Record the verdict into the closeout doc's Limitations/Verdict section (docs-only
   edit, committed from the isolated worktree).

### Stage 2 — pre-register the cap remediation
1. (When research-catalog MCP reconnects) run the K-budget estimate and record the bound;
   until then, mark the field as pending with the rationale that a single deterministic
   configuration is below the binding threshold.
2. Create the pre-registration document at
   `docs/audit/hypotheses/2026-06-05-c11-cap-x080-remediation-v1.yaml` containing:
   - metadata (canonical commit 9429c540, profile, instrument, date);
   - numbered hypothesis: a per-lane ORB-size cap at 0.80x current per-lane cap reduces
     strict 90-day max-drawdown below the 1,800-dollar Express budget while retaining at
     least 85 percent of edge — cited to the Carver volatility-targeting extract and
     framed per the self-funded-sizing doctrine as tail-risk truncation, not an earnings
     cap;
   - dimensions: a single deterministic configuration (the 0.80 cap), optionally with the
     0.75 cap co-registered as a robustness sibling;
   - acceptance gates, locked now: operational survival at least 70 percent; strict
     observed 90-day max-drawdown at most 1,800 dollars; breach-days equal to zero; edge
     retained at least 85 percent of the uncapped baseline; robustness across
     walk-forward eras and a deflated-Sharpe check;
   - baseline: uncapped drawdown 2,038.84 dollars at 9429c540, from the canonical loader;
   - holdout: 2026 frozen, descriptive only;
   - data sources: canonical profile lane definitions plus the canonical outcomes and
     daily-features layers — never the derived allocation json.
3. Validate the document through the prereg front door in text mode.
4. Commit docs-only from the isolated worktree.

### Test strategy
- Stage 1 produces no code, so the "test" is the auditor verdict plus the re-run of the
  cited baseline tests to anchor it.
- Stage 2 produces a document whose correctness gate is front-door acceptance plus a
  human read that the five acceptance gates and the baseline number match canon.

### Drift-check impact
None expected. No production code, schema, or canonical config changes in either stage.
Stage 2 adds a hypothesis file, which the prereg front door — not drift — validates.

---

## Turn 4 — VALIDATE

### Failure modes and risks
- The auditor returns CONDITIONAL or FAIL: this is a success of the gate, not a failure.
  Findings route into a fix iteration and Stage 3 stays blocked until closed or
  explicitly deferred in writing. Stop condition, not error.
- Editing the peer-owned profile file or the dirty main tree: forbidden; both stages run
  in a clean isolated worktree off origin/main.
- Writing a fabricated K-budget number because the MCP is down: avoided by marking the
  field pending rather than inventing a value.
- Quoting a remembered baseline instead of the canonical figure: avoided by pinning
  2,038.84 at 9429c540 from the loader.

### What proves correctness (behavior, not "it runs")
- Stage 1: an independent-context verdict with file-and-line evidence, plus the
  multi-event-per-bar case explicitly exercised — proving the fix holds under the input
  the implementer's tests did not send.
- Stage 2: front-door acceptance plus a confirmed match between the document's acceptance
  gates and the locked C11 GO criteria, with the baseline traceable to the canonical
  loader.

### Rollback plan
- Stage 1: the only artifact is a verdict recorded into an existing doc; revert the
  docs commit if the verdict needs revision.
- Stage 2: delete or supersede the hypothesis file; nothing downstream consumes it until
  Stage 4 runs.

### Guardian prompts
None required — no entry-model or pipeline-data change. The adversarial-audit gate itself
is the guardian for Stage 1.

---

## Dependencies and stop conditions
- Stage 1 and Stage 2: no peer dependency, no profile-file touch — runnable now in a
  clean isolated worktree off origin/main.
- Stage 3 (cap wiring) blocked by: Stage 1 verdict PASS, Stage 2 committed, and the peer
  releasing the profile file. Note Stage 3 is smaller than previously framed because the
  live cap-enforcement path already exists; Stage 3 populates the cap value, it does not
  build the gate.
- research-catalog MCP disconnected: defers only the Stage 2 K-budget field.
- Hard stops: any CONDITIONAL/FAIL verdict halts before wiring; any need to touch the
  profile file or dirty main tree halts; any step that would arm C11 to live halts for a
  separate operator GO.
