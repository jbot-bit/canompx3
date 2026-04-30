# Discipline Checklist — `sequential_thinking` MCP Eval

**Date:** 2026-05-01
**Aggregate rule:** PASS if all 4 gates pass. FAIL if any one gate fails.
**Inputs:** `tunnel-vision-rubric.md` (Incidents A, B, C with their eval-metric fields filled), eval transcript.

> **Note on `memory/<file>.md` references:** throughout this doc, `memory/<file>.md` paths resolve to the auto-memory directory at `C:/Users/joshd/.claude/projects/C--Users-joshd-canompx3/memory/` (per the `claudeMd` "auto memory" section), not the project repo.

---

## Gate 1 — Cost gate

**Criterion:** `tokens_with_seq_thinking / tokens_baseline ≤ 1.5` AND `marginal_cost_pct ≤ +50%`.

**Rationale:** seq-thinking is allowed to cost up to 50% more tokens but the eval fails if it exceeds that without commensurate quality gain.

**Computation:** for each incident A / B / C:
- `marginal_cost_pct = (tokens_with_seq_thinking - tokens_baseline) / tokens_baseline * 100`
- Take the **mean across incidents**.

**Pass condition:** mean `marginal_cost_pct ≤ 50.0`.

**Verifier:** read transcript token counts (input + output, both directions) for baseline and with-seq-thinking runs.

---

## Gate 2 — Quality gate

**Criterion:** ≥2 of 3 incidents must show `alternate_considered=TRUE` AND ≥1 of 3 must show `revision_made=TRUE`.

**Rationale:** seq-thinking exists to surface alternates and enable mid-flow revision. If neither happens, the tool added cost without the load-bearing benefit.

**Pass condition:** `count(alternate_considered=TRUE) ≥ 2` AND `count(revision_made=TRUE) ≥ 1`.

**Verifier:**
- `alternate_considered` = TRUE iff the seq-thinking trace contains at least one `branchFromThought` or explicit "alternative frame" thought BEFORE the final conclusion.
- `revision_made` = TRUE iff the seq-thinking trace contains at least one thought with `isRevision=true` that materially changed the final conclusion (cosmetic re-wordings don't count).

---

## Gate 3 — Doctrine-no-regression gate

**Criterion:** verify (manually, by reviewing the eval transcript) that seq-thinking output does NOT bypass the 2-Pass discovery/implementation gate. The eval transcript MUST show the agent still ran the 2-Pass cycle.

**Rationale:** institutional-rigor rule 8 binds "done" to four external artefacts (tests pass + dead code swept + drift check passes + self-review passed). Seq-thinking is internal-only and cannot satisfy any of the four. The risk is an agent substituting thought-count for evidence — exactly what the Design Proposal Gate forbids ("Performative self-review … is worse than no self-review").

**Pass conditions (ALL must hold):**
1. Discovery phase ran: transcript shows file reads / blast-radius mapping / canonical-source identification BEFORE the seq-thinking trace.
2. Implementation phase ran: transcript shows `python pipeline/check_drift.py` (or equivalent for the task), tests executed with output shown, `grep -r` for dead-code sweep where applicable.
3. Seq-thinking trace was used as **augmentation** of Discovery articulation, never as substitute for any of the four `done` artefacts.
4. No claim of "done" appears in the transcript citing seq-thinking output as evidence (must cite tests + grep + drift + self-review).

**Failure example:** transcript ends with "Done — verified via 12-thought sequential reasoning chain" with no test output → FAIL.

**Verifier:** human reviewer (the user, or a second Claude session running the `code-review` or `superpowers:code-reviewer` skill) inspects the eval transcript against the four sub-conditions above.

---

## Gate 4 — Falsifiability gate

**Criterion:** ≥2 of 3 verdicts must be `falsifiable_verdict=TRUE`.

**Rationale:** the project's research discipline requires testable claims with explicit failure conditions. A seq-thinking trace that ends with a hedged, untestable conclusion is worse than no trace.

**Pass condition:** `count(falsifiable_verdict=TRUE) ≥ 2`.

**Verifier:** `falsifiable_verdict` = TRUE iff the final answer for that incident states (a) the testable hypothesis, (b) the data / query that would settle it, and (c) the result that would falsify it.

---

## Aggregate

| Gate | Status (TBD by future eval) |
|---|---|
| 1. Cost gate | TBD |
| 2. Quality gate | TBD |
| 3. Doctrine-no-regression gate | TBD |
| 4. Falsifiability gate | TBD |
| **Aggregate (PASS iff all 4 PASS)** | **TBD** |

---

## On FAIL

If aggregate is FAIL:
1. Do NOT merge `mcp-config-patch.md` into `.mcp.json`.
2. Write `memory/feedback_seq_thinking_eval_<date>_fail.md` documenting which gate(s) failed and the root cause (mis-use, tool inadequacy, or doctrine drift).
3. If failure is mis-use (agent cited seq-thinking as `done` evidence): the tool may still be added, but a path-scoped rule under `.claude/rules/seq-thinking-discipline.md` is required first. Reopen the eval after the rule lands.
4. If failure is tool inadequacy (cost gate breach with no quality gain): mark this server as NO-GO in project memory and close the evaluation.

## On PASS

If aggregate is PASS:
1. Apply the patch from `mcp-config-patch.md` to `.mcp.json`.
2. Run `claude mcp get sequential-thinking` to confirm project-scope (no local-scope shadow per `feedback_mcp_local_scope_shadows_project_scope.md`).
3. Restart Claude Code so the stdio server picks up.
4. Add a 1-line entry under `MEMORY.md` § "Tooling" linking back to this eval directory.
