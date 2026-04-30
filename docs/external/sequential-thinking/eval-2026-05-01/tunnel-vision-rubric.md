# Tunnel-Vision Rubric — `sequential_thinking` MCP eval

**Date:** 2026-05-01
**Purpose:** replay three real project incidents with seq-thinking enabled and score whether the tool would have surfaced the alternate framing in time. Each incident has eval-metric placeholders to be filled by the future eval session.

**Source-of-truth note:** memory files named in the task spec (`memory/phase_2_9_xmes_reaudit_tunnel_lesson.md`, `memory/feedback_token_efficient_audit_loop.md`, `memory/feedback_d4_aistudio_audit_lessons.md`) are not present at those paths in the current worktree (`memory/` contains 10 dated files, none matching). The task prompt provided the authoritative gist for each incident; that gist is reproduced verbatim below in italics, and treated as the canonical incident description for this rubric. The future eval session SHOULD attempt to locate the originals (e.g. via `MEMORY.md` index references) before rerunning the rubric, and amend if the originals carry additional facts.

---

## Common eval-metric placeholders (apply to every incident)

| Field | Type | Definition |
|---|---|---|
| `alternate_considered` | BOOL | Did seq-thinking actually surface the alternate framing in the trace? |
| `revision_made` | BOOL | Did the agent revise an earlier conclusion mid-flow (isRevision=true)? |
| `falsifiable_verdict` | BOOL | Was the final answer testable, with explicit failure conditions? |
| `tokens_baseline` | int | Total tokens consumed without seq-thinking. |
| `tokens_with_seq_thinking` | int | Total tokens consumed with seq-thinking enabled. |
| `marginal_cost_pct` | float | `(tokens_with_seq_thinking - tokens_baseline) / tokens_baseline * 100` |

---

## Incident A — Phase 2-9 X_MES_ATR60 re-audit (2026-04-20)

**Source (gist as supplied by task prompt, treated as canonical):**

> *Ran a full session (pre-reg → script → refactor → result) on block-bootstrap null for 4 phase-2-9 framing-audit lanes. Only at end-of-session, while scoping a portfolio-overlap follow-up, queried the live allocator and found NONE of the 4 lanes were in the live 6 DEPLOY set. The re-audit was methodologically correct, decision-irrelevant. Tunnel-vision symptom: assumed "previously promoted" → "currently deployed" without checking.*

### Decision flow it represents

1. Receive an audit request referencing "previously promoted" lanes.
2. Build pre-reg + script for the audit.
3. Execute, refactor, produce result.

### Without-seq-thinking baseline

The agent went straight from step 1 → step 2 with no branching on the assumption "previously promoted == currently deployed." Tunnel began at step 1 (frame acceptance). Cost: a full session of pre-reg + script + refactor + result-write that turned out decision-irrelevant. Late-pivot at end-of-session when the portfolio-overlap follow-up incidentally exposed the gap.

### With-seq-thinking expected behavior

Seq-thinking should branch at **step 1 → step 2 transition** (the frame-acceptance step). The agent should generate at least one `branchFromThought` exploring "is this lane actually in the live DEPLOY set right now?" before committing to pre-reg. Concretely:

- Thought N: "Audit request received for 4 phase-2-9 lanes."
- Thought N+1 (branch): "Before scoping pre-reg, check live allocator (`lane_allocation.json`) — are these 4 lanes deployed, research-provisional, or stale?"
- Thought N+2 (revision if check fails): "Lanes not deployed → audit is academic-priority not real-capital-priority → reframe scope or defer."

**Verdict: seq-thinking SHOULD have helped.** This is a textbook case for `branchFromThought` because the missing operation is *cheap* (one MCP call) and *binary* (deployed or not). The cost of the branch is one extra thought + one query; the saved cost is a full audit session.

### Eval-metric placeholders for Incident A

| Field | Value |
|---|---|
| `alternate_considered` | TBD |
| `revision_made` | TBD |
| `falsifiable_verdict` | TBD |
| `tokens_baseline` | TBD |
| `tokens_with_seq_thinking` | TBD |
| `marginal_cost_pct` | TBD |

---

## Incident B — Token-efficient audit loop (2026-04-28)

**Source (gist as supplied by task prompt, treated as canonical):**

> *Ran 3 nested audits on the closed `scratch-eod-mtm-canonical-fix` plan, each re-deriving the same sign-flip count and t-stats already in `docs/audit/results/`. Tunnel-vision symptom: kept re-running SQL after the canonical result doc already had the answer; 1 verification query would have sufficed.*

### Decision flow it represents

1. Receive an audit request on a closed plan.
2. Decide the verification depth (re-derive vs. spot-check vs. re-read canonical doc).
3. Execute the chosen depth.

### Without-seq-thinking baseline

Agent skipped step 2 entirely — went 1 → 3 with maximum-depth re-derivation, then nested another two audits on top. Tunnel began at step 1 (no scoping pause). Cost: ~3× redundant SQL load and ~3× wall-clock vs. a single canonical-doc read + one verification query.

### With-seq-thinking expected behavior

Seq-thinking would surface the alternate at **step 1 → step 2**:

- Thought N: "Audit request on closed plan."
- Thought N+1 (branch): "Before re-deriving — does `docs/audit/results/` already contain the sign-flip count and t-stats? If yes, the verification depth is 'spot-check', not 're-derive'."
- Thought N+2 (revision): "Read canonical result doc first; only run SQL on load-bearing claims with no verification ref."

**Honest caveat:** the issue here was partly *discipline* (read the canonical doc first), not pure *branching*. An agent without seq-thinking but with a clear "audit-discipline" rule would solve this equally well. Seq-thinking helps because it forces the depth-selection step to be made explicit; it does NOT add new information. **Verdict: seq-thinking PROBABLY would have helped, but the marginal benefit is smaller than Incident A**, because the missing operation is a habit, not a fact-check.

### Eval-metric placeholders for Incident B

| Field | Value |
|---|---|
| `alternate_considered` | TBD |
| `revision_made` | TBD |
| `falsifiable_verdict` | TBD |
| `tokens_baseline` | TBD |
| `tokens_with_seq_thinking` | TBD |
| `marginal_cost_pct` | TBD |

---

## Incident C — D4 AIStudio "fragility" framing (2026-04-28)

**Source (gist as supplied by task prompt, treated as canonical):**

> *External reviewer called a signal "fragile" because Q5-rest dropped from +0.260R at 5m aperture to -0.053R at 30m. Project anchored on Carver Ch12 + Chan Ch7 reading without first asking "is the implied holding-period horizon different at each aperture? does the mechanism's stated horizon match?" Tunnel-vision symptom: accepted the fragility framing without testing the alternative (mechanism-horizon match) until literature grounding pushed back.*

### Decision flow it represents

1. Receive an external "fragility" claim with aperture-decay numbers.
2. Choose an interpretive frame (fragility-of-edge vs. mechanism-horizon-mismatch vs. selection-bias artefact vs. …).
3. Anchor on the chosen frame and gather supporting literature.

### Without-seq-thinking baseline

Agent skipped step 2 — went 1 → 3 by accepting the reviewer's "fragility" frame as given, then citing Carver Ch12 + Chan Ch7 to support it. Tunnel began at step 1→3 collapse (frame inheritance). Cost: prolonged anchoring on a frame that literature grounding eventually had to push back against; wrong-conclusion risk on what the signal's mechanism actually is.

### With-seq-thinking expected behavior

Seq-thinking should branch at **step 2** (frame selection), enumerating multiple competing frames before anchoring:

- Thought N: "Reviewer claims signal is fragile because Q5-rest decays across aperture."
- Thought N+1 (branch 1): "Frame: fragility-of-edge — predicts decay at all apertures."
- Thought N+2 (branch 2): "Frame: mechanism-horizon mismatch — implied holding period differs by aperture; the 5m aperture matches mechanism horizon and 30m doesn't, so 30m result is mis-specified, not fragile."
- Thought N+3 (branch 3): "Frame: selection-bias artefact in Q5-rest construction."
- Thought N+4 (decision): "Test branch 2 first because it's cheapest and most falsifiable: what is the mechanism's stated horizon? Does it match 5m or 30m?"

**Honest caveat:** part of this was *missing knowledge* (the literature pushback came from `docs/institutional/literature/`, not from the agent's reasoning). Seq-thinking cannot manufacture missing knowledge — but it CAN keep the alternate frame alive long enough for literature grounding to land. **Verdict: seq-thinking SHOULD have helped, specifically at step 2 (frame enumeration before anchoring).**

### Eval-metric placeholders for Incident C

| Field | Value |
|---|---|
| `alternate_considered` | TBD |
| `revision_made` | TBD |
| `falsifiable_verdict` | TBD |
| `tokens_baseline` | TBD |
| `tokens_with_seq_thinking` | TBD |
| `marginal_cost_pct` | TBD |

---

## Aggregate scoring (to be filled by future eval)

- Incidents where `alternate_considered=TRUE`: __ / 3 (gate: ≥2)
- Incidents where `revision_made=TRUE`: __ / 3 (gate: ≥1)
- Incidents where `falsifiable_verdict=TRUE`: __ / 3 (gate: ≥2)
- Mean `marginal_cost_pct`: __% (gate: ≤ +50%)

See `discipline-checklist.md` for pass/fail aggregation.
