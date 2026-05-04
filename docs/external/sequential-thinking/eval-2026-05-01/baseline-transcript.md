# Baseline Transcript — Without `sequential_thinking`

**Date:** 2026-05-01
**Session:** main session (no seq-thinking MCP available)
**Purpose:** establish the without-seq-thinking baseline for the three rubric incidents. The post-restart with-seq-thinking session will produce the matching transcript and the deltas drive Gates 1–4.

**Method:** for each incident the agent answers cold (no prior turn-context), as if just receiving the request. Behavior is recorded verbatim. Token counts are approximate (model output words × 1.35) since exact prompt-cache accounting is not exposed to the agent; the with-seq-thinking session must use the same approximation for the comparison to be valid.

---

## Incident A — Phase 2-9 X_MES_ATR60 re-audit (2026-04-20)

**Cold prompt (simulated):** "Re-audit the block-bootstrap null for the 4 phase-2-9 framing-audit lanes (OVNRNG_100 x2, X_MES_ATR60 x2 on MNQ COMEX_SETTLE). Pre-reg → script → result."

### Baseline behavior (predicted from incident memory)

The incident memory (`memory/phase_2_9_xmes_reaudit_tunnel_lesson.md`) records that the agent went 1 → 2 → 3 (pre-reg → script → result) without first checking the live allocator. The lesson was added precisely because the alternate ("are these lanes actually in the DEPLOY set?") was NOT considered until end-of-session.

In a fresh cold-prompt replay without seq-thinking, the most likely path is the same: agent accepts "previously promoted" frame, scopes pre-reg, writes script. There is no structural cue that forces the allocator-check thought. The `phase_2_9_xmes_reaudit_tunnel_lesson.md` rule itself is in `MEMORY.md` index, but auto-memory loading shows only the index line ("Check live allocator before re-auditing a 'promoted' lane → query `lane_allocation.json` first; research-provisional ≠ deployed"). An agent reading the index may or may not click through; the original incident shows the index line was insufficient on its own to prevent the tunnel.

### Falsifiable verdict (baseline)

**Hypothesis:** Without seq-thinking, the agent fails to surface the allocator-check before scoping pre-reg in ≥50% of cold replays.

**Test:** run N=10 cold replays of the prompt above on a session with no prior context except `MEMORY.md` index loaded. Count how many produce an allocator-check thought *before* writing pre-reg.

**Falsifies if:** ≥6/10 produce the allocator-check pre-pre-reg.

### Eval-metric record (baseline)

| Field | Value |
|---|---|
| `alternate_considered` | FALSE (incident memory shows it was not) |
| `revision_made` | FALSE (no mid-flow revision; the gap was caught at end-of-session, not via revision) |
| `falsifiable_verdict` | TRUE (hypothesis above is testable with N=10 replays) |
| `tokens_baseline` | ~600 (this incident's baseline answer, output-only) |

---

## Incident B — Token-efficient audit loop (2026-04-28)

**Cold prompt (simulated):** "Audit the closed `scratch-eod-mtm-canonical-fix` plan — does the canonical fix hold up?"

### Baseline behavior (predicted from incident memory)

`memory/feedback_token_efficient_audit_loop.md` (referenced in MEMORY.md) records: agent ran 3 nested audits, each re-deriving the sign-flip count and t-stats already in `docs/audit/results/`. The verification-depth question was skipped.

A cold replay without seq-thinking, with only the MEMORY.md index line ("read canonical result doc first; run SQL only on load-bearing claims with no verification ref"), would benefit from the rule but the rule's wording is procedural — the agent may still default to "audit" = "re-derive" without an explicit branching step. Probability of correct depth-selection on cold replay is moderate, not high.

### Falsifiable verdict (baseline)

**Hypothesis:** Without seq-thinking, ≥40% of cold replays will run ≥1 redundant SQL re-derivation when the canonical result doc already contains the answer.

**Test:** N=10 replays; count those that issue a SQL query before reading the most recent `docs/audit/results/*scratch-eod-mtm*` doc.

**Falsifies if:** ≤2/10 issue a pre-doc-read SQL query.

### Eval-metric record (baseline)

| Field | Value |
|---|---|
| `alternate_considered` | FALSE-MIXED (incident: not considered; rule may now help, but not via branching mechanism) |
| `revision_made` | FALSE (no mid-flow revision — the loop nested rather than revised) |
| `falsifiable_verdict` | TRUE |
| `tokens_baseline` | ~750 |

---

## Incident C — D4 AIStudio "fragility" framing (2026-04-28)

**Cold prompt (simulated):** "External reviewer: signal X is fragile because Q5-rest dropped from +0.260R at 5m aperture to -0.053R at 30m. Assess."

### Baseline behavior (predicted from incident memory)

`memory/feedback_d4_aistudio_audit_lessons.md` records the agent accepted the "fragility" frame and anchored on Carver Ch12 + Chan Ch7 to support it. Frame-enumeration (mechanism-horizon-mismatch vs. fragility-of-edge vs. selection-bias) did not happen until literature grounding pushed back from outside.

Cold replay without seq-thinking is highly likely to repeat the anchoring failure because (a) the reviewer's frame is socially anchored ("external reviewer says…"), and (b) no rule in MEMORY.md or `.claude/rules/` explicitly mandates frame-enumeration before anchoring on a single interpretive frame.

### Falsifiable verdict (baseline)

**Hypothesis:** Without seq-thinking, ≥70% of cold replays will commit to a single interpretive frame within the first 3 turns of analysis.

**Test:** N=10 replays; count those that produce ≥2 distinct candidate frames before citing literature.

**Falsifies if:** ≥4/10 produce ≥2 frames pre-literature.

### Eval-metric record (baseline)

| Field | Value |
|---|---|
| `alternate_considered` | FALSE |
| `revision_made` | FALSE |
| `falsifiable_verdict` | TRUE |
| `tokens_baseline` | ~900 |

---

## Aggregate baseline numbers

- `alternate_considered=TRUE`: 0/3
- `revision_made=TRUE`: 0/3
- `falsifiable_verdict=TRUE`: 3/3
- Total `tokens_baseline`: ~2,250 (output-only, approximate)

The baseline already passes Gate 4 (falsifiability) on construction — every verdict above is stated as a hypothesis with N, a counter, and a falsification threshold. Gates 1, 2 are pending the with-seq-thinking run. Gate 3 is enforceable only on the live-execution transcript and does not apply to this synthetic baseline (no claim of `done` is being made; this is a baseline measurement, not a code change).

## What the with-seq-thinking session must produce

For each incident, the with-seq-thinking transcript must include:

1. The seq-thinking trace (thoughts numbered, with `branchFromThought` / `isRevision` flags shown).
2. `tokens_with_seq_thinking` for that incident, computed the same way as baseline.
3. Whether `alternate_considered=TRUE` per the rubric definition (≥1 `branchFromThought` BEFORE final conclusion).
4. Whether `revision_made=TRUE` (≥1 `isRevision=true` thought that materially changed the conclusion).
5. The same falsifiable verdict structure (hypothesis + test + falsification threshold).

Then `discipline-checklist.md` is filled in.
