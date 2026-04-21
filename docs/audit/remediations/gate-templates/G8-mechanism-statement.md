# G8 — Mechanism Statement Certificate

**Candidate:** ________________________

---

## Purpose

Every edge must have a written mechanism hypothesis cited to local literature OR labelled UNGROUNDED (training-memory). Prevents the "works in backtest, no trader story" class of over-fit finding.

## Mechanism statement

In 2-4 sentences, state WHY the candidate should work in live trading.

```
<paste mechanism statement>
```

## Literature grounding

Choose ONE of:

### A — LOCAL-LITERATURE-GROUNDED

- [ ] Mechanism is supported by a passage in `docs/institutional/literature/________________.md` at page/section ______
- [ ] Passage verbatim: ________________________________________________________________
- [ ] The passage makes the mechanism claim; the candidate applies the claim to a specific market setup.

### B — PROJECT-CANON-GROUNDED

- [ ] Mechanism is supported by a prior verified finding in:
  - [ ] `memory/________.md` (must cite the specific verification, not just the claim)
  - [ ] `docs/audit/results/________.md` commit SHA ________
  - [ ] `docs/institutional/mechanism_priors.md` §______

### C — UNGROUNDED (training memory)

- [ ] Mechanism is based on training-memory / general intuition, NO local literature or project canon
- [ ] Candidate is explicitly flagged UNGROUNDED in all downstream documentation

Per CLAUDE.md Local Academic / Project-Source Grounding Rule, UNGROUNDED candidates cannot advance past RESEARCH_SURVIVOR without first earning local-literature grounding.

## Mechanism-prior-hierarchy check (`mechanism_priors.md` §4)

If mechanism involves a PRE-ENTRY FILTER role, check hierarchy:
- R1 FILTER (binary gate on instrument-level feature)
- R2 REGIME ALLOCATOR (capital-weight by volatility regime)
- R3 POSITION SIZE (continuous sizer)

Carver 2015 prior: tail/binary effects prefer R1; continuous gradients prefer R3.

- [ ] Candidate's proposed role matches the effect shape (e.g., Q5-only = R1 filter preferred over linear sizer when effect is tail-concentrated)
- [ ] If mismatched, candidate is re-framed to the preferred role OR justifies the mismatch with evidence

## Verdict

- [ ] GROUNDED — local literature or verified project-canon cite present
- [ ] UNGROUNDED — flagged, cannot advance past RESEARCH_SURVIVOR

## Literature citation

- `docs/institutional/mechanism_priors.md`
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` Ch 10 (tail effect prefers filter over sizer)
- CLAUDE.md Local Academic / Project-Source Grounding Rule

## Authored by / committed

- Author: ____________________________
- Commit SHA of this certificate: ________________
