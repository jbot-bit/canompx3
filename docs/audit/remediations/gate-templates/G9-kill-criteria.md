# G9 — Kill-Criteria Certificate

**Candidate:** ________________________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`

---

## Purpose

Kill criteria must be written BEFORE the first run. Post-hoc kill criteria are post-hoc rescue (if they save a failing run) or post-hoc rejection (if they kill a passing run). Both violate pre-registration discipline per `backtesting-methodology.md` RULE 3.5 (commit `631bda30`).

## Pre-registered kill criteria (verbatim from pre-reg yaml)

```yaml
<paste the pre-reg's kill_criteria / decision_rules / verdicts_per_* block>
```

## Evidence of write-before-run ordering

- Pre-reg commit SHA (kill criteria locked): ______________________________
- Pre-reg commit date: ______________________________
- First candidate run commit SHA: ______________________________
- First candidate run date: ______________________________
- Pre-reg lock < first run: [ ] YES [ ] NO (G9 FAIL)

Evidence:
```
$ git log --oneline --follow <pre-reg yaml>
<paste output>
$ git log --oneline --follow <candidate script>
<paste output>
```

## Kill criteria completeness check

- [ ] Criteria cover H1 (primary hypothesis) fail
- [ ] Criteria cover downstream non-waivable gate fail (C6 WFE, C8 OOS, C9 era)
- [ ] Criteria cover per-lane heterogeneity fail (per memory `feedback_per_lane_breakdown_required.md` — ≥25% cell-flip)
- [ ] Criteria cover positive-control fail (framework broken — G7 Control 3)
- [ ] Criteria are binary (pass/fail), not judgment calls

## No post-hoc additions check

If any kill criterion was ADDED after first run, this is post-hoc REJECTION (RULE 3.5 class). Check diff:

```
$ git diff <pre-reg-lock-sha> HEAD -- <pre-reg yaml>
<paste output — should show no new kill criteria>
```

- [ ] Zero kill criteria added post-lock
- [ ] OR: criteria added are purely cosmetic (typo fixes, clarifications) — reviewable in the diff

## Verdict

- [ ] CLEAR — kill criteria pre-committed, completeness-checked, no post-hoc additions
- [ ] FAIL — missing criteria class OR post-hoc additions present

## Failure disposition

FAIL → pre-reg must be re-locked with complete kill criteria. Current run's result stands as informational only, not confirmatory.

## Literature citation

- `.claude/rules/backtesting-methodology.md` RULE 3.5 (post-hoc rejection)
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` §3 (pre-registration discipline)

## Authored by / committed

- Author: ____________________________
- Commit SHA of this certificate: ________________
